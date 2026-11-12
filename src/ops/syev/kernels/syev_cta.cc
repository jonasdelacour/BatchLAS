// SPDX-License-Identifier: MIT
#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/mempool.hh>
#include <batchlas/backend_config.h>

#include <src/queue.hh>
#include <src/select/grid_chunk.hh>
#include <src/util/template-instantiations.hh>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <iostream>

#include <src/math-helpers.hh>
#include <src/util/batch_mask.hh>
#include <batchlas/util/env.hh>
#include <batchlas/settings.hh>
#include <batchlas/tuning_params.hh>
#include <batchlas/device.hh>
#include <batchlas/device/sg_partition.hh>
#include <batchlas/device/lapack/syev.hh>

namespace batchlas {


namespace {

// Diagnostics, not a safety override: it prints the stage name and drains the
// pipeline, removing no check and selecting no different kernel, so it is
// deliberately NOT one of the knobs BATCHLAS_ALLOW_UNSAFE_ENV gates. Parsed by
// env_truthy in settings.cc, exactly as here before.
inline bool cta_debug_sync_enabled() {
    return batchlas::settings().diagnostics.cta_debug_sync;
}

inline void cta_debug_log(const char* stage) {
    if (!cta_debug_sync_enabled()) return;
    std::cerr << "[cta-debug] " << stage << std::endl;
}

inline void cta_debug_sync(Queue& ctx, const char* stage) {
    if (!cta_debug_sync_enabled()) return;
    cta_debug_log(stage);
    try {
        ctx.wait_and_throw();
    } catch (const std::exception& e) {
        std::cerr << "[cta-debug] failure at stage: " << stage << "\n";
        std::cerr << "[cta-debug] exception: " << e.what() << std::endl;
        throw;
    }
}

inline std::byte* align_up(std::byte* p, std::size_t align) {
    const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(p);
    const std::uintptr_t aligned = (addr % align == 0) ? addr : ((addr + align - 1) & ~(align - 1));
    return reinterpret_cast<std::byte*>(aligned);
}

template <typename T>
Span<T> ws_alloc(Queue& ctx, Span<std::byte> ws, std::size_t& offset_bytes, std::size_t count) {
    if (count == 0) return {};

    const std::size_t align = BumpAllocator::alignment<T>(ctx.device());
    std::byte* base = ws.data();
    std::byte* p = base + offset_bytes;
    p = align_up(p, align);

    const std::size_t used_for_align = static_cast<std::size_t>(p - (base + offset_bytes));
    if (used_for_align > (ws.size() - offset_bytes)) {
        throw batchlas::workspace_error("syev_cta: workspace alignment overflow.");
    }
    offset_bytes += used_for_align;

    const std::size_t bytes_needed = count * sizeof(T);
    const std::size_t padded = BumpAllocator::allocation_size<T>(ctx, count);

    if (padded > (ws.size() - offset_bytes)) {
        throw batchlas::workspace_error("syev_cta: insufficient workspace.");
    }

    T* out = reinterpret_cast<T*>(base + offset_bytes);
    offset_bytes += padded;
    return Span<T>(out, count);
}

Span<std::byte> ws_remaining(Span<std::byte> ws, std::size_t offset_bytes) {
    if (offset_bytes > ws.size()) {
        throw batchlas::workspace_error("syev_cta: workspace bookkeeping overflow.");
    }
    return Span<std::byte>(ws.data() + offset_bytes, ws.size() - offset_bytes);
}


// The steqr parameters syev_cta solves its real tridiagonal with: back_transform off (the
// SYTRD back-transform is explicit), the work-group multiplier wg_mult, and, when the caller
// left max_sweeps and the shift strategy at their defaults, 400 / Wilkinson (the CTA STEQR
// defaults are speed-oriented; large batches want the robust pair). A complex T copies the
// fields over; its cta_update_scheme stays at the default.
template <typename T>
SteqrParams<typename base_type<T>::type> cta_steqr_params(SteqrParams<T> p, size_t wg_mult) {
    using Real = typename base_type<T>::type;
    const SteqrParams<T> defaults{};
    if (p.max_sweeps == defaults.max_sweeps && p.cta_shift_strategy == defaults.cta_shift_strategy) {
        p.max_sweeps = 400;
        p.cta_shift_strategy = SteqrShiftStrategy::Wilkinson;
    }
    SteqrParams<Real> r;
    if constexpr (detail::is_complex<T>::value) {
        r.block_size = p.block_size;
        r.max_sweeps = p.max_sweeps;
        r.block_rotations = p.block_rotations;
        r.sort = p.sort;
        r.transpose_working_vectors = p.transpose_working_vectors;
        r.sort_order = p.sort_order;
        r.cta_shift_strategy = p.cta_shift_strategy;
    } else {
        r = p;
    }
    r.back_transform = false;
    r.cta_wg_size_multiplier = wg_mult;
    return r;
}

template <Backend B, typename T>
Event syev_cta_part(Queue& ctx,
               const MatrixView<T, MatrixFormat::Dense>& a_in,
               Span<typename base_type<T>::type> eigenvalues,
               JobType jobz,
               Uplo uplo,
               const Span<std::byte>& ws,
               SteqrParams<T> steqr_params,
               size_t cta_wg_size_multiplier,
               Span<int32_t> info) {
    cta_debug_log("syev_cta: entry");
    if (a_in.rows() != a_in.cols()) {
        throw batchlas::invalid_argument("syev_cta: A must be square.");
    }
    if (jobz != JobType::NoEigenVectors && jobz != JobType::EigenVectors) {
        throw batchlas::invalid_argument("syev_cta: invalid JobType.");
    }

    const int64_t n64 = a_in.rows();
    const int64_t batch64 = a_in.batch_size();

    if (n64 < 1 || n64 > 32) {
        throw batchlas::invalid_argument("syev_cta currently supports 1 <= n <= 32.");
    }
    const int32_t n = static_cast<int32_t>(n64);
    const int32_t batch = static_cast<int32_t>(batch64);

    if (eigenvalues.size() < static_cast<std::size_t>(n) * static_cast<std::size_t>(batch)) {
        throw batchlas::invalid_argument("syev_cta: eigenvalues span too small for n*batch.");
    }

    // We overwrite A only when jobz==EigenVectors.
    auto& a = const_cast<MatrixView<T, MatrixFormat::Dense>&>(a_in);
    // The workspace views are rebuilt from raw pointers, so they take the part's mask back on.
    const Span<const std::uint8_t> mask_span = ::batchlas::detail::call_mask("syev_cta", a_in);
    const std::uint8_t* mask = ::batchlas::detail::mask_ptr(mask_span);

    // NOTE: the CTA SYTRD/SYEV Uplo::Lower path has a known correctness defect, so Lower
    // input is symmetrized (A_upper := conj(A_lower)) and runs the known-good Upper pipeline.
    // evidence: docs/design/known-defects.md#known-defects-the-cta-sytrd-lower-path
    Uplo uplo_eff = uplo;
    if (uplo == Uplo::Lower) {
        uplo_eff = Uplo::Upper;
        ctx->submit([&](sycl::handler& cgh) {
            auto A = a.kernel_view();
            const int32_t nn = n;
            const int32_t nb = batch;
            const int64_t total = static_cast<int64_t>(nb) * nn * nn;
            cgh.parallel_for(sycl::range<1>(static_cast<std::size_t>(total)), [=](sycl::id<1> tid) {
                const int64_t idx = static_cast<int64_t>(tid[0]);
                const int32_t b = device::detail::batched_item(idx, nn);
                if (mask != nullptr && mask[b] == 0) return;  // I3
                const auto [row, col] = device::detail::batched_rc(idx, b, nn);
                // The strict upper triangle from the lower (conj), Im(diag) dropped: element (row, col).
                device::mirror<Uplo::Lower, true>(device::ElemAt{row, col}, device::item(A, b, nn, nn));
            });
        });
    }

    // Workspace is treated as mutable storage by the implementation.
    Span<std::byte> ws_mut(const_cast<std::byte*>(ws.data()), ws.size());

    using Real = typename base_type<T>::type;
    constexpr bool cplx = detail::is_complex<T>::value;
    const std::size_t nb = static_cast<std::size_t>(batch);
    const std::size_t nn2 = static_cast<std::size_t>(n) * static_cast<std::size_t>(n) * nb;

    const SteqrParams<Real> steqr_params_local = cta_steqr_params(steqr_params, cta_wg_size_multiplier);

    // Workspace layout (syev_cta_part_buffer_size sums the same list, in this order):
    //  - d, e, tau (n, n-1, n-1) in T: SYTRD's tridiagonal (Hermitian for a complex T)
    //  - complex T only: d, e (n, n-1) real (|e| after the phase similarity) and S (n), the
    //    unit-modulus diagonal of that similarity
    //  - Z (n x n) real   (STEQR eigenvectors of T)
    //  - complex T only: ZC (n x n), S Z, then back-transformed
    //  - A_q (n x n)      (packed reflectors in QR/QL-compatible layout), tau_q (n)
    //  - remaining passed to steqr
    std::size_t ws_off = 0;
    auto d_t_span = ws_alloc<T>(ctx, ws_mut, ws_off, static_cast<std::size_t>(n) * nb);
    auto e_t_span = ws_alloc<T>(ctx, ws_mut, ws_off, static_cast<std::size_t>(n - 1) * nb);
    auto tau_span = ws_alloc<T>(ctx, ws_mut, ws_off, static_cast<std::size_t>(n - 1) * nb);
    const VectorView<T> d_t_view = VectorView<T>(d_t_span, /*size=*/n, /*batch_size=*/batch, /*inc=*/1, /*stride=*/n).with_batch_mask(mask_span);
    const VectorView<T> e_t_view = VectorView<T>(e_t_span, /*size=*/n - 1, /*batch_size=*/batch, /*inc=*/1, /*stride=*/n - 1).with_batch_mask(mask_span);
    const VectorView<T> tau_view = VectorView<T>(tau_span, /*size=*/n - 1, /*batch_size=*/batch, /*inc=*/1, /*stride=*/n - 1).with_batch_mask(mask_span);

    struct RealTridiag {
        VectorView<Real> d, e;
        VectorView<T> s;
    };
    const RealTridiag rt = [&]() {
        if constexpr (cplx) {
            auto d_span = ws_alloc<Real>(ctx, ws_mut, ws_off, static_cast<std::size_t>(n) * nb);
            auto e_span = ws_alloc<Real>(ctx, ws_mut, ws_off, static_cast<std::size_t>(n - 1) * nb);
            auto phase_span = ws_alloc<T>(ctx, ws_mut, ws_off, static_cast<std::size_t>(n) * nb);
            return RealTridiag{VectorView<Real>(d_span, /*size=*/n, /*batch_size=*/batch, /*inc=*/1, /*stride=*/n).with_batch_mask(mask_span),
                               VectorView<Real>(e_span, /*size=*/n - 1, /*batch_size=*/batch, /*inc=*/1, /*stride=*/n - 1).with_batch_mask(mask_span),
                               VectorView<T>(phase_span, /*size=*/n, /*batch_size=*/batch, /*inc=*/1, /*stride=*/n)};
        } else {
            return RealTridiag{d_t_view, e_t_view, VectorView<T>()};
        }
    }();
    const VectorView<Real>& d_view = rt.d;
    const VectorView<Real>& e_view = rt.e;
    const VectorView<T>& phase_view = rt.s;

    auto z_span = ws_alloc<Real>(ctx, ws_mut, ws_off, nn2);
    const MatrixView<Real, MatrixFormat::Dense> z_view = MatrixView<Real, MatrixFormat::Dense>(
        z_span.data(), n, n, /*ld=*/n, /*stride=*/n * n, /*batch_size=*/batch).with_batch_mask(mask_span);
    const MatrixView<T, MatrixFormat::Dense> zc_view = [&]() {
        if constexpr (cplx) {
            auto zc_span = ws_alloc<T>(ctx, ws_mut, ws_off, nn2);
            return MatrixView<T, MatrixFormat::Dense>(zc_span.data(), n, n, /*ld=*/n, /*stride=*/n * n, /*batch_size=*/batch)
                .with_batch_mask(mask_span);
        } else {
            return z_view;
        }
    }();
    auto aq_span = ws_alloc<T>(ctx, ws_mut, ws_off, nn2);
    auto tau_q_span = ws_alloc<T>(ctx, ws_mut, ws_off, static_cast<std::size_t>(n) * nb);
    const MatrixView<T, MatrixFormat::Dense> aq_view = MatrixView<T, MatrixFormat::Dense>(
        aq_span.data(), n, n, /*ld=*/n, /*stride=*/n * n, /*batch_size=*/batch).with_batch_mask(mask_span);
    const VectorView<T> tau_q_view = VectorView<T>(tau_q_span, /*size=*/n, /*batch_size=*/batch, /*inc=*/1, /*stride=*/n).with_batch_mask(mask_span);

    // Reduce A to tridiagonal: A overwritten with reflectors/tridiagonal.
    // Note: sytrd_cta's workspace is currently unused.
    // (void) on an Event: deliberate. This Queue is in-order, so the next submission
    // is already ordered after this one and the Event carries nothing the caller needs.
    (void)sytrd_cta<B, T>(ctx, a, d_t_view, e_t_view, tau_view, uplo_eff, Span<std::byte>(), cta_wg_size_multiplier);
    cta_debug_sync(ctx, "syev_cta: after sytrd_cta");

    if constexpr (cplx) {
        // Hermitian tridiagonal (complex off-diagonal) -> real symmetric tridiagonal by a
        // diagonal unitary similarity: T' = S^H T S with real off-diagonal |e|. S (phase_view)
        // lifts the eigenvectors back: v_T = S * v_T'. A real T runs steqr on SYTRD's (d, e).
        ctx->submit([&](sycl::handler& cgh) {
            auto D = d_t_view;
            auto E = e_t_view;
            auto Dr = d_view;
            auto Er = e_view;
            auto S = phase_view;
            const int32_t nn = n;
            const Uplo u = uplo_eff;
            cgh.parallel_for(sycl::range<1>(nb), [=](sycl::id<1> tid) {
                const int32_t b = static_cast<int32_t>(tid[0]);
                if (mask != nullptr && mask[b] == 0) return;  // I3
                // The subdiagonal becomes real; S is always recorded here.
                device::detail::phase_similarity(device::Thread{}, nn, device::item(D, b), device::item(E, b), device::item(Dr, b),
                                                 device::item(Er, b), device::item(S, b), u, /*want_s=*/true);
            });
        });
        cta_debug_sync(ctx, "syev_cta: after complex phase transform");
    }

    // Solve the tridiagonal eigenproblem in real arithmetic; Z from identity when
    // jobz == EigenVectors (back_transform is off: the SYTRD back-transform is explicit below).
    const VectorView<Real> evals_view = VectorView<Real>(eigenvalues.data(), /*size=*/n, /*batch_size=*/batch,
                                                         /*inc=*/1, /*stride=*/n).with_batch_mask(mask_span);
    auto steqr_ws = ws_remaining(ws_mut, ws_off);
    // syev_cta owns no convergence criterion of its own: the whole of it is this call, so
    // `info` is forwarded rather than reduced. steqr clears the span itself
    // (src/kernels/common/info_span.hh), and exactly one steqr call runs per syev_cta call.
    (void)steqr<B, Real>(ctx, d_view, e_view, evals_view, steqr_ws, jobz, steqr_params_local, z_view, info);
    cta_debug_sync(ctx, "syev_cta: after steqr");

    if (jobz == JobType::EigenVectors) {
        if constexpr (cplx) {
            // Lift real Z to complex and apply the diagonal phase scaling: Zc = S * Z.
            ctx->submit([&](sycl::handler& cgh) {
                auto Z = z_view.kernel_view();
                auto ZC = zc_view.kernel_view();
                auto S = phase_view;
                const int32_t nn = n;
                const int64_t total = static_cast<int64_t>(batch) * nn * nn;
                cgh.parallel_for(sycl::range<1>(static_cast<std::size_t>(total)), [=](sycl::id<1> tid) {
                    const int64_t idx = static_cast<int64_t>(tid[0]);
                    const int32_t b = device::detail::batched_item(idx, nn);
                    if (mask != nullptr && mask[b] == 0) return;  // I3
                    const auto [row, col] = device::detail::batched_rc(idx, b, nn);
                    device::detail::dgmm_left_at(device::item(S, b), device::item(Z, b, nn, nn), device::item(ZC, b, nn, nn), row, col);
                });
            });
            cta_debug_sync(ctx, "syev_cta: after lift Z and phase");
        }

        // Pack SYTRD Householder reflectors into a QR/QL-compatible layout for ormqx_cta.
        //
        // Lower (SYTRD Lower): reflectors act on trailing submatrices starting at row i+1.
        // We encode them as QR reflectors in columns 1..n-1 (column 0 is identity), i.e.
        // H(1) is a no-op and H(2..n) are the SYTRD reflectors.
        //
        // Upper (SYTRD Upper): reflectors act on leading submatrices ending at row k-1.
        // We encode them as QL reflectors in columns 0..n-2 (column n-1 is identity).
        // A real T folds the TAUQ shift into this kernel (row 0 writes it: one thread per
        // column already exists, a separate n*batch launch is pure overhead); a complex T
        // packs TAUQ in a second launch.
        ctx->submit([&](sycl::handler& cgh) {
            auto A = a.kernel_view();
            auto AQ = aq_view.kernel_view();
            auto TAU = tau_view;
            auto TAUQ = tau_q_view;
            const int32_t nn = n;
            const int64_t total = static_cast<int64_t>(batch) * static_cast<int64_t>(nn) * static_cast<int64_t>(nn);
            cgh.parallel_for(sycl::range<1>(static_cast<std::size_t>(total)), [=](sycl::id<1> tid) {
                const int64_t idx = static_cast<int64_t>(tid[0]);
                const int32_t b = device::detail::batched_item(idx, nn);
                if (mask != nullptr && mask[b] == 0) return;  // I3
                const auto [row, col] = device::detail::batched_rc(idx, b, nn);
                if constexpr (cplx) {
                    device::detail::sytrd_reflector_pack_at(device::item(A, b, nn, nn), device::item(AQ, b, nn, nn), uplo_eff, nn, row, col);
                } else {
                    device::detail::sytrd_reflector_pack_at(device::item(A, b, nn, nn), device::item(AQ, b, nn, nn), uplo_eff, nn, row, col,
                                                            device::item(TAU, b), device::item(TAUQ, b));
                }
            });
        });
        cta_debug_sync(ctx, "syev_cta: after pack AQ");

        if constexpr (cplx) {
            ctx->submit([&](sycl::handler& cgh) {
                auto TAU = tau_view;
                auto TAUQ = tau_q_view;
                const int32_t nn = n;
                const int64_t total = static_cast<int64_t>(batch) * static_cast<int64_t>(nn);
                cgh.parallel_for(sycl::range<1>(static_cast<std::size_t>(total)), [=](sycl::id<1> tid) {
                    const int64_t idx = static_cast<int64_t>(tid[0]);
                    const int32_t b = static_cast<int32_t>(idx / nn);
                    if (mask != nullptr && mask[b] == 0) return;  // I3
                    const int32_t j = static_cast<int32_t>(idx - static_cast<int64_t>(b) * nn);
                    device::detail::sytrd_tau_pack_at(device::item(TAU, b), device::item(TAUQ, b), uplo_eff, nn, j);
                });
            });
            cta_debug_sync(ctx, "syev_cta: after pack TAUQ");
        }

        // Apply Q (from SYTRD reflectors) to Z: Z := Q * Z.
        const Uplo ormq_factorization = (uplo_eff == Uplo::Lower) ? Uplo::Upper : Uplo::Lower;
        (void)ormqx_cta<B, T>(ctx,
                              aq_view,
                              tau_q_view,
                              zc_view,
                              ormq_factorization,
                              Side::Left,
                              Transpose::NoTrans,
                              /*k=*/n,
                              Span<std::byte>(),
                              cta_wg_size_multiplier);
        cta_debug_sync(ctx, "syev_cta: after ormqx_cta");

        // Overwrite A with eigenvectors.
        (void)MatrixView<T, MatrixFormat::Dense>::copy(ctx, a, zc_view);
        cta_debug_sync(ctx, "syev_cta: after copy eigenvectors out");
    }

    return ctx.get_event();
}

template <Backend B, typename T>
size_t syev_cta_part_buffer_size(Queue& ctx,
                            const MatrixView<T, MatrixFormat::Dense>& a,
                            JobType jobz,
                            SteqrParams<T> steqr_params) {
    using Real = typename base_type<T>::type;
    constexpr bool cplx = detail::is_complex<T>::value;
    if (a.rows() != a.cols()) {
        throw batchlas::invalid_argument("syev_cta_buffer_size: A must be square.");
    }

    const int64_t n64 = a.rows();
    const int64_t batch64 = a.batch_size();
    if (n64 < 1 || n64 > 32) {
        throw batchlas::invalid_argument("syev_cta_buffer_size currently supports 1 <= n <= 32.");
    }

    const int32_t n = static_cast<int32_t>(n64);
    const int32_t batch = static_cast<int32_t>(batch64);

    // The fixed allocations of syev_cta_part, in its order, before the steqr workspace.
    const std::size_t d_count = static_cast<std::size_t>(n) * static_cast<std::size_t>(batch);
    const std::size_t e_count = static_cast<std::size_t>(n - 1) * static_cast<std::size_t>(batch);
    const std::size_t z_count = static_cast<std::size_t>(n) * static_cast<std::size_t>(n) * static_cast<std::size_t>(batch);

    size_t bytes = 0;
    bytes += BumpAllocator::allocation_size<T>(ctx, d_count);
    bytes += BumpAllocator::allocation_size<T>(ctx, e_count);
    bytes += BumpAllocator::allocation_size<T>(ctx, e_count);  // tau
    if constexpr (cplx) {
        bytes += BumpAllocator::allocation_size<Real>(ctx, d_count);
        bytes += BumpAllocator::allocation_size<Real>(ctx, e_count);
        bytes += BumpAllocator::allocation_size<T>(ctx, d_count);  // S
    }
    bytes += BumpAllocator::allocation_size<Real>(ctx, z_count);
    if constexpr (cplx) bytes += BumpAllocator::allocation_size<T>(ctx, z_count);  // ZC
    bytes += BumpAllocator::allocation_size<T>(ctx, z_count);                      // A_q
    bytes += BumpAllocator::allocation_size<T>(ctx, d_count);                      // tau_q

    // Remaining: the steqr workspace, with the parameters the call uses (the multiplier the
    // call is given is not known here: steqr_params' own).
    VectorView<Real> d_dummy(nullptr, /*size=*/n, /*batch_size=*/batch, /*inc=*/1, /*stride=*/n);
    VectorView<Real> e_dummy(nullptr, /*size=*/n - 1, /*batch_size=*/batch, /*inc=*/1, /*stride=*/n - 1);
    VectorView<Real> evals_dummy(nullptr, /*size=*/n, /*batch_size=*/batch, /*inc=*/1, /*stride=*/n);
    const SteqrParams<Real> steqr_params_local = cta_steqr_params(steqr_params, steqr_params.cta_wg_size_multiplier);
    bytes += steqr_buffer_size<Real>(ctx, d_dummy, e_dummy, evals_dummy, jobz, steqr_params_local);

    return bytes;
}

// Items per pass: sytrd_cta, steqr and ormqx_cta hand A, Z and the packed reflectors (three n x n
// matrices) between launches through L2. The item is those three, not the part workspace: most of
// that is steqr_wg's rotation store, which the CTA kernels never touch. The query and the call must
// agree, so both use it. evidence: docs/perf/syev.md#syev-l2-batch-chunking
template <typename T>
int syev_cta_chunk(Queue& ctx, int n, int batch) {
    const auto dev = ctx.device();
    return select::grid_capped_chunk(ctx, static_cast<int>(tuning::blocked_chunk_items(
        batch, 3 * static_cast<int64_t>(n) * n * static_cast<int64_t>(sizeof(T)),
        static_cast<int64_t>(dev.get_property(DeviceProperty::GLOBAL_MEM_CACHE_SIZE)),
        static_cast<int64_t>(dev.get_property(DeviceProperty::MAX_COMPUTE_UNITS)),
        tuning::SYEV_CTA_L2_CHUNK_PERCENT)), batch);
}

} // namespace

template <Backend B, typename T>
Event syev_cta(Queue& ctx,
               const MatrixView<T, MatrixFormat::Dense>& a_in,
               Span<typename base_type<T>::type> eigenvalues,
               JobType jobz,
               Uplo uplo,
               const Span<std::byte>& ws,
               SteqrParams<T> steqr_params,
               size_t cta_wg_size_multiplier,
               Span<int32_t> info) {
    (void)::batchlas::detail::call_mask("syev_cta", a_in);  // a_in.batch_range carries each part's slice
    const int n = static_cast<int>(a_in.rows());
    const int batch = static_cast<int>(a_in.batch_size());
    if (n < 1 || n > 32 || batch < 1 || a_in.rows() != a_in.cols())
        return syev_cta_part<B, T>(ctx, a_in, eigenvalues, jobz, uplo, ws, steqr_params, cta_wg_size_multiplier, info);
    if (eigenvalues.size() < static_cast<std::size_t>(n) * static_cast<std::size_t>(batch))
        throw batchlas::invalid_argument("syev_cta: eigenvalues span too small for n*batch.");
    // Each part is a complete solve of its items, so the parts are independent; they share the
    // one part-sized workspace, hence the wait between parts on an out-of-order queue.
    const int chunk = syev_cta_chunk<T>(ctx, n, batch);
    const bool info_requested = info.size() >= static_cast<std::size_t>(batch);
    for (int b0 = 0, cb = 0; b0 < batch; b0 += cb) {
        cb = batch - b0 < 2 * chunk ? batch - b0 : chunk;   // the last part takes the remainder
        if (b0 > 0 && !ctx.in_order()) ctx.wait();
        (void)syev_cta_part<B, T>(
            ctx, a_in.batch_range(b0, cb),
            eigenvalues.subspan(static_cast<std::size_t>(b0) * n, static_cast<std::size_t>(cb) * n), jobz, uplo, ws,
            steqr_params, cta_wg_size_multiplier,
            info_requested ? info.subspan(static_cast<std::size_t>(b0), static_cast<std::size_t>(cb)) : Span<int32_t>());
    }
    return ctx.get_event();
}

template <Backend B, typename T>
size_t syev_cta_buffer_size(Queue& ctx,
                            const MatrixView<T, MatrixFormat::Dense>& a,
                            JobType jobz,
                            SteqrParams<T> steqr_params) {
    const int n = static_cast<int>(a.rows());
    const int batch = static_cast<int>(a.batch_size());
    if (n < 1 || n > 32 || batch < 1 || a.rows() != a.cols())
        return syev_cta_part_buffer_size<B, T>(ctx, a, jobz, steqr_params);
    const int chunk = syev_cta_chunk<T>(ctx, n, batch);
    return syev_cta_part_buffer_size<B, T>(ctx, a.batch_range(0, chunk + batch % chunk), jobz, steqr_params);
}

#define SYEV_CTA_INSTANTIATE(back, fp) \
    template Event syev_cta<back, BATCHLAS_UNPAREN fp>(Queue&, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
                                                       Span<typename base_type<BATCHLAS_UNPAREN fp>::type>, JobType, Uplo, \
                                                       const Span<std::byte>&, SteqrParams<BATCHLAS_UNPAREN fp>, size_t, Span<int32_t>); \
    template size_t syev_cta_buffer_size<back, BATCHLAS_UNPAREN fp>(Queue&, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
                                                                    JobType, SteqrParams<BATCHLAS_UNPAREN fp>);

    BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(SYEV_CTA_INSTANTIATE)

#undef SYEV_CTA_INSTANTIATE

} // namespace batchlas
