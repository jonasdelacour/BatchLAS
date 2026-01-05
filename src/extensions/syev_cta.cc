#include <blas/extensions.hh>
#include <blas/functions.hh>
#include <blas/matrix.hh>
#include <util/mempool.hh>
#include <batchlas/backend_config.h>

#include "../queue.hh"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include "../math-helpers.hh"

namespace batchlas {

namespace {

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
        throw std::runtime_error("syev_cta: workspace alignment overflow.");
    }
    offset_bytes += used_for_align;

    const std::size_t bytes_needed = count * sizeof(T);
    const std::size_t padded = BumpAllocator::allocation_size<T>(ctx, count);

    if (padded > (ws.size() - offset_bytes)) {
        throw std::runtime_error("syev_cta: insufficient workspace.");
    }

    T* out = reinterpret_cast<T*>(base + offset_bytes);
    offset_bytes += padded;
    return Span<T>(out, count);
}

Span<std::byte> ws_remaining(Span<std::byte> ws, std::size_t offset_bytes) {
    if (offset_bytes > ws.size()) {
        throw std::runtime_error("syev_cta: workspace bookkeeping overflow.");
    }
    return Span<std::byte>(ws.data() + offset_bytes, ws.size() - offset_bytes);
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
               size_t cta_wg_size_multiplier) {
    if (a_in.rows() != a_in.cols()) {
        throw std::invalid_argument("syev_cta: A must be square.");
    }
    if (jobz != JobType::NoEigenVectors && jobz != JobType::EigenVectors) {
        throw std::invalid_argument("syev_cta: invalid JobType.");
    }

    const int64_t n64 = a_in.rows();
    const int64_t batch64 = a_in.batch_size();

    if (n64 < 1 || n64 > 32) {
        throw std::invalid_argument("syev_cta currently supports 1 <= n <= 32.");
    }
    const int32_t n = static_cast<int32_t>(n64);
    const int32_t batch = static_cast<int32_t>(batch64);

    if (eigenvalues.size() < static_cast<std::size_t>(n) * static_cast<std::size_t>(batch)) {
        throw std::invalid_argument("syev_cta: eigenvalues span too small for n*batch.");
    }

    // We overwrite A only when jobz==EigenVectors.
    auto& a = const_cast<MatrixView<T, MatrixFormat::Dense>&>(a_in);

    // Workspace is treated as mutable storage by the implementation.
    Span<std::byte> ws_mut(const_cast<std::byte*>(ws.data()), ws.size());

    if constexpr (internal::is_complex<T>::value) {
        using Real = typename base_type<T>::type;

        // Workspace layout (complex Hermitian):
        //  - d_c (n) complex (diag should be real, imag~0)
        //  - e_c (n-1) complex (Hermitian tridiagonal off-diagonal, generally complex)
        //  - tau_c (n-1) complex
        //  - d (n) real
        //  - e (n-1) real (|offdiag| after diagonal phase similarity)
        //  - phase (n) complex (unit-modulus diagonal scaling)
        //  - Z (n x n) real        (STEQR eigenvectors of T)
        //  - Zc (n x n) complex    (lifted Z, then back-transformed)
        //  - A_q (n x n) complex   (packed reflectors)
        //  - tau_q (n) complex
        //  - remaining passed to steqr_cta<Real>
        std::size_t ws_off = 0;

        auto d_c_span = ws_alloc<T>(ctx, ws_mut, ws_off, static_cast<std::size_t>(n) * static_cast<std::size_t>(batch));
        auto e_c_span = ws_alloc<T>(ctx, ws_mut, ws_off, static_cast<std::size_t>(n - 1) * static_cast<std::size_t>(batch));
        auto tau_c_span = ws_alloc<T>(ctx, ws_mut, ws_off, static_cast<std::size_t>(n - 1) * static_cast<std::size_t>(batch));

        auto d_span = ws_alloc<Real>(ctx, ws_mut, ws_off, static_cast<std::size_t>(n) * static_cast<std::size_t>(batch));
        auto e_span = ws_alloc<Real>(ctx, ws_mut, ws_off, static_cast<std::size_t>(n - 1) * static_cast<std::size_t>(batch));

        auto phase_span = ws_alloc<T>(ctx, ws_mut, ws_off, static_cast<std::size_t>(n) * static_cast<std::size_t>(batch));

        auto z_span = ws_alloc<Real>(ctx, ws_mut, ws_off, static_cast<std::size_t>(n) * static_cast<std::size_t>(n) * static_cast<std::size_t>(batch));
        auto zc_span = ws_alloc<T>(ctx, ws_mut, ws_off, static_cast<std::size_t>(n) * static_cast<std::size_t>(n) * static_cast<std::size_t>(batch));
        auto aq_span = ws_alloc<T>(ctx, ws_mut, ws_off, static_cast<std::size_t>(n) * static_cast<std::size_t>(n) * static_cast<std::size_t>(batch));
        auto tau_q_span = ws_alloc<T>(ctx, ws_mut, ws_off, static_cast<std::size_t>(n) * static_cast<std::size_t>(batch));

        VectorView<T> d_c_view(d_c_span, /*size=*/n, /*batch_size=*/batch, /*inc=*/1, /*stride=*/n);
        VectorView<T> e_c_view(e_c_span, /*size=*/n - 1, /*batch_size=*/batch, /*inc=*/1, /*stride=*/n - 1);
        VectorView<T> tau_c_view(tau_c_span, /*size=*/n - 1, /*batch_size=*/batch, /*inc=*/1, /*stride=*/n - 1);

        VectorView<Real> d_view(d_span, /*size=*/n, /*batch_size=*/batch, /*inc=*/1, /*stride=*/n);
        VectorView<Real> e_view(e_span, /*size=*/n - 1, /*batch_size=*/batch, /*inc=*/1, /*stride=*/n - 1);
        VectorView<T> phase_view(phase_span, /*size=*/n, /*batch_size=*/batch, /*inc=*/1, /*stride=*/n);

        MatrixView<Real, MatrixFormat::Dense> z_view(z_span.data(), n, n, /*ld=*/n,
                                                    /*stride=*/n * n, /*batch_size=*/batch);
        MatrixView<T, MatrixFormat::Dense> zc_view(zc_span.data(), n, n, /*ld=*/n,
                                                  /*stride=*/n * n, /*batch_size=*/batch);
        MatrixView<T, MatrixFormat::Dense> aq_view(aq_span.data(), n, n, /*ld=*/n,
                                                  /*stride=*/n * n, /*batch_size=*/batch);
        VectorView<T> tau_q_view(tau_q_span, /*size=*/n, /*batch_size=*/batch, /*inc=*/1, /*stride=*/n);

        // Reduce Hermitian A to Hermitian tridiagonal (d_c real-ish, e_c generally complex).
        sytrd_cta<B, T>(ctx, a, d_c_view, e_c_view, tau_c_view, uplo, Span<std::byte>(), cta_wg_size_multiplier);

        // Convert Hermitian tridiagonal (complex off-diagonal) to real symmetric tridiagonal via a
        // diagonal unitary similarity: T' = S^H T S with real off-diagonal |e|.
        // Also record S (phase_view) so eigenvectors can be lifted back: v_T = S * v_T'.
        ctx->submit([&](sycl::handler& cgh) {
            auto D = d_c_view;
            auto E = e_c_view;
            auto Dr = d_view;
            auto Er = e_view;
            auto S = phase_view;
            const int32_t nn = n;
            const int32_t nb = batch;
            const Uplo u = uplo;

            cgh.parallel_for(sycl::range<1>(static_cast<std::size_t>(nb)), [=](sycl::id<1> tid) {
                const int32_t b = static_cast<int32_t>(tid[0]);
                // S(0) = 1
                S(0, b) = T(1, Real(0));

                for (int32_t i = 0; i < nn; ++i) {
                    Dr(i, b) = D(i, b).real();
                }

                for (int32_t i = 0; i < nn - 1; ++i) {
                    const T e = E(i, b);
                    const Real abs_e = sycl::hypot(e.real(), e.imag());
                    Er(i, b) = abs_e;

                    if (abs_e == Real(0)) {
                        S(i + 1, b) = S(i, b);
                    } else {
                        // We want the *subdiagonal* to become real.
                        const T sub = (u == Uplo::Lower) ? e : T(e.real(), -e.imag());
                        S(i + 1, b) = S(i, b) * (sub / abs_e);
                    }
                }
            });
        });

        // Solve tridiagonal eigenproblem in real arithmetic.
        SteqrParams<Real> steqr_params_local;
        steqr_params_local.block_size = steqr_params.block_size;
        steqr_params_local.max_sweeps = steqr_params.max_sweeps;
        steqr_params_local.zero_threshold = static_cast<Real>(std::abs(steqr_params.zero_threshold));
        steqr_params_local.back_transform = false;
        steqr_params_local.block_rotations = steqr_params.block_rotations;
        steqr_params_local.sort = steqr_params.sort;
        steqr_params_local.transpose_working_vectors = steqr_params.transpose_working_vectors;
        steqr_params_local.sort_order = steqr_params.sort_order;
        steqr_params_local.cta_wg_size_multiplier = cta_wg_size_multiplier;
        steqr_params_local.cta_shift_strategy = steqr_params.cta_shift_strategy;

        VectorView<Real> evals_view(eigenvalues.data(), /*size=*/n, /*batch_size=*/batch,
                                    /*inc=*/1, /*stride=*/n);

        auto steqr_ws = ws_remaining(ws_mut, ws_off);
        steqr_cta<B, Real>(ctx, d_view, e_view, evals_view, steqr_ws, jobz, steqr_params_local, z_view);

        if (jobz == JobType::EigenVectors) {
            // Lift real Z to complex and apply the diagonal phase scaling: Zc = S * Z.
            ctx->submit([&](sycl::handler& cgh) {
                auto Z = z_view.kernel_view();
                auto ZC = zc_view.kernel_view();
                auto S = phase_view;
                const int32_t nn = n;
                const int32_t nb = batch;
                const int64_t total = static_cast<int64_t>(nb) * nn * nn;
                cgh.parallel_for(sycl::range<1>(static_cast<std::size_t>(total)), [=](sycl::id<1> tid) {
                    const int64_t idx = static_cast<int64_t>(tid[0]);
                    const int32_t b = static_cast<int32_t>(idx / (static_cast<int64_t>(nn) * nn));
                    const int64_t rem = idx - static_cast<int64_t>(b) * nn * nn;
                    const int32_t row = static_cast<int32_t>(rem % nn);
                    const int32_t col = static_cast<int32_t>(rem / nn);
                    ZC(row, col, b) = S(row, b) * T(Z(row, col, b), Real(0));
                });
            });

            // Pack reflectors into QR/QL-compatible layout for ormqx_cta.
            ctx->submit([&](sycl::handler& cgh) {
                auto A = a.kernel_view();
                auto AQ = aq_view.kernel_view();
                auto TAU = tau_c_view;
                auto TAUQ = tau_q_view;

                const int32_t nb = batch;
                const int32_t nn = n;

                const int64_t total = static_cast<int64_t>(nb) * static_cast<int64_t>(nn) * static_cast<int64_t>(nn);
                cgh.parallel_for(sycl::range<1>(static_cast<std::size_t>(total)), [=](sycl::id<1> tid) {
                    const int64_t idx = static_cast<int64_t>(tid[0]);
                    const int32_t b = static_cast<int32_t>(idx / (static_cast<int64_t>(nn) * nn));
                    const int64_t rem = idx - static_cast<int64_t>(b) * nn * nn;
                    const int32_t row = static_cast<int32_t>(rem % nn);
                    const int32_t col = static_cast<int32_t>(rem / nn);

                    AQ(row, col, b) = T(0);

                    if (uplo == Uplo::Lower) {
                        if (col >= 1) {
                            const int32_t i = col - 1;
                            if (row >= (col + 1) && row < nn) {
                                AQ(row, col, b) = A(row, i, b);
                            }
                        }
                    } else {
                        if (col <= nn - 2) {
                            const int32_t p = col;
                            const int32_t k = p + 1;
                            if (row < p) {
                                AQ(row, col, b) = A(row, k, b);
                            }
                        }
                    }
                });
            });

            ctx->submit([&](sycl::handler& cgh) {
                auto TAU = tau_c_view;
                auto TAUQ = tau_q_view;
                const int32_t nb = batch;
                const int32_t nn = n;
                const int64_t total = static_cast<int64_t>(nb) * static_cast<int64_t>(nn);

                cgh.parallel_for(sycl::range<1>(static_cast<std::size_t>(total)), [=](sycl::id<1> tid) {
                    const int64_t idx = static_cast<int64_t>(tid[0]);
                    const int32_t b = static_cast<int32_t>(idx / nn);
                    const int32_t j = static_cast<int32_t>(idx - static_cast<int64_t>(b) * nn);

                    if (uplo == Uplo::Lower) {
                        if (j == 0) {
                            TAUQ(j, b) = T(0);
                        } else {
                            TAUQ(j, b) = (j - 1 < (nn - 1)) ? TAU(j - 1, b) : T(0);
                        }
                    } else {
                        if (j < (nn - 1)) {
                            TAUQ(j, b) = TAU(j, b);
                        } else {
                            TAUQ(j, b) = T(0);
                        }
                    }
                });
            });

            const Uplo ormq_factorization = (uplo == Uplo::Lower) ? Uplo::Upper : Uplo::Lower;
            ormqx_cta<B, T>(ctx,
                            aq_view,
                            tau_q_view,
                            zc_view,
                            ormq_factorization,
                            Side::Left,
                            Transpose::NoTrans,
                            /*k=*/n,
                            Span<std::byte>(),
                            cta_wg_size_multiplier);

            MatrixView<T, MatrixFormat::Dense>::copy(ctx, a, zc_view);
        }

        return ctx.get_event();
    }

    // Workspace layout:
    //  - d (n)
    //  - e (n-1)
    //  - tau (n-1)
    //  - Z (n x n)        (STEQR eigenvectors of T)
    //  - A_q (n x n)      (packed reflectors in QR/QL-compatible layout)
    //  - tau_q (n)
    //  - remaining passed to steqr_cta
    std::size_t ws_off = 0;

    auto d_span = ws_alloc<T>(ctx, ws_mut, ws_off, static_cast<std::size_t>(n) * static_cast<std::size_t>(batch));
    auto e_span = ws_alloc<T>(ctx, ws_mut, ws_off, static_cast<std::size_t>(n - 1) * static_cast<std::size_t>(batch));
    auto tau_span = ws_alloc<T>(ctx, ws_mut, ws_off, static_cast<std::size_t>(n - 1) * static_cast<std::size_t>(batch));

    auto z_span = ws_alloc<T>(ctx, ws_mut, ws_off, static_cast<std::size_t>(n) * static_cast<std::size_t>(n) * static_cast<std::size_t>(batch));
    auto aq_span = ws_alloc<T>(ctx, ws_mut, ws_off, static_cast<std::size_t>(n) * static_cast<std::size_t>(n) * static_cast<std::size_t>(batch));
    auto tau_q_span = ws_alloc<T>(ctx, ws_mut, ws_off, static_cast<std::size_t>(n) * static_cast<std::size_t>(batch));

    VectorView<T> d_view(d_span, /*size=*/n, /*batch_size=*/batch, /*inc=*/1, /*stride=*/n);
    VectorView<T> e_view(e_span, /*size=*/n - 1, /*batch_size=*/batch, /*inc=*/1, /*stride=*/n - 1);
    VectorView<T> tau_view(tau_span, /*size=*/n - 1, /*batch_size=*/batch, /*inc=*/1, /*stride=*/n - 1);

    MatrixView<T, MatrixFormat::Dense> z_view(z_span.data(), n, n, /*ld=*/n,
                                             /*stride=*/n * n, /*batch_size=*/batch);

    MatrixView<T, MatrixFormat::Dense> aq_view(aq_span.data(), n, n, /*ld=*/n,
                                              /*stride=*/n * n, /*batch_size=*/batch);

    VectorView<T> tau_q_view(tau_q_span, /*size=*/n, /*batch_size=*/batch, /*inc=*/1, /*stride=*/n);

    // Reduce A to tridiagonal: A overwritten with reflectors/tridiagonal.
    // Note: sytrd_cta's workspace is currently unused.
    sytrd_cta<B, T>(ctx, a, d_view, e_view, tau_view, uplo, Span<std::byte>(), cta_wg_size_multiplier);

    // Solve tridiagonal eigenproblem; compute Z from identity when jobz==EigenVectors.
    // We force back_transform=false here because we apply the sytrd back-transform explicitly.
    auto steqr_ws = ws_remaining(ws_mut, ws_off);
    SteqrParams<T> steqr_params_local = steqr_params;
    steqr_params_local.back_transform = false;
    steqr_params_local.cta_wg_size_multiplier = cta_wg_size_multiplier;

    VectorView<T> evals_view(reinterpret_cast<T*>(eigenvalues.data()), /*size=*/n, /*batch_size=*/batch,
                             /*inc=*/1, /*stride=*/n);

    steqr_cta<B, T>(ctx, d_view, e_view, evals_view, steqr_ws, jobz, steqr_params_local, z_view);

    if (jobz == JobType::EigenVectors) {
        // Pack SYTRD Householder reflectors into a QR/QL-compatible layout for ormqx_cta.
        //
        // Lower (SYTRD Lower): reflectors act on trailing submatrices starting at row i+1.
        // We encode them as QR reflectors in columns 1..n-1 (column 0 is identity), i.e.
        // H(1) is a no-op and H(2..n) are the SYTRD reflectors.
        //
        // Upper (SYTRD Upper): reflectors act on leading submatrices ending at row k-1.
        // We encode them as QL reflectors in columns 0..n-2 (column n-1 is identity).
        ctx->submit([&](sycl::handler& cgh) {
            auto A = a.kernel_view();
            auto AQ = aq_view.kernel_view();
            auto TAU = tau_view;
            auto TAUQ = tau_q_view;

            const int32_t nb = batch;
            const int32_t nn = n;

            const int64_t total = static_cast<int64_t>(nb) * static_cast<int64_t>(nn) * static_cast<int64_t>(nn);
            cgh.parallel_for(sycl::range<1>(static_cast<std::size_t>(total)), [=](sycl::id<1> tid) {
                const int64_t idx = static_cast<int64_t>(tid[0]);
                const int32_t b = static_cast<int32_t>(idx / (static_cast<int64_t>(nn) * nn));
                const int64_t rem = idx - static_cast<int64_t>(b) * nn * nn;
                const int32_t row = static_cast<int32_t>(rem % nn);
                const int32_t col = static_cast<int32_t>(rem / nn);

                // Default: zero fill.
                AQ(row, col, b) = T(0);

                if (uplo == Uplo::Lower) {
                    // Fill below-diagonal tails for QR reflectors stored in columns 1..n-1.
                    // Reflector i (0..n-2) from SYTRD column i maps to QR column (i+1).
                    if (col >= 1) {
                        const int32_t i = col - 1;
                        // Tail starts at row (i+2) == col+1.
                        if (row >= (col + 1) && row < nn) {
                            AQ(row, col, b) = A(row, i, b);
                        }
                    }
                } else {
                    // Upper -> QL reflectors stored in columns 0..n-2.
                    // Reflector with pivot p (0..n-2) comes from SYTRD column (p+1)
                    // with entries in rows 0..p-1.
                    if (col <= nn - 2) {
                        const int32_t p = col;
                        const int32_t k = p + 1;
                        if (row < p) {
                            AQ(row, col, b) = A(row, k, b);
                        }
                    }
                }
            });
        });

        ctx->submit([&](sycl::handler& cgh) {
            auto TAU = tau_view;
            auto TAUQ = tau_q_view;
            const int32_t nb = batch;
            const int32_t nn = n;
            const int64_t total = static_cast<int64_t>(nb) * static_cast<int64_t>(nn);

            cgh.parallel_for(sycl::range<1>(static_cast<std::size_t>(total)), [=](sycl::id<1> tid) {
                const int64_t idx = static_cast<int64_t>(tid[0]);
                const int32_t b = static_cast<int32_t>(idx / nn);
                const int32_t j = static_cast<int32_t>(idx - static_cast<int64_t>(b) * nn);

                if (uplo == Uplo::Lower) {
                    // tau_q[0] = 0, tau_q[j] = tau[j-1] for j>=1.
                    if (j == 0) {
                        TAUQ(j, b) = T(0);
                    } else {
                        TAUQ(j, b) = (j - 1 < (nn - 1)) ? TAU(j - 1, b) : T(0);
                    }
                } else {
                    // tau_q[j] = tau[j] for j<=n-2, tau_q[n-1]=0.
                    if (j < (nn - 1)) {
                        TAUQ(j, b) = TAU(j, b);
                    } else {
                        TAUQ(j, b) = T(0);
                    }
                }
            });
        });

        const Uplo ormq_factorization = (uplo == Uplo::Lower) ? Uplo::Upper : Uplo::Lower;
        // Apply Q (from SYTRD reflectors) to Z: Z := Q * Z.
        ormqx_cta<B, T>(ctx,
                        aq_view,
                        tau_q_view,
                        z_view,
                        ormq_factorization,
                        Side::Left,
                        Transpose::NoTrans,
                        /*k=*/n,
                        Span<std::byte>(),
                        cta_wg_size_multiplier);

        // Overwrite A with eigenvectors.
        MatrixView<T, MatrixFormat::Dense>::copy(ctx, a, z_view);
    }

    return ctx.get_event();
}

template <Backend B, typename T>
size_t syev_cta_buffer_size(Queue& ctx,
                            const MatrixView<T, MatrixFormat::Dense>& a,
                            JobType jobz,
                            SteqrParams<T> steqr_params) {
    if constexpr (internal::is_complex<T>::value) {
        using Real = typename base_type<T>::type;

        if (a.rows() != a.cols()) {
            throw std::invalid_argument("syev_cta_buffer_size: A must be square.");
        }

        const int64_t n64 = a.rows();
        const int64_t batch64 = a.batch_size();
        if (n64 < 1 || n64 > 32) {
            throw std::invalid_argument("syev_cta_buffer_size currently supports 1 <= n <= 32.");
        }

        const int32_t n = static_cast<int32_t>(n64);
        const int32_t batch = static_cast<int32_t>(batch64);

        // Fixed allocations used by complex syev_cta before calling steqr_cta<Real>.
        const std::size_t d_c_count = static_cast<std::size_t>(n) * static_cast<std::size_t>(batch);
        const std::size_t e_c_count = static_cast<std::size_t>(n - 1) * static_cast<std::size_t>(batch);
        const std::size_t tau_c_count = static_cast<std::size_t>(n - 1) * static_cast<std::size_t>(batch);

        const std::size_t d_count = d_c_count;
        const std::size_t e_count = e_c_count;
        const std::size_t phase_count = d_c_count;
        const std::size_t z_count = static_cast<std::size_t>(n) * static_cast<std::size_t>(n) * static_cast<std::size_t>(batch);
        const std::size_t zc_count = z_count;
        const std::size_t aq_count = z_count;
        const std::size_t tau_q_count = static_cast<std::size_t>(n) * static_cast<std::size_t>(batch);

        size_t bytes = 0;
        bytes += BumpAllocator::allocation_size<T>(ctx, d_c_count);
        bytes += BumpAllocator::allocation_size<T>(ctx, e_c_count);
        bytes += BumpAllocator::allocation_size<T>(ctx, tau_c_count);

        bytes += BumpAllocator::allocation_size<Real>(ctx, d_count);
        bytes += BumpAllocator::allocation_size<Real>(ctx, e_count);

        bytes += BumpAllocator::allocation_size<T>(ctx, phase_count);
        bytes += BumpAllocator::allocation_size<Real>(ctx, z_count);

        bytes += BumpAllocator::allocation_size<T>(ctx, zc_count);
        bytes += BumpAllocator::allocation_size<T>(ctx, aq_count);
        bytes += BumpAllocator::allocation_size<T>(ctx, tau_q_count);

        // STEQR workspace (real).
        VectorView<Real> d_dummy(nullptr, /*size=*/n, /*batch_size=*/batch, /*inc=*/1, /*stride=*/n);
        VectorView<Real> e_dummy(nullptr, /*size=*/n - 1, /*batch_size=*/batch, /*inc=*/1, /*stride=*/n - 1);
        VectorView<Real> evals_dummy(nullptr, /*size=*/n, /*batch_size=*/batch, /*inc=*/1, /*stride=*/n);

        SteqrParams<Real> steqr_params_local;
        steqr_params_local.block_size = steqr_params.block_size;
        steqr_params_local.max_sweeps = steqr_params.max_sweeps;
        steqr_params_local.zero_threshold = static_cast<Real>(std::abs(steqr_params.zero_threshold));
        steqr_params_local.back_transform = false;
        steqr_params_local.block_rotations = steqr_params.block_rotations;
        steqr_params_local.sort = steqr_params.sort;
        steqr_params_local.transpose_working_vectors = steqr_params.transpose_working_vectors;
        steqr_params_local.sort_order = steqr_params.sort_order;
        steqr_params_local.cta_wg_size_multiplier = steqr_params.cta_wg_size_multiplier;
        steqr_params_local.cta_shift_strategy = steqr_params.cta_shift_strategy;

        bytes += steqr_cta_buffer_size<Real>(ctx, d_dummy, e_dummy, evals_dummy, jobz, steqr_params_local);

        return bytes;
    }

    if (a.rows() != a.cols()) {
        throw std::invalid_argument("syev_cta_buffer_size: A must be square.");
    }

    const int64_t n64 = a.rows();
    const int64_t batch64 = a.batch_size();
    if (n64 < 1 || n64 > 32) {
        throw std::invalid_argument("syev_cta_buffer_size currently supports 1 <= n <= 32.");
    }

    const int32_t n = static_cast<int32_t>(n64);
    const int32_t batch = static_cast<int32_t>(batch64);

    // Fixed allocations used by syev_cta before calling steqr_cta.
    const std::size_t d_count = static_cast<std::size_t>(n) * static_cast<std::size_t>(batch);
    const std::size_t e_count = static_cast<std::size_t>(n - 1) * static_cast<std::size_t>(batch);
    const std::size_t tau_count = static_cast<std::size_t>(n - 1) * static_cast<std::size_t>(batch);
    const std::size_t z_count = static_cast<std::size_t>(n) * static_cast<std::size_t>(n) * static_cast<std::size_t>(batch);
    const std::size_t aq_count = z_count;
    const std::size_t tau_q_count = static_cast<std::size_t>(n) * static_cast<std::size_t>(batch);

    size_t bytes = 0;
    bytes += BumpAllocator::allocation_size<T>(ctx, d_count);
    bytes += BumpAllocator::allocation_size<T>(ctx, e_count);
    bytes += BumpAllocator::allocation_size<T>(ctx, tau_count);
    bytes += BumpAllocator::allocation_size<T>(ctx, z_count);
    bytes += BumpAllocator::allocation_size<T>(ctx, aq_count);
    bytes += BumpAllocator::allocation_size<T>(ctx, tau_q_count);

    // Remaining: steqr_cta internal workspace requirements.
    VectorView<T> d_dummy(nullptr, /*size=*/n, /*batch_size=*/batch, /*inc=*/1, /*stride=*/n);
    VectorView<T> e_dummy(nullptr, /*size=*/n - 1, /*batch_size=*/batch, /*inc=*/1, /*stride=*/n - 1);
    VectorView<T> evals_dummy(nullptr, /*size=*/n, /*batch_size=*/batch, /*inc=*/1, /*stride=*/n);

    SteqrParams<T> steqr_params_local = steqr_params;
    steqr_params_local.back_transform = false;

    bytes += steqr_cta_buffer_size<T>(ctx, d_dummy, e_dummy, evals_dummy, jobz, steqr_params_local);

    return bytes;
}

#if BATCHLAS_HAS_CUDA_BACKEND
    template Event syev_cta<Backend::CUDA, float>(Queue&, const MatrixView<float, MatrixFormat::Dense>&,
                                                  Span<typename base_type<float>::type>, JobType, Uplo,
                                                  const Span<std::byte>&, SteqrParams<float>, size_t);
    template Event syev_cta<Backend::CUDA, double>(Queue&, const MatrixView<double, MatrixFormat::Dense>&,
                                                   Span<typename base_type<double>::type>, JobType, Uplo,
                                                   const Span<std::byte>&, SteqrParams<double>, size_t);

    template Event syev_cta<Backend::CUDA, std::complex<float>>(Queue&, const MatrixView<std::complex<float>, MatrixFormat::Dense>&,
                                                                Span<typename base_type<std::complex<float>>::type>, JobType, Uplo,
                                                                const Span<std::byte>&, SteqrParams<std::complex<float>>, size_t);
    template Event syev_cta<Backend::CUDA, std::complex<double>>(Queue&, const MatrixView<std::complex<double>, MatrixFormat::Dense>&,
                                                                 Span<typename base_type<std::complex<double>>::type>, JobType, Uplo,
                                                                 const Span<std::byte>&, SteqrParams<std::complex<double>>, size_t);

    template size_t syev_cta_buffer_size<Backend::CUDA, float>(Queue&, const MatrixView<float, MatrixFormat::Dense>&,
                                                               JobType, SteqrParams<float>);
    template size_t syev_cta_buffer_size<Backend::CUDA, double>(Queue&, const MatrixView<double, MatrixFormat::Dense>&,
                                                                JobType, SteqrParams<double>);

    template size_t syev_cta_buffer_size<Backend::CUDA, std::complex<float>>(Queue&, const MatrixView<std::complex<float>, MatrixFormat::Dense>&,
                                                                              JobType, SteqrParams<std::complex<float>>);
    template size_t syev_cta_buffer_size<Backend::CUDA, std::complex<double>>(Queue&, const MatrixView<std::complex<double>, MatrixFormat::Dense>&,
                                                                               JobType, SteqrParams<std::complex<double>>);
#endif

#if BATCHLAS_HAS_HOST_BACKEND
    template Event syev_cta<Backend::NETLIB, float>(Queue&, const MatrixView<float, MatrixFormat::Dense>&,
                                                    Span<typename base_type<float>::type>, JobType, Uplo,
                                                    const Span<std::byte>&, SteqrParams<float>, size_t);
    template Event syev_cta<Backend::NETLIB, double>(Queue&, const MatrixView<double, MatrixFormat::Dense>&,
                                                     Span<typename base_type<double>::type>, JobType, Uplo,
                                                     const Span<std::byte>&, SteqrParams<double>, size_t);

    template Event syev_cta<Backend::NETLIB, std::complex<float>>(Queue&, const MatrixView<std::complex<float>, MatrixFormat::Dense>&,
                                                                  Span<typename base_type<std::complex<float>>::type>, JobType, Uplo,
                                                                  const Span<std::byte>&, SteqrParams<std::complex<float>>, size_t);
    template Event syev_cta<Backend::NETLIB, std::complex<double>>(Queue&, const MatrixView<std::complex<double>, MatrixFormat::Dense>&,
                                                                   Span<typename base_type<std::complex<double>>::type>, JobType, Uplo,
                                                                   const Span<std::byte>&, SteqrParams<std::complex<double>>, size_t);

    template size_t syev_cta_buffer_size<Backend::NETLIB, float>(Queue&, const MatrixView<float, MatrixFormat::Dense>&,
                                                                 JobType, SteqrParams<float>);
    template size_t syev_cta_buffer_size<Backend::NETLIB, double>(Queue&, const MatrixView<double, MatrixFormat::Dense>&,
                                                                  JobType, SteqrParams<double>);

    template size_t syev_cta_buffer_size<Backend::NETLIB, std::complex<float>>(Queue&, const MatrixView<std::complex<float>, MatrixFormat::Dense>&,
                                                                                JobType, SteqrParams<std::complex<float>>);
    template size_t syev_cta_buffer_size<Backend::NETLIB, std::complex<double>>(Queue&, const MatrixView<std::complex<double>, MatrixFormat::Dense>&,
                                                                                 JobType, SteqrParams<std::complex<double>>);
#endif

} // namespace batchlas
