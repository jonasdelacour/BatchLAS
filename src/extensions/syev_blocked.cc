#include <blas/extensions.hh>
#include <blas/functions.hh>
#include <blas/matrix.hh>
#include <blas/linalg.hh>
#include <internal/ormqr_blocked.hh>
#include <util/mempool.hh>
#include <batchlas/backend_config.h>

#include "../math-helpers.hh"
#include "../queue.hh"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace batchlas {

namespace {

template <typename T>
inline void validate_syev_blocked_dims(const MatrixView<T, MatrixFormat::Dense>& a,
                                      Span<typename base_type<T>::type> eigenvalues,
                                      JobType jobz,
                                      Uplo uplo) {
    if (a.rows() != a.cols()) {
        throw std::invalid_argument("syev_blocked: A must be square.");
    }
    if (jobz != JobType::NoEigenVectors && jobz != JobType::EigenVectors) {
        throw std::invalid_argument("syev_blocked: invalid JobType.");
    }
    if (uplo != Uplo::Lower) {
        // Current sytrd_blocked implementation supports only Lower.
        throw std::invalid_argument("syev_blocked: only Uplo::Lower is currently implemented.");
    }

    const int64_t n64 = a.rows();
    const int64_t batch64 = a.batch_size();
    if (n64 < 1 || batch64 < 1) {
        throw std::invalid_argument("syev_blocked: invalid n or batch size.");
    }

    const std::size_t need = static_cast<std::size_t>(n64) * static_cast<std::size_t>(batch64);
    if (eigenvalues.size() < need) {
        throw std::invalid_argument("syev_blocked: eigenvalues span too small for n*batch.");
    }
}

template <typename T>
inline void pack_sytrd_lower_to_qsub_qr_layout(Queue& ctx,
                                              const MatrixView<T, MatrixFormat::Dense>& a_sytrd,
                                              const MatrixView<T, MatrixFormat::Dense>& a_qsub_qr,
                                              const VectorView<T>& tau_sytrd,
                                              const VectorView<T>& tau_qsub,
                                              int32_t n) {
    // Pack SYTRD Lower reflectors into a QR-compatible layout for the trailing
    // (n-1)x(n-1) subproblem (Qsub). This matches the approach used by
    // tests/sytrd_blocked_tests.cc:
    //   - Q = [1, 0; 0, Qsub]
    //   - For i in [0, p-1) where p=n-1, reflector i acts on rows i..p-1
    //     in the subproblem and has tail entries from A_out((i+2):n-1, i).
    const int32_t batch = static_cast<int32_t>(a_sytrd.batch_size());
    const int32_t p = std::max<int32_t>(0, n - 1);
    if (p == 0) return;

    // Pack A (strictly lower triangle). Diagonal is implicit 1 in ormqr_blocked.
    ctx->submit([&](sycl::handler& cgh) {
        auto A = a_sytrd.kernel_view();
        auto AQ = a_qsub_qr.kernel_view();
        const int32_t pp = p;
        const int32_t nb = batch;

        const int64_t total = static_cast<int64_t>(nb) * static_cast<int64_t>(pp) * static_cast<int64_t>(pp);
        cgh.parallel_for(sycl::range<1>(static_cast<std::size_t>(total)), [=](sycl::id<1> tid) {
            const int64_t idx = static_cast<int64_t>(tid[0]);
            const int32_t b = static_cast<int32_t>(idx / (static_cast<int64_t>(pp) * pp));
            const int64_t rem = idx - static_cast<int64_t>(b) * pp * pp;
            const int32_t row = static_cast<int32_t>(rem % pp);
            const int32_t col = static_cast<int32_t>(rem / pp);

            T val = T(0);
            if (row > col) {
                // Local (row,col) maps to global (row+1, col) in A_out.
                val = A(row + 1, col, b);
            }
            AQ(row, col, b) = val;
        });
    });

    // Pack tau for the subproblem (size p).
    ctx->submit([&](sycl::handler& cgh) {
        auto TAU = tau_sytrd;
        auto TAUQ = tau_qsub;
        const int32_t pp = p;
        const int32_t nb = batch;
        const int64_t total = static_cast<int64_t>(nb) * static_cast<int64_t>(pp);

        cgh.parallel_for(sycl::range<1>(static_cast<std::size_t>(total)), [=](sycl::id<1> tid) {
            const int64_t idx = static_cast<int64_t>(tid[0]);
            const int32_t b = static_cast<int32_t>(idx / pp);
            const int32_t i = static_cast<int32_t>(idx - static_cast<int64_t>(b) * pp);
            TAUQ(i, b) = TAU(i, b);
        });
    });
}

} // namespace

template <Backend B, typename T>
Event syev_blocked(Queue& ctx,
                   const MatrixView<T, MatrixFormat::Dense>& a_in,
                   Span<typename base_type<T>::type> eigenvalues,
                   JobType jobz,
                   Uplo uplo,
                   const Span<std::byte>& ws,
                   int32_t sytrd_block_size,
                   int32_t ormqr_block_size,
                   StedcParams<typename base_type<T>::type> stedc_params) {
    validate_syev_blocked_dims(a_in, eigenvalues, jobz, uplo);

    if (!ctx.in_order()) {
        throw std::runtime_error("syev_blocked: requires an in-order Queue");
    }

    const int32_t n = static_cast<int32_t>(a_in.rows());
    const int32_t batch = static_cast<int32_t>(a_in.batch_size());
    const int32_t p = std::max<int32_t>(0, n - 1);

    // Overwrite A only when jobz==EigenVectors.
    auto& a = const_cast<MatrixView<T, MatrixFormat::Dense>&>(a_in);

    // Workspace treated as mutable.
    Span<std::byte> ws_mut(const_cast<std::byte*>(ws.data()), ws.size());
    BumpAllocator pool(ws_mut);

    if constexpr (internal::is_complex<T>::value) {
        using Real = typename base_type<T>::type;

        // Allocate SYTRD outputs (complex Hermitian tridiagonal).
        auto d_c_span = pool.allocate<T>(ctx, static_cast<std::size_t>(n) * static_cast<std::size_t>(batch));
        auto e_c_span = pool.allocate<T>(ctx, static_cast<std::size_t>(std::max(0, n - 1)) * static_cast<std::size_t>(batch));
        auto tau_c_span = pool.allocate<T>(ctx, static_cast<std::size_t>(std::max(0, n - 1)) * static_cast<std::size_t>(batch));

        VectorView<T> d_c_view(d_c_span, n, batch, 1, n);
        VectorView<T> e_c_view(e_c_span, std::max(0, n - 1), batch, 1, std::max(0, n - 1));
        VectorView<T> tau_c_view(tau_c_span, std::max(0, n - 1), batch, 1, std::max(0, n - 1));

        // Real tridiagonal for STEDC.
        auto d_span = pool.allocate<Real>(ctx, static_cast<std::size_t>(n) * static_cast<std::size_t>(batch));
        auto e_span = pool.allocate<Real>(ctx, static_cast<std::size_t>(std::max(0, n - 1)) * static_cast<std::size_t>(batch));
        auto phase_span = pool.allocate<T>(ctx, static_cast<std::size_t>(n) * static_cast<std::size_t>(batch));

        VectorView<Real> d_view(d_span, n, batch, 1, n);
        VectorView<Real> e_view(e_span, std::max(0, n - 1), batch, 1, std::max(0, n - 1));
        VectorView<T> phase_view(phase_span, n, batch, 1, n);

        // STEDC eigenvectors (real) and lifted eigenvectors (complex).
        auto z_span = pool.allocate<Real>(ctx, static_cast<std::size_t>(n) * static_cast<std::size_t>(n) * static_cast<std::size_t>(batch));
        auto zc_span = pool.allocate<T>(ctx, static_cast<std::size_t>(n) * static_cast<std::size_t>(n) * static_cast<std::size_t>(batch));

        MatrixView<Real, MatrixFormat::Dense> z_view(z_span.data(), n, n, n, n * n, batch);
        MatrixView<T, MatrixFormat::Dense> zc_view(zc_span.data(), n, n, n, n * n, batch);

        // Pack reflectors for Qsub into QR layout for ORMQR (size (n-1)x(n-1)).
        auto aq_span = pool.allocate<T>(ctx, static_cast<std::size_t>(p) * static_cast<std::size_t>(p) * static_cast<std::size_t>(batch));
        auto tau_q_span = pool.allocate<T>(ctx, static_cast<std::size_t>(p) * static_cast<std::size_t>(batch));
        MatrixView<T, MatrixFormat::Dense> aq_view(aq_span.data(), p, p, p, p * p, batch);
        VectorView<T> tau_q_view(tau_q_span, p, batch, 1, p);

        // Sub-workspaces.
        {
            auto sytrd_ws_bytes = sytrd_blocked_buffer_size<B, T>(ctx,
                                                                 a,
                                                                 d_c_view,
                                                                 e_c_view,
                                                                 tau_c_view,
                                                                 uplo,
                                                                 sytrd_block_size);
            auto sytrd_ws = pool.allocate<std::byte>(ctx, sytrd_ws_bytes);
            sytrd_blocked<B, T>(ctx, a, d_c_view, e_c_view, tau_c_view, uplo, sytrd_ws, sytrd_block_size);
        }

        // Convert complex Hermitian tridiagonal to real symmetric tridiagonal via diagonal phase similarity.
        ctx->submit([&](sycl::handler& cgh) {
            auto D = d_c_view;
            auto E = e_c_view;
            auto Dr = d_view;
            auto Er = e_view;
            auto S = phase_view;
            const int32_t nn = n;
            const int32_t nb = batch;

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
                        // Lower: subdiagonal is E(i)
                        S(i + 1, b) = S(i, b) * (e / abs_e);
                    }
                }
            });
        });

        // Solve tridiagonal eigenproblem in real arithmetic.
        // NOTE: The current STEDC implementation relies on eigenvectors during
        // recursion/merge, so we always run it in EigenVectors mode and only
        // backtransform/store vectors when requested by the public API.
        const JobType internal_jobz = JobType::EigenVectors;
        VectorView<Real> evals_view(eigenvalues.data(), n, batch, 1, n);
        {
            auto stedc_ws_bytes = stedc_workspace_size<B, Real>(ctx, static_cast<std::size_t>(n), static_cast<std::size_t>(batch), internal_jobz, stedc_params);
            auto stedc_ws = pool.allocate<std::byte>(ctx, stedc_ws_bytes);
            stedc<B, Real>(ctx, d_view, e_view, evals_view, stedc_ws, internal_jobz, stedc_params, z_view);
        }

        if (jobz == JobType::EigenVectors) {
            // Lift Z to complex and apply phase scaling: Zc = S * Z.
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

            if (p > 0) {
                pack_sytrd_lower_to_qsub_qr_layout<T>(ctx, a, aq_view, tau_c_view, tau_q_view, n);
            }

            // Apply Qsub from packed QR reflectors to rows 1..n-1: Zc_sub := Qsub * Zc_sub.
            {
                auto zc_sub = zc_view({1, SliceEnd()}, Slice());
                Span<T> tau_q_span_flat(tau_q_span.data(), static_cast<std::size_t>(p) * static_cast<std::size_t>(batch));

                if (p > 0) {
                    auto ormqr_ws_bytes = ormqr_blocked_buffer_size<B, T>(ctx,
                                                                          aq_view,
                                                                          zc_sub,
                                                                          Side::Left,
                                                                          Transpose::NoTrans,
                                                                          tau_q_span_flat,
                                                                          ormqr_block_size);
                    auto ormqr_ws = pool.allocate<std::byte>(ctx, ormqr_ws_bytes);

                    ormqr_blocked<B, T>(ctx,
                                        aq_view,
                                        zc_sub,
                                        Side::Left,
                                        Transpose::NoTrans,
                                        tau_q_span_flat,
                                        ormqr_ws,
                                        ormqr_block_size);
                }
            }

            MatrixView<T, MatrixFormat::Dense>::copy(ctx, a, zc_view);
        }

        return ctx.get_event();
    } else {
        // Real symmetric case.
        auto d_span = pool.allocate<T>(ctx, static_cast<std::size_t>(n) * static_cast<std::size_t>(batch));
        auto e_span = pool.allocate<T>(ctx, static_cast<std::size_t>(std::max(0, n - 1)) * static_cast<std::size_t>(batch));
        auto tau_span = pool.allocate<T>(ctx, static_cast<std::size_t>(std::max(0, n - 1)) * static_cast<std::size_t>(batch));
        auto phase_span = pool.allocate<T>(ctx, static_cast<std::size_t>(n) * static_cast<std::size_t>(batch));

        VectorView<T> d_view(d_span, n, batch, 1, n);
        VectorView<T> e_view(e_span, std::max(0, n - 1), batch, 1, std::max(0, n - 1));
        VectorView<T> tau_view(tau_span, std::max(0, n - 1), batch, 1, std::max(0, n - 1));
        VectorView<T> phase_view(phase_span, n, batch, 1, n);

        auto z_span = pool.allocate<T>(ctx, static_cast<std::size_t>(n) * static_cast<std::size_t>(n) * static_cast<std::size_t>(batch));
        MatrixView<T, MatrixFormat::Dense> z_view(z_span.data(), n, n, n, n * n, batch);

        auto aq_span = pool.allocate<T>(ctx, static_cast<std::size_t>(p) * static_cast<std::size_t>(p) * static_cast<std::size_t>(batch));
        auto tau_q_span = pool.allocate<T>(ctx, static_cast<std::size_t>(p) * static_cast<std::size_t>(batch));
        MatrixView<T, MatrixFormat::Dense> aq_view(aq_span.data(), p, p, p, p * p, batch);
        VectorView<T> tau_q_view(tau_q_span, p, batch, 1, p);

        {
            auto sytrd_ws_bytes = sytrd_blocked_buffer_size<B, T>(ctx, a, d_view, e_view, tau_view, uplo, sytrd_block_size);
            auto sytrd_ws = pool.allocate<std::byte>(ctx, sytrd_ws_bytes);
            sytrd_blocked<B, T>(ctx, a, d_view, e_view, tau_view, uplo, sytrd_ws, sytrd_block_size);
        }

        // STEDC's recursive merge path uses the split off-diagonal value and
        // currently assumes a non-negative coupling magnitude. Convert the
        // tridiagonal to one with non-negative off-diagonals via a diagonal
        // similarity transform S (entries +/-1):
        //   T' = S * T * S  =>  e'_i = |e_i|
        // and later recover eigenvectors by Z := S * Z.
        ctx->submit([&](sycl::handler& cgh) {
            auto E = e_view;
            auto S = phase_view;
            const int32_t nn = n;
            const int32_t nb = batch;
            cgh.parallel_for(sycl::range<1>(static_cast<std::size_t>(nb)), [=](sycl::id<1> tid) {
                const int32_t b = static_cast<int32_t>(tid[0]);
                S(0, b) = T(1);
                for (int32_t i = 0; i < nn - 1; ++i) {
                    const T ei = E(i, b);
                    const T sgn = (ei >= T(0)) ? T(1) : T(-1);
                    S(i + 1, b) = S(i, b) * sgn;
                    E(i, b) = sycl::fabs(ei);
                }
            });
        });

        const JobType internal_jobz = JobType::EigenVectors;
        VectorView<T> evals_view(eigenvalues.data(), n, batch, 1, n);
        {
            auto stedc_ws_bytes = stedc_workspace_size<B, T>(ctx, static_cast<std::size_t>(n), static_cast<std::size_t>(batch), internal_jobz, StedcParams<T>{stedc_params.recursion_threshold});
            auto stedc_ws = pool.allocate<std::byte>(ctx, stedc_ws_bytes);
            stedc<B, T>(ctx, d_view, e_view, evals_view, stedc_ws, internal_jobz, StedcParams<T>{stedc_params.recursion_threshold}, z_view);
        }

        if (jobz == JobType::EigenVectors) {
            // Recover eigenvectors for the original (signed) tridiagonal.
            ctx->submit([&](sycl::handler& cgh) {
                auto Z = z_view.kernel_view();
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
                    Z(row, col, b) *= S(row, b);
                });
            });

            if (p > 0) {
                pack_sytrd_lower_to_qsub_qr_layout<T>(ctx, a, aq_view, tau_view, tau_q_view, n);
            }

            {
                auto z_sub = z_view({1, SliceEnd()}, Slice());
                Span<T> tau_q_span_flat(tau_q_span.data(), static_cast<std::size_t>(p) * static_cast<std::size_t>(batch));
                if (p > 0) {
                    auto ormqr_ws_bytes = ormqr_blocked_buffer_size<B, T>(ctx,
                                                                          aq_view,
                                                                          z_sub,
                                                                          Side::Left,
                                                                          Transpose::NoTrans,
                                                                          tau_q_span_flat,
                                                                          ormqr_block_size);
                    auto ormqr_ws = pool.allocate<std::byte>(ctx, ormqr_ws_bytes);

                    ormqr_blocked<B, T>(ctx,
                                        aq_view,
                                        z_sub,
                                        Side::Left,
                                        Transpose::NoTrans,
                                        tau_q_span_flat,
                                        ormqr_ws,
                                        ormqr_block_size);
                }
            }

            MatrixView<T, MatrixFormat::Dense>::copy(ctx, a, z_view);
        }

        return ctx.get_event();
    }
}

template <Backend B, typename T>
size_t syev_blocked_buffer_size(Queue& ctx,
                                const MatrixView<T, MatrixFormat::Dense>& a,
                                JobType jobz,
                                Uplo uplo,
                                int32_t sytrd_block_size,
                                int32_t ormqr_block_size,
                                StedcParams<typename base_type<T>::type> stedc_params) {
    if (a.rows() != a.cols()) {
        throw std::invalid_argument("syev_blocked_buffer_size: A must be square.");
    }
    if (jobz != JobType::NoEigenVectors && jobz != JobType::EigenVectors) {
        throw std::invalid_argument("syev_blocked_buffer_size: invalid JobType.");
    }
    if (uplo != Uplo::Lower) {
        throw std::invalid_argument("syev_blocked_buffer_size: only Uplo::Lower is currently implemented.");
    }

    const int32_t n = static_cast<int32_t>(a.rows());
    const int32_t batch = static_cast<int32_t>(a.batch_size());

    size_t bytes = 0;

    if constexpr (internal::is_complex<T>::value) {
        using Real = typename base_type<T>::type;

        const int32_t p = std::max<int32_t>(0, n - 1);

        const std::size_t d_c_count = static_cast<std::size_t>(n) * static_cast<std::size_t>(batch);
        const std::size_t e_c_count = static_cast<std::size_t>(std::max(0, n - 1)) * static_cast<std::size_t>(batch);
        const std::size_t tau_c_count = e_c_count;
        const std::size_t d_count = d_c_count;
        const std::size_t e_count = e_c_count;
        const std::size_t phase_count = d_c_count;
        const std::size_t z_count = static_cast<std::size_t>(n) * static_cast<std::size_t>(n) * static_cast<std::size_t>(batch);
        const std::size_t zc_count = z_count;
        const std::size_t aq_count = static_cast<std::size_t>(p) * static_cast<std::size_t>(p) * static_cast<std::size_t>(batch);
        const std::size_t tau_q_count = static_cast<std::size_t>(p) * static_cast<std::size_t>(batch);

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

        // Sub-workspaces.
        VectorView<T> d_c_dummy(nullptr, n, batch, 1, n);
        VectorView<T> e_c_dummy(nullptr, std::max(0, n - 1), batch, 1, std::max(0, n - 1));
        VectorView<T> tau_c_dummy(nullptr, std::max(0, n - 1), batch, 1, std::max(0, n - 1));
        bytes += sytrd_blocked_buffer_size<B, T>(ctx, a, d_c_dummy, e_c_dummy, tau_c_dummy, uplo, sytrd_block_size);

        bytes += stedc_workspace_size<B, Real>(ctx, static_cast<std::size_t>(n), static_cast<std::size_t>(batch), JobType::EigenVectors, stedc_params);

        MatrixView<T, MatrixFormat::Dense> aq_dummy(nullptr, p, p, p, static_cast<int64_t>(p) * static_cast<int64_t>(p), batch);
        MatrixView<T, MatrixFormat::Dense> c_dummy(nullptr, p, n, p, static_cast<int64_t>(p) * static_cast<int64_t>(n), batch);
        Span<T> tau_q_dummy(nullptr, static_cast<std::size_t>(p) * static_cast<std::size_t>(batch));
        bytes += ormqr_blocked_buffer_size<B, T>(ctx, aq_dummy, c_dummy, Side::Left, Transpose::NoTrans, tau_q_dummy, ormqr_block_size);

        return bytes;
    } else {
        // Real.
        const int32_t p = std::max<int32_t>(0, n - 1);
        const std::size_t d_count = static_cast<std::size_t>(n) * static_cast<std::size_t>(batch);
        const std::size_t e_count = static_cast<std::size_t>(std::max(0, n - 1)) * static_cast<std::size_t>(batch);
        const std::size_t tau_count = e_count;
        const std::size_t phase_count = d_count;
        const std::size_t z_count = static_cast<std::size_t>(n) * static_cast<std::size_t>(n) * static_cast<std::size_t>(batch);
        const std::size_t aq_count = static_cast<std::size_t>(p) * static_cast<std::size_t>(p) * static_cast<std::size_t>(batch);
        const std::size_t tau_q_count = static_cast<std::size_t>(p) * static_cast<std::size_t>(batch);

        bytes += BumpAllocator::allocation_size<T>(ctx, d_count);
        bytes += BumpAllocator::allocation_size<T>(ctx, e_count);
        bytes += BumpAllocator::allocation_size<T>(ctx, tau_count);
        bytes += BumpAllocator::allocation_size<T>(ctx, phase_count);
        bytes += BumpAllocator::allocation_size<T>(ctx, z_count);
        bytes += BumpAllocator::allocation_size<T>(ctx, aq_count);
        bytes += BumpAllocator::allocation_size<T>(ctx, tau_q_count);

        VectorView<T> d_dummy(nullptr, n, batch, 1, n);
        VectorView<T> e_dummy(nullptr, std::max(0, n - 1), batch, 1, std::max(0, n - 1));
        VectorView<T> tau_dummy(nullptr, std::max(0, n - 1), batch, 1, std::max(0, n - 1));
        bytes += sytrd_blocked_buffer_size<B, T>(ctx, a, d_dummy, e_dummy, tau_dummy, uplo, sytrd_block_size);

        bytes += stedc_workspace_size<B, T>(ctx, static_cast<std::size_t>(n), static_cast<std::size_t>(batch), JobType::EigenVectors, StedcParams<T>{stedc_params.recursion_threshold});

        MatrixView<T, MatrixFormat::Dense> aq_dummy(nullptr, p, p, p, static_cast<int64_t>(p) * static_cast<int64_t>(p), batch);
        MatrixView<T, MatrixFormat::Dense> c_dummy(nullptr, p, n, p, static_cast<int64_t>(p) * static_cast<int64_t>(n), batch);
        Span<T> tau_q_dummy(nullptr, static_cast<std::size_t>(p) * static_cast<std::size_t>(batch));
        bytes += ormqr_blocked_buffer_size<B, T>(ctx, aq_dummy, c_dummy, Side::Left, Transpose::NoTrans, tau_q_dummy, ormqr_block_size);

        return bytes;
    }
}

#define SYEV_BLOCKED_INSTANTIATE(back, fp) \
    template Event syev_blocked<back, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Span<typename base_type<fp>::type>, \
        JobType, \
        Uplo, \
        const Span<std::byte>&, \
        int32_t, \
        int32_t, \
        StedcParams<typename base_type<fp>::type>); \
    template size_t syev_blocked_buffer_size<back, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        JobType, \
        Uplo, \
        int32_t, \
        int32_t, \
        StedcParams<typename base_type<fp>::type>);

#if BATCHLAS_HAS_CUDA_BACKEND
SYEV_BLOCKED_INSTANTIATE(Backend::CUDA, float)
SYEV_BLOCKED_INSTANTIATE(Backend::CUDA, double)
SYEV_BLOCKED_INSTANTIATE(Backend::CUDA, std::complex<float>)
SYEV_BLOCKED_INSTANTIATE(Backend::CUDA, std::complex<double>)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
SYEV_BLOCKED_INSTANTIATE(Backend::ROCM, float)
SYEV_BLOCKED_INSTANTIATE(Backend::ROCM, double)
SYEV_BLOCKED_INSTANTIATE(Backend::ROCM, std::complex<float>)
SYEV_BLOCKED_INSTANTIATE(Backend::ROCM, std::complex<double>)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
SYEV_BLOCKED_INSTANTIATE(Backend::NETLIB, float)
SYEV_BLOCKED_INSTANTIATE(Backend::NETLIB, double)
SYEV_BLOCKED_INSTANTIATE(Backend::NETLIB, std::complex<float>)
SYEV_BLOCKED_INSTANTIATE(Backend::NETLIB, std::complex<double>)
#endif

#undef SYEV_BLOCKED_INSTANTIATE

} // namespace batchlas
