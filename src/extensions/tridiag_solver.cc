#include "../linalg-impl.hh"
#include <batchlas/util/group-collectives.hh>
#include "../math-helpers.hh"
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/util/sycl-span.hh>
#include "../queue.hh"
#include <batchlas/util/mempool.hh>
#include <sycl/sycl.hpp>
#include <complex>
#include <string>
#include <batchlas/error.hh>
#include <batchlas/blas/linalg.hh>

#include "../util/template-instantiations.hh"

namespace batchlas {

// QR steps allowed per eigenvalue before the item is reported unconverged (LAPACK ?steqr's MAXIT).
inline constexpr int kTridiagMaxIterations = 30;

template <typename T>
void apply_all_reflections(const sycl::group<1> &cta, const Span<T> V, const int n, const int m, const int ld, T* Q) {
    auto tid = cta.get_local_linear_id();
    auto bdim = cta.get_local_range()[0];
    for(int k = 0; k < n; k++) {
        const T &v0 = V[2*k], &v1 = V[2*k+1];
        for(int l = tid; l < m; l += bdim) {
            T &q0 = Q[k*ld+l], &q1 = Q[(k+1)*ld+l];
            T vTA = q0*v0 + q1*v1;
            q0 -= 2*v0*vTA;
            q1 -= 2*v1*vTA;
        }
    }
}

template <typename T>
void T_QTQ(sycl::group<1>& cta, const int n, const Span<T> D, const Span<T> L, const Span<T> U, const Span<T> Vout, T shift=0) {
    using real_t = typename base_type<T>::type;
    using coord3d = std::array<T, 3>;

    int tix = cta.get_local_linear_id();
    int bdim = cta.get_local_range()[0];

    // L[n-1] couples the active block to the deflated part below it; L[n] may be past the end.
    const T l_nm1 = L[n-1];

    sycl::group_barrier(cta);

    real_t a[2], v[2];
    for(int k = tix; k < n; k += bdim){
        D[k] -= shift;
        U[n+1 + k] = real_t(0.);
        if(k < n-1){
            U[k] = L[k];
            Vout[2*k] = real_t(0.); Vout[2*k+1] = real_t(0.);
        } else {
            L[k] = real_t(0.);
            U[k] = real_t(0.);
        }
    }

    sycl::group_barrier(cta);

    if(tix == 0) {
        for(int k = 0; k < n-1; k++) {
            if (L[k] != real_t(0)) {
                a[0] = D[k]; a[1] = L[k];
                real_t anorm = sycl::sqrt(a[0]*a[0] + a[1]*a[1]);
                v[0] = D[k]; v[1] = L[k];
                real_t alpha = -sycl::copysign(anorm,a[0]);
                v[0] -= alpha;
                real_t vnorm = sycl::sqrt(v[0]*v[0]+v[1]*v[1]);
                real_t norm_inv = real_t(1.)/vnorm;
                v[0] *= norm_inv;  v[1] *= norm_inv;
                Vout[2*k] = v[0]; Vout[2*k+1] = v[1];
                coord3d vTA = { D[ k ]*v[0] + L[ k ]*v[1],
                                U[ k ]*v[0] + D[k+1]*v[1],
                                U[(n+1)+k]*v[0] + U[k+1]*v[1]};
                D[k]     -= real_t(2.)*v[0]*vTA[0];
                L[k]     -= real_t(2.)*v[1]*vTA[0];
                U[k]     -= real_t(2.)*v[0]*vTA[1];
                D[k+1]     -= real_t(2.)*v[1]*vTA[1];
                U[(n+1)+k] -= real_t(2.)*v[0]*vTA[2];
                U[k+1]     -= real_t(2.)*v[1]*vTA[2];
            }
        }
    }

    if(tix == 0) {
        int k = 0;
        const real_t *v = &Vout[0];
        real_t vTA[2] = {D[k]*v[0] + U[k]*v[1],
                        0 + D[k+1]*v[1]};
        D[k]       -= real_t(2.)*v[0]*vTA[0];
        U[k]       -= real_t(2.)*v[1]*vTA[0];
        L[k]       -= real_t(2.)*v[0]*vTA[1];
        D[k+1]     -= real_t(2.)*v[1]*vTA[1];
    }

    sycl::group_barrier(cta);

    if(tix == 0) {
        for(int k = 1; k < n-1; k++) {
            const real_t *v = &Vout[2*k];
            coord3d vTA = {U[k-1]*v[0] + U[(n+1)+k-1]*v[1],
                            D[k]*v[0] + U[k]*v[1],
                            L[k]*v[0] + D[k+1]*v[1]};
            U[k-1]     -= real_t(2.)*v[0]*vTA[0];
            U[(n+1)+(k-1)] -= real_t(2.)*v[1]*vTA[0];
            U[k]       -= real_t(2.)*v[1]*vTA[1];
            D[k]       -= real_t(2.)*v[0]*vTA[1];
            L[k]       -= real_t(2.)*v[0]*vTA[2];
            D[k+1]     -= real_t(2.)*v[1]*vTA[2];
        }
    }

    sycl::group_barrier(cta);

    for (int k = tix; k < n; k += bdim) {
        D[k] += shift;
        if(k < n-1) {
            L[k] = U[k];
        }
    }

    sycl::group_barrier(cta);

    if (tix == 0) {
        L[n-1] = l_nm1;
    }
    sycl::group_barrier(cta);
}

template <Backend B, typename T>
Event tridiagonal_solver(Queue& ctx,
                         Span<T> alphas,
                         Span<T> betas,
                         Span<typename base_type<T>::type> W,
                         Span<std::byte> workspace,
                         JobType jobz,
                         const MatrixView<T, MatrixFormat::Dense>& Q,
                         size_t n,
                         size_t batch_size) {
    using real_t = typename base_type<T>::type;
    auto pool = BumpAllocator(workspace);

    auto smem_required = sizeof(T) * (3 * n + 3 * (n + 1));
    bool smem_possible = ctx.device().get_property(DeviceProperty::LOCAL_MEM_SIZE) >= smem_required;

    auto V_reflectors = pool.allocate<T>(ctx, !smem_possible ? 2 * n * batch_size : 0);
    auto D_global = pool.allocate<T>(ctx, !smem_possible ? (n+1) * batch_size : 0);
    auto U_global = pool.allocate<T>(ctx, !smem_possible ? 2 * (n + 1) * batch_size : 0);
    auto L_global = pool.allocate<T>(ctx, !smem_possible ? n * batch_size : 0);
    UnifiedVector<int32_t> status(batch_size, 0);
    int32_t* status_ptr = status.data();

    ctx->submit([=](sycl::handler& h) {
        auto Vstride = Q.stride();
        auto Vdata = Q.data_ptr();
        auto Vld = Q.ld();

        sycl::local_accessor<T, 1> D_smem(smem_possible ? n + 1 : 0, h);
        sycl::local_accessor<T, 1> L_smem(smem_possible ? n : 0, h);
        sycl::local_accessor<T, 1> U_smem(smem_possible ? 2 * (n + 1) : 0, h);
        sycl::local_accessor<T, 1> V_smem(smem_possible ? 2 * n : 0, h);

        batchlas::impl::parallel_for(h, sycl::nd_range<1>(batch_size*32, 32), [=](sycl::nd_item<1> item) {
            auto bid = item.get_group_linear_id();
            auto tid = item.get_local_linear_id();
            auto bdim = item.get_local_range()[0];
            auto cta = item.get_group();

            auto D = Span(smem_possible ? &D_smem[0] : D_global.data() + bid*(n+1), n+1);
            auto L = Span(smem_possible ? &L_smem[0] : L_global.data() + bid*n, n);
            auto U = Span(smem_possible ? &U_smem[0] : U_global.data() + bid*2*(n+1), 2 * (n + 1));
            auto Vlocal = Span(smem_possible ? &V_smem[0] : V_reflectors.data() + bid*2*n, 2 * n);

            auto batch_alphas = Span(alphas.data() + bid * n, n);
            auto batch_betas = Span(betas.data() + bid * n, n);
            T* batch_Q = Vdata + bid * Vstride;
            real_t* batch_W = W.data() + bid * n;

            if (jobz == JobType::EigenVectors) {
                for (int i = tid; i < n*n; i += bdim) {
                    batch_Q[(i / n) * Vld + i % n] = T(0);
                }
                sycl::group_barrier(cta);
                for (int i = tid; i < n; i += bdim) {
                    batch_Q[i * (Vld + 1)] = T(1);
                }
            }

            for( int i = tid; i < n; i += bdim) {
                D[i] = batch_alphas[i];
                L[i] = batch_betas[i];
            }

            sycl::group_barrier(cta);

            // Every work-item reads the same local values after T_QTQ's closing barrier, so the
            // loop decisions agree across the group. Relative deflation as LAPACK ?steqr.
            const real_t eps = std::numeric_limits<real_t>::epsilon();
            bool failed = false;
            for (int k = int(n) - 1; k > 0; k--) {
                int it = 0;
                while (std::abs(L[k-1]) > eps * (std::abs(D[k-1]) + std::abs(D[k]))) {
                    if (it++ == kTridiagMaxIterations) { failed = true; break; }
                    const auto [l0, l1] = internal::eigenvalues_2x2(D[k-1], L[k-1], D[k]);
                    const T shift = std::abs(l0 - D[k]) < std::abs(l1 - D[k]) ? l0 : l1;
                    T_QTQ(cta, k+1, D, L, U, Vlocal, shift);
                    if (jobz == JobType::EigenVectors) {
                        apply_all_reflections(cta, Vlocal, k, int(n), int(Vld), batch_Q);
                    }
                }
            }
            if (tid == 0) status_ptr[bid] = failed ? 1 : 0;

            for (int i = tid; i < n; i += bdim) {
                if constexpr (internal::is_complex<T>::value) {
                    batch_W[i] = D[i].real();
                } else {
                    batch_W[i] = D[i];
                }
            }
        });
    });

    ctx.wait();
    for (size_t b = 0; b < batch_size; ++b) {
        if (status_ptr[b] != 0) {
            throw batchlas::convergence_error("tridiagonal_solver: an eigenvalue did not converge within " +
                                              std::to_string(kTridiagMaxIterations) + " QR steps");
        }
    }
    return ctx.get_event();
}

template <Backend B, typename T>
size_t tridiagonal_solver_buffer_size(Queue& ctx, size_t n, size_t batch_size, JobType /*jobz*/){
    auto smem_required = sizeof(T) * (3 * n + 3 * (n + 1));
    bool smem_possible = ctx.device().get_property(DeviceProperty::LOCAL_MEM_SIZE) >= smem_required;
    size_t size = 0;
    size += BumpAllocator::allocation_size<T>(ctx, !smem_possible ? 2*n*batch_size : 0);
    size += BumpAllocator::allocation_size<T>(ctx, !smem_possible ? (n+1)*batch_size : 0);
    size += BumpAllocator::allocation_size<T>(ctx, !smem_possible ? 2*(n+1)*batch_size : 0);
    size += BumpAllocator::allocation_size<T>(ctx, !smem_possible ? n*batch_size : 0);
    return size;
}

#define TRIDAG_INSTANTIATE(back, fp) \
    template Event tridiagonal_solver<back, BATCHLAS_UNPAREN fp>(Queue&, Span<BATCHLAS_UNPAREN fp>, Span<BATCHLAS_UNPAREN fp>, Span<typename base_type<BATCHLAS_UNPAREN fp>::type>, Span<std::byte>, JobType, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, size_t, size_t); \
    template size_t tridiagonal_solver_buffer_size<back, BATCHLAS_UNPAREN fp>(Queue&, size_t, size_t, JobType);

// Real only. The complex arms below have never been enabled:
//   TRIDAG_INSTANTIATE(back, std::complex<float>)
//   TRIDAG_INSTANTIATE(back, std::complex<double>)
BATCHLAS_INSTANTIATE_REAL_ALL_BACKENDS(TRIDAG_INSTANTIATE)

//TRIDAG_INSTANTIATE(Backend::CUDA, std::complex<float>)
//TRIDAG_INSTANTIATE(Backend::CUDA, std::complex<double>)

#undef TRIDAG_INSTANTIATE

} // namespace batchlas

