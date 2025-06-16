#include "../linalg-impl.hh"
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include "../queue.hh"
#include <util/mempool.hh>
#include <sycl/sycl.hpp>
#include <complex>
#include <oneapi/dpl/random>
#include <oneapi/dpl/algorithm>
#include <blas/linalg.hh>

namespace batchlas {

template <typename T>
std::array<T,2> eigvalsh2x2(const std::array<T,4> &A){
    auto [a,b,c,d] = A;
    T D = sycl::sqrt(4*b*c+(a-d)*(a-d));
    return {(a+d-D)/2, (a+d+D)/2};
}

template <typename T>
void apply_all_reflections(const sycl::group<1> &cta, const Span<T> V, const int n, const int m, T* Q) {
    auto tid = cta.get_local_linear_id();
    auto bdim = cta.get_local_range()[0];
    for(int k = 0; k < n; k++) {
        const T &v0 = V[2*k], &v1 = V[2*k+1];
        for(int l = tid; l < m; l += bdim) {
            T &q0 = Q[k*m+l], &q1 = Q[(k+1)*m+l];
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

    T local_max = T(0.);
    for (int i = tix; i < n; i += bdim){
        local_max = std::max(local_max, std::abs(D[i]) + 2*std::abs(L[i]));
    }
    T max_norm = sycl::reduce_over_group(cta, local_max, sycl::maximum<T>());
    (void)max_norm;
    T numerical_zero = 10*std::numeric_limits<T>::epsilon();
    T d_n, l_n, l_nm1;

    d_n = D[n]; l_n = L[n]; l_nm1 = L[n-1];

    sycl::group_barrier(cta);

    real_t a[2], v[2];
    for(int k = tix; k < n + 1; k += bdim){
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
            if (std::abs(L[k]) > numerical_zero) {
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
        D[n] = d_n;
        L[n-1] = l_nm1;
        L[n] = l_n;
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

    ctx->submit([=](sycl::handler& h) {
        auto Vstride = Q.stride();
        auto Vdata = Q.data_ptr();
        auto Vld = Q.ld();

        sycl::local_accessor<T, 1> D_smem(smem_possible ? n + 1 : 0, h);
        sycl::local_accessor<T, 1> L_smem(smem_possible ? n : 0, h);
        sycl::local_accessor<T, 1> U_smem(smem_possible ? 2 * (n + 1) : 0, h);
        sycl::local_accessor<T, 1> V_smem(smem_possible ? 2 * n : 0, h);

        h.parallel_for(sycl::nd_range<1>(batch_size*32, 32), [=](sycl::nd_item<1> item) {
            auto bid = item.get_group_linear_id();
            auto tid = item.get_local_linear_id();
            auto bdim = item.get_local_range()[0];
            auto cta = item.get_group();

            auto D = Span(smem_possible ? D_smem.begin() : D_global.data() + bid*(n+1), n+1);
            auto L = Span(smem_possible ? L_smem.begin() : L_global.data() + bid*n, n);
            auto U = Span(smem_possible ? U_smem.begin() : U_global.data() + bid*2*(n+1), 2 * (n + 1));
            auto Vlocal = Span(smem_possible ? V_smem.begin() : V_reflectors.data() + bid*2*n, 2 * n);

            auto batch_alphas = Span(alphas.data() + bid * n, n);
            auto batch_betas = Span(betas.data() + bid * n, n);
            T* batch_Q = Vdata + bid * Vstride;
            real_t* batch_W = W.data() + bid * n;

            if (jobz == JobType::EigenVectors) {
                for (int i = tid; i < n*n; i += bdim) {
                    batch_Q[i] = T(0);
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

            for (int k = n-1; k >= 0; k--) {
                T d = D[k];
                T shift = d;
                int i = 0;
                real_t GR = (k > 0 ? std::abs(L[k-1]) : 0) + std::abs(L[k]);
                int not_done = 1;
                while (not_done > 0) {
                    i++;
                    T_QTQ(cta, k+1, D, L, U, Vlocal, shift);
                    if (jobz == JobType::EigenVectors) {
                        apply_all_reflections(cta, Vlocal, k, n, batch_Q);
                    }
                    GR = (k > 0 ? std::abs(L[k-1]) : 0) + (k+1 < n ? std::abs(L[k]) : 0);
                    if (k > 0) {
                        std::array<T,4> args = {D[k-1], L[k-1], L[k-1], D[k]};
                        auto [l0, l1] = eigvalsh2x2(args);
                        shift = std::abs(l0-d) < std::abs(l1-d) ? l0 : l1;
                    } else {
                        shift = D[k];
                    }
                    if (GR <= std::numeric_limits<real_t>::epsilon() * real_t(10.0)) {
                        not_done--;
                    }
                    if (i > 5) {
                        break;
                    }
                }
            }

            for (int i = tid; i < n; i += bdim) {
                if constexpr (sycl::detail::is_complex<T>::value) {
                    batch_W[i] = D[i].real();
                } else {
                    batch_W[i] = D[i];
                }
            }
        });
    });

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

#define TRIDAG_INSTANTIATE(fp) \
    template Event tridiagonal_solver<Backend::CUDA, fp>(Queue&, Span<fp>, Span<fp>, Span<typename base_type<fp>::type>, Span<std::byte>, JobType, const MatrixView<fp, MatrixFormat::Dense>&, size_t, size_t); \
    template size_t tridiagonal_solver_buffer_size<Backend::CUDA, fp>(Queue&, size_t, size_t, JobType);

TRIDAG_INSTANTIATE(float)
TRIDAG_INSTANTIATE(double)
//TRIDAG_INSTANTIATE(std::complex<float>)
//TRIDAG_INSTANTIATE(std::complex<double>)

#undef TRIDAG_INSTANTIATE

} // namespace batchlas

