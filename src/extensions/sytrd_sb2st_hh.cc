// Stage-2 band -> tridiagonal by Householder bulge chasing, retaining the
// reflectors for the eigenvector back-transform. See sytrd_sb2st_hh.hh for the
// rationale and for the schedule, which is validated in
// playground/sb2st_hh_sequential.py.

#include <blas/extensions.hh>
#include <blas/matrix.hh>
#include <util/mempool.hh>

#include <sycl/sycl.hpp>

#include <batchlas/backend_config.h>

#include "../math-helpers.hh"
#include "../queue.hh"
#include "../util/template-instantiations.hh"

#include "sytrd_sb2st_hh.hh"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace batchlas {
namespace internal {

namespace {

template <typename U>
inline U conj_if(const U& x) {
    if constexpr (internal::is_complex<U>::value) {
        return U(x.real(), -x.imag());
    } else {
        return x;
    }
}

template <typename T>
inline T real_part_as_T(const T& x) {
    if constexpr (internal::is_complex<T>::value) {
        return T(x.real(), typename base_type<T>::type(0));
    } else {
        return x;
    }
}

template <typename T>
inline typename base_type<T>::type abs2(const T& x) {
    using R = typename base_type<T>::type;
    if constexpr (internal::is_complex<T>::value) {
        return static_cast<R>(x.real()) * static_cast<R>(x.real()) +
               static_cast<R>(x.imag()) * static_cast<R>(x.imag());
    } else {
        return static_cast<R>(x) * static_cast<R>(x);
    }
}

// Group-wide sum that also works for std::complex, which sycl::plus<> does not
// accept directly.
template <typename Group, typename T>
inline T group_sum(Group g, T v) {
    using R = typename base_type<T>::type;
    if constexpr (internal::is_complex<T>::value) {
        const R re = sycl::reduce_over_group(g, static_cast<R>(v.real()), sycl::plus<R>());
        const R im = sycl::reduce_over_group(g, static_cast<R>(v.imag()), sycl::plus<R>());
        return T(re, im);
    } else {
        return sycl::reduce_over_group(g, v, sycl::plus<T>());
    }
}

template <Backend B, typename T>
class Sb2stHhChaseKernel;

constexpr int kWg = 32;

} // namespace

template <Backend B, typename T>
size_t sytrd_sb2st_hh_buffer_size(Queue& ctx, int32_t n, int32_t kd, int32_t batch) {
    if (n <= 0 || batch <= 0) return 0;
    const int32_t kdw = sb2st_hh_work_bandwidth(n, kd);
    size_t size = 0;
    size += BumpAllocator::allocation_size<T>(
        ctx, static_cast<size_t>(kdw + 1) * static_cast<size_t>(n) * static_cast<size_t>(batch));
    return size;
}

// ab_in      : (kd+1) x n  lower band, read-only
// ab_tri_out : 2 x n       row 0 = diagonal, row 1 = *signed* subdiagonal
//              (kept signed so build_phase_from_kd1_band works unchanged)
// d_out/e_out: real diagonal and |subdiagonal|
// v_out      : kd x nrefl  reflector k in column k, v[0] = 1, zero-padded
// tau_out    : nrefl
template <Backend B, typename T>
Event sytrd_sb2st_hh(Queue& ctx,
                     const MatrixView<T, MatrixFormat::Dense>& ab_in,
                     const MatrixView<T, MatrixFormat::Dense>& ab_tri_out,
                     const VectorView<typename base_type<T>::type>& d_out,
                     const VectorView<typename base_type<T>::type>& e_out,
                     const MatrixView<T, MatrixFormat::Dense>& v_out,
                     const VectorView<T>& tau_out,
                     Uplo uplo,
                     int32_t kd,
                     const Span<std::byte>& ws) {
    using Real = typename base_type<T>::type;

    if (!ctx.in_order()) {
        throw std::runtime_error("sytrd_sb2st_hh: requires an in-order Queue");
    }
    if (uplo != Uplo::Lower) {
        throw std::runtime_error("sytrd_sb2st_hh: only Uplo::Lower is implemented");
    }

    const int32_t n = static_cast<int32_t>(ab_in.cols());
    const int32_t batch = static_cast<int32_t>(ab_in.batch_size());
    if (n <= 0 || batch <= 0) return ctx.get_event();

    const int32_t kd_i = std::max<int32_t>(0, kd);
    const int32_t kdw = sb2st_hh_work_bandwidth(n, kd_i);

    BumpAllocator pool(ws);

    // Expanded working band: transient bulge fill reaches kd rows below the band.
    auto abw = pool.allocate<T>(
        ctx, static_cast<size_t>(kdw + 1) * static_cast<size_t>(n) * static_cast<size_t>(batch));

    const int32_t ldw = kdw + 1;

    auto ABsrc = ab_in.kernel_view();
    auto ABtri = ab_tri_out.kernel_view();
    auto Vout = v_out.kernel_view();
    const int32_t nrefl = static_cast<int32_t>(v_out.cols());
    const int32_t ldv = static_cast<int32_t>(v_out.ld());
    T* abw_ptr = abw.data();

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<T, 1> vloc(sycl::range<1>(static_cast<size_t>(std::max(1, kd_i))), h);
        sycl::local_accessor<T, 1> wloc(sycl::range<1>(static_cast<size_t>(std::max(1, kd_i))), h);

        auto Dv = d_out;
        auto Ev = e_out;
        auto TAUv = tau_out;

        h.parallel_for<Sb2stHhChaseKernel<B, T>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(batch) * kWg),
                              sycl::range<1>(kWg)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const auto wg = it.get_group();
                const int32_t b = static_cast<int32_t>(wg.get_group_linear_id());
                const int32_t lid = static_cast<int32_t>(it.get_local_linear_id());
                if (b >= batch) return;

                T* AB = abw_ptr + static_cast<size_t>(b) * static_cast<size_t>(ldw) *
                                      static_cast<size_t>(n);
                auto ABs = ABsrc.batch_item(b);
                auto ABt = ABtri.batch_item(b);
                auto Vb = Vout.batch_item(b);

                // --- Load the input band into the expanded working band.
                for (int32_t idx = lid; idx < ldw * n; idx += kWg) {
                    const int32_t col = idx / ldw;
                    const int32_t row = idx - col * ldw;
                    AB[idx] = (row <= kd_i) ? ABs(row, col) : T(0);
                }
                sycl::group_barrier(wg);

                // Hermitian input: force a real diagonal before we start.
                if constexpr (internal::is_complex<T>::value) {
                    for (int32_t j = lid; j < n; j += kWg) {
                        AB[j * ldw] = real_part_as_T(AB[j * ldw]);
                    }
                    sycl::group_barrier(wg);
                }

                auto bget = [&](int32_t i, int32_t j) -> T {
                    if (i >= j) {
                        const int32_t r = i - j;
                        return (r <= kdw) ? AB[r + j * ldw] : T(0);
                    }
                    const int32_t r = j - i;
                    return (r <= kdw) ? conj_if(AB[r + i * ldw]) : T(0);
                };
                auto bset = [&](int32_t i, int32_t j, T val) {
                    if (i >= j) {
                        const int32_t r = i - j;
                        if (r <= kdw) AB[r + j * ldw] = val;
                    } else {
                        const int32_t r = j - i;
                        if (r <= kdw) AB[r + i * ldw] = conj_if(val);
                    }
                };

                // W <- H W H^H on the principal block [a..b], H = I - tau v v^H,
                // v held in vloc with length m. Only the lower triangle is
                // written; the update keeps the diagonal real by construction.
                auto two_sided = [&](int32_t a, int32_t bb, T tau) {
                    const int32_t m = bb - a + 1;
                    if (m <= 0) return;
                    for (int32_t i = lid; i < m; i += kWg) {
                        T acc = T(0);
                        for (int32_t j = 0; j < m; ++j) {
                            acc += bget(a + i, a + j) * vloc[j];
                        }
                        wloc[i] = acc;
                    }
                    sycl::group_barrier(wg);

                    // kappa = v^H W v
                    T part = T(0);
                    for (int32_t i = lid; i < m; i += kWg) {
                        part += conj_if(vloc[i]) * wloc[i];
                    }
                    const T kappa = group_sum(wg, part);

                    // w = conj(tau) p - (|tau|^2 kappa / 2) v
                    const T coef = tau * conj_if(tau) * kappa * T(Real(0.5));
                    for (int32_t i = lid; i < m; i += kWg) {
                        wloc[i] = conj_if(tau) * wloc[i] - coef * vloc[i];
                    }
                    sycl::group_barrier(wg);

                    for (int32_t i = lid; i < m; i += kWg) {
                        for (int32_t j = 0; j <= i; ++j) {
                            T val = bget(a + i, a + j) - wloc[i] * conj_if(vloc[j]) -
                                    vloc[i] * conj_if(wloc[j]);
                            if (i == j) val = real_part_as_T(val);
                            bset(a + i, a + j, val);
                        }
                    }
                    sycl::group_barrier(wg);
                };

                // B <- B (I - tau v v^H), v of length (c1-c0+1) in vloc.
                auto right_apply = [&](int32_t r0, int32_t r1, int32_t c0, int32_t c1, T tau) {
                    if (r0 > r1 || c0 > c1) return;
                    for (int32_t i = r0 + lid; i <= r1; i += kWg) {
                        T y = T(0);
                        for (int32_t j = c0; j <= c1; ++j) {
                            y += bget(i, j) * vloc[j - c0];
                        }
                        y = tau * y;
                        for (int32_t j = c0; j <= c1; ++j) {
                            bset(i, j, bget(i, j) - y * conj_if(vloc[j - c0]));
                        }
                    }
                    sycl::group_barrier(wg);
                };

                // C <- (I - tau v v^H) C, v of length (r1-r0+1) in vloc.
                auto left_apply = [&](int32_t r0, int32_t r1, int32_t c0, int32_t c1, T tau) {
                    if (r0 > r1 || c0 > c1) return;
                    for (int32_t j = c0 + lid; j <= c1; j += kWg) {
                        T z = T(0);
                        for (int32_t i = r0; i <= r1; ++i) {
                            z += conj_if(vloc[i - r0]) * bget(i, j);
                        }
                        z = tau * z;
                        for (int32_t i = r0; i <= r1; ++i) {
                            bset(i, j, bget(i, j) - vloc[i - r0] * z);
                        }
                    }
                    sycl::group_barrier(wg);
                };

                // Build a reflector from column `col`, rows [r0..r1]; write it to
                // vloc, store it at reflector slot `slot`, and overwrite the
                // column with (beta, 0, ..., 0). Returns tau.
                auto make_reflector = [&](int32_t col, int32_t r0, int32_t r1,
                                          int32_t slot) -> T {
                    const int32_t m = r1 - r0 + 1;
                    Real partial = Real(0);
                    for (int32_t k = lid + 1; k < m; k += kWg) {
                        partial += abs2(bget(r0 + k, col));
                    }
                    const Real ss = sycl::reduce_over_group(wg, partial, sycl::plus<Real>());
                    const Real xnorm = sycl::sqrt(ss);
                    const T alpha = bget(r0, col);
                    const auto res = internal::larfg<T>(alpha, xnorm, m);

                    // v[0] = 1, v[k] = x[k] * scale. Read the column before it is
                    // overwritten below.
                    if (lid == 0) vloc[0] = T(1);
                    for (int32_t k = lid + 1; k < m; k += kWg) {
                        vloc[k] = (res.tau == T(0)) ? T(0) : bget(r0 + k, col) * res.scale;
                    }
                    sycl::group_barrier(wg);

                    if (lid == 0) bset(r0, col, res.beta);
                    for (int32_t k = lid + 1; k < m; k += kWg) {
                        bset(r0 + k, col, T(0));
                    }

                    for (int32_t k = lid; k < kd_i; k += kWg) {
                        Vb(k, slot) = (k < m) ? vloc[k] : T(0);
                    }
                    if (lid == 0) TAUv(slot, b) = res.tau;
                    sycl::group_barrier(wg);
                    return res.tau;
                };

                // --- Sequential chase schedule. The reflector counter must match
                // build_sb2st_hh_schedule() exactly, since the back-transform
                // indexes V by the host-side schedule.
                int32_t slot = 0;
                if (kd_i > 1) {
                    for (int32_t st = 0; st + 2 < n; ++st) {
                        int32_t r0 = st + 1;
                        int32_t r1 = (st + kd_i < n - 1) ? (st + kd_i) : (n - 1);
                        if (r1 <= r0) continue;

                        T tau = make_reflector(st, r0, r1, slot++);
                        two_sided(r0, r1, tau);

                        while (true) {
                            const int32_t p0 = r1 + 1;
                            const int32_t p1 = (r1 + kd_i < n - 1) ? (r1 + kd_i) : (n - 1);
                            if (p0 > p1) break;

                            right_apply(p0, p1, r0, r1, tau);
                            tau = make_reflector(r0, p0, p1, slot++);
                            left_apply(p0, p1, r0 + 1, r1, tau);
                            two_sided(p0, p1, tau);

                            r0 = p0;
                            r1 = p1;
                        }
                    }
                }
                (void)nrefl;
                (void)ldv;

                // --- Extract the tridiagonal.
                for (int32_t j = lid; j < n; j += kWg) {
                    const T diag = bget(j, j);
                    ABt(0, j) = real_part_as_T(diag);
                    if constexpr (internal::is_complex<T>::value) {
                        Dv(j, b) = static_cast<Real>(diag.real());
                    } else {
                        Dv(j, b) = static_cast<Real>(diag);
                    }
                }
                for (int32_t j = lid; j < n - 1; j += kWg) {
                    const T sub = bget(j + 1, j);
                    ABt(1, j) = sub;  // signed: the phase pass consumes this
                    Ev(j, b) = internal::abs(sub);
                }
                if (lid == 0 && n > 0) ABt(1, n - 1) = T(0);
            });
    });

    return ctx.get_event();
}

#define SB2ST_HH_INSTANTIATE(back, fp)                                                  \
    template Event sytrd_sb2st_hh<back, BATCHLAS_UNPAREN fp>(                           \
        Queue&,                                                                          \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&,                     \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&,                     \
        const VectorView<typename base_type<BATCHLAS_UNPAREN fp>::type>&,                \
        const VectorView<typename base_type<BATCHLAS_UNPAREN fp>::type>&,                \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&,                     \
        const VectorView<BATCHLAS_UNPAREN fp>&,                                          \
        Uplo,                                                                            \
        int32_t,                                                                         \
        const Span<std::byte>&);                                                         \
    template size_t sytrd_sb2st_hh_buffer_size<back, BATCHLAS_UNPAREN fp>(               \
        Queue&, int32_t, int32_t, int32_t);

#define SB2ST_HH_INSTANTIATE_FOR_BACKEND(back) \
    BATCHLAS_FOR_EACH_SCALAR_TYPE_1(SB2ST_HH_INSTANTIATE, back)

#if BATCHLAS_HAS_CUDA_BACKEND
SB2ST_HH_INSTANTIATE_FOR_BACKEND(Backend::CUDA)
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
SB2ST_HH_INSTANTIATE_FOR_BACKEND(Backend::ROCM)
#endif
#if BATCHLAS_HAS_HOST_BACKEND
SB2ST_HH_INSTANTIATE_FOR_BACKEND(Backend::NETLIB)
#endif

#undef SB2ST_HH_INSTANTIATE_FOR_BACKEND
#undef SB2ST_HH_INSTANTIATE

} // namespace internal
} // namespace batchlas
