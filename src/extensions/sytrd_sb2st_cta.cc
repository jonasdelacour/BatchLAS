// Sub-group (CTA-style) SB2ST implementation.
//
// This translation unit intentionally contains ONLY the sub-group/chunked-partition
// kernel path. The baseline implementation and the public sytrd_sb2st entrypoints
// live in sytrd_sb2st.cc.

#include "sytrd_sb2st_cta.hh"

#include <util/group-invoke.hh>

#include <sycl/sycl.hpp>

#include <batchlas/backend_config.h>

#include "../math-helpers.hh"

#include <array>
#include <cstdint>
#include <type_traits>

namespace batchlas::internal {

namespace {

template <typename T>
struct StridedView {
    T* ptr = nullptr;
    int inc = 1;
    inline T& operator[](int i) const { return ptr[i * inc]; }
};

template <typename T>
struct ConstStridedView {
    const T* ptr = nullptr;
    int inc = 1;
    inline const T& operator[](int i) const { return ptr[i * inc]; }
};

template <typename T>
inline StridedView<T> make_strided(T* ptr, int inc) {
    return StridedView<T>{ptr, inc};
}

template <typename T>
inline ConstStridedView<T> make_strided(const T* ptr, int inc) {
    return ConstStridedView<T>{ptr, inc};
}

template <typename T>
inline ConstStridedView<T> make_const_strided(T* ptr, int inc) {
    return ConstStridedView<T>{ptr, inc};
}

template <typename Ops, typename T>
inline void lartv_impl(int nrot, StridedView<T> x, StridedView<T> y, ConstStridedView<typename Ops::Real> c,
                       ConstStridedView<T> s) {
    for (int t = 0; t < nrot; ++t) {
        Ops::rot_pair(x[t], y[t], c[t], s[t]);
    }
}

template <typename Ops, typename T>
inline void largv_impl(int nrot, StridedView<T> x, StridedView<T> w, StridedView<typename Ops::Real> c) {
    for (int t = 0; t < nrot; ++t) {
        const T f = x[t];
        const T g = w[t];
        typename Ops::Real ct = typename Ops::Real(0);
        T st = T(0);
        T rt = T(0);
        Ops::lartg(f, g, ct, st, rt);
        c[t] = ct;
        w[t] = st;
        x[t] = rt;
    }
}

template <typename U>
inline U conj_if_needed(const U& x) {
    if constexpr (internal::is_complex<U>::value) {
        return U(x.real(), -x.imag());
    } else {
        return x;
    }
}

template <typename T>
inline T real_as_T(const T& x) {
    if constexpr (internal::is_complex<T>::value) {
        return T(x.real(), typename T::value_type(0));
    } else {
        return x;
    }
}

template <typename T>
inline typename base_type<T>::type real_part(const T& x) {
    if constexpr (internal::is_complex<T>::value) {
        return static_cast<typename base_type<T>::type>(x.real());
    } else {
        return static_cast<typename base_type<T>::type>(x);
    }
}

// === Band-only SB2ST subgroup path (lower storage) ===

// Complex Hermitian helpers.
template <typename T>
inline typename base_type<T>::type abs_complex(const T& z) {
    using Real = typename base_type<T>::type;
    return sycl::hypot(static_cast<Real>(z.real()), static_cast<Real>(z.imag()));
}

template <typename T, bool IsComplex = internal::is_complex<T>::value>
struct btrd_givens_ops;

// Real/symmetric variant (DSBTRD-style primitives).
template <typename T>
struct btrd_givens_ops<T, false> {
    using Real = typename base_type<T>::type;

    static inline void lartg(const T& f_in, const T& g_in, Real& c, T& s, T& r) {
        auto [ct, st, rt] = internal::lartg(f_in, g_in);
        c = ct;
        s = T(st);
        r = T(rt);
    }

    static inline void rot_pair(T& x, T& y, Real c, const T& s_in) {
        const Real s = static_cast<Real>(s_in);
        const Real xt = static_cast<Real>(x);
        const Real yt = static_cast<Real>(y);
        x = T(c * xt + s * yt);
        y = T(c * yt - s * xt);
    }

    static inline void rot(int nrot, StridedView<T> x, StridedView<T> y, Real c, const T& s_in) {
        for (int t = 0; t < nrot; ++t) {
            rot_pair(x[t], y[t], c, s_in);
        }
    }

    static inline void lartv(int nrot,
                             StridedView<T> x,
                             StridedView<T> y,
                             ConstStridedView<Real> c,
                             ConstStridedView<T> s) {
        lartv_impl<btrd_givens_ops<T, false>, T>(nrot, x, y, c, s);
    }

    static inline void largv(int nrot, StridedView<T> x, StridedView<T> w, StridedView<Real> c) {
        largv_impl<btrd_givens_ops<T, false>, T>(nrot, x, w, c);
    }

    static inline void lar2v_pair(T& x, T& y, T& z, Real ci, const T& si_in) {
        const Real xi = static_cast<Real>(x);
        const Real yi = static_cast<Real>(y);
        const Real zi = static_cast<Real>(z);
        const Real si = static_cast<Real>(si_in);

        const Real t1 = si * zi;
        const Real t2 = ci * zi;
        const Real t3 = t2 - si * xi;
        const Real t4 = t2 + si * yi;
        const Real t5 = ci * xi + t1;
        const Real t6 = ci * yi - t1;

        x = T(ci * t5 + si * t4);
        y = T(ci * t6 - si * t3);
        z = T(ci * t4 - si * t5);
    }

    static inline void lar2v(int nrot,
                             StridedView<T> x,
                             StridedView<T> y,
                             StridedView<T> z,
                             ConstStridedView<Real> c,
                             ConstStridedView<T> s) {
        for (int t = 0; t < nrot; ++t) {
            lar2v_pair(x[t], y[t], z[t], c[t], s[t]);
        }
    }

    static inline void prepare_work_for_right(int /*nrot*/, T* /*s*/, int /*stride*/) {}
    static inline void ensure_band_diag_real(T* /*AB*/, int /*ldab*/, int /*n*/) {}

    static inline void finalize_tridiag(const KernelMatrixView<T, MatrixFormat::Dense>& AB,
                                        int n,
                                        VectorView<Real> D,
                                        VectorView<Real> E,
                                        VectorView<T> TAU) {
        auto diag = AB(int32_t(0), Slice{});
        auto sub = AB(int32_t(1), Slice{});
        for (int j = 0; j < n; ++j) {
            D[j] = static_cast<Real>(diag[j]);
            if (j < n - 1) {
                E[j] = static_cast<Real>(sub[j]);
                TAU[j] = T(0);
            }
        }
    }
};

// Complex/Hermitian variant (ZHBTRD-style primitives).
template <typename T>
struct btrd_givens_ops<T, true> {
    using Real = typename base_type<T>::type;

    static inline void lartg(const T& f, const T& g, Real& c, T& s, T& r) {
        auto [ct, st, rt] = internal::lartg(f, g);
        c = ct;
        s = st;
        r = rt;
    }

    static inline void rot_pair(T& x, T& y, Real c, const T& s) {
        const T xt = x;
        const T yt = y;
        x = T(c) * xt + s * yt;
        y = T(c) * yt - conj_if_needed(s) * xt;
    }

    static inline void rot(int nrot, StridedView<T> x, StridedView<T> y, Real c, const T& s) {
        for (int t = 0; t < nrot; ++t) {
            rot_pair(x[t], y[t], c, s);
        }
    }

    static inline void lartv(int nrot,
                             StridedView<T> x,
                             StridedView<T> y,
                             ConstStridedView<Real> c,
                             ConstStridedView<T> s) {
        lartv_impl<btrd_givens_ops<T, true>, T>(nrot, x, y, c, s);
    }

    static inline void largv(int nrot, StridedView<T> x, StridedView<T> w, StridedView<Real> c) {
        largv_impl<btrd_givens_ops<T, true>, T>(nrot, x, w, c);
    }

    static inline void lar2v_pair(T& x, T& y, T& z, Real ci, const T& si) {
        // Exact port of Netlib LAPACK CLAR2V logic.
        const Real xi = static_cast<Real>(x.real());
        const Real yi = static_cast<Real>(y.real());
        const T zi = z;

        const Real zir = static_cast<Real>(zi.real());
        const Real zii = static_cast<Real>(zi.imag());

        const Real sir = static_cast<Real>(si.real());
        const Real sii = static_cast<Real>(si.imag());

        const Real t1r = sir * zir - sii * zii;
        const Real t1i = sir * zii + sii * zir;

        const T t2 = T(ci * zir, ci * zii);
        const T t3 = t2 - conj_if_needed(si) * T(xi, Real(0));
        const T t4 = conj_if_needed(t2) + si * T(yi, Real(0));

        const Real t5 = ci * xi + t1r;
        const Real t6 = ci * yi - t1r;

        const Real xnew = ci * t5 + (sir * static_cast<Real>(t4.real()) + sii * static_cast<Real>(t4.imag()));
        const Real ynew = ci * t6 - (sir * static_cast<Real>(t3.real()) - sii * static_cast<Real>(t3.imag()));
        const T znew = T(ci) * t3 + conj_if_needed(si) * T(t6, t1i);

        x = T(xnew, Real(0));
        y = T(ynew, Real(0));
        z = znew;
    }

    static inline void lar2v(int nrot,
                             StridedView<T> x,
                             StridedView<T> y,
                             StridedView<T> z,
                             ConstStridedView<Real> c,
                             ConstStridedView<T> s) {
        for (int t = 0; t < nrot; ++t) {
            lar2v_pair(x[t], y[t], z[t], c[t], s[t]);
        }
    }

    static inline void prepare_work_for_right(int nrot, T* s, int stride) {
        // ZHBTRD requirement: conjugate WORK(J1) vector before applying from the right.
        for (int t = 0; t < nrot; ++t) {
            s[t * stride] = conj_if_needed(s[t * stride]);
        }
    }

    static inline void ensure_band_diag_real(T* AB, int ldab, int n) {
        // Ensure diagonal is real (ZHBTRD does this).
        for (int j = 0; j < n; ++j) {
            const int idx = 0 + j * ldab;
            AB[idx] = real_as_T(AB[idx]);
        }
    }

    static inline void finalize_tridiag(KernelMatrixView<T, MatrixFormat::Dense> AB,
                                        int n,
                                        VectorView<Real> D,
                                        VectorView<Real> E,
                                        VectorView<T> TAU) {
        auto diag = AB(int32_t(0), Slice{});
        auto sub = AB(int32_t(1), Slice{});

        // Make diagonal real.
        for (int j = 0; j < n; ++j) {
            diag[j] = real_as_T(diag[j]);
        }

        // Make subdiagonal real (phase chaining) and extract D/E.
        for (int j = 0; j < n - 1; ++j) {
            const T t = sub[j];
            const Real a = abs_complex(t);
            E[j] = a;
            T phase = T(1);
            if (a != Real(0)) {
                phase = t / T(a);
            }
            sub[j] = T(a, Real(0));
            if (j < n - 2) {
                sub[j + 1] *= phase;
            }
            TAU[j] = T(0);
        }

        for (int j = 0; j < n; ++j) {
            D[j] = static_cast<Real>(real_part(diag[j]));
        }
    }
};

template <typename T, int32_t P>
class BtrdLowerSubgroupKernel;

template <typename T, int32_t P>
Event btrd_lower_inplace_subgroup(Queue& q,
                                  KernelMatrixView<T, MatrixFormat::Dense> ab,
                                  int n,
                                  int kd,
                                  VectorView<typename base_type<T>::type> c,
                                  VectorView<T> work,
                                  VectorView<typename base_type<T>::type> d,
                                  VectorView<typename base_type<T>::type> e,
                                  VectorView<T> tau) {
    static_assert(P == 2 || P == 4 || P == 8 || P == 16 || P == 32);
    constexpr int32_t sg_size = 32;
    static_assert(sg_size % P == 0);

    using Real = typename base_type<T>::type;

    // --- Optional: stage the band matrix AB in local memory (tight-packed ldab = kd+1).
    // This is gated by device local memory size and the per-work-group footprint.
    const size_t lmem_bytes = q->get_device().get_info<sycl::info::device::local_mem_size>();
    constexpr int32_t wg_size = 32;
    constexpr int32_t probs_per_wg = wg_size / P;

    // Tight-packed band leading dimension in local memory.
    const int ldab_local_host = kd + 1;
    const int cw_len_host = n + kd + 2; // size of C/WORK scratch per problem

    const size_t ab_bytes_per_wg =
        size_t(ldab_local_host) * size_t(n) * sizeof(T) * size_t(probs_per_wg);
    const size_t cw_bytes_per_wg =
        size_t(cw_len_host) * (sizeof(Real) + sizeof(T)) * size_t(probs_per_wg);

    // Leave headroom (25%) for compiler/runtime usage.
    const bool use_local_ab = (ab_bytes_per_wg > 0) && ((ab_bytes_per_wg + cw_bytes_per_wg) <= (lmem_bytes * 3) / 4);

    // local_accessor must be non-zero sized.
    const size_t ab_local_elems = use_local_ab
                                      ? (size_t(ldab_local_host) * size_t(n) * size_t(probs_per_wg))
                                      : size_t(1);
    const size_t cw_local_elems = (cw_len_host > 0)
                                      ? (size_t(cw_len_host) * size_t(probs_per_wg))
                                      : size_t(1);
    (void)q->submit([&](sycl::handler& h) {
        const int32_t batch = static_cast<int32_t>(ab.batch_size());

        // Tuneable: keep modest to avoid spilling local memory; SB2ST is fairly heavy.
        constexpr int32_t wg_size = 32;
        static_assert(wg_size % P == 0);
        const int32_t probs_per_wg = wg_size / P;
        const int32_t num_wg = (batch + probs_per_wg - 1) / probs_per_wg;
        const int32_t global = num_wg * wg_size;

        sycl::local_accessor<T, 1> ab_local(sycl::range<1>(ab_local_elems), h);
        sycl::local_accessor<Real, 1> c_local(sycl::range<1>(cw_local_elems), h);
        sycl::local_accessor<T, 1> work_local(sycl::range<1>(cw_local_elems), h);

        h.parallel_for<BtrdLowerSubgroupKernel<T, P>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(global)),
                              sycl::range<1>(static_cast<size_t>(wg_size))),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                using Real = typename base_type<T>::type;
                using Ops = btrd_givens_ops<T>;

                const auto wg = it.get_group();
                const int32_t wg_id = static_cast<int32_t>(wg.get_group_linear_id());
                const auto sg = it.get_sub_group();
                const auto partition = sycl::ext::oneapi::experimental::chunked_partition<P>(sg);

                // NOTE: chunked_partition<P>(sg) partitions *within a sub-group*.
                // If the work-group contains multiple sub-groups, partition.get_group_linear_id()
                // repeats for each sub-group. Make part_id unique within the whole work-group.
                const int32_t sg_id = static_cast<int32_t>(sg.get_group_linear_id());
                const int32_t parts_per_sg = static_cast<int32_t>(partition.get_group_linear_range());
                const int32_t part_id = sg_id * parts_per_sg + static_cast<int32_t>(partition.get_group_linear_id());
                const int32_t lane = static_cast<int32_t>(partition.get_local_linear_id());
                const int32_t prob_id = wg_id * probs_per_wg + part_id;
                if (prob_id >= batch) return;

                auto ABv = ab.batch_item(prob_id);
                auto Dv = d.batch_item(prob_id);
                auto Ev = e.batch_item(prob_id);
                auto TAUv = tau.batch_item(prob_id);

                const int ldab_g = ABv.ld();
                const int ldab_l = kd + 1;

                T* AB = ABv.data();
                T* ABg = AB; // original global pointer for copy-back

                Real* C = c_local.template get_multi_ptr<sycl::access::decorated::no>().get() + part_id * (n + kd + 2);
                T* WORK = work_local.template get_multi_ptr<sycl::access::decorated::no>().get() + part_id * (n + kd + 2);

                auto& D = Dv;
                auto& E = Ev;
                auto& TAU = TAUv;

                // Select the active leading dimension (global or tight-packed local).
                int ldab = ldab_g;

                for (int i = lane; i < n + kd + 2; i += P) {
                    C[i] = Real(0);
                    WORK[i] = T(0);
                }
                sycl::group_barrier(partition);

                if (use_local_ab) {
                    T* ABl_all = ab_local.template get_multi_ptr<sycl::access::decorated::no>().get();
                    const int ab_elems_l = ldab_l * n;
                    const int local_base = part_id * ab_elems_l;

                    // Copy (r,c) from global (ldab_g) into local (ldab_l) so shared accesses are bank-friendlier.
                    for (int t = lane; t < ab_elems_l; t += P) {
                        const int ccol = t / ldab_l;
                        const int rrow = t - ccol * ldab_l;
                        ABl_all[local_base + t] = ABg[rrow + ccol * ldab_g];
                    }
                    sycl::group_barrier(partition);
                    AB = ABl_all + local_base;
                    ldab = ldab_l;
                }

                const int kd1 = kd + 1;
                const int kdm1 = kd - 1;
                const int incx = ldab - 1;
                const int inca = kd1 * ldab;
                const int kdn = (n - 1 < kd) ? (n - 1) : kd;

                auto idx = [&](int r, int c) -> int { return r + c * ldab; };


                // Unified accessor for AB that works whether AB points to global or local storage.
                auto AB_at = [&](int r, int c) -> T& { return AB[idx(r, c)]; };

                int nr = 0;
                int j1 = kdn + 1;
                int j2 = 0;

                // Ensure diagonal is real (for Hermitian path). Parallelize across columns.
                if constexpr (internal::is_complex<T>::value) {
                    for (int j = lane; j < n; j += P) {
                        AB_at(0, j) = real_as_T(AB_at(0, j));
                    }
                    sycl::group_barrier(partition);
                }

                for (int i = 0; i < n - 2; ++i) {
                    for (int k = kdn; k >= 1; --k) {
                        // All lanes share the same scalar state for the partition's problem.
                        j1 += kdn;
                        j2 += kdn;

                        if (nr > 0) {
                            // Parallel LARGV over rotations.
                            for (int t = lane; t < nr; t += P) {
                                const int j = j1 + t * kd1;
                                const T f = AB_at(kd, j - kd1);
                                const T g = WORK[j];
                                Real ct = Real(0);
                                T st = T(0);
                                T rt = T(0);
                                Ops::lartg(f, g, ct, st, rt);
                                C[j] = ct;
                                WORK[j] = st;
                                AB_at(kd, j - kd1) = rt;
                            }
                            sycl::group_barrier(partition);

                            if (nr > 2 * kd - 1) {
                                // Keep the l-loop sequential, but parallelize across rotations within each l.
                                // Rotations are independent across t; we only need one barrier after all l steps.
                                for (int l = 0; l < kd - 1; ++l) {
                                    for (int t = lane; t < nr; t += P) {
                                        const int j = j1 + t * kd1;
                                        Ops::rot_pair(AB_at(kd - 1 - l, j - kd1 + l + 1),
                                                      AB_at(kd - l, j - kd1 + l + 1),
                                                      C[j],
                                                      WORK[j]);
                                    }
                                }
                                sycl::group_barrier(partition);
                            } else {
                                // Parallelize across rotations (t). Each lane owns a subset of j = j1 + t*kd1.
                                if (kdm1 > 0) {
                                    for (int t = lane; t < nr; t += P) {
                                        const int j = j1 + t * kd1;
                                        Ops::rot(kdm1,
                                                 make_strided(AB + idx(kd - 1, j - kd), incx),
                                                 make_strided(AB + idx(kd, j - kd), incx),
                                                 C[j],
                                                 WORK[j]);
                                    }
                                }
                                sycl::group_barrier(partition);
                            }
                        }

                        // Scalar bulge creation and scalar-state update; broadcast updated (nr, j1).
                        const auto nr_j1 = invoke_one_broadcast(partition, [&]() {
                            int nr_out = nr;
                            int j1_out = j1;
                            if (k > 1) {
                                    if (k <= n - i - 1) {
                                        const T f = AB_at(k - 1, i);
                                        const T g = AB_at(k, i);
                                        Real ct = Real(0);
                                        T st = T(0);
                                        T rt = T(0);
                                        Ops::lartg(f, g, ct, st, rt);
                                        C[i + k] = ct;
                                        WORK[i + k] = st;
                                        AB_at(k - 1, i) = rt;

                                    if (k > 2) {
                                        const Real ct2 = C[i + k];
                                        const T st2 = WORK[i + k];
                                        Ops::rot(k - 2,
                                                 make_strided(AB + idx(k - 2, i + 1), ldab - 1),
                                                 make_strided(AB + idx(k - 1, i + 1), ldab - 1),
                                                 ct2,
                                                 st2);
                                    }
                                }
                                nr_out += 1;
                                j1_out = j1_out - kdn - 1;
                            }
                            return std::array<int, 2>{nr_out, j1_out};
                        });
                        nr = nr_j1[0];
                        j1 = nr_j1[1];

                        if (nr > 0) {
                            // Parallel LAR2V over rotations (nrot==1 fast path).
                            for (int t = lane; t < nr; t += P) {
                                const int j = j1 + t * kd1;
                                Ops::lar2v_pair(AB_at(0, j - 1),
                                               AB_at(0, j),
                                               AB_at(1, j - 1),
                                               C[j],
                                               WORK[j]);
                            }
                            sycl::group_barrier(partition);

                            // ZHBTRD requirement: conjugate WORK(J1) before right-application.
                            if constexpr (internal::is_complex<T>::value) {
                                for (int t = lane; t < nr; t += P) {
                                    const int j = j1 + t * kd1;
                                    WORK[j] = conj_if_needed(WORK[j]);
                                }
                                sycl::group_barrier(partition);
                            }

                            if (nr > 2 * kd - 1) {
                                for (int l = 0; l < kd - 1; ++l) {
                                    const int nrt = (j2 + l + 2 > n) ? (nr - 1) : nr;
                                    if (nrt > 0) {
                                        for (int t = lane; t < nrt; t += P) {
                                            const int j = j1 + t * kd1;
                                            Ops::rot_pair(AB_at(l + 2, j - 1),
                                                          AB_at(l + 1, j),
                                                          C[j],
                                                          WORK[j]);
                                        }
                                    }
                                }
                                sycl::group_barrier(partition);
                            } else {
                                // Parallelize across rotations (t) for the main body (0..nr-2).
                                const int main_nrot = nr - 1;
                                if (kdm1 > 0 && main_nrot > 0) {
                                    for (int t = lane; t < main_nrot; t += P) {
                                        const int j = j1 + t * kd1;
                                        Ops::rot(kdm1,
                                                 make_strided(AB + idx(2, j - 1), 1),
                                                 make_strided(AB + idx(1, j), 1),
                                                 C[j],
                                                 WORK[j]);
                                    }
                                }

                                // Tail rotation ("last") has different length (lend). Do it once.
                                if (lane == 0) {
                                    const int j1end = j1 + kd1 * (nr - 2);
                                    const int lend = (n - j2 - 1 < kdm1) ? (n - j2 - 1) : kdm1;
                                    const int last = j1end + kd1;
                                    if (lend > 0) {
                                        Ops::rot(lend,
                                                 make_strided(AB + idx(2, last - 1), 1),
                                                 make_strided(AB + idx(1, last), 1),
                                                 C[last],
                                                 WORK[last]);
                                    }
                                }
                                sycl::group_barrier(partition);
                            }
                        }

                            // Scalar-state update based on j2; broadcast updated (nr, j2).
                            const auto nr_j2 = invoke_one_broadcast(partition, [&]() {
                                int nr_out = nr;
                                int j2_out = j2;
                                if (j2_out + kdn >= n) {
                                    nr_out -= 1;
                                    j2_out = j2_out - kdn - 1;
                                }
                                return std::array<int, 2>{nr_out, j2_out};
                            });
                            nr = nr_j2[0];
                            j2 = nr_j2[1];

                        if (nr > 0) {
                            for (int t = lane; t < nr; t += P) {
                                const int j = j1 + t * kd1;
                                WORK[j + kd] = WORK[j] * AB_at(kd, j);
                                AB_at(kd, j) = T(C[j]) * AB_at(kd, j);
                            }
                            sycl::group_barrier(partition);
                        }
                    }
                }

                // Write back staged AB to global memory before finalization.
                if (use_local_ab) {
                    sycl::group_barrier(partition);
                    const int ab_elems_l = ldab_l * n;
                    for (int t = lane; t < ab_elems_l; t += P) {
                        const int ccol = t / ldab_l;
                        const int rrow = t - ccol * ldab_l;
                        ABg[rrow + ccol * ldab_g] = AB[t];
                    }
                    sycl::group_barrier(partition);
                }

                invoke_one(partition, [&]() {
                    Ops::finalize_tridiag(ABv, n, D, E, TAU);
                });
            });
    });
    return q.get_event();
}

} // namespace

template <typename T>
Event btrd_lower_inplace_subgroup_dispatch(Queue& q,
                                           KernelMatrixView<T, MatrixFormat::Dense> ab,
                                           int n,
                                           int kd,
                                           VectorView<typename base_type<T>::type> c,
                                           VectorView<T> work,
                                           VectorView<typename base_type<T>::type> d,
                                           VectorView<typename base_type<T>::type> e,
                                           VectorView<T> tau) {
    // Choose a partition size that keeps enough lanes for typical kd<=32 while
    // allowing smaller partitions when kd is very small.
    if (kd <= 2) {
        return btrd_lower_inplace_subgroup<T, 2>(q, ab, n, kd, c, work, d, e, tau);
    }
    if (kd <= 4) {
        return btrd_lower_inplace_subgroup<T, 4>(q, ab, n, kd, c, work, d, e, tau);
    }
    if (kd <= 8) {
        return btrd_lower_inplace_subgroup<T, 8>(q, ab, n, kd, c, work, d, e, tau);
    }
    if (kd <= 16) {
        return btrd_lower_inplace_subgroup<T, 16>(q, ab, n, kd, c, work, d, e, tau);
    }
    return btrd_lower_inplace_subgroup<T, 32>(q, ab, n, kd, c, work, d, e, tau);
}

// ---- Explicit template instantiations (needed because sytrd_sb2st.cc only sees the declaration) ----
template Event btrd_lower_inplace_subgroup_dispatch<float>(Queue&,
                                                           KernelMatrixView<float, MatrixFormat::Dense>,
                                                           int,
                                                           int,
                                                           VectorView<float>,
                                                           VectorView<float>,
                                                           VectorView<float>,
                                                           VectorView<float>,
                                                           VectorView<float>);
template Event btrd_lower_inplace_subgroup_dispatch<double>(Queue&,
                                                            KernelMatrixView<double, MatrixFormat::Dense>,
                                                            int,
                                                            int,
                                                            VectorView<double>,
                                                            VectorView<double>,
                                                            VectorView<double>,
                                                            VectorView<double>,
                                                            VectorView<double>);
template Event btrd_lower_inplace_subgroup_dispatch<std::complex<float>>(
    Queue&,
    KernelMatrixView<std::complex<float>, MatrixFormat::Dense>,
    int,
    int,
    VectorView<float>,
    VectorView<std::complex<float>>,
    VectorView<float>,
    VectorView<float>,
    VectorView<std::complex<float>>);
template Event btrd_lower_inplace_subgroup_dispatch<std::complex<double>>(
    Queue&,
    KernelMatrixView<std::complex<double>, MatrixFormat::Dense>,
    int,
    int,
    VectorView<double>,
    VectorView<std::complex<double>>,
    VectorView<double>,
    VectorView<double>,
    VectorView<std::complex<double>>);

} // namespace batchlas::internal
