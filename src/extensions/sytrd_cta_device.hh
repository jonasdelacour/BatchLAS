#pragma once

// Device-side CTA SYTRD, shared by sytrd_cta.cc and syev_cta_fused.cc: one definition keeps
// the fused and partitioned syev numerically identical, so comparing them measures fusion only.

#include <batchlas/blas/matrix.hh>
#include <batchlas/util/group-invoke.hh>
#include "sg_compat.hh"
#include "../math-helpers.hh"

#include <cstdint>

namespace batchlas {

    template <typename U>
    inline U conj_if_complex(const U& x) {
        if constexpr (internal::is_complex<U>::value) {
            return U(x.real(), -x.imag());
        } else {
            return x;
        }
    }

    template <typename U>
    inline typename base_type<U>::type abs2_if_complex(const U& x) {
        using Real = typename base_type<U>::type;
        if constexpr (internal::is_complex<U>::value) {
            const Real re = x.real();
            const Real im = x.imag();
            return re * re + im * im;
        } else {
            return x * x;
        }
    }

    // XOR-butterfly reductions: the group size must be a power of two.
    template <typename T, typename Group>
    inline T group_reduce_sum(const Group& g, T v) {
        for (uint32_t offset = static_cast<uint32_t>(g.get_local_linear_range() / 2);
             offset > 0;
             offset >>= 1) {
            v += permute_group_by_xor(g, v, offset);
        }
        return v;
    }

    template <typename T, typename Group>
    inline T group_reduce_max(const Group& g, T v) {
        for (uint32_t offset = static_cast<uint32_t>(g.get_local_linear_range() / 2);
             offset > 0;
             offset >>= 1) {
            const T other = permute_group_by_xor(g, v, offset);
            v = sycl::fmax(v, other);
        }
        return v;
    }

    // Shuffles, not reduce_over_group: DPC++'s CUDA path cannot reduce floating point over a
    // chunked partition. The result is replicated in every lane.
    template <typename T, typename Group>
    inline T group_reduce_sum_select_from_group(const Group& g, T v) {
        const uint32_t lanes = static_cast<uint32_t>(g.get_local_linear_range());
        (void)lanes;

        if constexpr (internal::is_complex<T>::value) {
            using Real = typename base_type<T>::type;
            Real re = v.real();
            Real im = v.imag();
            for (uint32_t offset = lanes / 2; offset > 0; offset >>= 1) {
                re += permute_group_by_xor(g, re, offset);
                im += permute_group_by_xor(g, im, offset);
            }
            return T(re, im);
        } else {
            for (uint32_t offset = lanes / 2; offset > 0; offset >>= 1) {
                v += permute_group_by_xor(g, v, offset);
            }
            return v;
        }
    }

    // DLARFG over a partition: alpha lives in lane alpha_lane, x in the others (inactive lanes
    // pass 0). Overwrites alpha with beta and x with the scaled v (implicit 1); returns tau.
    template <typename T, typename Partition>
    inline T larfg_small(const Partition& part,
                         int32_t len,
                         int32_t lane,
                         int32_t alpha_lane,
                         T& alpha,
                         T& x,
                         bool x_active) {
        using Real = typename base_type<T>::type;

        const Real xsq = x_active ? abs2_if_complex(x) : Real(0);
        const Real sumsq = group_reduce_sum_select_from_group(part, xsq);
        // sumsq is replicated, so every lane computes the scalars instead of serialising on a leader.
        const Real xnorm = sycl::sqrt(sumsq);

        T tau = T(0);

        const T alpha_leader = select_from_group(part, alpha, static_cast<uint32_t>(alpha_lane));

        T beta_b = alpha_leader;
        T tau_b = T(0);
        T scale_b = T(0);
        if (len > 1) {
            const auto scalars = internal::larfg(alpha_leader, xnorm, len);
            beta_b = scalars.beta;
            tau_b = scalars.tau;
            scale_b = scalars.scale;
        }

        tau = tau_b;

        if (lane == alpha_lane) {
            alpha = beta_b;
        } else if (x_active && tau != T(0)) {
            x *= scale_b;
        }

        return tau;
    }

    // SYTD2 on the UPPER triangle of a partition-resident P x P tile (column-major, LDA), which
    // must hold the full Hermitian matrix zero-padded past n; V_local/W_local are length-P
    // scratch. Leaves {s,d}sytd2 / {c,z}hetd2 packing: d(i) = A[i,i], e(i) = A[i,i+1], reflector
    // i in rows 0..i-1 of column i+1. Returns this lane's tau(lane) (zero for lane >= n-1).
    // Step k touches only the leading k x k block, so column k is final once step k has run.
    template <typename T, int32_t LDA, typename LocalPtr, typename Partition>
    inline T sytd2_cta_upper_partition(const Partition& part,
                                       LocalPtr A_local,
                                       LocalPtr V_local,
                                       LocalPtr W_local,
                                       int32_t n,
                                       int32_t lane) {
        T tau_lane = T(0);

        // Step k annihilates A(0:k-2, k); [x; alpha] is column k, rows 0..k-1.
        for (int32_t k = n - 1; k >= 1; --k) {
            const int32_t m = k;
            const int32_t alpha_row = k - 1;
            const int32_t col = k;

            const bool in_vec = (lane < m);
            const bool is_alpha = (lane == alpha_row);
            const bool x_active = (lane < (m - 1));

            T alpha = T(0);
            T x = T(0);
            if (in_vec) {
                const T a_val = A_local[lane + col * LDA];
                if (is_alpha) {
                    alpha = a_val;
                } else {
                    x = a_val;
                }
            }

            const T taui = larfg_small<T>(part, m, lane, alpha_row, alpha, x, x_active);

            // Both triangles are written: the rank-2 update below reads the full block.
            if (x_active) {
                A_local[lane + col * LDA] = x;
                A_local[col + lane * LDA] = conj_if_complex(x);
            }
            if (is_alpha) {
                A_local[lane + col * LDA] = alpha;
                A_local[col + lane * LDA] = conj_if_complex(alpha);
                tau_lane = taui;
            }

            if (taui != T(0)) {
                const T v_lane = (lane < m)
                                    ? ((lane == alpha_row) ? T(1) : A_local[lane + col * LDA])
                                    : T(0);
                V_local[lane] = v_lane;
                group_barrier(part);

                // Temporarily set A(alpha_row, col) = 1 (LAPACK convention) for the math.
                if (is_alpha) {
                    A_local[alpha_row + col * LDA] = T(1);
                    A_local[col + alpha_row * LDA] = T(1);
                }
                group_barrier(part);

                // DSYTD2: x = tau*A*v, w = x - (tau/2)(v^H x) v, A -= v w^H + w v^H.
                T y = T(0);
                if (lane < m) {
                    for (int32_t c = 0; c < m; ++c) {
                        const T a_rc = A_local[lane + c * LDA];
                        const T v_c = V_local[c];
                        y += a_rc * v_c;
                    }
                    y *= taui;
                }
                W_local[lane] = y;
                group_barrier(part);

                const T dot_lane = (lane < m) ? (conj_if_complex(V_local[lane]) * W_local[lane]) : T(0);
                const T dot = group_reduce_sum_select_from_group(part, dot_lane);
                const T alpha2 = T(-0.5) * taui * dot;

                if (lane < m) {
                    W_local[lane] = W_local[lane] + alpha2 * V_local[lane];
                }
                group_barrier(part);

                if (lane < m) {
                    const T v_r = V_local[lane];
                    const T w_r = W_local[lane];
                    for (int32_t c = 0; c < m; ++c) {
                        const T v_c = V_local[c];
                        const T w_c = W_local[c];
                        const int32_t idx = lane + c * LDA;
                        A_local[idx] = A_local[idx] - (v_r * conj_if_complex(w_c) + w_r * conj_if_complex(v_c));
                    }
                }
                group_barrier(part);

                if (is_alpha) {
                    A_local[alpha_row + col * LDA] = alpha;
                    A_local[col + alpha_row * LDA] = conj_if_complex(alpha);
                }
                group_barrier(part);
            } else {
                if (is_alpha) {
                    A_local[col + alpha_row * LDA] = conj_if_complex(alpha);
                }
            }

            group_barrier(part);
        }

        return tau_lane;
    }

} // namespace batchlas
