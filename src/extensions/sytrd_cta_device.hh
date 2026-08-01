#pragma once

// Device-side building blocks of the CTA (sub-group-partition) SYTRD reduction.
//
// Shared by:
//   - sytrd_cta.cc      : the standalone tridiagonalization kernel
//   - syev_cta_fused.cc : the monolithic SYEV kernel, which runs the same
//                         reduction on a tile it then keeps resident
//
// A single definition keeps the fused and partitioned SYEV paths numerically
// identical, so a benchmark between them measures fusion and nothing else.

#include <blas/matrix.hh>
#include <util/group-invoke.hh>
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

    // Unblocked symmetric tridiagonal reduction (LAPACK SYTD2-style) for very small matrices.
    //
    // This is intended as a building block for batched eigensolvers: it overwrites A with the
    // tridiagonal (diag + first offdiag) and stores Householder reflectors in the same layout
    // as LAPACK's {s,d}sytd2.
    //
    // References:
    // - DSYTD2 reference algorithm (unblocked) in LAPACK.

    template <typename T, typename Group>
    inline T group_reduce_sum(const Group& g, T v) {
        // Butterfly reduction using XOR shuffles; assumes power-of-two group size.
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

    // NOTE: DPC++'s CUDA path for non-uniform group collectives currently has
    // limitations for floating-point reductions on chunked partitions.
    // This reduction uses XOR shuffles (butterfly), which is O(log P) and keeps
    // the result replicated in all lanes.
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

    // Generate a Householder reflector H = I - tau * v v^T for a vector [alpha; x].
    // Mirrors DLARFG for the real case.
    //
    // Input:
    //  - alpha: scalar (lane==alpha_lane)
    //  - x elements in other lanes (inactive lanes must pass 0)
    // Output:
    //  - alpha overwritten with beta
    //  - x elements overwritten with v (scaled), and the implicit element becomes 1
    //  - tau returned
    template <typename T, typename Partition>
    inline T larfg_small(const Partition& part,
                         int32_t len,
                         int32_t lane,
                         int32_t alpha_lane,
                         T& alpha,
                         T& x,
                         bool x_active) {
        using Real = typename base_type<T>::type;

        // Compute xnorm.
        const Real xsq = x_active ? abs2_if_complex(x) : Real(0);
        const Real sumsq = group_reduce_sum_select_from_group(part, xsq);
        // `sumsq` is already replicated across the partition, so evaluate the
        // reflector scalars redundantly instead of serializing onto the leader.
        const Real xnorm = sycl::sqrt(sumsq);

        T tau = T(0);

        // Ensure every lane sees the correct alpha value regardless of where
        // alpha lives inside the partition.
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

        // Apply scaling to x and set alpha=beta.
        if (lane == alpha_lane) {
            alpha = beta_b;
        } else if (x_active && tau != T(0)) {
            x *= scale_b;
        }

        return tau;
    }

    // LAPACK SYTD2-style reduction of the *upper* triangle of a
    // partition-resident tile to tridiagonal form.
    //
    // `A_local` points at this problem's P x P tile (column-major, leading
    // dimension LDA); the tile must hold the full symmetric/Hermitian matrix,
    // zero-padded outside the leading n x n block. `V_local` and `W_local` are
    // this problem's two length-P scratch vectors.
    //
    // On return the tile holds the tridiagonal in its diagonal and
    // superdiagonal, and the Householder reflector for index i in rows 0..i-1 of
    // column i+1 (with an implicit 1 at row i) -- exactly LAPACK's
    // {s,d}sytd2 / {c,z}hetd2 packing. So the caller can read
    //   d(i)   = A_local[i + i*LDA]
    //   e(i)   = A_local[i + (i+1)*LDA]
    // straight off the tile: iteration k only ever touches the leading k x k
    // block, so column k is final once iteration k has run.
    //
    // The return value is this lane's tau: lane i holds tau(i) for i < n-1, and
    // zero elsewhere. Keeping it in a register (rather than a shared array) is
    // free -- tau is partition-uniform where it is produced -- and it is the
    // form the back-transform wants, which broadcasts tau(ii) per reflector.
    template <typename T, int32_t LDA, typename LocalPtr, typename Partition>
    inline T sytd2_cta_upper_partition(const Partition& part,
                                       LocalPtr A_local,
                                       LocalPtr V_local,
                                       LocalPtr W_local,
                                       int32_t n,
                                       int32_t lane) {
        T tau_lane = T(0);

        // For k = n-1 .. 1, annihilate A(0:k-2, k).
        for (int32_t k = n - 1; k >= 1; --k) {
            const int32_t m = k;          // active submatrix size (0..m-1)
            const int32_t alpha_row = k - 1;
            const int32_t col = k;

            // Vector [x; alpha] lives in column 'col' rows [0..m-1], alpha at row alpha_row.
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

            // Form reflector.
            const T taui = larfg_small<T>(part, m, lane, alpha_row, alpha, x, x_active);

            // Write back scaled vector and beta (alpha).
            if (x_active) {
                A_local[lane + col * LDA] = x;
                // Keep Hermitian/symmetric storage consistent.
                A_local[col + lane * LDA] = conj_if_complex(x);
            }
            if (is_alpha) {
                A_local[lane + col * LDA] = alpha;
                A_local[col + lane * LDA] = conj_if_complex(alpha);
                tau_lane = taui;
            }

            if (taui != T(0)) {
                // Build v (length m) with v(m-1)=1.
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

                // Compute x := tau * A(0:m-1,0:m-1) * v, store in W_local.
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

                // dot = v^H x
                const T dot_lane = (lane < m) ? (conj_if_complex(V_local[lane]) * W_local[lane]) : T(0);
                const T dot = group_reduce_sum_select_from_group(part, dot_lane);
                const T alpha2 = T(-0.5) * taui * dot;

                // w := x + alpha2 * v
                if (lane < m) {
                    W_local[lane] = W_local[lane] + alpha2 * V_local[lane];
                }
                group_barrier(part);

                // Rank-2 update on the leading m x m block (Hermitian-safe):
                // A := A - v*w^H - w*v^H
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

                // Restore superdiagonal element and keep symmetry consistent.
                if (is_alpha) {
                    A_local[alpha_row + col * LDA] = alpha;
                    A_local[col + alpha_row * LDA] = conj_if_complex(alpha);
                }
                group_barrier(part);
            } else {
                // If tau==0, ensure we don't leave a "1" in A.
                if (is_alpha) {
                    // alpha already stored.
                    A_local[col + alpha_row * LDA] = conj_if_complex(alpha);
                }
            }

            group_barrier(part);
        }

        return tau_lane;
    }

} // namespace batchlas
