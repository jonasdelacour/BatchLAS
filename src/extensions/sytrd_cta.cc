#include <blas/matrix.hh>
#include <blas/functions.hh>
#include <blas/extensions.hh>
#include <blas/extra.hh>
#include <util/kernel-heuristics.hh>
#include <util/mempool.hh>
#include <util/group-invoke.hh>
#include <batchlas/backend_config.h>
#include "../math-helpers.hh"
#include "../queue.hh"
#include <internal/sort.hh>
#include <complex>
#include <numeric>
#include <array>
using namespace sycl::ext::oneapi;

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

    template <typename Real>
    inline Real sign_nonzero_real(Real x) {
        return (sycl::signbit(x) ? Real(-1) : Real(1));
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
            v += sycl::permute_group_by_xor(g, v, offset);
        }
        return v;
    }

    template <typename T, typename Group>
    inline T group_reduce_max(const Group& g, T v) {
        for (uint32_t offset = static_cast<uint32_t>(g.get_local_linear_range() / 2);
             offset > 0;
             offset >>= 1) {
            const T other = sycl::permute_group_by_xor(g, v, offset);
            v = sycl::fmax(v, other);
        }
        return v;
    }

    // NOTE: DPC++'s CUDA path for non-uniform group collectives currently has
    // limitations for floating-point reductions on chunked partitions.
    // This fallback implements a reduction using `select_from_group`, which is
    // supported and correct for the small fixed sizes used here (P <= 32).
    template <typename T, typename Group>
    inline T group_reduce_sum_select_from_group(const Group& g, T v) {
        const uint32_t lanes = static_cast<uint32_t>(g.get_local_linear_range());
        T sum = T(0);
        for (uint32_t j = 0; j < lanes; ++j) {
            sum += sycl::select_from_group(g, v, j);
        }
        return sum;
    }

    template <typename T>
    inline T sign_nonzero(T x) {
        // Return +/-1 with sign of x, treating +0 as +.
        return sign_nonzero_real(x);
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
        const Real xnorm = invoke_one_broadcast(part, [&]() {
            return sycl::sqrt(sumsq);
        });

        T tau = T(0);

        // Ensure the leader sees the correct alpha value regardless of where
        // alpha lives inside the partition.
        const T alpha_leader = sycl::select_from_group(part, alpha, static_cast<uint32_t>(alpha_lane));

        // Leader computes beta/tau/scale.
        const auto [beta_b, tau_b, scale_b] = invoke_one_broadcast(part, [&]() {
            T beta = alpha_leader;
            T tau_l = T(0);
            T scale = T(0);

            if (len <= 1) {
                tau_l = T(0);
                beta = alpha_leader;
                scale = T(0);
            } else if constexpr (internal::is_complex<T>::value) {
                // Complex Householder (ZLARFG/CLARFG-style): choose beta with the same phase as alpha.
                // NOTE: For Hermitian tridiagonal reduction, the off-diagonal can be made real later
                // via a diagonal phase similarity on the tridiagonal (see syev_cta complex path).
                const Real alpha_abs = sycl::hypot(alpha_leader.real(), alpha_leader.imag());

                if (xnorm == Real(0) && alpha_leader.imag() == Real(0)) {
                    tau_l = T(0);
                    beta = alpha_leader;
                    scale = T(0);
                } else {
                    const Real beta_abs = sycl::hypot(alpha_abs, xnorm);
                    const T alpha_sign = (alpha_abs == Real(0)) ? T(1) : (alpha_leader / alpha_abs);
                    beta = -alpha_sign * T(beta_abs);
                    tau_l = (beta - alpha_leader) / beta;
                    scale = T(1) / (alpha_leader - beta);
                }
            } else {
                if (xnorm == Real(0)) {
                    tau_l = T(0);
                    beta = alpha_leader;
                    scale = T(0);
                } else {
                    beta = -sign_nonzero(alpha_leader) * sycl::hypot(alpha_leader, xnorm);
                    tau_l = (beta - alpha_leader) / beta;
                    scale = T(1) / (alpha_leader - beta);
                }
            }

            return std::array<T, 3>{beta, tau_l, scale};
        });

        tau = tau_b;

        // Apply scaling to x and set alpha=beta.
        if (lane == alpha_lane) {
            alpha = beta_b;
        } else if (x_active && tau != T(0)) {
            x *= scale_b;
        }

        return tau;
    }

    template <typename T, size_t P, bool Upper>
    class Sytd2CTAKernel;

    template <typename T, size_t P, bool Upper>
    inline void sytd2_cta_impl(Queue& ctx,
                              MatrixView<T, MatrixFormat::Dense>& a,
                              VectorView<T>& d,
                              VectorView<T>& e,
                              VectorView<T>& tau,
                              int32_t n,
                              size_t cta_wg_size_multiplier) {
        const auto batch_size = a.batch_size();
        if (n < 1 || n > static_cast<int32_t>(P) || a.rows() != n || a.cols() != n) {
            throw std::runtime_error("sytd2_cta_impl: invalid n or matrix sizes for CTA partition.");
        }
        if (d.size() != n || e.size() != (n - 1) || tau.size() != (n - 1)) {
            throw std::runtime_error("sytd2_cta_impl: invalid d/e/tau sizes.");
        }
        if (d.batch_size() != batch_size || e.batch_size() != batch_size || tau.batch_size() != batch_size) {
            throw std::runtime_error("sytd2_cta_impl: batch size mismatch.");
        }

        ctx->submit([&](sycl::handler& cgh) {
            auto A_view = a.kernel_view();
            auto D_view = d;
            auto E_view = e;
            auto TAU_view = tau;

            const auto dev = ctx->get_device();
            (void)dev;

            // CTA path assumes warp-sized sub-groups on NVIDIA.
            const int32_t sg_size = 32;

            // Baseline work-group size is LCM(P, sg_size), so we can form fixed-size partitions of size P.
            const int32_t base_wg_size = std::lcm<int32_t>(static_cast<int32_t>(P), static_cast<int32_t>(sg_size));
            int32_t wg_size_multiplier = std::max<int32_t>(int32_t(1), static_cast<int32_t>(cta_wg_size_multiplier));
            int32_t wg_size = base_wg_size * wg_size_multiplier;

            const int32_t max_wg_size = static_cast<int32_t>(dev.get_info<sycl::info::device::max_work_group_size>());
            if (wg_size > max_wg_size) {
                const int32_t max_mul = std::max<int32_t>(int32_t(1), max_wg_size / base_wg_size);
                wg_size_multiplier = std::min(wg_size_multiplier, max_mul);
                wg_size = base_wg_size * wg_size_multiplier;
            }

            // Clamp by local memory usage: per-problem local storage is
            //   A_local: P*P, V_local: P, W_local: P
            // and probs_per_wg = wg_size / P.
            {
                const std::size_t local_mem_bytes = dev.get_info<sycl::info::device::local_mem_size>();
                const std::size_t elems_per_prob = static_cast<std::size_t>(P) * static_cast<std::size_t>(P)
                                                 + static_cast<std::size_t>(2) * static_cast<std::size_t>(P);
                const std::size_t bytes_per_prob = elems_per_prob * sizeof(T);
                const int32_t max_probs = (bytes_per_prob == 0)
                                              ? int32_t(1)
                                              : std::max<int32_t>(int32_t(1), static_cast<int32_t>(local_mem_bytes / bytes_per_prob));
                wg_size_multiplier = std::min(wg_size_multiplier, max_probs);
                wg_size = base_wg_size * wg_size_multiplier;
            }

            const int32_t probs_per_wg = wg_size / static_cast<int32_t>(P);
            const int32_t num_wg = (static_cast<int32_t>(batch_size) + probs_per_wg - 1) / probs_per_wg;
            const int32_t global_size = num_wg * wg_size;

            // Local storage per problem:
            // - Full P x P tile for A (column-major)
            // - Two length-P vectors (v and w)
            auto A_local = sycl::local_accessor<T, 1>(sycl::range<1>(probs_per_wg * P * P), cgh);
            auto V_local = sycl::local_accessor<T, 1>(sycl::range<1>(probs_per_wg * P), cgh);
            auto W_local = sycl::local_accessor<T, 1>(sycl::range<1>(probs_per_wg * P), cgh);

            cgh.parallel_for<Sytd2CTAKernel<T, P, Upper>>(
                sycl::nd_range<1>(global_size, wg_size),
                [=](sycl::nd_item<1> it) {
                    const auto wg = it.get_group();
                    const int32_t wg_id = static_cast<int32_t>(wg.get_group_linear_id());

                    const auto sg = it.get_sub_group();
                    const auto part = sycl::ext::oneapi::experimental::chunked_partition<P>(sg);

                    // Make part_id unique across sub-groups in the work-group.
                    const int32_t sg_id = static_cast<int32_t>(sg.get_group_linear_id());
                    const int32_t parts_per_sg = static_cast<int32_t>(part.get_group_linear_range());
                    const int32_t part_id = sg_id * parts_per_sg + static_cast<int32_t>(part.get_group_linear_id());

                    const int32_t lane = static_cast<int32_t>(part.get_local_linear_id());
                    const int32_t prob_id = wg_id * probs_per_wg + part_id;
                    if (prob_id >= static_cast<int32_t>(batch_size)) return;

                    auto A_prob = A_view.batch_item(prob_id);
                    auto D_prob = D_view.batch_item(prob_id);
                    auto E_prob = E_view.batch_item(prob_id);
                    auto TAU_prob = TAU_view.batch_item(prob_id);

                    const int32_t base_a = part_id * static_cast<int32_t>(P) * static_cast<int32_t>(P);
                    const int32_t base_v = part_id * static_cast<int32_t>(P);
                    const int32_t base_w = part_id * static_cast<int32_t>(P);

                    // Load A into local memory (column-major).
                    if (lane < n) {
                        for (int32_t c = 0; c < n; ++c) {
                            A_local[base_a + lane + c * static_cast<int32_t>(P)] = A_prob(lane, c);
                        }
                        // Pad remaining columns to keep reads defined.
                        for (int32_t c = n; c < static_cast<int32_t>(P); ++c) {
                            A_local[base_a + lane + c * static_cast<int32_t>(P)] = T(0);
                        }
                    } else {
                        // Pad remaining rows.
                        for (int32_t c = 0; c < static_cast<int32_t>(P); ++c) {
                            A_local[base_a + lane + c * static_cast<int32_t>(P)] = T(0);
                        }
                    }
                    sycl::group_barrier(part);

                    if constexpr (Upper) {
                        // Reduce the upper triangle of A.
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
                                const T a_val = A_local[base_a + lane + col * static_cast<int32_t>(P)];
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
                                A_local[base_a + lane + col * static_cast<int32_t>(P)] = x;
                                // Keep Hermitian/symmetric storage consistent.
                                A_local[base_a + col + lane * static_cast<int32_t>(P)] = conj_if_complex(x);
                            }
                            if (is_alpha) {
                                A_local[base_a + lane + col * static_cast<int32_t>(P)] = alpha;
                                A_local[base_a + col + lane * static_cast<int32_t>(P)] = conj_if_complex(alpha);
                            }

                            // Offdiagonal element for T.
                            // NOTE: select_from_group is a collective; all lanes must participate.
                            const T alpha_out = sycl::select_from_group(part, alpha, static_cast<uint32_t>(alpha_row));
                            if (lane == 0) {
                                // E index is k-1.
                                E_prob(k - 1) = alpha_out;
                                TAU_prob(k - 1) = taui;
                            }

                            if (taui != T(0)) {
                                // Build v (length m) with v(m-1)=1.
                                const T v_lane = (lane < m)
                                                    ? ((lane == alpha_row) ? T(1)
                                                                           : A_local[base_a + lane + col * static_cast<int32_t>(P)])
                                                    : T(0);
                                V_local[base_v + lane] = v_lane;
                                sycl::group_barrier(part);

                                // Temporarily set A(alpha_row, col) = 1 (LAPACK convention) for the math.
                                if (is_alpha) {
                                    A_local[base_a + alpha_row + col * static_cast<int32_t>(P)] = T(1);
                                    A_local[base_a + col + alpha_row * static_cast<int32_t>(P)] = T(1);
                                }
                                sycl::group_barrier(part);

                                // Compute x := tau * A(0:m-1,0:m-1) * v, store in W_local.
                                T y = T(0);
                                if (lane < m) {
                                    for (int32_t c = 0; c < m; ++c) {
                                        const T a_rc = A_local[base_a + lane + c * static_cast<int32_t>(P)];
                                        const T v_c = V_local[base_v + c];
                                        y += a_rc * v_c;
                                    }
                                    y *= taui;
                                }
                                W_local[base_w + lane] = y;
                                sycl::group_barrier(part);

                                // dot = v^H x
                                const T dot_lane = (lane < m) ? (conj_if_complex(V_local[base_v + lane]) * W_local[base_w + lane]) : T(0);
                                const T dot = group_reduce_sum_select_from_group(part, dot_lane);
                                const T alpha2 = invoke_one_broadcast(part, [&]() {
                                    return T(-0.5) * taui * dot;
                                });

                                // w := x + alpha2 * v
                                if (lane < m) {
                                    W_local[base_w + lane] = W_local[base_w + lane] + alpha2 * V_local[base_v + lane];
                                }
                                sycl::group_barrier(part);

                                // Rank-2 update on the leading m x m block (Hermitian-safe):
                                // A := A - v*w^H - w*v^H
                                if (lane < m) {
                                    const T v_r = V_local[base_v + lane];
                                    const T w_r = W_local[base_w + lane];
                                    for (int32_t c = 0; c < m; ++c) {
                                        const T v_c = V_local[base_v + c];
                                        const T w_c = W_local[base_w + c];
                                        const int32_t idx = base_a + lane + c * static_cast<int32_t>(P);
                                        A_local[idx] = A_local[idx] - (v_r * conj_if_complex(w_c) + w_r * conj_if_complex(v_c));
                                    }
                                }
                                sycl::group_barrier(part);

                                // Restore superdiagonal element and keep symmetry consistent.
                                if (is_alpha) {
                                    A_local[base_a + alpha_row + col * static_cast<int32_t>(P)] = alpha;
                                    A_local[base_a + col + alpha_row * static_cast<int32_t>(P)] = conj_if_complex(alpha);
                                }
                                sycl::group_barrier(part);
                            } else {
                                // If tau==0, ensure we don't leave a "1" in A.
                                if (is_alpha) {
                                    // alpha already stored.
                                    A_local[base_a + col + alpha_row * static_cast<int32_t>(P)] = conj_if_complex(alpha);
                                }
                            }

                            // Store diagonal element D(k).
                            if (lane == 0) {
                                D_prob(k) = A_local[base_a + k + k * static_cast<int32_t>(P)];
                            }
                            sycl::group_barrier(part);
                        }
                        if (lane == 0) {
                            D_prob(0) = A_local[base_a + 0 + 0 * static_cast<int32_t>(P)];
                        }
                    } else {
                        // Reduce the lower triangle of A.
                        // For i = 0 .. n-2, annihilate A(i+2:n-1, i).
                        for (int32_t i = 0; i <= n - 2; ++i) {
                            const int32_t s = i + 1;             // start row/col of trailing block
                            const int32_t m = n - s;             // size of trailing principal block
                            const int32_t col = i;

                            // Vector [alpha; x] lives in column 'col' rows [s..n-1], alpha at row s.
                            const bool in_vec = (lane < m);
                            const bool is_alpha = (lane == 0);
                            const bool x_active = (lane >= 1 && lane < m);

                            T alpha = T(0);
                            T x = T(0);
                            if (in_vec) {
                                const int32_t row = s + lane;
                                const T a_val = A_local[base_a + row + col * static_cast<int32_t>(P)];
                                if (is_alpha) {
                                    alpha = a_val;
                                } else {
                                    x = a_val;
                                }
                            }

                            // Form reflector of length m.
                            const T taui = larfg_small<T>(part, m, lane, /*alpha_lane=*/0, alpha, x, x_active);

                            // Write back scaled vector and beta.
                            if (x_active) {
                                const int32_t row = s + lane;
                                A_local[base_a + row + col * static_cast<int32_t>(P)] = x;
                                // Keep symmetry (optional but helpful for later stages).
                                A_local[base_a + col + row * static_cast<int32_t>(P)] = conj_if_complex(x);
                            }
                            if (is_alpha) {
                                A_local[base_a + s + col * static_cast<int32_t>(P)] = alpha;
                                A_local[base_a + col + s * static_cast<int32_t>(P)] = conj_if_complex(alpha);
                            }

                            if (lane == 0) {
                                E_prob(i) = alpha;
                                TAU_prob(i) = taui;
                                D_prob(i) = A_local[base_a + i + i * static_cast<int32_t>(P)];
                            }

                            if (taui != T(0)) {
                                // Build v (length m) with v(0)=1.
                                const T v_lane = (lane < m) ? ((lane == 0) ? T(1)
                                                                            : A_local[base_a + (s + lane) + col * static_cast<int32_t>(P)])
                                                            : T(0);
                                V_local[base_v + lane] = v_lane;
                                sycl::group_barrier(part);

                                // Temporarily set A(s, col) = 1 (LAPACK convention).
                                if (lane == 0) {
                                    A_local[base_a + s + col * static_cast<int32_t>(P)] = T(1);
                                    A_local[base_a + col + s * static_cast<int32_t>(P)] = T(1);
                                }
                                sycl::group_barrier(part);

                                // Compute x := tau * A(s:s+m-1, s:s+m-1) * v, store in W_local.
                                T y = T(0);
                                if (lane < m) {
                                    const int32_t row = s + lane;
                                    for (int32_t c = 0; c < m; ++c) {
                                        const int32_t col2 = s + c;
                                        const T a_rc = A_local[base_a + row + col2 * static_cast<int32_t>(P)];
                                        const T v_c = V_local[base_v + c];
                                        y += a_rc * v_c;
                                    }
                                    y *= taui;
                                }
                                W_local[base_w + lane] = y;
                                sycl::group_barrier(part);

                                // dot = v^H x
                                const T dot_lane = (lane < m) ? (conj_if_complex(V_local[base_v + lane]) * W_local[base_w + lane]) : T(0);
                                const T dot = group_reduce_sum_select_from_group(part, dot_lane);
                                const T alpha2 = invoke_one_broadcast(part, [&]() {
                                    return T(-0.5) * taui * dot;
                                });

                                // w := x + alpha2 * v
                                if (lane < m) {
                                    W_local[base_w + lane] = W_local[base_w + lane] + alpha2 * V_local[base_v + lane];
                                }
                                sycl::group_barrier(part);

                                // Rank-2 update on trailing m x m block (Hermitian-safe):
                                // A := A - v*w^H - w*v^H
                                if (lane < m) {
                                    const int32_t row = s + lane;
                                    const T v_r = V_local[base_v + lane];
                                    const T w_r = W_local[base_w + lane];
                                    for (int32_t c = 0; c < m; ++c) {
                                        const int32_t col2 = s + c;
                                        const T v_c = V_local[base_v + c];
                                        const T w_c = W_local[base_w + c];
                                        const int32_t idx = base_a + row + col2 * static_cast<int32_t>(P);
                                        A_local[idx] = A_local[idx] - (v_r * conj_if_complex(w_c) + w_r * conj_if_complex(v_c));
                                    }
                                }
                                sycl::group_barrier(part);

                                // Restore subdiagonal element.
                                if (lane == 0) {
                                    A_local[base_a + s + col * static_cast<int32_t>(P)] = alpha;
                                    A_local[base_a + col + s * static_cast<int32_t>(P)] = conj_if_complex(alpha);
                                }
                                sycl::group_barrier(part);
                            }
                        }
                        if (lane == 0) {
                            D_prob(n - 1) = A_local[base_a + (n - 1) + (n - 1) * static_cast<int32_t>(P)];
                        }
                    }

                    sycl::group_barrier(part);

                    // Write back A.
                    if (lane < n) {
                        for (int32_t c = 0; c < n; ++c) {
                            A_prob(lane, c) = A_local[base_a + lane + c * static_cast<int32_t>(P)];
                        }
                    }
                });
        });
    }

    // Public wrapper: pick a compile-time P in {4,8,16,32} and choose triangle at runtime.
    // This is unblocked (SYTD2-style), so it is meant for very small n.
    template <Backend B, typename T>
    Event sytrd_cta(Queue& ctx,
                    const MatrixView<T, MatrixFormat::Dense>& a_in,
                    const VectorView<T>& d_out,
                    const VectorView<T>& e_out,
                    const VectorView<T>& tau_out,
                    Uplo uplo,
                    const Span<std::byte>& ws,
                    size_t cta_wg_size_multiplier) {
        (void)ws;
        if (a_in.rows() != a_in.cols()) {
            throw std::invalid_argument("sytrd_cta: A must be square.");
        }

        const int64_t n64 = a_in.rows();
        const int64_t batch_size = a_in.batch_size();
        if (batch_size != d_out.batch_size() || batch_size != e_out.batch_size() || batch_size != tau_out.batch_size()) {
            throw std::invalid_argument("sytrd_cta: batch size mismatch.");
        }

        const int32_t n = static_cast<int32_t>(n64);
        if (n < 1 || n > 32) {
            throw std::invalid_argument("sytrd_cta currently supports 1 <= n <= 32.");
        }

        // Make mutable views (A is overwritten, D/E/TAU are outputs).
        auto& a = const_cast<MatrixView<T, MatrixFormat::Dense>&>(a_in);
        auto& d = const_cast<VectorView<T>&>(d_out);
        auto& e = const_cast<VectorView<T>&>(e_out);
        auto& tau = const_cast<VectorView<T>&>(tau_out);

        // CTA backend: requires subgroup size 32 on NVIDIA-like devices.
        {
            const auto dev = ctx->get_device();
            const auto sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
            bool has32 = false;
            for (auto sgs : sg_sizes) {
                if (static_cast<int32_t>(sgs) == 32) {
                    has32 = true;
                    break;
                }
            }
            if (!has32) {
                throw std::runtime_error("sytrd_cta: device does not support subgroup size 32 required for CTA kernels.");
            }
        }

        auto launch = [&](auto P_tag, bool upper) {
            constexpr int32_t P = decltype(P_tag)::value;
            if (upper) {
                sytd2_cta_impl<T, P, true>(ctx, a, d, e, tau, n, cta_wg_size_multiplier);
            } else {
                sytd2_cta_impl<T, P, false>(ctx, a, d, e, tau, n, cta_wg_size_multiplier);
            }
        };

        const bool upper = (uplo == Uplo::Upper);

        if (n <= 4) {
            launch(std::integral_constant<int32_t, 4>{}, upper);
        } else if (n <= 8) {
            launch(std::integral_constant<int32_t, 8>{}, upper);
        } else if (n <= 16) {
            launch(std::integral_constant<int32_t, 16>{}, upper);
        } else {
            launch(std::integral_constant<int32_t, 32>{}, upper);
        }

        return ctx.get_event();
    }

#if BATCHLAS_HAS_CUDA_BACKEND
    template Event sytrd_cta<Backend::CUDA, float>(Queue&, const MatrixView<float, MatrixFormat::Dense>&, const VectorView<float>&, const VectorView<float>&, const VectorView<float>&, Uplo, const Span<std::byte>&, size_t);
    template Event sytrd_cta<Backend::CUDA, double>(Queue&, const MatrixView<double, MatrixFormat::Dense>&, const VectorView<double>&, const VectorView<double>&, const VectorView<double>&, Uplo, const Span<std::byte>&, size_t);
    template Event sytrd_cta<Backend::CUDA, std::complex<float>>(Queue&, const MatrixView<std::complex<float>, MatrixFormat::Dense>&, const VectorView<std::complex<float>>&, const VectorView<std::complex<float>>&, const VectorView<std::complex<float>>&, Uplo, const Span<std::byte>&, size_t);
    template Event sytrd_cta<Backend::CUDA, std::complex<double>>(Queue&, const MatrixView<std::complex<double>, MatrixFormat::Dense>&, const VectorView<std::complex<double>>&, const VectorView<std::complex<double>>&, const VectorView<std::complex<double>>&, Uplo, const Span<std::byte>&, size_t);
#endif

#if BATCHLAS_HAS_HOST_BACKEND
    template Event sytrd_cta<Backend::NETLIB, float>(Queue&, const MatrixView<float, MatrixFormat::Dense>&, const VectorView<float>&, const VectorView<float>&, const VectorView<float>&, Uplo, const Span<std::byte>&, size_t);
    template Event sytrd_cta<Backend::NETLIB, double>(Queue&, const MatrixView<double, MatrixFormat::Dense>&, const VectorView<double>&, const VectorView<double>&, const VectorView<double>&, Uplo, const Span<std::byte>&, size_t);
    template Event sytrd_cta<Backend::NETLIB, std::complex<float>>(Queue&, const MatrixView<std::complex<float>, MatrixFormat::Dense>&, const VectorView<std::complex<float>>&, const VectorView<std::complex<float>>&, const VectorView<std::complex<float>>&, Uplo, const Span<std::byte>&, size_t);
    template Event sytrd_cta<Backend::NETLIB, std::complex<double>>(Queue&, const MatrixView<std::complex<double>, MatrixFormat::Dense>&, const VectorView<std::complex<double>>&, const VectorView<std::complex<double>>&, const VectorView<std::complex<double>>&, Uplo, const Span<std::byte>&, size_t);
#endif

}

