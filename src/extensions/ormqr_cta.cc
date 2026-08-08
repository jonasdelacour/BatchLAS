

#include <blas/matrix.hh>
#include <blas/device.hh>
#include <blas/functions.hh>
#include <blas/extensions.hh>
#include <blas/extra.hh>
#include <util/kernel-heuristics.hh>
#include <util/mempool.hh>
#include <util/group-invoke.hh>
#include "sg_compat.hh"
#include <batchlas/backend_config.h>
#include "../math-helpers.hh"
#include "../queue.hh"

#include <complex>
#include <numeric>
#include <array>
#include <type_traits>

using namespace sycl::ext::oneapi;

namespace batchlas {

// Small-matrix CTA implementations of ORMQR/ORMQL/UNMQR/UNMQL semantics.
//
// These apply the implicit orthogonal/unitary matrix Q represented by Householder
// reflectors (A, TAU) from a QR factorization (GEQRF) or QL factorization (GEQLF)
// to a matrix C.
//
// Notes:
// - This file is intended for very small sizes (n <= 32).
// - Current implementation focuses on the n-by-n case (C is square), which is
//   exactly what we need to post-process tridiagonal eigenvectors in SYEVD.
// - Order/storage match LAPACK's DORMQR/ZUNMQR and DORMQL/ZUNMQL contracts.

namespace detail {

template <typename U>
inline U conj_val(const U& x) {
    if constexpr (internal::is_complex<U>::value) {
        return U(x.real(), -x.imag());
    } else {
        return x;
    }
}

// Reduce sum over a group.
// NOTE: For non-uniform chunked partitions on CUDA, reduce_over_group may be
// constrained; use XOR shuffles (butterfly) which is O(log P) and keeps the
// reduced value replicated in all lanes.
template <typename T, typename Group>
inline T group_sum(const Group& g, T v) {
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

} // namespace detail

template <typename T, size_t P, bool QL, bool Left, bool TransposeOrConj>
class OrmQxCTAKernel;

// Apply one Householder reflector H = I - tau * v * v^H to C.
//
// For LEFT:  C := H * C  (or H^H * C)
// For RIGHT: C := C * H  (or C * H^H)
//
// The caller supplies:
// - v vector in V_local[0:len-1]
// - tau scalar
// - (offset, len) describing which rows/cols are affected
// - left/right, and whether we're applying conjugate-transpose (for complex)
//
template <typename T, typename Partition>
inline void apply_reflector_small(const Partition& part,
                                  T* C_local,
                                  int32_t P_i,
                                  int32_t n,
                                  int32_t lane,
                                  const T* V_local,
                                  T* Work_local,
                                  int32_t offset,
                                  int32_t len,
                                  T tau,
                                  bool left,
                                  bool conj_trans) {
    // tau_eff = tau for H, conj(tau) for H^H
    const T tau_eff = conj_trans ? detail::conj_val(tau) : tau;

    if (tau_eff == T(0) || len <= 0) return;

    auto v_view = VectorView<T>(const_cast<T*>(V_local), len);

    if (left) {
        if (lane < n) {
            T dot = T(0);
            for (int32_t r = 0; r < len; ++r) {
                const T v_r = V_local[r];
                dot += detail::conj_val(v_r) * C_local[(offset + r) + lane * P_i];
            }
            Work_local[lane] = tau_eff * dot;
        }
        group_barrier(part);

        auto gamma_view = VectorView<T>(Work_local, n);
        auto c_sub = KernelMatrixView<T, MatrixFormat::Dense>(C_local + offset, len, n, P_i, P_i * n);
        if constexpr (internal::is_complex<T>::value) {
            batchlas::device::geru(part, v_view, gamma_view, c_sub, T(-1));
        } else {
            batchlas::device::ger(part, v_view, gamma_view, c_sub, T(-1));
        }
    } else {
        if (lane < n) {
            T dot = T(0);
            for (int32_t c = 0; c < len; ++c) {
                dot += C_local[lane + (offset + c) * P_i] * V_local[c];
            }
            Work_local[lane] = tau_eff * dot;
        }
        group_barrier(part);

        auto gamma_view = VectorView<T>(Work_local, n);
        auto c_sub = KernelMatrixView<T, MatrixFormat::Dense>(C_local + offset * P_i, n, len, P_i, P_i * n);
        batchlas::device::gerc(part, gamma_view, v_view, c_sub, T(-1));
    }
}

template <typename T, size_t P, bool QL, bool Left, bool TransposeOrConj>
inline void ormqx_cta_impl(Queue& ctx,
                           MatrixView<T, MatrixFormat::Dense>& a,
                           VectorView<T>& tau,
                           MatrixView<T, MatrixFormat::Dense>& c,
                           int32_t n,
                           int32_t k,
                           size_t cta_wg_size_multiplier) {
    const auto batch_size = a.batch_size();

    if (n < 0 || n > static_cast<int32_t>(P) || a.rows() != n || a.cols() != n || c.rows() != n || c.cols() != n) {
        throw std::runtime_error("ormqx_cta_impl: currently requires square A and C with n <= P.");
    }
    if (k < 0 || k > n) {
        throw std::runtime_error("ormqx_cta_impl: invalid k.");
    }
    if (tau.size() < k) {
        throw std::runtime_error("ormqx_cta_impl: tau too small for k.");
    }
    if (tau.batch_size() != batch_size || c.batch_size() != batch_size) {
        throw std::runtime_error("ormqx_cta_impl: batch size mismatch.");
    }

    ctx->submit([&](sycl::handler& cgh) {
        auto A_view = a.kernel_view();
        auto TAU_view = tau;
        auto C_view = c.kernel_view();

        const auto dev = ctx->get_device();

        // CTA path assumes warp-sized sub-groups on NVIDIA.
        const int32_t sg_size = 32;

        const int32_t base_wg_size = std::lcm<int32_t>(static_cast<int32_t>(P), static_cast<int32_t>(sg_size));
        int32_t wg_size_multiplier = std::max<int32_t>(int32_t(1), static_cast<int32_t>(cta_wg_size_multiplier));
        int32_t wg_size = base_wg_size * wg_size_multiplier;

        const int32_t max_wg_size = static_cast<int32_t>(dev.get_info<sycl::info::device::max_work_group_size>());
        if (wg_size > max_wg_size) {
            const int32_t max_mul = std::max<int32_t>(int32_t(1), max_wg_size / base_wg_size);
            wg_size_multiplier = std::min(wg_size_multiplier, max_mul);
            wg_size = base_wg_size * wg_size_multiplier;
        }

        // Clamp by local memory usage. Per problem we allocate:
        // - LEFT specialization: V_local: P
        // - non-LEFT: C_local: P*P, V_local: P
        // and probs_per_wg = wg_size / P.
        {
            const std::size_t local_mem_bytes = dev.get_info<sycl::info::device::local_mem_size>();
            const std::size_t elems_per_prob = Left ? static_cast<std::size_t>(P)
                                                     : (static_cast<std::size_t>(P) * static_cast<std::size_t>(P) + static_cast<std::size_t>(P));
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

        if constexpr (Left) {
            // LEFT specialization: lane->column and keep that column in registers.
            auto V_local = sycl::local_accessor<T, 1>(sycl::range<1>(probs_per_wg * P), cgh);

            // Coalescing: lane j owns column j, and a column is contiguous in memory.
            // Consecutive lanes are therefore `ld` elements apart, so a scalar per-row
            // load/store issues one 32-sector request per instruction (8x more L1<->L2
            // traffic than the data requires). Moving 16 bytes per lane per access cuts
            // the request count - and the resulting L2 traffic - by a factor of VW.
            constexpr int32_t VW = static_cast<int32_t>(16 / sizeof(T));
            const bool vec_ok =
                !internal::is_complex<T>::value && VW > 1 && (n % VW) == 0 &&
                ((static_cast<std::size_t>(c.ld()) * sizeof(T)) % 16u) == 0 &&
                ((static_cast<std::size_t>(c.stride()) * sizeof(T)) % 16u) == 0 &&
                (reinterpret_cast<std::uintptr_t>(c.data_ptr()) % 16u) == 0;

            cgh.parallel_for<OrmQxCTAKernel<T, P, QL, Left, TransposeOrConj>>(
                sycl::nd_range<1>(global_size, wg_size),
                [=](sycl::nd_item<1> it) {
                const auto sg = it.get_sub_group();
                const auto part = make_partition<P>(sg);

                const int32_t sg_id = static_cast<int32_t>(sg.get_group_linear_id());
                const int32_t parts_per_sg = static_cast<int32_t>(part.get_group_linear_range());
                const int32_t part_id = sg_id * parts_per_sg + static_cast<int32_t>(part.get_group_linear_id());

                const int32_t lane = static_cast<int32_t>(part.get_local_linear_id());

                const int32_t wg_id = static_cast<int32_t>(it.get_group().get_group_linear_id());
                const int32_t prob_id = wg_id * probs_per_wg + part_id;
                if (prob_id >= static_cast<int32_t>(batch_size)) return;

                auto A_prob = A_view.batch_item(prob_id);
                auto TAU_prob = TAU_view.batch_item(prob_id);
                auto C_prob = C_view.batch_item(prob_id);
                const int32_t base_v = part_id * static_cast<int32_t>(P);

                const bool active_col = (lane < n);
                const int32_t j = lane;

                // Register column buffer for column j.
                //
                // Every index into `C_col` below is a compile-time constant so that the
                // array really is promoted to registers.  Indexing it with a runtime
                // value (the reflector support `offset + r`, or a bound of `n`) forces
                // the whole array into local memory, and the reflector loop then pays a
                // dependent L1/global round trip for each of its ~n^2 accesses.
                T C_col[P];
#pragma unroll
                for (int32_t r = 0; r < static_cast<int32_t>(P); ++r) {
                    C_col[r] = T(0);
                }
                if (vec_ok && active_col) {
                    using VecT = sycl::vec<T, VW>;
                    auto* col_ptr = reinterpret_cast<const VecT*>(&C_prob(0, j));
#pragma unroll
                    for (int32_t r = 0; r < static_cast<int32_t>(P); r += VW) {
                        if (r < n) {
                            const VecT v = col_ptr[r / VW];
#pragma unroll
                            for (int32_t t = 0; t < VW; ++t) {
                                C_col[r + t] = v[t];
                            }
                        }
                    }
                } else if (active_col) {
#pragma unroll
                    for (int32_t r = 0; r < static_cast<int32_t>(P); ++r) {
                        if (r < n) {
                            C_col[r] = C_prob(r, j);
                        }
                    }
                }

                // Determine reflector application order.
                //
                // QR (DORMQR/ZUNMQR): Q = H(1) H(2) ... H(k)
                //   LEFT:  C := Q*C      => apply H(k) ... H(1)
                //          C := Q^T/H*C  => apply H(1) ... H(k)
                //   RIGHT: C := C*Q      => apply H(1) ... H(k)
                //          C := C*Q^T/H  => apply H(k) ... H(1)
                //
                // QL (DORMQL/ZUNMQL): Q = H(k) ... H(2) H(1)
                //   LEFT:  C := Q*C      => apply H(1) ... H(k)
                //          C := Q^T/H*C  => apply H(k) ... H(1)
                //   RIGHT: C := C*Q      => apply H(k) ... H(1)
                //          C := C*Q^T/H  => apply H(1) ... H(k)
                //
                // Compactly: descending = (left XOR QL) ? !transpose_or_conj : transpose_or_conj.
                const bool descending = (Left ^ QL) ? (!TransposeOrConj) : TransposeOrConj;

                const int32_t i0 = descending ? (k - 1) : 0;
                const int32_t i1 = descending ? -1 : k;
                const int32_t step = descending ? -1 : 1;

                for (int32_t ii = i0; ii != i1; ii += step) {
                    const T tau_i = (ii >= 0 && ii < k) ? TAU_prob(ii) : T(0);
                    const T tau_eff = TransposeOrConj ? detail::conj_val(tau_i) : tau_i;

                    // Stage the reflector indexed by *absolute row*, explicitly zeroed
                    // outside its support.  The rank-1 update can then run over the
                    // full compile-time row range with no predication and, crucially,
                    // with constant `C_col` subscripts.
                    if constexpr (!QL) {
                        // QR: reflector ii is stored in column ii with v(ii) == 1 and
                        // v(0:ii-1) == 0; its support ends at row n-1.
                        const bool in_support = (lane >= ii) && (lane < n);
                        V_local[base_v + lane] =
                            !in_support ? T(0) : ((lane == ii) ? T(1) : A_prob(lane, ii));
                    } else {
                        // QL: reflector ii is H(ii+1); with m == n the pivot row is
                        // (n-k)+ii, v(pivot) == 1 and v(pivot+1:) == 0.
                        const int32_t col = (n - k) + ii;
                        const int32_t pivot = (n - k) + ii;
                        const bool in_support = (lane <= pivot) && (lane < n);
                        V_local[base_v + lane] =
                            !in_support ? T(0) : ((lane == pivot) ? T(1) : A_prob(lane, col));
                    }

                    group_barrier(part);

                    if (active_col && tau_eff != T(0)) {
                        T dot = T(0);
#pragma unroll
                        for (int32_t r = 0; r < static_cast<int32_t>(P); ++r) {
                            dot += detail::conj_val(V_local[base_v + r]) * C_col[r];
                        }
                        const T gamma = tau_eff * dot;
#pragma unroll
                        for (int32_t r = 0; r < static_cast<int32_t>(P); ++r) {
                            C_col[r] -= V_local[base_v + r] * gamma;
                        }
                    }

                    // Ensure all lanes have finished consuming v before overwrite.
                    group_barrier(part);
                }

                if (vec_ok && active_col) {
                    using VecT = sycl::vec<T, VW>;
                    auto* col_ptr = reinterpret_cast<VecT*>(&C_prob(0, j));
#pragma unroll
                    for (int32_t r = 0; r < static_cast<int32_t>(P); r += VW) {
                        if (r < n) {
                            VecT v{};
#pragma unroll
                            for (int32_t t = 0; t < VW; ++t) {
                                v[t] = C_col[r + t];
                            }
                            col_ptr[r / VW] = v;
                        }
                    }
                } else if (active_col) {
#pragma unroll
                    for (int32_t r = 0; r < static_cast<int32_t>(P); ++r) {
                        if (r < n) {
                            C_prob(r, j) = C_col[r];
                        }
                    }
                }
            });
        } else {
            // Local storage per problem:
            // - Full P x P tile for C (column-major)
            // - One length-P vector v
            // - One length-P work vector for reflector coefficients
            auto C_local = sycl::local_accessor<T, 1>(sycl::range<1>(probs_per_wg * P * P), cgh);
            auto V_local = sycl::local_accessor<T, 1>(sycl::range<1>(probs_per_wg * P), cgh);
            auto Work_local = sycl::local_accessor<T, 1>(sycl::range<1>(probs_per_wg * P), cgh);

            cgh.parallel_for<OrmQxCTAKernel<T, P, QL, Left, TransposeOrConj>>(
                sycl::nd_range<1>(global_size, wg_size),
                [=](sycl::nd_item<1> it) {
                    const auto sg = it.get_sub_group();
                    const auto part = make_partition<P>(sg);

                    const int32_t sg_id = static_cast<int32_t>(sg.get_group_linear_id());
                    const int32_t parts_per_sg = static_cast<int32_t>(part.get_group_linear_range());
                    const int32_t part_id = sg_id * parts_per_sg + static_cast<int32_t>(part.get_group_linear_id());

                    const int32_t lane = static_cast<int32_t>(part.get_local_linear_id());

                    const int32_t wg_id = static_cast<int32_t>(it.get_group().get_group_linear_id());
                    const int32_t prob_id = wg_id * probs_per_wg + part_id;
                    if (prob_id >= static_cast<int32_t>(batch_size)) return;

                    auto A_prob = A_view.batch_item(prob_id);
                    auto TAU_prob = TAU_view.batch_item(prob_id);
                    auto C_prob = C_view.batch_item(prob_id);

                    const int32_t base_c = part_id * static_cast<int32_t>(P) * static_cast<int32_t>(P);
                    const int32_t base_v = part_id * static_cast<int32_t>(P);
                    const int32_t base_work = part_id * static_cast<int32_t>(P);

                    const int32_t P_i = static_cast<int32_t>(P);

                    // Load C into local memory (column-major).
                    if (lane < n) {
                        for (int32_t col = 0; col < n; ++col) {
                            C_local[base_c + lane + col * P_i] = C_prob(lane, col);
                        }
                        for (int32_t col = n; col < P_i; ++col) {
                            C_local[base_c + lane + col * P_i] = T(0);
                        }
                    } else {
                        for (int32_t col = 0; col < P_i; ++col) {
                            C_local[base_c + lane + col * P_i] = T(0);
                        }
                    }
                    group_barrier(part);

                    const bool descending = (Left ^ QL) ? (!TransposeOrConj) : TransposeOrConj;

                    const int32_t i0 = descending ? (k - 1) : 0;
                    const int32_t i1 = descending ? -1 : k;
                    const int32_t step = descending ? -1 : 1;

                    for (int32_t ii = i0; ii != i1; ii += step) {
                        const T tau_i = (ii >= 0 && ii < k) ? TAU_prob(ii) : T(0);

                        // Build v and compute (offset,len) for this reflector.
                        int32_t offset = 0;
                        int32_t len = 0;

                        if constexpr (!QL) {
                            // QR: reflector ii stored in column ii, v has leading zeros of length ii.
                            offset = ii;
                            len = n - ii;

                            if (lane < len) {
                                const int32_t row = offset + lane;
                                const int32_t col = ii;
                                V_local[base_v + lane] = (lane == 0) ? T(1) : A_prob(row, col);
                            } else {
                                V_local[base_v + lane] = T(0);
                            }
                        } else {
                            // QL: reflector ii is H(ii+1), stored in last k columns.
                            // From LAPACK DGEQLF/ZGEQLF: v(pivot) = 1, v(pivot+1:m)=0, and v(0:pivot-1)
                            // stored in A(0:pivot-1, n-k+ii).
                            const int32_t col = (n - k) + ii;
                            const int32_t pivot = (n - k) + ii; // since m=n in our current restriction
                            offset = 0;
                            len = pivot + 1;

                            if (lane < len) {
                                const int32_t row = lane;
                                V_local[base_v + lane] = (row == pivot) ? T(1) : A_prob(row, col);
                            } else {
                                V_local[base_v + lane] = T(0);
                            }
                        }

                        group_barrier(part);

                        apply_reflector_small<T>(
                            part,
                            &C_local[base_c],
                            P_i,
                            n,
                            lane,
                            &V_local[base_v],
                            &Work_local[base_work],
                            offset,
                            len,
                            tau_i,
                            Left,
                            TransposeOrConj
                        );

                        group_barrier(part);
                    }

                    // Write back C.
                    if (lane < n) {
                        for (int32_t col = 0; col < n; ++col) {
                            C_prob(lane, col) = C_local[base_c + lane + col * P_i];
                        }
                    }
                });
        }
    });
}

// Public entrypoint.
//
// CTA-only small-n routine (n <= 32) to apply Q from a QR/QL factorization.
// Semantics match LAPACK ORMQx/UNMQx via (factorization, side, trans).

template <Backend B, typename T>
Event ormqx_cta(Queue& ctx,
               const MatrixView<T, MatrixFormat::Dense>& a_in,
               const VectorView<T>& tau_in,
               const MatrixView<T, MatrixFormat::Dense>& c_in,
               Uplo factorization,
               Side side,
               Transpose trans,
               int32_t k,
               const Span<std::byte>& ws,
               size_t cta_wg_size_multiplier) {
    (void)ws;

    const bool left = (side == Side::Left);
    const bool transpose_or_conj = (trans != Transpose::NoTrans);
    const bool ql = (factorization == Uplo::Lower);

    if (a_in.rows() != a_in.cols() || c_in.rows() != c_in.cols() || a_in.rows() != c_in.rows()) {
        throw std::invalid_argument("ormqx_cta: currently requires square A and C of the same order.");
    }

    const int32_t n = static_cast<int32_t>(a_in.rows());
    if (n < 0 || n > 32) {
        throw std::invalid_argument("ormqx_cta: currently supports 0 <= n <= 32.");
    }
    if (k < 0 || k > n) {
        throw std::invalid_argument("ormqx_cta: invalid k.");
    }

    auto& a = const_cast<MatrixView<T, MatrixFormat::Dense>&>(a_in);
    auto& tau = const_cast<VectorView<T>&>(tau_in);
    auto& c = const_cast<MatrixView<T, MatrixFormat::Dense>&>(c_in);

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
            throw std::runtime_error("ormqx_cta: device does not support subgroup size 32 required for CTA kernels.");
        }
    }

    auto launch = [&](auto P_tag) {
        constexpr int32_t P = decltype(P_tag)::value;
        if (ql) {
            if (left) {
                if (transpose_or_conj) {
                    ormqx_cta_impl<T, P, /*QL=*/true, /*Left=*/true, /*TransposeOrConj=*/true>(ctx, a, tau, c, n, k, cta_wg_size_multiplier);
                } else {
                    ormqx_cta_impl<T, P, /*QL=*/true, /*Left=*/true, /*TransposeOrConj=*/false>(ctx, a, tau, c, n, k, cta_wg_size_multiplier);
                }
            } else {
                if (transpose_or_conj) {
                    ormqx_cta_impl<T, P, /*QL=*/true, /*Left=*/false, /*TransposeOrConj=*/true>(ctx, a, tau, c, n, k, cta_wg_size_multiplier);
                } else {
                    ormqx_cta_impl<T, P, /*QL=*/true, /*Left=*/false, /*TransposeOrConj=*/false>(ctx, a, tau, c, n, k, cta_wg_size_multiplier);
                }
            }
        } else {
            if (left) {
                if (transpose_or_conj) {
                    ormqx_cta_impl<T, P, /*QL=*/false, /*Left=*/true, /*TransposeOrConj=*/true>(ctx, a, tau, c, n, k, cta_wg_size_multiplier);
                } else {
                    ormqx_cta_impl<T, P, /*QL=*/false, /*Left=*/true, /*TransposeOrConj=*/false>(ctx, a, tau, c, n, k, cta_wg_size_multiplier);
                }
            } else {
                if (transpose_or_conj) {
                    ormqx_cta_impl<T, P, /*QL=*/false, /*Left=*/false, /*TransposeOrConj=*/true>(ctx, a, tau, c, n, k, cta_wg_size_multiplier);
                } else {
                    ormqx_cta_impl<T, P, /*QL=*/false, /*Left=*/false, /*TransposeOrConj=*/false>(ctx, a, tau, c, n, k, cta_wg_size_multiplier);
                }
            }
        }
    };

    if (n <= 4) {
        launch(std::integral_constant<int32_t, 4>{});
    } else if (n <= 8) {
        launch(std::integral_constant<int32_t, 8>{});
    } else if (n <= 16) {
        launch(std::integral_constant<int32_t, 16>{});
    } else {
        launch(std::integral_constant<int32_t, 32>{});
    }

    return ctx.get_event();
}

#if BATCHLAS_HAS_CUDA_BACKEND
    template Event ormqx_cta<Backend::CUDA, float>(Queue&, const MatrixView<float, MatrixFormat::Dense>&, const VectorView<float>&, const MatrixView<float, MatrixFormat::Dense>&, Uplo, Side, Transpose, int32_t, const Span<std::byte>&, size_t);
    template Event ormqx_cta<Backend::CUDA, double>(Queue&, const MatrixView<double, MatrixFormat::Dense>&, const VectorView<double>&, const MatrixView<double, MatrixFormat::Dense>&, Uplo, Side, Transpose, int32_t, const Span<std::byte>&, size_t);
    template Event ormqx_cta<Backend::CUDA, std::complex<float>>(Queue&, const MatrixView<std::complex<float>, MatrixFormat::Dense>&, const VectorView<std::complex<float>>&, const MatrixView<std::complex<float>, MatrixFormat::Dense>&, Uplo, Side, Transpose, int32_t, const Span<std::byte>&, size_t);
    template Event ormqx_cta<Backend::CUDA, std::complex<double>>(Queue&, const MatrixView<std::complex<double>, MatrixFormat::Dense>&, const VectorView<std::complex<double>>&, const MatrixView<std::complex<double>, MatrixFormat::Dense>&, Uplo, Side, Transpose, int32_t, const Span<std::byte>&, size_t);
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
    template Event ormqx_cta<Backend::ROCM, float>(Queue&, const MatrixView<float, MatrixFormat::Dense>&, const VectorView<float>&, const MatrixView<float, MatrixFormat::Dense>&, Uplo, Side, Transpose, int32_t, const Span<std::byte>&, size_t);
    template Event ormqx_cta<Backend::ROCM, double>(Queue&, const MatrixView<double, MatrixFormat::Dense>&, const VectorView<double>&, const MatrixView<double, MatrixFormat::Dense>&, Uplo, Side, Transpose, int32_t, const Span<std::byte>&, size_t);
    template Event ormqx_cta<Backend::ROCM, std::complex<float>>(Queue&, const MatrixView<std::complex<float>, MatrixFormat::Dense>&, const VectorView<std::complex<float>>&, const MatrixView<std::complex<float>, MatrixFormat::Dense>&, Uplo, Side, Transpose, int32_t, const Span<std::byte>&, size_t);
    template Event ormqx_cta<Backend::ROCM, std::complex<double>>(Queue&, const MatrixView<std::complex<double>, MatrixFormat::Dense>&, const VectorView<std::complex<double>>&, const MatrixView<std::complex<double>, MatrixFormat::Dense>&, Uplo, Side, Transpose, int32_t, const Span<std::byte>&, size_t);
#endif

#if BATCHLAS_HAS_HOST_BACKEND
    template Event ormqx_cta<Backend::NETLIB, float>(Queue&, const MatrixView<float, MatrixFormat::Dense>&, const VectorView<float>&, const MatrixView<float, MatrixFormat::Dense>&, Uplo, Side, Transpose, int32_t, const Span<std::byte>&, size_t);
    template Event ormqx_cta<Backend::NETLIB, double>(Queue&, const MatrixView<double, MatrixFormat::Dense>&, const VectorView<double>&, const MatrixView<double, MatrixFormat::Dense>&, Uplo, Side, Transpose, int32_t, const Span<std::byte>&, size_t);
    template Event ormqx_cta<Backend::NETLIB, std::complex<float>>(Queue&, const MatrixView<std::complex<float>, MatrixFormat::Dense>&, const VectorView<std::complex<float>>&, const MatrixView<std::complex<float>, MatrixFormat::Dense>&, Uplo, Side, Transpose, int32_t, const Span<std::byte>&, size_t);
    template Event ormqx_cta<Backend::NETLIB, std::complex<double>>(Queue&, const MatrixView<std::complex<double>, MatrixFormat::Dense>&, const VectorView<std::complex<double>>&, const MatrixView<std::complex<double>, MatrixFormat::Dense>&, Uplo, Side, Transpose, int32_t, const Span<std::byte>&, size_t);
#endif

} // namespace batchlas