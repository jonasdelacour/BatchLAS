#include <blas/matrix.hh>
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
#include "../util/template-instantiations.hh"
#include <complex>
#include <limits>
#include <numeric>
using namespace sycl::ext::oneapi;

namespace batchlas {

// Kernel name tag. Must live outside the anonymous namespace below so it does
// not depend on internal-linkage entities.
template <typename T, size_t P, bool ComputeVectors, bool Upper>
class SyevJacobiCTAKernel;

// ---------------------------------------------------------------------------
// Tier-A Jacobi eigensolver: partition-resident cyclic two-sided Jacobi.
//
// One SubGroupPartition<P> owns one problem; A and (optionally) Z live in local
// memory for the whole solve, so a full eigendecomposition is a single kernel
// launch with no global-memory traffic beyond the initial load and final store.
//
// This is an accuracy-oriented alternative to the sytrd_cta -> steqr_cta ->
// ormqx_cta pipeline. With the *relative* off-diagonal threshold used below,
// Jacobi's eigenvalue error is governed by the condition number of the
// column-equilibrated matrix rather than that of the (tridiagonalized) matrix
// itself, so graded / badly scaled inputs come out with small relative error
// where a tridiagonalizing method loses the small eigenvalues entirely.
//
// The guarantee is proved for symmetric positive definite input; indefinite
// matrices are handled correctly but do not inherit the relative-accuracy bound.
//
// References:
// - Demmel & Veselic, "Jacobi's Method is More Accurate than QR",
//   SIAM J. Matrix Anal. Appl. 13(4), 1992.  (accuracy theorem, relative
//   stopping criterion)
// - Drmac & Veselic, LAPACK Working Notes 169/170.  (threshold form, backward
//   error, convergence test)
// - Golub & Van Loan, Matrix Computations, Alg. 8.5.1.  (2x2 rotation formulas)
// - See JACOBI_EIGENSOLVER_PLAN.md for the full design rationale.
// ---------------------------------------------------------------------------

namespace {

template <typename U>
inline U conj_if_complex_j(const U& x) {
    if constexpr (internal::is_complex<U>::value) {
        return U(x.real(), -x.imag());
    } else {
        return x;
    }
}

template <typename U>
inline typename base_type<U>::type abs_if_complex_j(const U& x) {
    using Real = typename base_type<U>::type;
    if constexpr (internal::is_complex<U>::value) {
        return sycl::hypot(x.real(), x.imag());
    } else {
        return sycl::fabs(x);
    }
}

// Zero the imaginary part (no-op for real types). Used to keep Hermitian
// diagonals exactly real after an update.
template <typename U>
inline U force_real_j(const U& x) {
    if constexpr (internal::is_complex<U>::value) {
        return U(x.real(), typename base_type<U>::type(0));
    } else {
        return x;
    }
}

template <typename U>
inline typename base_type<U>::type real_part_j(const U& x) {
    if constexpr (internal::is_complex<U>::value) {
        return x.real();
    } else {
        return x;
    }
}

// Butterfly reduction over a chunked partition. Mirrors the pattern used by
// sytrd_cta: DPC++'s CUDA path has limitations for group collectives on
// non-uniform groups, so we use XOR shuffles and keep the result replicated.
template <typename T, typename Group>
inline T partition_reduce_sum_j(const Group& g, T v) {
    const uint32_t lanes = static_cast<uint32_t>(g.get_local_linear_range());
    for (uint32_t offset = lanes / 2; offset > 0; offset >>= 1) {
        v += permute_group_by_xor(g, v, offset);
    }
    return v;
}

// Round-robin ("chess tournament" / circle method) pairing.
//
// For an even m, round t in [0, m-2] produces m/2 disjoint pairs, and the m-1
// rounds together cover all m(m-1)/2 index pairs exactly once. Index 0 is held
// fixed and the remaining m-1 indices rotate.
//
// This schedule is a permutation of a serial sweep into disjoint (hence
// commuting) pivot pairs, so by the weak-equivalence theorem of Hari &
// Begovic Kovac (ETNA 46, 2017, Thm 2.11) it produces the same matrix as the
// cyclic-by-rows ordering after each full sweep and inherits its convergence.
inline void round_robin_pair(int32_t m, int32_t t, int32_t k, int32_t& p, int32_t& q) {
    const int32_t ring = m - 1;
    if (k == 0) {
        p = 0;
        q = (t % ring) + 1;
    } else {
        p = ((t + k) % ring) + 1;
        q = (((t - k) % ring + ring) % ring) + 1;
    }
    if (p > q) {
        const int32_t tmp = p;
        p = q;
        q = tmp;
    }
}

template <typename T, size_t P, bool ComputeVectors, bool Upper>
inline void syev_jacobi_cta_impl(Queue& ctx,
                                 MatrixView<T, MatrixFormat::Dense>& a,
                                 typename base_type<T>::type* w_ptr,
                                 int32_t n,
                                 JacobiParams<T> params) {
    using Real = typename base_type<T>::type;

    const auto batch_size = a.batch_size();
    if (n < 1 || n > static_cast<int32_t>(P) || a.rows() != n || a.cols() != n) {
        throw std::runtime_error("syev_jacobi_cta_impl: invalid n or matrix sizes for CTA partition.");
    }

    // Pivot index space is padded to an even size so the round-robin schedule is
    // well defined; the (single) padded index is simply never paired with a real
    // one because pairs touching index >= n are skipped.
    const int32_t m = (n % 2 == 0) ? n : (n + 1);

    ctx->submit([&](sycl::handler& cgh) {
        auto A_view = a.kernel_view();

        const auto dev = ctx->get_device();

        // CTA path assumes warp-sized sub-groups on NVIDIA.
        const int32_t sg_size = 32;

        // Local-memory leading dimension is padded to P+1.
        //
        // This matters a great deal. The row-update phase has lane == column, so
        // lane i touches address (row + i*LD). With LD == P == 32 every lane in a
        // warp lands in the same 32-bit bank and the access serializes 32 ways;
        // padding to 33 makes consecutive lanes differ by 33 == 1 (mod 32) and
        // the access becomes conflict-free.
        constexpr int32_t LD = static_cast<int32_t>(P) + 1;
        constexpr std::size_t kTileElems = static_cast<std::size_t>(LD) * P;

        const int32_t base_wg_size = std::lcm<int32_t>(static_cast<int32_t>(P), sg_size);
        int32_t wg_size_multiplier = std::max<int32_t>(int32_t(1),
                                                       static_cast<int32_t>(params.cta_wg_size_multiplier));
        int32_t wg_size = base_wg_size * wg_size_multiplier;

        const int32_t max_wg_size = static_cast<int32_t>(dev.get_info<sycl::info::device::max_work_group_size>());
        if (wg_size > max_wg_size) {
            const int32_t max_mul = std::max<int32_t>(int32_t(1), max_wg_size / base_wg_size);
            wg_size_multiplier = std::min(wg_size_multiplier, max_mul);
            wg_size = base_wg_size * wg_size_multiplier;
        }

        // Clamp by local memory: per problem we hold A, optionally Z, and the
        // per-round rotation coefficients. The pivot-pair table is shared by the
        // whole work-group, so it is accounted for separately below.
        constexpr std::size_t kRotSlots = (P / 2 > 0) ? (P / 2) : 1;
        constexpr std::size_t kPairSlots = (P - 1) * kRotSlots;
        constexpr std::size_t kPairTabBytes = kPairSlots * sizeof(int16_t);
        constexpr bool kNeedPhase = internal::is_complex<T>::value;
        {
            const std::size_t local_mem_bytes = dev.get_info<sycl::info::device::local_mem_size>();
            const std::size_t avail = (local_mem_bytes > kPairTabBytes) ? (local_mem_bytes - kPairTabBytes) : 1;
            const std::size_t z_elems = ComputeVectors ? kTileElems : 0;
            const std::size_t bytes_per_prob = (kTileElems + z_elems) * sizeof(T)
                                             + (kNeedPhase ? kRotSlots * sizeof(T) : 0)
                                             + 2 * kRotSlots * sizeof(Real);
            const int32_t max_probs = (bytes_per_prob == 0)
                                          ? int32_t(1)
                                          : std::max<int32_t>(int32_t(1),
                                                              static_cast<int32_t>(avail / bytes_per_prob));
            wg_size_multiplier = std::min(wg_size_multiplier, max_probs);
            wg_size = base_wg_size * wg_size_multiplier;
        }

        const int32_t probs_per_wg = wg_size / static_cast<int32_t>(P);
        const int32_t num_wg = (static_cast<int32_t>(batch_size) + probs_per_wg - 1) / probs_per_wg;
        const int32_t global_size = num_wg * wg_size;
        const int32_t wg_sz = wg_size;

        auto A_local = sycl::local_accessor<T, 1>(sycl::range<1>(probs_per_wg * kTileElems), cgh);
        auto Z_local = sycl::local_accessor<T, 1>(
            sycl::range<1>(ComputeVectors ? (probs_per_wg * kTileElems) : 1), cgh);
        // The rotation cosine/sine pair is stored as one vector so the update
        // loops issue a single LDS load instead of two. These are broadcast
        // reads (every lane in a partition reads the same slot), but they are
        // still LDS *instructions*, and before packing they outnumbered the
        // actual matrix accesses in the inner loop.
        auto Rcs_local = sycl::local_accessor<sycl::vec<Real, 2>, 1>(
            sycl::range<1>(probs_per_wg * kRotSlots), cgh);
        // The diagonal phase is identically 1 for real types, so it is neither
        // stored nor loaded there.
        auto Rd_local = sycl::local_accessor<T, 1>(
            sycl::range<1>(kNeedPhase ? (probs_per_wg * kRotSlots) : 1), cgh);
        // Round-robin pivot pairs, precomputed once per work-group and shared by
        // every problem in it, with the two indices packed into one 16-bit slot.
        // Computing them inline costs three integer modulos per pair per lane per
        // phase, which dominated the inner loops.
        auto Pair_local = sycl::local_accessor<int16_t, 1>(sycl::range<1>(kPairSlots), cgh);

        const int32_t nn = n;
        const int32_t mm = m;
        const int32_t nb = static_cast<int32_t>(batch_size);
        const int32_t max_sweeps = std::max<int32_t>(int32_t(1), static_cast<int32_t>(params.max_sweeps));
        const bool do_sort = params.sort;
        const bool ascending = (params.sort_order == SortOrder::Ascending);

        // Relative off-diagonal threshold. A rotation is applied only when
        //     |a_pq| > tol * sqrt(|a_pp| * |a_qq|)
        // (Demmel & Veselic; LAWN 169 Remark 2.2). Using the classical absolute
        // test |a_pq| <= tol * max|a_kl| instead would forfeit the entire
        // relative-accuracy advantage that motivates this kernel.
        const Real tol = params.tol_multiplier
                       * static_cast<Real>(nn)
                       * std::numeric_limits<Real>::epsilon();
        // Only ever treat truly denormal/zero off-diagonals as unconditionally
        // converged. This is a guard against churn when a diagonal entry passes
        // through zero on an indefinite matrix (which makes the relative
        // threshold demand |a_pq| == 0 exactly); it sits far below any magnitude
        // that affects the accuracy bound.
        const Real tiny = std::numeric_limits<Real>::min();
        // Above this magnitude tau*tau would overflow, so use the asymptotic
        // branch t ~ 1/(2*tau) instead.
        const Real tau_big = Real(1) / sycl::sqrt(std::numeric_limits<Real>::epsilon());

        Real* W = w_ptr;

        cgh.parallel_for<SyevJacobiCTAKernel<T, P, ComputeVectors, Upper>>(
            sycl::nd_range<1>(global_size, wg_size),
            [=](sycl::nd_item<1> it) {
                const auto wg = it.get_group();
                const int32_t wg_id = static_cast<int32_t>(wg.get_group_linear_id());
                const int32_t local_id = static_cast<int32_t>(it.get_local_linear_id());

                const int32_t pairs_per_round = mm / 2;
                const int32_t rounds = mm - 1;

                // Build the shared pivot-pair table. This must happen before any
                // early return, since it ends in a work-group barrier.
                for (int32_t idx = local_id; idx < rounds * pairs_per_round; idx += wg_sz) {
                    const int32_t t = idx / pairs_per_round;
                    const int32_t k = idx - t * pairs_per_round;
                    int32_t p = 0;
                    int32_t q = 0;
                    round_robin_pair(mm, t, k, p, q);
                    Pair_local[idx] = static_cast<int16_t>(p | (q << 8));
                }
                sycl::group_barrier(wg);

                const auto sg = it.get_sub_group();
                const auto part = make_partition<P>(sg);

                const int32_t sg_id = static_cast<int32_t>(sg.get_group_linear_id());
                const int32_t parts_per_sg = static_cast<int32_t>(part.get_group_linear_range());
                const int32_t part_id = sg_id * parts_per_sg + static_cast<int32_t>(part.get_group_linear_id());

                const int32_t lane = static_cast<int32_t>(part.get_local_linear_id());
                const int32_t prob_id = wg_id * probs_per_wg + part_id;
                if (prob_id >= nb) return;

                auto A_prob = A_view.batch_item(prob_id);

                const int32_t base_a = part_id * static_cast<int32_t>(kTileElems);
                const int32_t base_z = ComputeVectors ? (part_id * static_cast<int32_t>(kTileElems)) : 0;
                const int32_t base_r = part_id * static_cast<int32_t>(kRotSlots);

                // ---- Load: symmetrize from the requested triangle. ----
                // The pad region is zeroed so that no uninitialized value can ever
                // reach the arithmetic, even though the pivot schedule never
                // selects a padded index.
                for (int32_t c = 0; c < static_cast<int32_t>(P); ++c) {
                    T v = T(0);
                    if (lane < nn && c < nn) {
                        if constexpr (Upper) {
                            v = (lane <= c) ? A_prob(lane, c) : conj_if_complex_j(A_prob(c, lane));
                        } else {
                            v = (lane >= c) ? A_prob(lane, c) : conj_if_complex_j(A_prob(c, lane));
                        }
                        if (lane == c) {
                            // The diagonal of a Hermitian matrix is real by
                            // definition; drop any imaginary noise in the input.
                            if constexpr (internal::is_complex<T>::value) {
                                v = T(v.real(), Real(0));
                            }
                        }
                    }
                    A_local[base_a + lane + c * LD] = v;
                    if constexpr (ComputeVectors) {
                        Z_local[base_z + lane + c * LD] = (lane == c && lane < nn) ? T(1) : T(0);
                    }
                }
                group_barrier(part);

                const bool pair_lane = (lane < pairs_per_round);
                const bool row_lane = (lane < nn);

                // ---- Sweeps ----
                for (int32_t sweep = 0; sweep < max_sweeps; ++sweep) {
                    int32_t rot_count = 0;

                    for (int32_t t = 0; t < rounds; ++t) {
                        const int32_t tab_base = t * pairs_per_round;

                        int32_t p = 0;
                        int32_t q = 0;
                        bool active = false;

                        if (pair_lane) {
                            const int32_t pq = static_cast<int32_t>(Pair_local[tab_base + lane]);
                            p = pq & 0xFF;
                            q = (pq >> 8) & 0xFF;
                            active = (q < nn);
                        }

                        // ---- Compute the 2x2 diagonalizing transform. ----
                        Real c_rot = Real(1);
                        Real s_rot = Real(0);
                        T d_rot = T(1);

                        if (active) {
                            const T apq = A_local[base_a + p + q * LD];
                            const Real app = real_part_j(A_local[base_a + p + p * LD]);
                            const Real aqq = real_part_j(A_local[base_a + q + q * LD]);
                            const Real g_abs = abs_if_complex_j(apq);

                            const Real thresh = tol * sycl::sqrt(sycl::fabs(app) * sycl::fabs(aqq));

                            if (g_abs > thresh && g_abs > tiny) {
                                // Reduce to a real symmetric 2x2 by a diagonal
                                // phase similarity, then apply the classic real
                                // Jacobi rotation. For real T the phase is 1 and
                                // g carries the sign of a_pq, which is exactly
                                // Golub & Van Loan Alg. 8.5.1.
                                Real g;
                                if constexpr (internal::is_complex<T>::value) {
                                    g = g_abs;
                                    d_rot = T(apq.real() / g_abs, -apq.imag() / g_abs);
                                } else {
                                    g = apq;
                                    d_rot = T(1);
                                }

                                const Real tau = (aqq - app) / (Real(2) * g);
                                Real tt;
                                if (sycl::fabs(tau) > tau_big) {
                                    tt = Real(1) / (Real(2) * tau);
                                } else {
                                    tt = sycl::copysign(Real(1), tau)
                                       / (sycl::fabs(tau) + sycl::sqrt(Real(1) + tau * tau));
                                }
                                c_rot = Real(1) / sycl::sqrt(Real(1) + tt * tt);
                                s_rot = tt * c_rot;
                                // A rotation that rounds to the identity would
                                // never annihilate a_pq, so counting it would
                                // keep the sweep loop alive forever.
                                if (s_rot == Real(0)) active = false;
                            } else {
                                active = false;
                            }
                        }

                        if (!active) {
                            c_rot = Real(1);
                            s_rot = Real(0);
                            d_rot = T(1);
                        }

                        if (pair_lane) {
                            Rcs_local[base_r + lane] = sycl::vec<Real, 2>(c_rot, s_rot);
                            if constexpr (kNeedPhase) {
                                Rd_local[base_r + lane] = d_rot;
                            }
                        }
                        group_barrier(part);

                        // One reduction serves both as the sweep's rotation
                        // counter and as an early-out for rounds in which every
                        // pair was skipped. Late sweeps are almost all such
                        // rounds, and skipping them avoids two full O(n) passes
                        // over the tile.
                        const int32_t round_active = partition_reduce_sum_j(part, active ? 1 : 0);
                        rot_count += round_active;
                        if (round_active == 0) continue;

                        // ---- Phase 1: A <- A * U, and Z <- Z * U. ----
                        // Lane owns row `lane`; pairs touch disjoint column
                        // pairs, so there is no cross-lane hazard within a phase.
                        if (row_lane) {
                            for (int32_t k = 0; k < pairs_per_round; ++k) {
                                const sycl::vec<Real, 2> cs = Rcs_local[base_r + k];
                                const Real ck = cs[0];
                                const Real sk = cs[1];
                                if (sk == Real(0)) continue;

                                const int32_t pq = static_cast<int32_t>(Pair_local[tab_base + k]);
                                const int32_t pk = pq & 0xFF;
                                const int32_t qk = (pq >> 8) & 0xFF;

                                T u11 = T(ck);
                                T u12 = T(sk);
                                T u21 = T(-sk);
                                T u22 = T(ck);
                                if constexpr (kNeedPhase) {
                                    const T dk = Rd_local[base_r + k];
                                    u21 = -(dk * T(sk));
                                    u22 = dk * T(ck);
                                }

                                const int32_t ip = base_a + lane + pk * LD;
                                const int32_t iq = base_a + lane + qk * LD;
                                const T ap = A_local[ip];
                                const T aq = A_local[iq];
                                A_local[ip] = ap * u11 + aq * u21;
                                A_local[iq] = ap * u12 + aq * u22;

                                if constexpr (ComputeVectors) {
                                    const int32_t zp = base_z + lane + pk * LD;
                                    const int32_t zq = base_z + lane + qk * LD;
                                    const T zp_v = Z_local[zp];
                                    const T zq_v = Z_local[zq];
                                    Z_local[zp] = zp_v * u11 + zq_v * u21;
                                    Z_local[zq] = zp_v * u12 + zq_v * u22;
                                }
                            }
                        }
                        group_barrier(part);

                        // ---- Phase 2: A <- U^H * A. ----
                        // Lane now owns column `lane`. The annihilated entries are
                        // stored as exact zeros here rather than in a separate
                        // pass: the rotation makes them zero in exact arithmetic,
                        // and forcing it keeps the convergence test from chasing
                        // rounding noise.
                        if (row_lane) {
                            for (int32_t k = 0; k < pairs_per_round; ++k) {
                                const sycl::vec<Real, 2> cs = Rcs_local[base_r + k];
                                const Real ck = cs[0];
                                const Real sk = cs[1];
                                if (sk == Real(0)) continue;

                                const int32_t pq = static_cast<int32_t>(Pair_local[tab_base + k]);
                                const int32_t pk = pq & 0xFF;
                                const int32_t qk = (pq >> 8) & 0xFF;

                                // c, s are real, so conj(u11) = c and
                                // conj(u12) = s; only the phase conjugates.
                                const T cu11 = T(ck);
                                const T cu12 = T(sk);
                                T cu21 = T(-sk);
                                T cu22 = T(ck);
                                if constexpr (kNeedPhase) {
                                    const T dk = Rd_local[base_r + k];
                                    cu21 = -(conj_if_complex_j(dk) * T(sk));
                                    cu22 = conj_if_complex_j(dk) * T(ck);
                                }

                                const int32_t ip = base_a + pk + lane * LD;
                                const int32_t iq = base_a + qk + lane * LD;
                                const T ap = A_local[ip];
                                const T aq = A_local[iq];

                                T new_p = cu11 * ap + cu21 * aq;
                                T new_q = cu12 * ap + cu22 * aq;

                                if (lane == qk) {
                                    new_p = T(0);                       // annihilated a_pq
                                    new_q = force_real_j(new_q);        // Hermitian diagonal
                                } else if (lane == pk) {
                                    new_q = T(0);                       // annihilated a_qp
                                    new_p = force_real_j(new_p);        // Hermitian diagonal
                                }

                                A_local[ip] = new_p;
                                A_local[iq] = new_q;
                            }
                        }
                        group_barrier(part);
                    }

                    // Converged when a full sweep applied no rotation, i.e. all
                    // n(n-1)/2 pivot pairs passed the relative threshold test.
                    if (rot_count == 0) break;
                }

                // ---- Sort by rank and write back. ----
                // Rank is computed in parallel (one lane per eigenvalue) rather
                // than by a serial sort on the leader; ties break on index so the
                // permutation is a bijection.
                if (row_lane) {
                    const Real wj = real_part_j(A_local[base_a + lane + lane * LD]);
                    int32_t dst = lane;
                    if (do_sort) {
                        int32_t rank = 0;
                        for (int32_t k = 0; k < nn; ++k) {
                            const Real wk = real_part_j(A_local[base_a + k + k * LD]);
                            const bool before = ascending
                                ? (wk < wj || (wk == wj && k < lane))
                                : (wk > wj || (wk == wj && k < lane));
                            if (before) ++rank;
                        }
                        dst = rank;
                    }

                    W[static_cast<int64_t>(prob_id) * nn + dst] = wj;

                    if constexpr (ComputeVectors) {
                        for (int32_t r = 0; r < nn; ++r) {
                            A_prob(r, dst) = Z_local[base_z + r + lane * LD];
                        }
                    }
                }
            });
    });
}

} // namespace

template <Backend B, typename T>
Event syev_jacobi_cta(Queue& ctx,
                      const MatrixView<T, MatrixFormat::Dense>& a_in,
                      Span<typename base_type<T>::type> eigenvalues,
                      JobType jobz,
                      Uplo uplo,
                      const Span<std::byte>& ws,
                      JacobiParams<T> params) {
    (void)ws;

    if (a_in.rows() != a_in.cols()) {
        throw std::invalid_argument("syev_jacobi_cta: A must be square.");
    }
    if (jobz != JobType::NoEigenVectors && jobz != JobType::EigenVectors) {
        throw std::invalid_argument("syev_jacobi_cta: invalid JobType.");
    }

    const int64_t n64 = a_in.rows();
    const int64_t batch64 = a_in.batch_size();

    if (n64 < 1 || n64 > 32) {
        throw std::invalid_argument("syev_jacobi_cta currently supports 1 <= n <= 32.");
    }

    const int32_t n = static_cast<int32_t>(n64);

    if (eigenvalues.size() < static_cast<std::size_t>(n64) * static_cast<std::size_t>(batch64)) {
        throw std::invalid_argument("syev_jacobi_cta: eigenvalues span too small for n*batch.");
    }

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
            throw std::runtime_error("syev_jacobi_cta: device does not support subgroup size 32 required for CTA kernels.");
        }
    }

    auto& a = const_cast<MatrixView<T, MatrixFormat::Dense>&>(a_in);
    auto* w_ptr = eigenvalues.data();

    const bool upper = (uplo == Uplo::Upper);
    const bool vectors = (jobz == JobType::EigenVectors);

    auto launch = [&](auto P_tag) {
        constexpr size_t P = decltype(P_tag)::value;
        if (vectors) {
            if (upper) syev_jacobi_cta_impl<T, P, true, true>(ctx, a, w_ptr, n, params);
            else       syev_jacobi_cta_impl<T, P, true, false>(ctx, a, w_ptr, n, params);
        } else {
            if (upper) syev_jacobi_cta_impl<T, P, false, true>(ctx, a, w_ptr, n, params);
            else       syev_jacobi_cta_impl<T, P, false, false>(ctx, a, w_ptr, n, params);
        }
    };

    if (n <= 4) {
        launch(std::integral_constant<size_t, 4>{});
    } else if (n <= 8) {
        launch(std::integral_constant<size_t, 8>{});
    } else if (n <= 16) {
        launch(std::integral_constant<size_t, 16>{});
    } else {
        launch(std::integral_constant<size_t, 32>{});
    }

    return ctx.get_event();
}

template <Backend B, typename T>
size_t syev_jacobi_cta_buffer_size(Queue& ctx,
                                   const MatrixView<T, MatrixFormat::Dense>& a,
                                   JobType jobz,
                                   JacobiParams<T> params) {
    (void)ctx;
    (void)jobz;
    (void)params;

    if (a.rows() != a.cols()) {
        throw std::invalid_argument("syev_jacobi_cta_buffer_size: A must be square.");
    }
    if (a.rows() < 1 || a.rows() > 32) {
        throw std::invalid_argument("syev_jacobi_cta_buffer_size currently supports 1 <= n <= 32.");
    }

    // Everything is resident in local memory for the lifetime of the kernel.
    return 0;
}

#define SYEV_JACOBI_CTA_INSTANTIATE(back, fp) \
    template Event syev_jacobi_cta<back, BATCHLAS_UNPAREN fp>(Queue&, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
                                                              Span<typename base_type<BATCHLAS_UNPAREN fp>::type>, JobType, Uplo, \
                                                              const Span<std::byte>&, JacobiParams<BATCHLAS_UNPAREN fp>); \
    template size_t syev_jacobi_cta_buffer_size<back, BATCHLAS_UNPAREN fp>(Queue&, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
                                                                           JobType, JacobiParams<BATCHLAS_UNPAREN fp>);

#define SYEV_JACOBI_CTA_INSTANTIATE_FOR_BACKEND(back) \
    BATCHLAS_FOR_EACH_SCALAR_TYPE_1(SYEV_JACOBI_CTA_INSTANTIATE, back)

#if BATCHLAS_HAS_CUDA_BACKEND
    SYEV_JACOBI_CTA_INSTANTIATE_FOR_BACKEND(Backend::CUDA)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
    SYEV_JACOBI_CTA_INSTANTIATE_FOR_BACKEND(Backend::ROCM)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
    SYEV_JACOBI_CTA_INSTANTIATE_FOR_BACKEND(Backend::NETLIB)
#endif

#undef SYEV_JACOBI_CTA_INSTANTIATE_FOR_BACKEND
#undef SYEV_JACOBI_CTA_INSTANTIATE

} // namespace batchlas
