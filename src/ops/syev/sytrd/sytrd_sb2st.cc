// SPDX-License-Identifier: MIT
#include <batchlas/blas/extensions.hh>
#include <src/util/batch_mask.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/mempool.hh>

#include <sycl/sycl.hpp>

#include <batchlas/backend_config.h>

#include <src/util/template-instantiations.hh>

#include <src/math-helpers.hh>
#include <src/queue.hh>

#include <src/ops/syev/sytrd/sb2st_v2.hh>
#include <src/ops/syev/sytrd/sytrd_sb2st_cta.hh>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <batchlas/settings.hh>

namespace batchlas {

namespace {

namespace v2 = batchlas::device::v2;
namespace sbv = batchlas::sb2st_v2;

template <typename T>
inline void validate_sytrd_sb2st_dims(const MatrixView<T, MatrixFormat::Dense>& ab,
                                     const VectorView<typename base_type<T>::type>& d,
                                     const VectorView<typename base_type<T>::type>& e,
                                     const VectorView<T>& tau,
                                     Uplo uplo,
                                     int32_t kd) {
    if (kd < 0) {
        throw batchlas::invalid_argument("sytrd_sb2st: kd must be non-negative");
    }
    if (uplo != Uplo::Lower && uplo != Uplo::Upper) {
        throw batchlas::invalid_argument("sytrd_sb2st: invalid uplo");
    }

    const int n = ab.cols();
    if (ab.rows() != kd + 1) {
        throw batchlas::invalid_argument("sytrd_sb2st: AB must be (kd+1) x n");
    }

    if (d.size() != n || e.size() != std::max(0, n - 1) || tau.size() != std::max(0, n - 1)) {
        throw batchlas::invalid_argument("sytrd_sb2st: invalid d/e/tau sizes");
    }

    if (ab.batch_size() != d.batch_size() || ab.batch_size() != e.batch_size() || ab.batch_size() != tau.batch_size()) {
        throw batchlas::invalid_argument("sytrd_sb2st: batch size mismatch");
    }
    if (ab.batch_size() < 1) {
        throw batchlas::invalid_argument("sytrd_sb2st: invalid batch size");
    }
}

template <typename T>
class Kd0ExtractKernel;

template <typename T>
Event kd0_extract(Queue& q,
                  const MatrixView<T, MatrixFormat::Dense>& ab,
                  const VectorView<typename base_type<T>::type>& d,
                  const VectorView<typename base_type<T>::type>& e,
                  const VectorView<T>& tau,
                  const std::uint8_t* mask) {
    const int n = ab.cols();
    const int batch = ab.batch_size();
    using Real = typename base_type<T>::type;

    auto ABv = ab.kernel_view();
    auto dv = d;
    auto ev = e;
    auto tauv = tau;

    (void)q->submit([&](sycl::handler& h) {
        h.parallel_for<Kd0ExtractKernel<T>>(sycl::range<1>(static_cast<size_t>(batch)),
                                            [=](sycl::id<1> idx) {
            const int b = static_cast<int>(idx[0]);
            if (mask != nullptr && mask[b] == 0) return;  // I3
            auto AB = ABv.batch_item(b);
            auto Db = dv.batch_item(b);
            auto Eb = ev.batch_item(b);
            auto Taub = tauv.batch_item(b);

            for (int i = 0; i < n; ++i) {
                Db[i] = detail::real_part(AB(0, i));
                if (i < n - 1) {
                    Eb[i] = Real(0);
                    Taub[i] = T(0);
                }
            }
        });
    });
    return q.get_event();
}

template <typename T>
class Kd1ExtractKernel;

template <Backend B, typename T>
class AbWorkCopyKernel;

template <typename T>
Event kd1_extract_lower(Queue& q,
                        const MatrixView<T, MatrixFormat::Dense>& ab,
                        const VectorView<typename base_type<T>::type>& d,
                        const VectorView<typename base_type<T>::type>& e,
                        const VectorView<T>& tau,
                  const std::uint8_t* mask) {
    const int n = ab.cols();
    const int batch = ab.batch_size();
    using Real = typename base_type<T>::type;

    auto ABv = ab.kernel_view();
    auto dv = d;
    auto ev = e;
    auto tauv = tau;

    (void)q->submit([&](sycl::handler& h) {
        h.parallel_for<Kd1ExtractKernel<T>>(sycl::range<1>(static_cast<size_t>(batch)),
                                            [=](sycl::id<1> idx) {
            const int b = static_cast<int>(idx[0]);
            if (mask != nullptr && mask[b] == 0) return;  // I3
            auto AB = ABv.batch_item(b);
            auto Db = dv.batch_item(b);
            auto Eb = ev.batch_item(b);
            auto Taub = tauv.batch_item(b);

            for (int i = 0; i < n; ++i) {
                Db[i] = detail::real_part(AB(0, i));
            }

            // Make off-diagonal real via chained phase, as in LAPACK KD=1 special-case.
            T phase = T(1);
            for (int i = 0; i < n - 1; ++i) {
                const T t = AB(1, i);
                const Real a = detail::abs(t);
                Eb[i] = a;
                Taub[i] = T(0);
                if constexpr (detail::is_complex<T>::value) {
                    if (a != Real(0)) {
                        const T u = t / T(a);
                        phase *= u;
                    }
                } else {
                    (void)phase;
                }
            }
        });
    });
    return q.get_event();
}

// === Band-only SB2ST path (real symmetric, lower storage) ===

template <typename T>
class BtrdLowerKernel;

template <typename T>
Event btrd_lower_inplace(Queue& q,
                         KernelMatrixView<T, MatrixFormat::Dense> ab,
                         int n,
                         int kd,
                         VectorView<typename base_type<T>::type> c,
                         VectorView<T> work,
                         VectorView<typename base_type<T>::type> d,
                         VectorView<typename base_type<T>::type> e,
                         VectorView<T> tau,
                         const std::uint8_t* mask) {
    (void)q->submit([&](sycl::handler& h) {
        constexpr int kWorkGroupSize = 128;
        const int batch = ab.batch_size();
        const size_t groups = static_cast<size_t>(batch);
        const size_t global = groups * static_cast<size_t>(kWorkGroupSize);
        sycl::local_accessor<int, 1> shared_ints(4, h); // {nr, j1, j2, k}

        batchlas::impl::parallel_for<BtrdLowerKernel<T>>(h, sycl::nd_range<1>(sycl::range<1>(global),
                                                            sycl::range<1>(static_cast<size_t>(kWorkGroupSize))),
                                           [=](sycl::nd_item<1> it) {
            using Real = typename base_type<T>::type;
            using Ops = detail::btrd_givens_ops<T>;

            const int b = static_cast<int>(it.get_group(0));
            if (mask != nullptr && mask[b] == 0) return;  // I1: one item per work-group, before any barrier
            // Runtime stride, as legacy get_local_range(0): a constexpr one lets ptxas unroll the store loops.
            const v2::RtGroup<v2::Scope::Group> ex{it};

            auto ABv = ab.batch_item(b);
            auto Cv = c.batch_item(b);
            auto WORKv = work.batch_item(b);
            auto Dv = d.batch_item(b);
            auto Ev = e.batch_item(b);
            auto TAUv = tau.batch_item(b);

            const int ldab = ABv.ld();
            T* AB = ABv.data();
            auto& C = Cv;
            auto& WORK = WORKv;
            auto& D = Dv;
            auto& E = Ev;
            auto& TAU = TAUv;
            Real* const Cp = C.data_ptr();  // C / WORK are built with inc = 1
            T* const Wp = WORK.data_ptr();

            v2::for_each_owned(ex, n + kd + 2, [&](int i) BATCHLAS_DLA_BODY {
                C[i] = Real(0);
                WORK[i] = T(0);
            });
            v2::sync(ex, v2::Mem::Global);

            const int kd1 = kd + 1;
            const int kdm1 = kd - 1;
            const int incx = ldab - 1;
            const int kdn = (n - 1 < kd) ? (n - 1) : kd;

            auto idx = [&](int r, int c) BATCHLAS_DLA_BODY -> int { return r + c * ldab; };

            int nr = 0;
            int j1 = kdn + 1;
            int j2 = 0;

            // The rotation sequence j = j1 + t kd1, t < cnt, read through at(j): C(j), WORK(j) and the
            // band entries AB(r, j + dc), addressed as the scalar loops address them.
            auto rseq = [&](int cnt, auto at) BATCHLAS_DLA_BODY {
                return sbv::fn_vec(cnt, [at, j1 = j1, kd1](int t) BATCHLAS_DLA_BODY -> decltype(auto) {
                    return at(j1 + t * kd1);
                });
            };
            auto Cj = [&](int j) BATCHLAS_DLA_BODY -> Real& { return C[j]; };
            auto Wj = [&](int j) BATCHLAS_DLA_BODY -> T& { return WORK[j]; };
            auto Cpj = [&](int j) BATCHLAS_DLA_BODY -> Real& { return Cp[j]; };
            auto Wpj = [&](int j) BATCHLAS_DLA_BODY -> T& { return Wp[j]; };
            auto band = [&](int r, int dc) BATCHLAS_DLA_BODY {
                return [&ABv, r, dc](int j) BATCHLAS_DLA_BODY -> T& { return ABv(r, j + dc); };
            };
            auto band_p = [&](int r, int dc) BATCHLAS_DLA_BODY {
                return [&idx, AB, r, dc](int j) BATCHLAS_DLA_BODY -> T& { return AB[idx(r, j + dc)]; };
            };

            if (ex.leader()) {
                shared_ints[0] = nr;
                shared_ints[1] = j1;
                shared_ints[2] = j2;
                shared_ints[3] = 0;
            }
            v2::sync(ex, v2::Mem::Global);

            // Ensure diagonal is real (for Hermitian path). Parallelize across columns.
            if constexpr (detail::is_complex<T>::value) {
                auto diag = ABv(int32_t(0), Slice{});
                v2::for_each_owned(ex, n, [&](int j) BATCHLAS_DLA_BODY { diag[j] = detail::real_as_T(diag[j]); });
                v2::sync(ex, v2::Mem::Global);
            }

            for (int i = 0; i < n - 2; ++i) {
                for (int k = kdn; k >= 1; --k) {
                    if (ex.leader()) {
                        j1 += kdn;
                        j2 += kdn;
                        shared_ints[0] = nr;
                        shared_ints[1] = j1;
                        shared_ints[2] = j2;
                        shared_ints[3] = k;
                    }
                    v2::sync(ex, v2::Mem::Global);

                    nr = shared_ints[0];
                    j1 = shared_ints[1];
                    j2 = shared_ints[2];
                    // k is only used for scalar work (leader) below.

                    if (nr > 0) {
                        // xLARGV: the rotations annihilating the fill WORK(j) against AB(kd, j - kd1).
                        sbv::largv<Ops>(ex, rseq(nr, band(kd, -kd1)), rseq(nr, Wj), rseq(nr, Cj));
                        v2::sync(ex, v2::Mem::Global);

                        if (nr > 2 * kd - 1) {
                            // xLARTV down band diagonal l. Iteration l touches (kd-1-l, c) and
                            // (kd-l, c) with c = j-kd1+l+1, so successive l address distinct rows and
                            // columns and rotations are kd1 columns apart: one barrier for the nest
                            // (mirrors btrd_lower_inplace_subgroup).
                            for (int l = 0; l < kd - 1; ++l) {
                                sbv::lartv<Ops>(ex, rseq(nr, band(kd - 1 - l, -kd1 + l + 1)),
                                               rseq(nr, band(kd - l, -kd1 + l + 1)), rseq(nr, Cj), rseq(nr, Wj));
                            }
                            v2::sync(ex, v2::Mem::Global);
                        } else {
                            if (ex.leader()) {
                                const int jend = j1 + (nr - 1) * kd1;
                                for (int jinc = j1; jinc <= jend; jinc += kd1) {
                                    if (kdm1 > 0) {
                                        sbv::rot_by<Ops>(v2::Thread{}, v2::vec(AB + idx(kd - 1, jinc - kd), kdm1, incx),
                                                     v2::vec(AB + idx(kd, jinc - kd), kdm1, incx), C[jinc], WORK[jinc]);
                                    }
                                }
                            }
                            v2::sync(ex, v2::Mem::Global);
                        }
                    }

                    if (ex.leader()) {
                        if (k > 1) {
                            if (k <= n - i - 1) {
                                const T f = ABv(k - 1, i);
                                const T g = ABv(k, i);
                                Real ct = Real(0);
                                T st = T(0);
                                T rt = T(0);
                                Ops::lartg(f, g, ct, st, rt);
                                C[i + k] = ct;
                                WORK[i + k] = st;
                                ABv(k - 1, i) = rt;

                                if (k > 2) {
                                    const Real ct = C[i + k];
                                    const T st = WORK[i + k];
                                    sbv::rot_by<Ops>(v2::Thread{}, v2::vec(AB + idx(k - 2, i + 1), k - 2, ldab - 1),
                                                 v2::vec(AB + idx(k - 1, i + 1), k - 2, ldab - 1), ct, st);
                                }
                            }
                            nr += 1;
                            j1 = j1 - kdn - 1;
                        }
                        shared_ints[0] = nr;
                        shared_ints[1] = j1;
                        shared_ints[2] = j2;
                    }
                    v2::sync(ex, v2::Mem::Global);

                    nr = shared_ints[0];
                    j1 = shared_ints[1];
                    j2 = shared_ints[2];

                    if (nr > 0) {
                        // xLAR2V: the two-sided rotation of each 2x2 diagonal block.
                        sbv::lar2v<Ops>(ex, rseq(nr, band_p(0, -1)), rseq(nr, band_p(0, 0)), rseq(nr, band_p(1, -1)),
                                       rseq(nr, Cpj), rseq(nr, Wpj));
                        v2::sync(ex, v2::Mem::Global);

                        // ZHBTRD requirement: conjugate WORK(J1) before right-application.
                        if constexpr (detail::is_complex<T>::value) {
                            const auto ws = rseq(nr, Wj);
                            v2::map(ex, ws, [](const T& w) { return sbv::native_t<T>::conj(w); }, ws);
                            v2::sync(ex, v2::Mem::Global);
                        }

                        if (nr > 2 * kd - 1) {
                            // xLARTV from the right. Iteration l touches (l+2, j-1) and (l+1, j);
                            // l+1 touches (l+3, j-1) and (l+2, j): disjoint rows and columns, and the
                            // rotations are kd1 columns apart. One trailing barrier suffices.
                            for (int l = 0; l < kd - 1; ++l) {
                                const int nrt = (j2 + l + 2 > n) ? (nr - 1) : nr;
                                if (nrt > 0) {
                                    sbv::lartv<Ops>(ex, rseq(nrt, band(l + 2, -1)), rseq(nrt, band(l + 1, 0)),
                                                   rseq(nrt, Cj), rseq(nrt, Wj));
                                }
                            }
                            v2::sync(ex, v2::Mem::Global);
                        } else {
                            if (ex.leader()) {
                                const int j1end = j1 + kd1 * (nr - 2);
                                if (j1end >= j1) {
                                    for (int j1inc = j1; j1inc <= j1end; j1inc += kd1) {
                                        if (kdm1 > 0) {
                                            sbv::rot_by<Ops>(v2::Thread{}, v2::vec(AB + idx(2, j1inc - 1), kdm1),
                                                         v2::vec(AB + idx(1, j1inc), kdm1), C[j1inc], WORK[j1inc]);
                                        }
                                    }
                                }
                                const int lend = (n - j2 - 1 < kdm1) ? (n - j2 - 1) : kdm1;
                                const int last = j1end + kd1;
                                if (lend > 0) {
                                    sbv::rot_by<Ops>(v2::Thread{}, v2::vec(AB + idx(2, last - 1), lend),
                                                 v2::vec(AB + idx(1, last), lend), C[last], WORK[last]);
                                }
                            }
                            v2::sync(ex, v2::Mem::Global);
                        }
                    }

                    if (ex.leader()) {
                        if (j2 + kdn >= n) {
                            nr -= 1;
                            j2 = j2 - kdn - 1;
                        }
                        shared_ints[0] = nr;
                        shared_ints[1] = j1;
                        shared_ints[2] = j2;
                    }
                    v2::sync(ex, v2::Mem::Global);

                    nr = shared_ints[0];
                    j1 = shared_ints[1];
                    j2 = shared_ints[2];

                    if (nr > 0) {
                        // The next fill: WORK(j + kd) = WORK(j) AB(kd, j), then AB(kd, j) *= C(j).
                        v2::for_each_owned(ex, nr, [&](int t) BATCHLAS_DLA_BODY {
                            const int j = j1 + t * kd1;
                            WORK[j + kd] = WORK[j] * ABv(kd, j);
                            ABv(kd, j) = T(C[j]) * ABv(kd, j);
                        });
                        v2::sync(ex, v2::Mem::Global);
                    }
                }
            }

            if (ex.leader()) {
                Ops::finalize_tridiag(ABv, n, D, E, TAU);
            }
        });
    });
    return q.get_event();
}

} // namespace

template <typename T>
struct Sb2stWorkspace {
    Span<T> ab_work;                        // AB work copy: (kd+1) x n per batch
    Span<typename base_type<T>::type> c;    // Givens cosines: n+kd+2 per batch
    Span<T> work;                           // Givens sines / scratch: n+kd+2 per batch
};

// Single description of sytrd_sb2st's unified band-only (SBTRD/HBTRD-style)
// workspace; see workspace_bytes() in util/mempool.hh.
template <typename T>
Sb2stWorkspace<T> sytrd_sb2st_layout(Queue& ctx, BumpAllocator& pool, int n, int kd_i, int batch) {
    using Real = typename base_type<T>::type;
    const int ldab = kd_i + 1;
    const size_t ab_elems = static_cast<size_t>(ldab) * static_cast<size_t>(n) * static_cast<size_t>(batch);
    const size_t rot_elems = static_cast<size_t>(n + kd_i + 2) * static_cast<size_t>(batch);
    auto ab_work = pool.allocate<T>(ctx, ab_elems);
    auto c = pool.allocate<Real>(ctx, rot_elems);
    auto work = pool.allocate<T>(ctx, rot_elems);
    return {ab_work, c, work};
}

template <Backend B, typename T>
size_t sytrd_sb2st_buffer_size(Queue& ctx,
                               const MatrixView<T, MatrixFormat::Dense>& ab_in,
                               const VectorView<typename base_type<T>::type>& d_out,
                               const VectorView<typename base_type<T>::type>& e_out,
                               const VectorView<T>& tau_out,
                               Uplo uplo,
                               int32_t kd,
                               int32_t block_size) {
    validate_sytrd_sb2st_dims(ab_in, d_out, e_out, tau_out, uplo, kd);

    const int n = ab_in.cols();
    const int batch = ab_in.batch_size();

    (void)block_size;

    const int kd_i = std::max<int>(0, kd);
    if (n <= 0) return 0;

    return workspace_bytes([&](BumpAllocator& pool) {
        return sytrd_sb2st_layout<T>(ctx, pool, n, kd_i, batch);
    });
}

template <Backend B, typename T>
Event sytrd_sb2st(Queue& ctx,
                  const MatrixView<T, MatrixFormat::Dense>& ab_in,
                  const VectorView<typename base_type<T>::type>& d_out,
                  const VectorView<typename base_type<T>::type>& e_out,
                  const VectorView<T>& tau_out,
                  Uplo uplo,
                  int32_t kd,
                  const Span<std::byte>& ws,
                  int32_t block_size) {
    const std::uint8_t* mask =
        batchlas::detail::mask_ptr(batchlas::detail::call_mask("sytrd_sb2st", ab_in, d_out, e_out, tau_out));
    validate_sytrd_sb2st_dims(ab_in, d_out, e_out, tau_out, uplo, kd);

    if (!ctx.in_order()) {
        throw batchlas::invalid_argument("sytrd_sb2st: requires an in-order Queue");
    }

    if (uplo != Uplo::Lower) {
        throw batchlas::unsupported("sytrd_sb2st: only Uplo::Lower is implemented");
    }

    const int n = ab_in.cols();
    const int batch = ab_in.batch_size();
    const int kd_i = std::max<int>(0, kd);

    if (n <= 0) return ctx.get_event();

    // Fast paths.
    if (kd_i == 0) {
        (void)kd0_extract<T>(ctx, ab_in, d_out, e_out, tau_out, mask);
        return ctx.get_event();
    }
    if (kd_i == 1) {
        (void)kd1_extract_lower<T>(ctx, ab_in, d_out, e_out, tau_out, mask);
        return ctx.get_event();
    }

    BumpAllocator pool(ws);

    // Unified band-only SB2ST/HBTRD: copy AB into a mutable workspace and run bulge chasing.
    auto sb2st_ws = sytrd_sb2st_layout<T>(ctx, pool, n, kd_i, batch);
    const int ldab = kd_i + 1;
    T* ab_ptr = sb2st_ws.ab_work.data();

    {
        const int stride_out = ldab * n;
        auto ABsrc = ab_in.kernel_view();
        KernelMatrixView<T, MatrixFormat::Dense> ABdst(ab_ptr, ldab, n, ldab, stride_out, batch);
        (void)ctx->submit([&](sycl::handler& h) {
            h.parallel_for<AbWorkCopyKernel<B, T>>(sycl::range<1>(static_cast<size_t>(batch)),
                                                   [=](sycl::id<1> idx_b) {
                const int b = static_cast<int>(idx_b[0]);
                v2::lacpy(v2::Thread{}, v2::mat(ABsrc.batch_item(b)).sub(0, 0, ldab, n),
                          v2::mat(ABdst.batch_item(b)).sub(0, 0, ldab, n));
            });
        });
    }

    using Real = typename base_type<T>::type;
    auto c_buf = sb2st_ws.c;
    auto work_buf = sb2st_ws.work;

    KernelMatrixView<T, MatrixFormat::Dense> ABwork(ab_ptr, ldab, n, ldab, ldab * n, batch);
    VectorView<Real> Cv(c_buf.data(), n + kd_i + 2, batch, 1, (n + kd_i + 2));
    VectorView<T> WORKv(work_buf.data(), n + kd_i + 2, batch, 1, (n + kd_i + 2));

    enum class Sb2stSubgroupMode {
        Auto,
        ForceOn,
        ForceOff,
    };

    auto parse_mode = []() -> Sb2stSubgroupMode {
        const char* p = batchlas::settings().selection.sb2st_subgroup.get();
        if (!p || !*p) return Sb2stSubgroupMode::Auto;
        std::string v(p);
        for (char& ch : v) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        if (v == "1" || v == "true" || v == "on") return Sb2stSubgroupMode::ForceOn;
        if (v == "0" || v == "false" || v == "off") return Sb2stSubgroupMode::ForceOff;
        if (v == "auto") return Sb2stSubgroupMode::Auto;
        // Unknown value: keep behavior unchanged.
        return Sb2stSubgroupMode::Auto;
    };

    const Sb2stSubgroupMode mode = parse_mode();

    const bool device_has_sg32 = ctx.device().supports_sub_group_size(32);

    bool can_use_subgroup = false;
    if (mode != Sb2stSubgroupMode::ForceOff) {
        can_use_subgroup = (kd_i <= 32) && device_has_sg32;
    }

    if (mode == Sb2stSubgroupMode::ForceOn && !can_use_subgroup) {
        throw batchlas::unsupported(
            "sytrd_sb2st: BATCHLAS_SB2ST_SUBGROUP=1 requires kd<=32 and a device supporting sub_group_size=32");
    }

    if (can_use_subgroup) {
        (void)detail::btrd_lower_inplace_subgroup_dispatch<T>(ctx, ABwork, n, kd_i, Cv, WORKv, d_out, e_out, tau_out, mask);
    } else {
        (void)btrd_lower_inplace<T>(ctx, ABwork, n, kd_i, Cv, WORKv, d_out, e_out, tau_out, mask);
    }

    return ctx.get_event();
}

#define SYTRD_SB2ST_INSTANTIATE(back, fp) \
    template Event sytrd_sb2st<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const VectorView<typename base_type<BATCHLAS_UNPAREN fp>::type>&, \
        const VectorView<typename base_type<BATCHLAS_UNPAREN fp>::type>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        Uplo, \
        int32_t, \
        const Span<std::byte>&, \
        int32_t); \
    template size_t sytrd_sb2st_buffer_size<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const VectorView<typename base_type<BATCHLAS_UNPAREN fp>::type>&, \
        const VectorView<typename base_type<BATCHLAS_UNPAREN fp>::type>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        Uplo, \
        int32_t, \
        int32_t);

BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(SYTRD_SB2ST_INSTANTIATE)

#undef SYTRD_SB2ST_INSTANTIATE

} // namespace batchlas
