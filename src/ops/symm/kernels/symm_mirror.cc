// SPDX-License-Identifier: MIT
// symm / hemm `mirror`: wide v2's double-buffered 64x64 gemm (register_64x64_k16_wide.hh), its
// lane maps, swizzled staging, fragment loop and epilogue, with the symmetric operand staged from
// its named triangle. A slab strictly inside the named triangle is staged with the gemm's own
// load, one strictly inside the other triangle with the transposed form's load of the mirror (and
// its conjugate for hemm), and a slab touching the diagonal element by element. double has no
// wide v2 tile, so it runs gemm's 16x16 tiled kernel with the same mirrored element read.
// evidence: docs/perf/level3.md#level-3-the-mirror-family

#include <src/ops/symm/kernels/symm_mirror.hh>
#include <src/ops/symm/mirror_tile.hh>

#include <src/ops/gemm/kernels/register_64x64_k16_wide.hh>
#include <src/ops/gemm/kernels/epilogue_linear.hh>
#include <src/linalg-impl.hh>
#include <src/sycl/kernel_attrs.hh>
#include <src/util/batch_mask.hh>
#include <src/util/kernel-trace.hh>

#include <sycl/sycl.hpp>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace batchlas::sycl_symm {

namespace {

using sycl_gemm::GemmWide64Body;
using sycl_gemm::Vec4;
namespace wide_scalar = sycl_gemm::wide_scalar;
namespace wide_v2 = sycl_gemm::wide_v2;

template <typename T>
using MV = MatrixView<T, MatrixFormat::Dense>;

#if !BATCHLAS_SYCL_IMPL_ACPP && BATCHLAS_NVPTX_REQD_WG_OK && defined(__SYCL_DEVICE_ONLY__) && \
    defined(__NVPTX__)
#define BATCHLAS_SYMM_MIRROR_REQD_WG [[sycl::reqd_work_group_size(1, 1, 256)]]
#else
#define BATCHLAS_SYMM_MIRROR_REQD_WG
#endif

// Right: the symmetric matrix is the gemm's B operand (C = B A). Lower: the named triangle in the
// operand's own (i, kk) coordinates is i >= kk (Left Lower, Right Upper), else i <= kk. The gemm
// body G is NN for Left; for Right its B operand is the NT instance's, whose rows are unpadded.
template <typename T, bool Right, bool Herm, bool VecLoads, int TileK>
struct SymmMirrorBody {
    using G = GemmWide64Body<T, Transpose::NoTrans, Right ? Transpose::Trans : Transpose::NoTrans, VecLoads, TileK>;
    using D = typename G::D;
    static constexpr int CF = G::CF, VecN = G::VecN, Threads = G::Threads, TTM = G::TTM, TTN = G::TTN;
    static constexpr int NBM = G::NBM, NBN = G::NBN, MSep = G::MSep, NSep = G::NSep;
    static constexpr int GPR = G::GPR, KStep = G::KStep, OpThreads = G::OpThreads, KG = G::KG;
    static constexpr int RowA = G::RowA, RowB = G::RowB, BufA = G::BufA, BufB = G::BufB, RB0 = G::RB0;
    static constexpr bool Split = G::Split, PA = G::PA, PB = G::PB;
    static constexpr bool KD = Right ? G::KA : G::KB;  // the dense operand's staging orientation
    static constexpr int RS = Right ? RB0 : 0, RD = Right ? 0 : RB0;
    static constexpr int kMinGroupsPerCu = G::kMinGroupsPerCu, EpiBatch = G::EpiBatch;
    enum Mode { Nat, Mir, Mix };
    static_assert(G::TileM == 64 && G::TileN == 64 && !G::kPipe, "one 64-wide half per operand, register prefetch");
    static_assert(!(Right ? PB : PA), "both lane maps of the symmetric operand store one unpadded, swizzled row");
    static_assert(!G::CA && !G::CB, "no transpose-conjugate form: hemm conjugates at staging");
    static_assert(std::int64_t(2) * (BufA + BufB) * 4 <= ops::mirror::slm_bytes<T>(), "can_run's slm_bytes bound");
    static_assert(TileK == ops::mirror::tile_k<T>() && ops::mirror::tile<T>() == 64, "mirror_tile.hh's limits");

    const float* a_ptr;
    const float* b_ptr;
    D* c_ptr;
    int m, n, k, lda, ldb, ldc;
    std::ptrdiff_t stride_a, stride_b, stride_c;
    int batch;
    bool lower;
    D alpha, beta;
    sycl::local_accessor<wide_scalar::Vec16<float>, 1> tile_a, tile_b;
    const std::uint8_t* mask;  // the batch mask; null unmasked

    // The k-contiguous gload map (G::first_k<true> is the element leg's map when that leg pads).
    static int kc_k(int t) {
        if constexpr (CF != 2 || TileK == 8) return (t % KG) * VecN;
        else return ((t & 3) | ((t >> 1) & 4)) * VecN;
    }
    static int kc_i(int t) {
        if constexpr (CF != 2 || TileK == 8) return (t / KG) * CF;
        else return (((t >> 2) & 1) | ((t >> 3) & ~1)) * CF;
    }
    template <bool KC>
    __attribute__((always_inline)) static void sload(Vec4<float> (&r)[CF], const float* p, int ld, int k0, int kb,
                                                     int ib, int k, int lim, bool full) {
        if (full) wide_v2::gload<false, VecLoads, KC, CF, KStep>(r, p, ld, k0, kb, ib, k, lim);
        else wide_v2::gload<true, VecLoads, KC, CF, KStep>(r, p, ld, k0, kb, ib, k, lim);
    }
    // A diagonal slab on the along-i map: element (i, kk) from the named triangle, else its mirror.
    // Address i + kk ld is the stored element on the Left and the mirror on the Right.
    __attribute__((always_inline)) static void mixed(Vec4<float> (&r)[CF], const float* base, int ld, int s0, int k0,
                                                     int kb, int ib, int k, bool lower) {
BATCHLAS_UNROLL_FULL
        for (int c = 0; c < CF; ++c) {
BATCHLAS_UNROLL_FULL
            for (int q = 0; q < VecN; ++q) {
                const int kk = k0 + kb + c * KStep, i = s0 + ib + q;
                const bool named = lower ? i >= kk : i <= kk;
                const std::ptrdiff_t off = named != Right ? i + static_cast<std::ptrdiff_t>(kk) * ld
                                                          : kk + static_cast<std::ptrdiff_t>(i) * ld;
                const bool ok = kk < k && i < k;
                float v[CF];
BATCHLAS_UNROLL_FULL
                for (int e = 0; e < CF; ++e) v[e] = ok ? base[off * CF + e] : 0.0f;
                if constexpr (Herm) {
                    if (i == kk) {
BATCHLAS_UNROLL_FULL
                        for (int e = CF / 2; e < CF; ++e) v[e] = 0.0f;
                    } else if (!named) {
                        v[CF - 1] = -v[CF - 1];  // the imaginary part's sign (complex<double>: its high word)
                    }
                }
BATCHLAS_UNROLL_FULL
                for (int e = 0; e < CF; ++e) r[c][q * CF + e] = v[e];
            }
        }
    }
    __attribute__((always_inline)) static void conj(Vec4<float> (&r)[CF]) {
BATCHLAS_UNROLL_FULL
        for (int c = 0; c < CF; ++c) {
BATCHLAS_UNROLL_FULL
            for (int q = 0; q < VecN; ++q) r[c][q * CF + CF - 1] = -r[c][q * CF + CF - 1];
        }
    }

    BATCHLAS_LAUNCH_BOUNDS(Threads, kMinGroupsPerCu)
    BATCHLAS_SYMM_MIRROR_REQD_WG
    __attribute__((always_inline)) void operator()(sycl::nd_item<3> item) const {
        using namespace wide_v2;
        const int bid = static_cast<int>(item.get_group(0));
        if (bid >= batch || (mask != nullptr && mask[bid] == 0)) return;  // I1: before any barrier
        const int tid = static_cast<int>(item.get_local_id(2));
        const int ty = tid & 15, tx = tid >> 4;
        const int m0 = static_cast<int>(item.get_group(1)) * 64;
        const int n0 = static_cast<int>(item.get_group(2)) * 64;
        D* Cb = c_ptr + static_cast<std::ptrdiff_t>(bid) * stride_c;
        float* sa = reinterpret_cast<float*>(tile_a.template get_multi_ptr<sycl::access::decorated::no>().get());
        float* sb = reinterpret_cast<float*>(tile_b.template get_multi_ptr<sycl::access::decorated::no>().get());

        const int st = tid % OpThreads;
        const bool stage_a = !Split || tid < OpThreads;  // warp-uniform
        const bool stage_s = Right ? (!Split || !stage_a) : stage_a;
        const bool stage_d = Right ? stage_a : (!Split || !stage_a);
        const float* abase = a_ptr + static_cast<std::ptrdiff_t>(bid) * stride_a * CF;
        const float* bbase = b_ptr + static_cast<std::ptrdiff_t>(bid) * stride_b * CF;

        const int d0 = Right ? m0 : n0, d_lim = (Right ? m : n) - d0, ldd = Right ? lda : ldb;
        const int d_kb = G::template first_k<KD>(st), d_ib = G::template first_i<KD>(st);
        const float* pd = (Right ? abase : bbase) +
                          (KD ? static_cast<std::ptrdiff_t>(d0 + d_ib) * ldd + d_kb
                              : d0 + d_ib + static_cast<std::ptrdiff_t>(d_kb) * ldd) * CF;
        const bool d_in = d_lim >= 64;

        const int s0 = Right ? n0 : m0, lds = Right ? ldb : lda;
        const float* sbase = Right ? bbase : abase;
        const int n_kb = st / GPR, n_ib = (st % GPR) * VecN;  // along i: address i + kk ld
        const int t_kb = kc_k(st), t_ib = kc_i(st);           // along k: address i ld + kk
        const float* pn = sbase + (s0 + n_ib + static_cast<std::ptrdiff_t>(n_kb) * lds) * CF;
        const float* pt = sbase + (static_cast<std::ptrdiff_t>(s0 + t_ib) * lds + t_kb) * CF;
        const bool s_in = s0 + 64 <= k;

        D accum[TTM][TTN];
BATCHLAS_UNROLL_FULL
        for (int i = 0; i < TTM; ++i) {
BATCHLAS_UNROLL_FULL
            for (int j = 0; j < TTN; ++j) accum[i][j] = D{};
        }

        Vec4<float> r[G::NR][CF];
        bool s_kc = false;  // the staged slab's map: uniform over the group, as its mode is
        auto load = [&](int k0) __attribute__((always_inline)) {
            const bool kin = k0 + TileK <= k;
            if (stage_d) G::template fetch<KD>(r[RD], pd, ldd, k0, d_kb, d_ib, k, d_lim, d_in && kin);
            if (stage_s) {
                const bool below = s0 >= k0 + TileK, above = s0 + 64 <= k0;
                const Mode mode = (lower ? below : above) ? Nat : (lower ? above : below) ? Mir : Mix;
                s_kc = mode != Mix && (mode == Nat) == Right;
                if (mode == Mix) mixed(r[RS], sbase, lds, s0, k0, n_kb, n_ib, k, lower);
                else if (s_kc) sload<true>(r[RS], pt, lds, k0, t_kb, t_ib, k, k - s0, s_in && kin);
                else sload<false>(r[RS], pn, lds, k0, n_kb, n_ib, k, k - s0, s_in && kin);
                if constexpr (Herm) {
                    if (mode == Mir) conj(r[RS]);
                }
            }
        };
        auto store = [&](int buf) __attribute__((always_inline)) {
            if (stage_d) {
                if constexpr (Right) G::template put<KD, RowA, 64>(sa + buf * BufA, r[RD], d_kb, d_ib);
                else G::template put<KD, RowB, 64>(sb + buf * BufB, r[RD], d_kb, d_ib);
            }
            if (stage_s) {
                float* s = Right ? sb + buf * BufB : sa + buf * BufA;
                if (s_kc) sstore<true, CF, KStep, TileK, 64>(s, r[RS], t_kb, t_ib);
                else sstore<false, CF, KStep, TileK, 64>(s, r[RS], n_kb, n_ib);
            }
        };
        auto compute = [&](int buf) __attribute__((always_inline)) {
            const float* xa = sa + buf * BufA;
            const float* xb = sb + buf * BufB;
BATCHLAS_UNROLL_FULL
            for (int kk = 0; kk < TileK; ++kk) {
                D af[TTM], bf[TTN];
BATCHLAS_UNROLL_FULL
                for (int band = 0; band < NBM; ++band) {
                    const Vec4<float> ta =
                        ld16(xa + kk * RowA + (((band * (MSep / VecN) + ty) ^ G::template sw<PA>(kk)) << 2));
BATCHLAS_UNROLL_FULL
                    for (int w = 0; w < VecN; ++w) af[band * VecN + w] = elem<D>(ta, w ^ (PA ? ekx<CF, TileK>(kk) : 0));
                }
BATCHLAS_UNROLL_FULL
                for (int band = 0; band < NBN; ++band) {
                    const Vec4<float> tb =
                        ld16(xb + kk * RowB + (((band * (NSep / VecN) + tx) ^ G::template sw<PB>(kk)) << 2));
BATCHLAS_UNROLL_FULL
                    for (int w = 0; w < VecN; ++w) bf[band * VecN + w] = elem<D>(tb, w ^ (PB ? ekx<CF, TileK>(kk) : 0));
                }
BATCHLAS_UNROLL_FULL
                for (int i = 0; i < TTM; ++i) {
BATCHLAS_UNROLL_FULL
                    for (int j = 0; j < TTN; ++j) fma_signed<false, false>(accum[i][j], af[i], bf[j]);
                }
            }
        };

        const int slabs = (k + TileK - 1) / TileK;
        load(0);
        store(0);
        item.barrier(sycl::access::fence_space::local_space);
        int s = 0;
        for (; s + 1 < slabs; s += 2) {
            load((s + 1) * TileK);
            compute(0);
            store(1);
            item.barrier(sycl::access::fence_space::local_space);
            const bool more = s + 2 < slabs;
            if (more) load((s + 2) * TileK);
            compute(1);
            if (more) store(0);
            item.barrier(sycl::access::fence_space::local_space);
        }
        if (s < slabs) compute(0);

        const bool beta_zero = wide_scalar::dev_is_zero(beta);
        const bool c_vec =
            m0 + 64 <= m && n0 + 64 <= n && ldc % VecN == 0 && reinterpret_cast<std::uintptr_t>(Cb) % 16u == 0;
        if (!c_vec) {
            if (slabs % 2 == 1) item.barrier(sycl::access::fence_space::local_space);
            wide_scalar::staged_c_store<64, 64, Threads, 2 * TileK * 64, 2 * TileK * 64, EpiBatch>(
                item, reinterpret_cast<D*>(sa), reinterpret_cast<D*>(sb), Cb, ldc, m0, n0, m, n, tid, alpha, beta,
                beta_zero, [&](int c0, int cw, auto slot) {
BATCHLAS_UNROLL_FULL
                    for (int bm = 0; bm < NBM; ++bm) {
BATCHLAS_UNROLL_FULL
                        for (int j = 0; j < TTN; ++j) {
                            const int c = (j / VecN) * NSep + tx * VecN + j % VecN - c0;
                            if (c < 0 || c >= cw) continue;
                            wide_scalar::Vec16<D> v;
BATCHLAS_UNROLL_FULL
                            for (int w = 0; w < VecN; ++w) v.v[w] = accum[bm * VecN + w][j];
                            wide_scalar::vec_ref(slot(c, bm * MSep + ty * VecN)) = v;
                        }
                    }
                });
            return;
        }
BATCHLAS_UNROLL_FULL
        for (int bm = 0; bm < NBM; ++bm) {
            const int gm = m0 + bm * MSep + ty * VecN;
BATCHLAS_UNROLL_FULL
            for (int j = 0; j < TTN; ++j) {
                const int gn = n0 + (j / VecN) * NSep + tx * VecN + j % VecN;
                D* p = &Cb[gm + static_cast<std::ptrdiff_t>(gn) * ldc];
                wide_scalar::Vec16<D> out;
                wide_scalar::Vec16<D> prior{};
                if (!beta_zero) prior = wide_scalar::vec_ref(const_cast<const D*>(p));
BATCHLAS_UNROLL_FULL
                for (int w = 0; w < VecN; ++w)
                    out.v[w] = wide_scalar::lin_epi(alpha, beta_zero ? D{} : beta, accum[bm * VecN + w][j], prior.v[w]);
                wide_scalar::vec_ref(p) = out;
            }
        }
    }
};
#undef BATCHLAS_SYMM_MIRROR_REQD_WG

template <typename T, bool Right, bool Herm, bool VecLoads>
class SymmMirrorKernel;

template <typename T, bool Right, bool Herm, bool VecLoads>
Event launch_wide(Queue& q, const MV<T>& A, const MV<T>& Bm, const MV<T>& C, T alpha, T beta, bool lower) {
    using Body = SymmMirrorBody<T, Right, Herm, VecLoads, ops::mirror::tile_k<T>()>;
    using D = typename Body::D;
    const MV<T>& Ag = Right ? Bm : A;
    const MV<T>& Bg = Right ? A : Bm;
    const int m = C.rows(), n = C.cols(), k = A.rows();
    const sycl::range<3> local(1, 1, Body::Threads);
    const sycl::range<3> global(static_cast<size_t>(C.batch_size()), static_cast<size_t>((m + 63) / 64),
                                static_cast<size_t>((n + 63) / 64) * Body::Threads);
    q->submit([&](sycl::handler& h) {
        sycl::local_accessor<wide_scalar::Vec16<float>, 1> tile_a(sycl::range<1>(2 * Body::BufA / 4), h);
        sycl::local_accessor<wide_scalar::Vec16<float>, 1> tile_b(sycl::range<1>(2 * Body::BufB / 4), h);
        D alpha_d, beta_d;
        __builtin_memcpy(&alpha_d, &alpha, sizeof(D));
        __builtin_memcpy(&beta_d, &beta, sizeof(D));
        Body body{reinterpret_cast<const float*>(Ag.data_ptr()), reinterpret_cast<const float*>(Bg.data_ptr()),
                  reinterpret_cast<D*>(C.data_ptr()), m, n, k, static_cast<int>(Ag.ld()), static_cast<int>(Bg.ld()),
                  static_cast<int>(C.ld()), Ag.stride(), Bg.stride(), C.stride(), static_cast<int>(C.batch_size()),
                  lower, alpha_d, beta_d, tile_a, tile_b,
                  batchlas::detail::mask_ptr(batchlas::detail::call_mask("symm", A, Bm, C))};
        batchlas::impl::parallel_for<SymmMirrorKernel<T, Right, Herm, VecLoads>>(h, sycl::nd_range<3>(global, local),
                                                                                body);
    });
    return q.get_event();
}

template <typename T, bool Right>
class SymmMirrorTiledKernel;

// double: gemm's tiled kernel (tiled_general.hh) with the symmetric operand read element by element
// from its named triangle (storage coordinates: lower means row >= col is stored).
template <typename T, bool Right>
Event launch_tiled(Queue& q, const MV<T>& A, const MV<T>& Bm, const MV<T>& C, T alpha, T beta, bool lower) {
    constexpr int Tile = ops::mirror::tile<T>();
    static_assert(std::int64_t(2) * Tile * Tile * sizeof(T) <= ops::mirror::slm_bytes<T>() && Tile * Tile == 256,
                  "mirror_tile.hh's limits");
    const int m = C.rows(), n = C.cols(), k = A.rows();
    const sycl::range<3> local(1, Tile, Tile);
    const sycl::range<3> global(static_cast<size_t>(C.batch_size()), static_cast<size_t>((m + Tile - 1) / Tile * Tile),
                                static_cast<size_t>((n + Tile - 1) / Tile * Tile));
    q->submit([&](sycl::handler& h) {
        sycl::local_accessor<T, 1> tile_a(sycl::range<1>(Tile * Tile), h);
        sycl::local_accessor<T, 1> tile_b(sycl::range<1>(Tile * Tile), h);
        const T* a_ptr = A.data_ptr();
        const T* b_ptr = Bm.data_ptr();
        T* c_ptr = C.data_ptr();
        const int lda = A.ld(), ldb = Bm.ld(), ldc = C.ld(), batch = C.batch_size();
        const std::ptrdiff_t stride_a = A.stride(), stride_b = Bm.stride(), stride_c = C.stride();
        const std::uint8_t* mask = batchlas::detail::mask_ptr(batchlas::detail::call_mask("symm", A, Bm, C));
        batchlas::impl::parallel_for<SymmMirrorTiledKernel<T, Right>>(
            h, sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item) {
                const int bid = static_cast<int>(item.get_group(0));
                if (bid >= batch || (mask != nullptr && mask[bid] == 0)) return;  // I1: before any barrier
                const int lr = static_cast<int>(item.get_local_id(1)), lc = static_cast<int>(item.get_local_id(2));
                const int row = static_cast<int>(item.get_group(1)) * Tile + lr;
                const int col = static_cast<int>(item.get_group(2)) * Tile + lc;
                const T* a_item = a_ptr + static_cast<std::ptrdiff_t>(bid) * stride_a;
                const T* b_item = b_ptr + static_cast<std::ptrdiff_t>(bid) * stride_b;
                auto sym = [&](int r, int c) {
                    const bool named = lower ? r >= c : r <= c;
                    return named ? a_item[r + static_cast<std::ptrdiff_t>(c) * lda]
                                 : a_item[c + static_cast<std::ptrdiff_t>(r) * lda];
                };
                auto dense = [&](int r, int c) { return b_item[r + static_cast<std::ptrdiff_t>(c) * ldb]; };
                T sum = T(0);
                for (int kk0 = 0; kk0 < k; kk0 += Tile) {
                    const int ka = kk0 + lc, kb = kk0 + lr;
                    tile_a[lr * Tile + lc] = row < m && ka < k ? (Right ? dense(row, ka) : sym(row, ka)) : T(0);
                    tile_b[lr * Tile + lc] = col < n && kb < k ? (Right ? sym(kb, col) : dense(kb, col)) : T(0);
                    item.barrier(sycl::access::fence_space::local_space);
                    for (int t = 0; t < Tile && kk0 + t < k; ++t) sum += tile_a[lr * Tile + t] * tile_b[t * Tile + lc];
                    item.barrier(sycl::access::fence_space::local_space);
                }
                if (row < m && col < n)
                    sycl_gemm::LinearEpilogue<T>::store(
                        c_ptr + static_cast<std::ptrdiff_t>(bid) * stride_c + row + static_cast<std::ptrdiff_t>(col) * ldc,
                        alpha, beta, sum);
            });
    });
    return q.get_event();
}

// The chunk's items with its slice of the batch mask.
template <typename T>
MV<T> items(const MV<T>& v, int first, int count) {
    return v.batch_range(first, count);
}

}  // namespace

template <typename T, bool Herm>
Event symm_mirror(Queue& q, const MV<T>& A, const MV<T>& Bm, const MV<T>& C, T alpha, T beta, Side side, Uplo uplo,
                  int chunk) {
    const bool right = side == Side::Right, lower = uplo == Uplo::Lower;
    auto one = [&](const MV<T>& a, const MV<T>& b, const MV<T>& c) -> Event {
        if constexpr (std::is_same_v<T, double>) {
            BATCHLAS_KERNEL_TRACE_SCOPE("symm.mirror.tiled");
            return right ? launch_tiled<T, true>(q, a, b, c, alpha, beta, lower)
                         : launch_tiled<T, false>(q, a, b, c, alpha, beta, lower);
        } else {
            // The operand's own (i, kk) triangle: transposed on the Right (the gemm's B is A(kk, i)).
            const bool lower_ik = lower != right;
            const bool vec = right ? sycl_gemm::wide64_v2_vec_loads<T>(b, a) : sycl_gemm::wide64_v2_vec_loads<T>(a, b);
            BATCHLAS_KERNEL_TRACE_SCOPE(Herm ? (vec ? "hemm.mirror.vec" : "hemm.mirror.elem")
                                             : (vec ? "symm.mirror.vec" : "symm.mirror.elem"));
            if (right)
                return vec ? launch_wide<T, true, Herm, true>(q, a, b, c, alpha, beta, lower_ik)
                           : launch_wide<T, true, Herm, false>(q, a, b, c, alpha, beta, lower_ik);
            return vec ? launch_wide<T, false, Herm, true>(q, a, b, c, alpha, beta, lower_ik)
                       : launch_wide<T, false, Herm, false>(q, a, b, c, alpha, beta, lower_ik);
        }
    };
    const int batch = C.batch_size();
    if (batch <= chunk) return one(A, Bm, C);
    Event last;
    for (int first = 0; first < batch; first += chunk) {
        const int count = std::min(chunk, batch - first);
        last = one(items(A, first, count), items(Bm, first, count), items(C, first, count));
        if (!q.in_order()) last.wait();
    }
    return last;
}

#define SYMM_MIRROR_INSTANTIATE(T, H)                                                                              \
    template Event symm_mirror<T, H>(Queue&, const MV<T>&, const MV<T>&, const MV<T>&, T, T, Side, Uplo, int);
SYMM_MIRROR_INSTANTIATE(float, false)
SYMM_MIRROR_INSTANTIATE(double, false)
SYMM_MIRROR_INSTANTIATE(std::complex<float>, false)
SYMM_MIRROR_INSTANTIATE(std::complex<double>, false)
SYMM_MIRROR_INSTANTIATE(std::complex<float>, true)
SYMM_MIRROR_INSTANTIATE(std::complex<double>, true)
#undef SYMM_MIRROR_INSTANTIATE

}  // namespace batchlas::sycl_symm
