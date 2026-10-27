// sygst's Blocked family (sygst_native.hh): LAPACK's dsygst/zhegst schedule on the uplo triangles, the
// SLM diagonal kernel (sygst_diag.cc) and the public trsm, gemm, her2k/syr2k and trmm as its children.
// evidence: docs/perf/sygvd.md#sygst-the-blocked-family

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/her2k.hh>
#include <batchlas/blas/functions/syr2k.hh>
#include <batchlas/blas/functions/trmm.hh>
#include <batchlas/blas/functions/trsm.hh>

#include <batchlas/tuning_params.hh>
#include <batchlas/util/mempool.hh>

#include <src/ops/sygst/sygst_native.hh>
#include <src/util/template-instantiations.hh>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>

namespace batchlas::ops::sygst {

// Whether the call is one diagonal-kernel launch: no child, no workspace, no chunking.
inline bool single_block(int n, int nb) { return n <= std::min(nb, sycl_sygst::kDiagMaxOrder); }

// Fewest items per pass, in CUs: parts of 128 or 256 items starve the children (float n = 480, batch
// 1024: 22.0 ms at potrf's chunk, 14.0 ms at 512). Here, not in tuning_params.hh, which getrf, potrf,
// geqrf and syev hash as a kernel dependency. evidence: docs/perf/sygvd.md#sygst-measured
inline constexpr std::int64_t kBlockedMinChunkCus = 4;

// Items per pass: potrf's L2 chunk, at least kBlockedMinChunkCus x CU unless
// BATCHLAS_TUNE_L2_CHUNK_ITEMS forces it. One diagonal block never splits. Query and call both use it.
template <class T>
int blocked_chunk(Queue& q, int n, int nb, int batch) {
    if (single_block(n, nb)) return batch;
    const auto dev = q.device();
    const std::int64_t cu = std::max<std::int64_t>(1, dev.get_property(DeviceProperty::MAX_COMPUTE_UNITS));
    std::int64_t c = tuning::blocked_chunk_items(
        batch, static_cast<std::int64_t>(n) * (n + 1) * static_cast<std::int64_t>(sizeof(T)),
        static_cast<std::int64_t>(dev.get_property(DeviceProperty::GLOBAL_MEM_CACHE_SIZE)), cu);
    if (tuning::detail::tuning_env_override(settings().geometry.tune.l2_chunk_items, 0) <= 0)
        c = std::max(c, kBlockedMinChunkCus * cu);
    return static_cast<int>(batch >= 2 * c ? c : batch);
}

inline int blocked_part(int chunk, int batch) { return batch < 2 * chunk ? batch : chunk + batch % chunk; }

// One pointer array per trsm role (the vendor trsm reads them), the whole Hermitian diagonal block
// (nb x nb per item) that the gemm updates read, and, for itypes 2 and 3, the n x nb panel product per
// item that the out-of-place trmm needs. Nothing for a single diagonal block.
template <class T>
struct BlockedLayout {
    Span<T*> pf, pa;
    Span<T> h, w;
};

template <class T>
BlockedLayout<T> blocked_layout(Queue& q, BumpAllocator& pool, int n, int nb, int itype, int part) {
    BlockedLayout<T> l;
    if (single_block(n, nb)) return l;
    const auto p = static_cast<std::size_t>(part), un = static_cast<std::size_t>(n), unb = static_cast<std::size_t>(nb);
    l.pf = pool.allocate<T*>(q, p);
    l.pa = pool.allocate<T*>(q, p);
    l.h = pool.allocate<T>(q, unb * unb * p);
    if (itype != 1) l.w = pool.allocate<T>(q, un * unb * p);
    return l;
}

template <class T>
std::size_t blocked_workspace(Queue& q, const MV<T>& A, int nb, int itype) {
    const int n = A.rows(), batch = A.batch_size();
    const int part = blocked_part(blocked_chunk<T>(q, n, nb, batch), batch);
    return workspace_bytes([&](BumpAllocator& p) { return blocked_layout<T>(q, p, n, nb, itype, part); });
}

// One pass over cb items: views at absolute block offsets into A and B, and the workspace blocks.
template <class T>
struct Pass {
    T* a;
    const T* b;
    int lda, ldb, sa, sb, cb, n, nb;
    BlockedLayout<T> l;
    MV<T> A(int r0, int nr, int c0, int nc, T** p = nullptr) const {
        return MV<T>(a + static_cast<std::ptrdiff_t>(c0) * lda + r0, nr, nc, lda, sa, cb, p);
    }
    MV<T> Bm(int r0, int nr, int c0, int nc, T** p = nullptr) const {
        return MV<T>(const_cast<T*>(b) + static_cast<std::ptrdiff_t>(c0) * ldb + r0, nr, nc, ldb, sb, cb, p);
    }
    MV<T> W(int nr, int nc, int ld) const { return MV<T>(l.w.data(), nr, nc, ld, n * nb, cb); }
    MV<T> H(int kb) const { return MV<T>(l.h.data(), kb, kb, nb, nb * nb, cb); }
    T* at_a(int o) const { return a + static_cast<std::ptrdiff_t>(o) * lda + o; }
    const T* at_b(int o) const { return b + static_cast<std::ptrdiff_t>(o) * ldb + o; }
};

// C := C + alpha X Y^H + conj(alpha) Y X^H (NoTrans) or alpha X^H Y + conj(alpha) Y^H X, on C's uplo triangle.
template <Backend B, class T>
void sym_r2k(Queue& q, const MV<T>& X, const MV<T>& Y, const MV<T>& C, T alpha, Uplo uplo, bool trans) {
    if constexpr (is_std_complex_v<T>)
        (void)::batchlas::her2k<B, T>(q, X, Y, C, alpha, float_t<T>(1), uplo, trans ? Transpose::ConjTrans : Transpose::NoTrans);
    else (void)::batchlas::syr2k<B, T>(q, X, Y, C, alpha, T(1), uplo, trans ? Transpose::Trans : Transpose::NoTrans);
}

// C := alpha X Y + C. The Hermitian factor is the mirrored block H, so this is LAPACK's hemm / symm as
// one gemm. evidence: docs/perf/sygvd.md#sygst-the-blocked-family
template <Backend B, class T>
void mm(Queue& q, const MV<T>& X, const MV<T>& Y, const MV<T>& C, T alpha) {
    (void)::batchlas::gemm<B, T>(q, X, Y, C, alpha, T(1), Transpose::NoTrans, Transpose::NoTrans);
}

template <class T>
void mirror(Queue& q, const Pass<T>& p, int at, int kb, Uplo uplo) {
    (void)sycl_sygst::sygst_mirror<T>(q, p.at_a(at), p.lda, p.sa, kb, p.cb, uplo, p.l.h.data(), p.nb,
                                      std::ptrdiff_t(p.nb) * p.nb);
}

// The order-n block at offset o, nb at a time (dsygst / zhegst). A diagonal block wider than the kernel
// recurses at a quarter of its order (at least the kernel's): the same reduction with its own factor.
template <Backend B, class T>
void blocked_range(Queue& q, const Pass<T>& p, int o, int n, int nb, int itype, Uplo uplo) {
    const bool lower = uplo == Uplo::Lower;
    const T one(1), half(0.5), mhalf(-0.5), mone(-1);
    // Reduce the diagonal block at `at`; with `full`, also leave the whole Hermitian result in H.
    auto diag = [&](int at, int kb, bool full) {
        if (kb <= sycl_sygst::kDiagMaxOrder) {
            (void)sycl_sygst::sygst_diag<T>(q, p.at_a(at), p.lda, p.sa, p.at_b(at), p.ldb, p.sb, kb, p.cb, itype, uplo,
                                            full ? p.l.h.data() : nullptr, p.nb, std::ptrdiff_t(p.nb) * p.nb);
            return;
        }
        blocked_range<B, T>(q, p, at, kb, std::max(sycl_sygst::kDiagMaxOrder, kb / 4), itype, uplo);
        if (full) mirror(q, p, at, kb, uplo);
    };
    T** const pf = p.l.pf.data();
    T** const pa = p.l.pa.data();
    for (int k = 0; k < n; k += nb) {
        const int kb = std::min(nb, n - k), at = o + k, m = n - k - kb, r = at + kb;
        const auto H = p.H(kb);
        if (itype == 1) {
            diag(at, kb, m > 0);
            if (m == 0) break;
            const auto F11 = p.Bm(at, kb, at, kb, pf);
            const auto F22 = p.Bm(r, m, r, m, pf);
            const auto A22 = p.A(r, m, r, m);
            if (lower) {  // A21 := L22^-1 (A21 L11^-H - L21 C11), A22 -= P L21^H + L21 P^H
                const auto P = p.A(r, m, at, kb, pa);
                const auto L21 = p.Bm(r, m, at, kb);
                (void)::batchlas::trsm<B, T>(q, F11, P, one, Side::Right, uplo, kH<T>, Diag::NonUnit);
                mm<B, T>(q, L21, H, P, mhalf);
                sym_r2k<B, T>(q, P, L21, A22, mone, uplo, false);
                mm<B, T>(q, L21, H, P, mhalf);
                (void)::batchlas::trsm<B, T>(q, F22, P, one, Side::Left, uplo, Transpose::NoTrans, Diag::NonUnit);
            } else {  // A12 := (U11^-H A12 - C11 U12) U22^-1, A22 -= P^H U12 + U12^H P
                const auto P = p.A(at, kb, r, m, pa);
                const auto U12 = p.Bm(at, kb, r, m);
                (void)::batchlas::trsm<B, T>(q, F11, P, one, Side::Left, uplo, kH<T>, Diag::NonUnit);
                mm<B, T>(q, H, U12, P, mhalf);
                sym_r2k<B, T>(q, P, U12, A22, mone, uplo, true);
                mm<B, T>(q, H, U12, P, mhalf);
                (void)::batchlas::trsm<B, T>(q, F22, P, one, Side::Right, uplo, Transpose::NoTrans, Diag::NonUnit);
            }
            continue;
        }
        if (k > 0) {  // itypes 2, 3: extend the reduced leading k x k block by this block row/column
            const auto A11 = p.A(o, k, o, k);
            const auto F11 = p.Bm(o, k, o, k);
            const auto F22 = p.Bm(at, kb, at, kb);
            mirror(q, p, at, kb, uplo);  // H := A22, whole
            if (lower) {  // W = A21 L11 + A22 L21 / 2, A11 += W^H L21 + L21^H W, A21 = L22^H (W + A22 L21 / 2)
                const auto P = p.A(at, kb, o, k);
                const auto L21 = p.Bm(at, kb, o, k);
                const auto W = p.W(kb, k, p.nb);
                (void)::batchlas::trmm<B, T>(q, F11, P, W, one, Side::Right, uplo, Transpose::NoTrans, Diag::NonUnit);
                mm<B, T>(q, H, L21, W, half);
                sym_r2k<B, T>(q, W, L21, A11, one, uplo, true);
                mm<B, T>(q, H, L21, W, half);
                (void)::batchlas::trmm<B, T>(q, F22, W, P, one, Side::Left, uplo, kH<T>, Diag::NonUnit);
            } else {  // W = U11 A12 + U12 A22 / 2, A11 += W U12^H + U12 W^H, A12 = (W + U12 A22 / 2) U22^H
                const auto P = p.A(o, k, at, kb);
                const auto U12 = p.Bm(o, k, at, kb);
                const auto W = p.W(k, kb, p.n);
                (void)::batchlas::trmm<B, T>(q, F11, P, W, one, Side::Left, uplo, Transpose::NoTrans, Diag::NonUnit);
                mm<B, T>(q, U12, H, W, half);
                sym_r2k<B, T>(q, W, U12, A11, one, uplo, false);
                mm<B, T>(q, U12, H, W, half);
                (void)::batchlas::trmm<B, T>(q, F22, W, P, one, Side::Right, uplo, kH<T>, Diag::NonUnit);
            }
        }
        diag(at, kb, false);
    }
}

template <Backend B, class T>
Event blocked(Queue& q, const MV<T>& A, const MV<T>& Bf, int nb, int itype, Uplo uplo, Span<std::byte> ws) {
    const int n = A.rows(), batch = A.batch_size();
    const int chunk = blocked_chunk<T>(q, n, nb, batch);
    BumpAllocator pool(ws);
    const auto l = blocked_layout<T>(q, pool, n, nb, itype, blocked_part(chunk, batch));
    for (int b0 = 0, cb = 0; b0 < batch; b0 += cb) {
        cb = batch - b0 < 2 * chunk ? batch - b0 : chunk;  // the last part takes the remainder
        const Pass<T> p{A.data_ptr() + static_cast<std::ptrdiff_t>(b0) * A.stride(),
                        Bf.data_ptr() + static_cast<std::ptrdiff_t>(b0) * Bf.stride(),
                        A.ld(), Bf.ld(), A.stride(), Bf.stride(), cb, n, nb, l};
        blocked_range<B, T>(q, p, 0, n, nb, itype, uplo);
    }
    return q.get_event();
}

#define SYGST_BLOCKED_INSTANTIATE(B_, fp)                                                                     \
    template Event blocked<B_, BATCHLAS_UNPAREN fp>(Queue&, const MV<BATCHLAS_UNPAREN fp>&,                     \
                                                    const MV<BATCHLAS_UNPAREN fp>&, int, int, Uplo, Span<std::byte>);
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(SYGST_BLOCKED_INSTANTIATE)
#undef SYGST_BLOCKED_INSTANTIATE

template std::size_t blocked_workspace<float>(Queue&, const MV<float>&, int, int);
template std::size_t blocked_workspace<double>(Queue&, const MV<double>&, int, int);
template std::size_t blocked_workspace<std::complex<float>>(Queue&, const MV<std::complex<float>>&, int, int);
template std::size_t blocked_workspace<std::complex<double>>(Queue&, const MV<std::complex<double>>&, int, int);

}  // namespace batchlas::ops::sygst
