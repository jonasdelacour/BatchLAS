// SPDX-License-Identifier: MIT
// device_core_level3_tests: the register-tier Level 3 forms of level3.hh (design A.2) against
// long-double host loops.
// Guards (agent-guide section 8): every TA x TB and every Side x Uplo x Trans x Diag; complex
// data with nonzero imaginary parts; the unstored triangle of A and (for Unit) its diagonal
// poisoned with NaN; alpha != 1; beta == Zero over a NaN-poisoned C (never read); a well-
// conditioned graded triangle (diagonal 2 .. 3) so the solve tolerance is meaningful; 1024 items
// per launch, each with its own data.
// Red set (each break was made once, 2026-11-04, and this suite failed): trsm Right with the
// solve direction of the other triangle; the MacChain epilogue started from acc (Zero-beta gemm).
#include <tests/device_core_common.hh>
#include <batchlas/device/level3.hh>

#include <cmath>

using namespace device_core_test;
using batchlas::Diag;
using batchlas::Side;
using batchlas::Transpose;
using batchlas::Uplo;

namespace {

template <class T> class DeviceCoreLevel3Test : public DeviceCoreTest<T> {};
TYPED_TEST_SUITE(DeviceCoreLevel3Test, AllTypes);

constexpr int kItems = 1024;

template <class T> LD<T> opw(const std::vector<T>& a, std::size_t base, int ld, int i, int j, Transpose tr) {
    if (tr == Transpose::NoTrans) return wide(a[base + i + j * ld]);
    const LD<T> v = wide(a[base + j + i * ld]);
    if constexpr (is_cplx<T>) return tr == Transpose::ConjTrans ? std::conj(v) : v;
    else return v;
}

template <class T, Transpose TA, Transpose TB> void check_gemm(Dev& d) {
    constexpr int M = 3, N = 4, K = 5;
    constexpr int AR = TA == Transpose::NoTrans ? M : K, AC = TA == Transpose::NoTrans ? K : M;
    constexpr int BR = TB == Transpose::NoTrans ? K : N, BC = TB == Transpose::NoTrans ? N : K;
    const auto a_h = vals<T>(std::size_t(kItems) * AR * AC, 31), b_h = vals<T>(std::size_t(kItems) * BR * BC, 32);
    const auto c_h = vals<T>(std::size_t(kItems) * M * N, 33);
    const T alpha = val<T>(7), beta = val<T>(8);
    T* a = d.up(a_h);
    T* b = d.up(b_h);
    T* c = d.up(c_h);
    T* c0 = d.template alloc<T>(std::size_t(kItems) * M * N, dv::qnan<T>());  // beta == Zero: poisoned, never read
    d.template run<1>(kItems, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int q = static_cast<int>(it.get_group_linear_id());
        dv::Tile<T, AR, AC> A;
        dv::Tile<T, BR, BC> B;
        dv::Tile<T, M, N> C, C0;
        dv::load(A, dv::Mat<T, dv::Ext<AR>, dv::Ext<AC>, dv::Ext<AR>>{a + q * AR * AC, {}, {}, {}});
        dv::load(B, dv::Mat<T, dv::Ext<BR>, dv::Ext<BC>, dv::Ext<BR>>{b + q * BR * BC, {}, {}, {}});
        dv::load(C, dv::Mat<T, dv::Ext<M>, dv::Ext<N>, dv::Ext<M>>{c + q * M * N, {}, {}, {}});
        dv::load(C0, dv::Mat<T, dv::Ext<M>, dv::Ext<N>, dv::Ext<M>>{c0 + q * M * N, {}, {}, {}});
        dv::gemm<TA, TB>(dv::Thread{}, alpha, A, B, beta, C);
        dv::gemm<TA, TB, dv::epi::MacChain, dv::arith::Fma>(dv::Thread{}, alpha, A, B, dv::Zero{}, C0);
        dv::store(C, dv::Mat<T, dv::Ext<M>, dv::Ext<N>, dv::Ext<M>>{c + q * M * N, {}, {}, {}});
        dv::store(C0, dv::Mat<T, dv::Ext<M>, dv::Ext<N>, dv::Ext<M>>{c0 + q * M * N, {}, {}, {}});
    });
    const auto got = d.down(c, c_h.size()), got0 = d.down(c0, c_h.size());
    for (int q = 0; q < kItems; q += 37)
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < M; ++i) {
                LD<T> ab = 0;
                for (int k = 0; k < K; ++k) ab += opw(a_h, std::size_t(q) * AR * AC, AR, i, k, TA) * opw(b_h, std::size_t(q) * BR * BC, BR, k, j, TB);
                const std::size_t e = std::size_t(q) * M * N + i + j * M;
                EXPECT_LE(err(got[e], wide(alpha) * ab + wide(beta) * wide(c_h[e])), tol<T>(K)) << q << " " << i << "," << j;
                EXPECT_LE(err(got0[e], wide(alpha) * ab), tol<T>(K)) << "Zero beta " << q << " " << i << "," << j;
            }
}

TYPED_TEST(DeviceCoreLevel3Test, RegisterGemmEveryOp) {
    using T = TypeParam;
    check_gemm<T, Transpose::NoTrans, Transpose::NoTrans>(this->dev);
    check_gemm<T, Transpose::Trans, Transpose::NoTrans>(this->dev);
    check_gemm<T, Transpose::NoTrans, Transpose::ConjTrans>(this->dev);
    check_gemm<T, Transpose::ConjTrans, Transpose::Trans>(this->dev);
}

TYPED_TEST(DeviceCoreLevel3Test, RegisterGemmSubSubtracts) {
    using T = TypeParam;
    auto& d = this->dev;
    constexpr int M = 4, K = 3, N = 2;
    const auto a_h = vals<T>(M * K, 41), b_h = vals<T>(K * N, 42), c_h = vals<T>(M * N, 43);
    T* a = d.up(a_h);
    T* b = d.up(b_h);
    T* c = d.up(c_h);
    d.template run<1>(1, 0, [=](sycl::nd_item<1>, unsigned char*) {
        dv::Tile<T, M, K> A;
        dv::Tile<T, K, N> B;
        dv::Tile<T, M, N> C;
        dv::load(A, dv::Mat<T, dv::Ext<M>, dv::Ext<K>, dv::Ext<M>>{a, {}, {}, {}});
        dv::load(B, dv::Mat<T, dv::Ext<K>, dv::Ext<N>, dv::Ext<K>>{b, {}, {}, {}});
        dv::load(C, dv::Mat<T, dv::Ext<M>, dv::Ext<N>, dv::Ext<M>>{c, {}, {}, {}});
        dv::gemm_sub<dv::arith::FmaSplit>(dv::Thread{}, A, B, C);
        dv::store(C, dv::Mat<T, dv::Ext<M>, dv::Ext<N>, dv::Ext<M>>{c, {}, {}, {}});
    });
    const auto got = d.down(c, M * N);
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < M; ++i) {
            LD<T> ref = wide(c_h[i + j * M]);
            for (int k = 0; k < K; ++k) ref -= wide(a_h[i + k * M]) * wide(b_h[k + j * K]);
            EXPECT_LE(err(got[i + j * M], ref), tol<T>(K)) << i << "," << j;
        }
}

template <class T, Side SD, Uplo UL, Transpose TR, Diag DG> void check_trsm(Dev& d) {
    constexpr int M = 4, N = 3, NA = SD == Side::Left ? M : N;
    using R = real_of<T>;
    // A: the stored triangle graded (diag 2 .. 3, off-diagonal in [-0.5, 0.5)), the rest NaN;
    // a Unit diagonal is NaN too (never read).
    std::vector<T> a_h(std::size_t(kItems) * NA * NA, dv::qnan<T>());
    for (int q = 0; q < kItems; ++q)
        for (int j = 0; j < NA; ++j)
            for (int i = 0; i < NA; ++i) {
                const bool stored = UL == Uplo::Lower ? i > j : i < j;
                const std::size_t e = std::size_t(q) * NA * NA + i + j * NA;
                if (stored) a_h[e] = val<T>(q * 97 + i * 7 + j) * T(R(0.5));
                if (i == j && DG == Diag::NonUnit) a_h[e] = T(R(2) + R(i) / R(NA)) + (is_cplx<T> ? val<T>(q + i) * T(R(0.25)) : T(0));
            }
    const auto b_h = vals<T>(std::size_t(kItems) * M * N, 51);
    const T alpha = val<T>(9);
    T* a = d.up(a_h);
    T* b = d.up(b_h);
    d.template run<1>(kItems, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const int q = static_cast<int>(it.get_group_linear_id());
        dv::Tile<T, NA, NA> A;
        dv::Tile<T, M, N> B;
        dv::load(A, dv::Mat<T, dv::Ext<NA>, dv::Ext<NA>, dv::Ext<NA>>{a + q * NA * NA, {}, {}, {}});
        dv::load(B, dv::Mat<T, dv::Ext<M>, dv::Ext<N>, dv::Ext<M>>{b + q * M * N, {}, {}, {}});
        dv::trsm<SD, UL, TR, DG>(dv::Thread{}, alpha, A, B);
        dv::store(B, dv::Mat<T, dv::Ext<M>, dv::Ext<N>, dv::Ext<M>>{b + q * M * N, {}, {}, {}});
    });
    const auto X = d.down(b, b_h.size());
    // Residual: op(A) X - alpha B (Left) or X op(A) - alpha B (Right), A with its implied diagonal.
    auto aw = [&](int q, int i, int j) -> LD<T> {
        const int r = TR == Transpose::NoTrans ? i : j, c = TR == Transpose::NoTrans ? j : i;
        const bool stored = UL == Uplo::Lower ? r > c : r < c;
        if (r == c) return DG == Diag::Unit ? LD<T>(1) : opw(a_h, std::size_t(q) * NA * NA, NA, i, j, TR);
        return stored ? opw(a_h, std::size_t(q) * NA * NA, NA, i, j, TR) : LD<T>(0);
    };
    for (int q = 0; q < kItems; q += 53)
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < M; ++i) {
                LD<T> lhs = 0;
                if (SD == Side::Left)
                    for (int k = 0; k < M; ++k) lhs += aw(q, i, k) * wide(X[std::size_t(q) * M * N + k + j * M]);
                else
                    for (int k = 0; k < N; ++k) lhs += wide(X[std::size_t(q) * M * N + i + k * M]) * aw(q, k, j);
                const std::size_t e = std::size_t(q) * M * N + i + j * M;
                EXPECT_LE(static_cast<double>(std::abs(lhs - wide(alpha) * wide(b_h[e]))), tol<T>(8))
                    << "side " << int(SD) << " uplo " << int(UL) << " tr " << int(TR) << " diag " << int(DG) << " item " << q << " " << i << "," << j;
            }
}

template <class T, Side SD, Uplo UL> void check_trsm_ops(Dev& d) {
    check_trsm<T, SD, UL, Transpose::NoTrans, Diag::NonUnit>(d);
    check_trsm<T, SD, UL, Transpose::Trans, Diag::Unit>(d);
    check_trsm<T, SD, UL, Transpose::ConjTrans, Diag::NonUnit>(d);
    check_trsm<T, SD, UL, Transpose::NoTrans, Diag::Unit>(d);
}

TYPED_TEST(DeviceCoreLevel3Test, RegisterTrsmEverySideUploTransDiag) {
    using T = TypeParam;
    check_trsm_ops<T, Side::Left, Uplo::Lower>(this->dev);
    check_trsm_ops<T, Side::Left, Uplo::Upper>(this->dev);
    check_trsm_ops<T, Side::Right, Uplo::Lower>(this->dev);
    check_trsm_ops<T, Side::Right, Uplo::Upper>(this->dev);
}

}  // namespace
