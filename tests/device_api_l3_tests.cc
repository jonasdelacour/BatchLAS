// SPDX-License-Identifier: MIT
// device_api_l3_tests: the in-group Level 3 ops (include/batchlas/device/level3/group.hh, v22
// design B.3) called from a user-style kernel, one matrix per work-group of 256 items, against
// long-double host references and (float gemm) the host reg route pinned to the same tile.
// Guards (agent-guide section 8): every form of every op (TA x TB; Side x Uplo; Uplo x Trans;
// Side x Uplo x Trans x Diag); complex data with nonzero imaginary parts and a nonzero Im(diag)
// for hemm / herk / her2k (ignored, resp. written as 0); ld = rows + 3 and a batch stride past
// ld * cols, the pads holding a large finite value that must stay; the unread triangle of A (and
// a Unit diagonal) NaN; beta == Zero over a NaN C (never read); the unwritten triangle of C
// checked bit for bit; m, n, k off the 32-tile grid (2 x 3 tiles, a k tail); trsm over 2 and 3
// diagonal blocks of a graded well-conditioned triangle; a 1024-item batch of one repeated
// problem bit-identical to item 0; the float gemm bit-identical to the pinned, reached host route.
// Red set (each seen red, sm_89): TriMask keep() flipped (syrk/herk, syr2k/her2k); trsm's
// backward block order reversed (trsm); run_tile's second pass dropped (syr2k/her2k); sym_at's
// Hermitian diagonal kept complex (hemm); trmm's lower k range ending at m0 (trmm). A Left upper
// k range from 0 stays green: tri_at is zero off the triangle, so it only costs work.
#include <tests/device_core_common.hh>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/options.hh>
#include <batchlas/device/level3/group.hh>
#include <batchlas/verify/tolerance.hh>
#include <src/ops/gemm/choice.hh>
#include <src/select/select.hh>
#include <tests/test_support/coverage_rows.hh>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

using namespace device_core_test;
using batchlas::Diag;
using batchlas::Side;
using batchlas::Transpose;
using batchlas::Uplo;
using batchlas::verify::Check;

namespace {

constexpr int kWG = 256;
template <class T> using Cfg = dv::l3::cfg::RegTiled<T, 32, 32, 8, 2, 2>;
static_assert(Cfg<float>::threads == kWG);
constexpr int kItems = 3;
constexpr int kM = 45, kN = 70, kK = 37;
constexpr Transpose kNT = Transpose::NoTrans, kTr = Transpose::Trans, kCT = Transpose::ConjTrans;

template <class T> T pad_value() { return T(real_of<T>(1e30)); }

// A batch of column-major rows x cols matrices, ld = rows + 3, stride = ld * cols + 5; every
// element outside the matrices is pad_value().
template <class T> struct HostMat {
    int rows, cols, ld, items;
    std::size_t stride;
    std::vector<T> h;
    HostMat(int r, int c, int n_items, std::uint64_t seed)
        : rows(r), cols(c), ld(r + 3), items(n_items), stride(std::size_t(r + 3) * c + 5), h(stride * n_items, pad_value<T>()) {
        for (int q = 0; q < items; ++q)
            for (int j = 0; j < cols; ++j)
                for (int i = 0; i < rows; ++i) at(q, i, j) = val<T>(seed * 1000003u + off(q, i, j));
    }
    std::size_t off(int q, int i, int j) const { return std::size_t(q) * stride + i + std::size_t(j) * ld; }
    T& at(int q, int i, int j) { return h[off(q, i, j)]; }
    const T& at(int q, int i, int j) const { return h[off(q, i, j)]; }
    LD<T> w(int q, int i, int j) const { return wide(at(q, i, j)); }
    bool inside(std::size_t e) const {
        const std::size_t r = e % stride;
        return r < std::size_t(ld) * cols && int(r % ld) < rows;
    }
};

template <class T> struct DevMat {
    T* p;
    int rows, cols, ld;
    std::size_t stride;
    dv::l3::Global<T> g(int q) const { return {p + std::size_t(q) * stride, ld, rows, cols}; }
};
template <class T> DevMat<T> to_dev(Dev& d, const HostMat<T>& m) { return {d.up(m.h), m.rows, m.cols, m.ld, m.stride}; }

template <class T> LD<T> opw(const HostMat<T>& a, int q, int i, int j, Transpose tr) {
    if (tr == Transpose::NoTrans) return a.w(q, i, j);
    const LD<T> v = a.w(q, j, i);
    if constexpr (is_cplx<T>) return tr == Transpose::ConjTrans ? std::conj(v) : v;
    else return v;
}
template <class T> LD<T> cj(LD<T> v) {
    if constexpr (is_cplx<T>) return std::conj(v);
    else return v;
}

// |got - ref| / scale, NaN-propagating; the worst over a check.
struct Worst {
    double v = 0;
    void add(double num, double scale) { v = batchlas::verify::nanmax(v, scale > 0 ? num / scale : num); }
};

// Elements outside the matrices (ld and stride pads) are unchanged, bit for bit.
template <class T> int pads_changed(const HostMat<T>& before, const std::vector<T>& after) {
    int bad = 0;
    for (std::size_t e = 0; e < after.size(); ++e)
        if (!before.inside(e) && !same_bits(before.h[e], after[e])) ++bad;
    return bad;
}

// The kernel every case launches: items work-groups of kWG, f(ex, item, smem).
template <class T, class F> void launch(Dev& d, int items, F f) {
    d.template run<kWG>(items, dv::l3::smem_bytes<Cfg<T>>(), [=](sycl::nd_item<1> it, unsigned char* lm) {
        const dv::WorkGroup<kWG> ex{it};
        f(ex, static_cast<int>(it.get_group_linear_id()), dv::l3::Smem<Cfg<T>>{reinterpret_cast<T*>(lm)});
    });
}

template <class T> class DeviceApiL3Test : public DeviceCoreTest<T> {};
TYPED_TEST_SUITE(DeviceApiL3Test, AllTypes);

// ---- gemm ---------------------------------------------------------------------------------

template <class T, Transpose TA, Transpose TB> void check_gemm(Dev& d) {
    const HostMat<T> A(TA == kNT ? kM : kK, TA == kNT ? kK : kM, kItems, 11), B(TB == kNT ? kK : kN, TB == kNT ? kN : kK, kItems, 12);
    const HostMat<T> C(kM, kN, kItems, 13);
    HostMat<T> C0 = C;
    for (int q = 0; q < kItems; ++q)
        for (int j = 0; j < kN; ++j)
            for (int i = 0; i < kM; ++i) C0.at(q, i, j) = dv::qnan<T>();
    const T alpha = val<T>(7), beta = val<T>(8);
    const auto a = to_dev(d, A), b = to_dev(d, B), c = to_dev(d, C), c0 = to_dev(d, C0);
    const batchlas::KernelMatrixView<T, batchlas::MatrixFormat::Dense> av(a.p, a.rows, a.cols, a.ld, int(a.stride), kItems);
    launch<T>(d, kItems, [=](const auto& ex, int q, auto sm) {
        const dv::Mat<T> cm{c.p + std::size_t(q) * c.stride, dv::Dyn{c.rows}, dv::Dyn{c.cols}, dv::Dyn{c.ld}};
        dv::gemm<TA, TB, Cfg<T>>(ex, dv::item(av, q, a.rows, a.cols), b.g(q), cm, alpha, beta, sm);
        dv::gemm<TA, TB, Cfg<T>>(ex, dv::item(av, q, a.rows, a.cols), b.g(q), c0.g(q), alpha, dv::Zero{}, sm);
    });
    const auto got = d.down(c.p, C.h.size()), got0 = d.down(c0.p, C0.h.size());
    Worst w, w0;
    for (int q = 0; q < kItems; ++q)
        for (int j = 0; j < kN; ++j)
            for (int i = 0; i < kM; ++i) {
                LD<T> ab = 0;
                long double s = 0;
                for (int k = 0; k < kK; ++k) {
                    const LD<T> t = opw(A, q, i, k, TA) * opw(B, q, k, j, TB);
                    ab += t;
                    s += std::abs(t);
                }
                const LD<T> ref = wide(alpha) * ab + wide(beta) * C.w(q, i, j);
                const double scale = double(std::abs(wide(alpha)) * s + std::abs(wide(beta) * C.w(q, i, j)));
                w.add(err(got[C.off(q, i, j)], ref), scale);
                w0.add(err(got0[C.off(q, i, j)], wide(alpha) * ab), double(std::abs(wide(alpha)) * s));
            }
    EXPECT_VERIFY(T, Check::blas, kK, w.v) << "TA " << int(TA) << " TB " << int(TB);
    EXPECT_VERIFY(T, Check::blas, kK, w0.v) << "beta Zero over NaN C, TA " << int(TA) << " TB " << int(TB);
    EXPECT_EQ(pads_changed(C, got), 0);
    EXPECT_EQ(pads_changed(C0, got0), 0);
}

TYPED_TEST(DeviceApiL3Test, GemmEveryForm) {
    using T = TypeParam;
    check_gemm<T, kNT, kNT>(this->dev);
    check_gemm<T, kTr, kNT>(this->dev);
    check_gemm<T, kNT, kCT>(this->dev);
    check_gemm<T, kCT, kTr>(this->dev);
}

// One problem repeated over 1024 work-groups: every C bit-identical to item 0's (SLM races).
TYPED_TEST(DeviceApiL3Test, GemmAndTrsmSaturatingBatchBitIdentical) {
    using T = TypeParam;
    auto& d = this->dev;
    constexpr int items = 1024;
    HostMat<T> A(kM, kK, items, 31), B(kK, kN, items, 32), C(kM, kN, items, 33), L(kM, kM, items, 34);
    for (int j = 0; j < kM; ++j)
        for (int i = 0; i < kM; ++i) L.at(0, i, j) = i == j ? T(real_of<T>(2)) : L.at(0, i, j) * T(real_of<T>(0.5 / kM));
    for (int q = 1; q < items; ++q)
        for (int j = 0; j < kN; ++j)
            for (int i = 0; i < kM; ++i) {
                if (j < kK) A.at(q, i, j) = A.at(0, i, j);
                if (i < kK) B.at(q, i, j) = B.at(0, i, j);
                C.at(q, i, j) = C.at(0, i, j);
                if (j < kM) L.at(q, i, j) = L.at(0, i, j);
            }
    const T alpha = val<T>(5), beta = val<T>(6);
    const auto a = to_dev(d, A), b = to_dev(d, B), c = to_dev(d, C), l = to_dev(d, L);
    launch<T>(d, items, [=](const auto& ex, int q, auto sm) {
        dv::gemm<kNT, kNT, Cfg<T>>(ex, a.g(q), b.g(q), c.g(q), alpha, beta, sm);
        dv::trsm<Side::Left, Uplo::Lower, kNT, Diag::NonUnit, Cfg<T>>(ex, l.g(q), c.g(q), alpha, sm);
    });
    const auto got = d.down(c.p, C.h.size());
    int differ = 0;
    for (int q = 1; q < items; ++q)
        for (int j = 0; j < kN; ++j)
            for (int i = 0; i < kM; ++i) differ += !same_bits(got[C.off(q, i, j)], got[C.off(0, i, j)]);
    EXPECT_EQ(differ, 0);
}

class DeviceApiL3Host : public DeviceCoreTest<float> {};

// The in-group gemm with Cfg = reg_configs' 32x32x8 row (TR = TC = 2, one stage, u = 1) is the
// host reg route's arithmetic: bit-identical C, pads included, with the route pinned and reached.
TEST_F(DeviceApiL3Host, GemmFloatIsTheHostRegRouteBitForBit) {
    using T = float;
    auto& d = this->dev;
    constexpr int items = 4;
    const HostMat<T> A(kM, kK, items, 41), B(kK, kN, items, 42), C(kM, kN, items, 43);
    const T alpha = val<T>(9), beta = val<T>(10);
    const auto a = to_dev(d, A), b = to_dev(d, B), c = to_dev(d, C), ch = to_dev(d, C);
    launch<T>(d, items, [=](const auto& ex, int q, auto sm) { dv::gemm<kNT, kNT, Cfg<T>>(ex, a.g(q), b.g(q), c.g(q), alpha, beta, sm); });
    const batchlas::MatrixView<T, batchlas::MatrixFormat::Dense> av(a.p, a.rows, a.cols, a.ld, int(a.stride), items),
        bv(b.p, b.rows, b.cols, b.ld, int(b.stride), items), cv(ch.p, ch.rows, ch.cols, ch.ld, int(ch.stride), items);
    const std::string spelling = "reg:m=32:n=32:k=8:u=1";
    const auto rows = test_support::reached_rows([&] {
        const batchlas::select::ScopedPin<batchlas::ops::gemm::GemmChoice> pin("gemm", spelling, batchlas::select::StrictPin{});
        batchlas::gemm(*d.ctx, av, bv, cv, batchlas::GemmOptions<T>{alpha, beta, kNT, kNT});
        d.ctx->wait();
    });
    bool reached = false;
    for (const auto& r : rows) reached = reached || (r.op == "gemm" && r.spelling == spelling);
    EXPECT_TRUE(reached) << "the pinned host route was not reached";
    const auto dev_c = d.down(c.p, C.h.size()), host_c = d.down(ch.p, C.h.size());
    int differ = 0;
    for (std::size_t e = 0; e < dev_c.size(); ++e) differ += !same_bits(dev_c[e], host_c[e]);
    EXPECT_EQ(differ, 0);
}

// ---- symm, hemm ---------------------------------------------------------------------------

template <class T, Side SD, Uplo UL, bool Herm> void check_symm(Dev& d) {
    const int na = SD == Side::Left ? kM : kN;
    HostMat<T> A(na, na, kItems, 51);
    const HostMat<T> B(kM, kN, kItems, 52), C(kM, kN, kItems, 53);
    for (int q = 0; q < kItems; ++q)
        for (int j = 0; j < na; ++j)
            for (int i = 0; i < na; ++i)
                if (UL == Uplo::Lower ? i < j : i > j) A.at(q, i, j) = dv::qnan<T>();  // unread triangle
    auto sym = [&](int q, int i, int j) -> LD<T> {
        if (Herm && i == j) return LD<T>(std::real(A.w(q, i, i)));
        const bool stored = UL == Uplo::Lower ? i >= j : i <= j;
        return stored ? A.w(q, i, j) : (Herm ? cj<T>(A.w(q, j, i)) : A.w(q, j, i));
    };
    const T alpha = val<T>(3), beta = val<T>(4);
    const auto a = to_dev(d, A), b = to_dev(d, B), c = to_dev(d, C);
    launch<T>(d, kItems, [=](const auto& ex, int q, auto sm) {
        if constexpr (Herm) dv::hemm<SD, UL, Cfg<T>>(ex, a.g(q), b.g(q), c.g(q), alpha, beta, sm);
        else dv::symm<SD, UL, Cfg<T>>(ex, a.g(q), b.g(q), c.g(q), alpha, beta, sm);
    });
    const auto got = d.down(c.p, C.h.size());
    Worst w;
    for (int q = 0; q < kItems; ++q)
        for (int j = 0; j < kN; ++j)
            for (int i = 0; i < kM; ++i) {
                LD<T> ab = 0;
                long double s = 0;
                for (int k = 0; k < na; ++k) {
                    const LD<T> t = SD == Side::Left ? sym(q, i, k) * B.w(q, k, j) : B.w(q, i, k) * sym(q, k, j);
                    ab += t;
                    s += std::abs(t);
                }
                w.add(err(got[C.off(q, i, j)], wide(alpha) * ab + wide(beta) * C.w(q, i, j)),
                      double(std::abs(wide(alpha)) * s + std::abs(wide(beta) * C.w(q, i, j))));
            }
    EXPECT_VERIFY(T, Check::blas, na, w.v) << (Herm ? "hemm" : "symm") << " side " << int(SD) << " uplo " << int(UL);
    EXPECT_EQ(pads_changed(C, got), 0);
}

TYPED_TEST(DeviceApiL3Test, SymmHemmEverySideUplo) {
    using T = TypeParam;
    check_symm<T, Side::Left, Uplo::Lower, false>(this->dev);
    check_symm<T, Side::Right, Uplo::Upper, false>(this->dev);
    check_symm<T, Side::Left, Uplo::Upper, true>(this->dev);
    check_symm<T, Side::Right, Uplo::Lower, true>(this->dev);
}

// ---- syrk, herk, syr2k, her2k -------------------------------------------------------------

// Op 0 syrk, 1 herk, 2 syr2k, 3 her2k; C is kN x kN, op(A), op(B) kN x kK.
template <class T, int Op, Uplo UL, Transpose TR> void check_rank(Dev& d) {
    constexpr bool herm = Op == 1 || Op == 3, two = Op >= 2;
    const int ar = TR == kNT ? kN : kK, ac = TR == kNT ? kK : kN;
    const HostMat<T> A(ar, ac, kItems, 61), B(ar, ac, kItems, 62);
    HostMat<T> C(kN, kN, kItems, 63);
    if constexpr (!herm || !is_cplx<T>) {
        // a real-T herk / her2k must not read Im(diag): there is none
    } else {
        for (int q = 0; q < kItems; ++q)
            for (int i = 0; i < kN; ++i) C.at(q, i, i) = T(std::real(C.at(q, i, i)), real_of<T>(0.75));  // ignored on input
    }
    const real_of<T> alpha_r = real_of<T>(0.625), beta_r = real_of<T>(-0.375);
    const T alpha = Op == 1 ? T(alpha_r) : val<T>(1);
    const T beta = herm ? T(beta_r) : val<T>(2);
    const auto a = to_dev(d, A), b = to_dev(d, B), c = to_dev(d, C);
    launch<T>(d, kItems, [=](const auto& ex, int q, auto sm) {
        if constexpr (Op == 0) dv::syrk<UL, TR, Cfg<T>>(ex, a.g(q), c.g(q), alpha, beta, sm);
        else if constexpr (Op == 1) dv::herk<UL, TR, Cfg<T>>(ex, a.g(q), c.g(q), alpha_r, beta_r, sm);
        else if constexpr (Op == 2) dv::syr2k<UL, TR, Cfg<T>>(ex, a.g(q), b.g(q), c.g(q), alpha, beta, sm);
        else dv::her2k<UL, TR, Cfg<T>>(ex, a.g(q), b.g(q), c.g(q), alpha, beta_r, sm);
    });
    const auto got = d.down(c.p, C.h.size());
    auto opa = [&](const HostMat<T>& X, int q, int i, int k) { return opw(X, q, i, k, TR); };
    auto tr2 = [&](LD<T> v) { return herm ? cj<T>(v) : v; };
    Worst w;
    int other = 0, diag_im = 0;
    for (int q = 0; q < kItems; ++q)
        for (int j = 0; j < kN; ++j)
            for (int i = 0; i < kN; ++i) {
                const std::size_t e = C.off(q, i, j);
                if (UL == Uplo::Lower ? i < j : i > j) {
                    other += !same_bits(got[e], C.h[e]);
                    continue;
                }
                LD<T> acc = 0;
                long double s = 0;
                for (int k = 0; k < kK; ++k) {
                    LD<T> t = wide(alpha) * opa(A, q, i, k) * tr2(opa(two ? B : A, q, j, k));
                    if (two) {
                        const LD<T> al2 = herm ? cj<T>(wide(alpha)) : wide(alpha);
                        const LD<T> t2 = al2 * opa(B, q, i, k) * tr2(opa(A, q, j, k));
                        s += std::abs(t2);
                        t += t2;
                    }
                    s += std::abs(t);
                    acc += t;
                }
                LD<T> cin = C.w(q, i, j);
                if (herm && i == j) cin = LD<T>(std::real(cin));
                LD<T> ref = acc + wide(beta) * cin;
                if (herm && i == j) ref = LD<T>(std::real(ref));
                w.add(err(got[e], ref), double(s + std::abs(wide(beta) * cin)));
                if constexpr (is_cplx<T>)
                    if (herm && i == j) diag_im += std::imag(got[e]) != 0;
            }
    EXPECT_VERIFY(T, Check::blas, two ? 2 * kK : kK, w.v) << "op " << Op << " uplo " << int(UL) << " trans " << int(TR);
    EXPECT_EQ(other, 0) << "the other triangle of C was written";
    EXPECT_EQ(diag_im, 0) << "Im C(i, i) must be 0";
    EXPECT_EQ(pads_changed(C, got), 0);
}

TYPED_TEST(DeviceApiL3Test, SyrkHerkEveryUploTrans) {
    using T = TypeParam;
    check_rank<T, 0, Uplo::Lower, kNT>(this->dev);
    check_rank<T, 0, Uplo::Upper, kTr>(this->dev);
    check_rank<T, 1, Uplo::Upper, kNT>(this->dev);
    check_rank<T, 1, Uplo::Lower, kCT>(this->dev);
}

TYPED_TEST(DeviceApiL3Test, Syr2kHer2kEveryUploTrans) {
    using T = TypeParam;
    check_rank<T, 2, Uplo::Upper, kNT>(this->dev);
    check_rank<T, 2, Uplo::Lower, kTr>(this->dev);
    check_rank<T, 3, Uplo::Lower, kNT>(this->dev);
    check_rank<T, 3, Uplo::Upper, kCT>(this->dev);
}

// ---- trmm, trsm ---------------------------------------------------------------------------

// op(A) with the implied structure, from a triangle whose other half (and a Unit diagonal) is NaN.
template <class T, Uplo UL, Transpose TR, Diag DG> LD<T> tri_w(const HostMat<T>& a, int q, int i, int j) {
    const int r = TR == kNT ? i : j, c = TR == kNT ? j : i;
    if (r == c) return DG == Diag::Unit ? LD<T>(1) : opw(a, q, i, j, TR);
    const bool stored = UL == Uplo::Lower ? r > c : r < c;
    return stored ? opw(a, q, i, j, TR) : LD<T>(0);
}
// A graded triangle: diagonal 2 .. 3 (+ an imaginary part), off-diagonal |a| < 0.5 / n.
template <class T, Uplo UL, Diag DG> HostMat<T> graded_triangle(int na, std::uint64_t seed) {
    HostMat<T> A(na, na, kItems, seed);
    using R = real_of<T>;
    for (int q = 0; q < kItems; ++q)
        for (int j = 0; j < na; ++j)
            for (int i = 0; i < na; ++i) {
                const bool stored = UL == Uplo::Lower ? i > j : i < j;
                if (i == j) A.at(q, i, j) = DG == Diag::Unit ? dv::qnan<T>() : T(R(2) + R(i) / R(na)) + (is_cplx<T> ? val<T>(q + i) * T(R(0.25)) : T(0));
                else A.at(q, i, j) = stored ? A.at(q, i, j) * T(R(0.5) / R(na)) : dv::qnan<T>();
            }
    return A;
}

template <class T, Side SD, Uplo UL, Transpose TR, Diag DG> void check_trmm(Dev& d) {
    const int na = SD == Side::Left ? kM : kN;
    const HostMat<T> A = graded_triangle<T, UL, DG>(na, 71), B(kM, kN, kItems, 72);
    HostMat<T> C(kM, kN, kItems, 73);
    for (int q = 0; q < kItems; ++q)
        for (int j = 0; j < kN; ++j)
            for (int i = 0; i < kM; ++i) C.at(q, i, j) = dv::qnan<T>();  // output only: never read
    const T alpha = val<T>(5);
    const auto a = to_dev(d, A), b = to_dev(d, B), c = to_dev(d, C);
    launch<T>(d, kItems, [=](const auto& ex, int q, auto sm) { dv::trmm<SD, UL, TR, DG, Cfg<T>>(ex, a.g(q), b.g(q), c.g(q), alpha, sm); });
    const auto got = d.down(c.p, C.h.size()), b_after = d.down(b.p, B.h.size());
    Worst w;
    for (int q = 0; q < kItems; ++q)
        for (int j = 0; j < kN; ++j)
            for (int i = 0; i < kM; ++i) {
                LD<T> acc = 0;
                long double s = 0;
                for (int k = 0; k < na; ++k) {
                    const LD<T> t = SD == Side::Left ? tri_w<T, UL, TR, DG>(A, q, i, k) * B.w(q, k, j) : B.w(q, i, k) * tri_w<T, UL, TR, DG>(A, q, k, j);
                    acc += t;
                    s += std::abs(t);
                }
                w.add(err(got[C.off(q, i, j)], wide(alpha) * acc), double(std::abs(wide(alpha)) * s));
            }
    EXPECT_VERIFY(T, Check::blas, na, w.v) << "side " << int(SD) << " uplo " << int(UL) << " tr " << int(TR) << " diag " << int(DG);
    int b_changed = 0;
    for (std::size_t e = 0; e < b_after.size(); ++e) b_changed += !same_bits(b_after[e], B.h[e]);
    EXPECT_EQ(b_changed, 0) << "trmm is out of place";
    EXPECT_EQ(pads_changed(C, got), 0);
}

template <class T, Side SD, Uplo UL> void check_trmm_ops(Dev& d) {
    check_trmm<T, SD, UL, kNT, Diag::NonUnit>(d);
    check_trmm<T, SD, UL, kTr, Diag::Unit>(d);
    check_trmm<T, SD, UL, kCT, Diag::NonUnit>(d);
    check_trmm<T, SD, UL, kNT, Diag::Unit>(d);
}

TYPED_TEST(DeviceApiL3Test, TrmmEverySideUploTransDiag) {
    using T = TypeParam;
    check_trmm_ops<T, Side::Left, Uplo::Lower>(this->dev);
    check_trmm_ops<T, Side::Left, Uplo::Upper>(this->dev);
    check_trmm_ops<T, Side::Right, Uplo::Lower>(this->dev);
    check_trmm_ops<T, Side::Right, Uplo::Upper>(this->dev);
}

// B := alpha op(A)^-1 B; Left: 45 rows = 2 diagonal blocks of 32, Right: 70 columns = 3.
template <class T, Side SD, Uplo UL, Transpose TR, Diag DG, bool AlphaOne> void check_trsm(Dev& d) {
    const int na = SD == Side::Left ? kM : kN;
    const HostMat<T> A = graded_triangle<T, UL, DG>(na, 81), B(kM, kN, kItems, 82);
    const T alpha = AlphaOne ? T(1) : val<T>(6);
    const auto a = to_dev(d, A), b = to_dev(d, B);
    launch<T>(d, kItems, [=](const auto& ex, int q, auto sm) {
        if constexpr (AlphaOne) dv::trsm<SD, UL, TR, DG, Cfg<T>>(ex, a.g(q), b.g(q), dv::One{}, sm);
        else dv::trsm<SD, UL, TR, DG, Cfg<T>>(ex, a.g(q), b.g(q), alpha, sm);
    });
    const auto got = d.down(b.p, B.h.size());
    constexpr bool op_lower = (UL == Uplo::Lower) == (TR == kNT);
    auto opa = [&](int q, int i, int j) { return tri_w<T, UL, TR, DG>(A, q, i, j); };
    Worst w;
    for (int q = 0; q < kItems; ++q) {
        std::vector<LD<T>> x(std::size_t(kM) * kN);
        auto X = [&](int i, int j) -> LD<T>& { return x[i + std::size_t(j) * kM]; };
        for (int j = 0; j < kN; ++j)
            for (int i = 0; i < kM; ++i) X(i, j) = wide(alpha) * B.w(q, i, j);
        if (SD == Side::Left) {
            for (int j = 0; j < kN; ++j)
                for (int t = 0; t < kM; ++t) {
                    const int i = op_lower ? t : kM - 1 - t;
                    for (int p = 0; p < kM; ++p)
                        if (op_lower ? p < i : p > i) X(i, j) -= opa(q, i, p) * X(p, j);
                    X(i, j) /= opa(q, i, i);
                }
        } else {
            for (int i = 0; i < kM; ++i)
                for (int t = 0; t < kN; ++t) {
                    const int j = op_lower ? kN - 1 - t : t;
                    for (int p = 0; p < kN; ++p)
                        if (op_lower ? p > j : p < j) X(i, j) -= X(i, p) * opa(q, p, j);
                    X(i, j) /= opa(q, j, j);
                }
        }
        long double xmax = 0, emax = 0;
        for (int j = 0; j < kN; ++j)
            for (int i = 0; i < kM; ++i) {
                xmax = std::max<long double>(xmax, std::abs(X(i, j)));
                emax = std::max<long double>(emax, std::abs(wide(got[B.off(q, i, j)]) - X(i, j)));
                if (std::isnan(double(std::abs(wide(got[B.off(q, i, j)]))))) emax = std::numeric_limits<long double>::quiet_NaN();
            }
        w.add(double(emax), double(xmax));
    }
    EXPECT_VERIFY(T, Check::solve, na, w.v) << "side " << int(SD) << " uplo " << int(UL) << " tr " << int(TR) << " diag " << int(DG);
    EXPECT_EQ(pads_changed(B, got), 0);
}

template <class T, Side SD, Uplo UL> void check_trsm_ops(Dev& d) {
    check_trsm<T, SD, UL, kNT, Diag::NonUnit, false>(d);
    check_trsm<T, SD, UL, kTr, Diag::Unit, false>(d);
    check_trsm<T, SD, UL, kCT, Diag::NonUnit, true>(d);
    check_trsm<T, SD, UL, kNT, Diag::Unit, true>(d);
}

TYPED_TEST(DeviceApiL3Test, TrsmEverySideUploTransDiag) {
    using T = TypeParam;
    check_trsm_ops<T, Side::Left, Uplo::Lower>(this->dev);
    check_trsm_ops<T, Side::Left, Uplo::Upper>(this->dev);
    check_trsm_ops<T, Side::Right, Uplo::Lower>(this->dev);
    check_trsm_ops<T, Side::Right, Uplo::Upper>(this->dev);
}

}  // namespace
