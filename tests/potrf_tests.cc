// Native batched POTRF -- the CTA kernel's tests.
//
// ---------------------------------------------------------------------------
// THE ORACLE IS NEVER THE VENDOR, AND THE ROUTE IS NEVER TRUSTED
// ---------------------------------------------------------------------------
// Two independent reasons, both of which have burned this repository:
//
//   * A VENDOR REFERENCE IS INERT IN A VENDOR-FREE BUILD. resolve_route falls
//     back to a supported NATIVE route when no vendor exists -- which is the
//     code under test -- so a "compare against cuSOLVER" test compares the
//     kernel with itself in exactly the build this work package exists for.
//
//   * A FORCED ROUTE THAT supports() REJECTS SILENTLY BECOMES THE VENDOR.
//     route_resolve.hh:101 tests `if (Table::supports(forced, s)) return forced;`
//     and falls through to automatic() at :111, so a test that sets
//     BATCHLAS_POTRF_ROUTE=cta and gets one gate wrong runs cuSOLVER and passes
//     GREEN over a kernel nothing executed.
//
// So: every numerical test below calls sycl_potrf::potrf_cta_dispatch<T>
// DIRECTLY -- a call a vendor cannot serve -- and checks a HOST MULTIPLY-BACK
// residual, ||L L^H - A||_F / ||A||_F, computed here from the returned factor
// and the input this file generated. That oracle depends on no other
// implementation in the tree. Exactly one test (FacadeReachesTheCtaKernel) is
// about routing, and it asserts the resolved route is native BEFORE it believes
// the numbers.
//
// ---------------------------------------------------------------------------
// EVERY CLAIM BELOW WAS EXECUTED, AND ONE OF THEM WAS WRONG THE FIRST TIME
// ---------------------------------------------------------------------------
// This project has three recorded incidents of a guard test that could not fail
// by construction (trmm uplo/diag; a conjugation test blind by construction; a
// ConjTrans test too small to reach the tile it guarded). Five deliberate breaks
// were run against this file. What each turned red, MEASURED:
//
//  1. THE STALE PIVOT reintroduced in (P1) (read the pivot from the tile instead
//     of shuffling lane k's register): 18 of 42 red -- every residual test.
//     InfoIndexIsExact and InfoReportsTheFirstFailure STAYED GREEN. That was not
//     the prediction, and it is the fourth instance of this repository's blind
//     guard: with a plain planted L0 D L0^H the ORIGINAL diagonal at the failure
//     column is still negative, so a stale-pivot reader names the same column.
//     make_planted_ldl now normalises row c's prefix so that the original
//     diagonal there is POSITIVE (+1) and only the updated Schur diagonal is
//     negative; the test asserts that property of its own input. Re-run with the
//     same break: 26 of 42 red, InfoIndexIsExact among them, reporting
//     info == 33 where 17 was planted.
//
//  2. THE (P1) PUBLISH GUARD `lane < ib` removed: 30 of 42 red, including
//     PackedBatchMatchesSolo, which is the only test that can see the half of
//     that defect that writes into a NEIGHBOURING MATRIX
//     ("packed vs solo differ at n=9 b=1 (0,0): 0 vs 1.53606").
//
//  3. THE (P3) FORCED-REAL HERMITIAN DIAGONAL removed: NOTHING went red, and
//     that is correct rather than a gap. Every diagonal entry of the OUTPUT is
//     written by (P1)'s publish, which stores dev_from_real(sqrt(akk)); the only
//     consumer of a tile diagonal is (P1)'s pivot, which takes dev_real(). The
//     residue that line discards is never read. It is defence in depth and the
//     kernel says so where it lives.
//
//  4. THE LOAD-SIDE diagonal real-forcing removed: also nothing red, for the
//     same reason.
//
//  5. BREAK 4 PLUS scaling (P1)'s diagonal publish instead of rebuilding it from
//     the real square root: ComplexDiagonalIsExactlyReal RED for both complex
//     types ("imag(L(1,1)) != 0", -2.08e-11 for complex<float>). So that test
//     CAN fail -- the diagonal-real property is enforced redundantly at three
//     points and any single removal is masked by the other two.
//
// All five were restored and the suite re-run green (42 passed, 46 skipped --
// the skips are the four NETLIB/CPU instantiations, which have no native route,
// plus the complex-only test on the two real types).
//
#include <gtest/gtest.h>

#include <batchlas/blas/linalg.hh>
#include <batchlas/blas/functions/potrf.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include "test_utils.hh"

#include "../src/extensions/potrf_native.hh"
#include "../src/backends/potrf_route.hh"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace batchlas;

namespace {

template <typename T>
using RealOf = typename batchlas::base_type<T>::type;

template <typename T>
inline T host_conj(T v) {
    if constexpr (test_utils::is_complex<T>::value) return std::conj(v);
    else return v;
}

template <typename T>
inline RealOf<T> host_real(T v) {
    if constexpr (test_utils::is_complex<T>::value) return v.real();
    else return v;
}

template <typename T>
inline RealOf<T> host_imag(T v) {
    if constexpr (test_utils::is_complex<T>::value) return v.imag();
    else return RealOf<T>(0);
}

template <typename T>
inline T make_scalar(RealOf<T> re, RealOf<T> im) {
    if constexpr (test_utils::is_complex<T>::value) return T(re, im);
    else return re;
}

template <typename T>
inline T host_rand(std::mt19937& gen) {
    std::uniform_real_distribution<RealOf<T>> d(RealOf<T>(-1), RealOf<T>(1));
    if constexpr (test_utils::is_complex<T>::value) return T(d(gen), d(gen));
    else return d(gen);
}

// A dense, host-side, column-major Hermitian positive-definite matrix.
//
// A = (M M^H)/n + shift*I with M's entries uniform in [-1,1]. The shift keeps
// the condition number O(1) so a residual failure is a BUG and not the kappa^2 u
// cliff Cholesky legitimately falls off; the M M^H term is what puts a
// non-trivial imaginary part in every off-diagonal for complex T, which is the
// only way a residual test can see a missing conjugate at all.
template <typename T>
std::vector<T> make_spd(int n, unsigned seed, RealOf<T> shift = RealOf<T>(2)) {
    using R = RealOf<T>;
    std::mt19937 gen(seed);
    std::vector<T> M(static_cast<size_t>(n) * n);
    for (auto& v : M) v = host_rand<T>(gen);

    std::vector<T> A(static_cast<size_t>(n) * n, T{});
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            T acc{};
            for (int k = 0; k < n; ++k) {
                acc += M[i + static_cast<size_t>(k) * n] *
                       host_conj(M[j + static_cast<size_t>(k) * n]);
            }
            A[i + static_cast<size_t>(j) * n] = acc / T(R(n));
        }
    }
    for (int i = 0; i < n; ++i) {
        A[i + static_cast<size_t>(i) * n] =
            make_scalar<T>(host_real(A[i + static_cast<size_t>(i) * n]) + shift, R(0));
    }
    // Force exact Hermitian symmetry and an exactly real diagonal: the kernel is
    // contractually allowed to ignore imag(diag(A)), so the reference must too.
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            A[j + static_cast<size_t>(i) * n] = host_conj(A[i + static_cast<size_t>(j) * n]);
        }
    }
    return A;
}

// A = L0 D L0^H with L0 UNIT lower triangular and D real diagonal.
//
// The point of this generator is that Cholesky's k-th updated Schur diagonal
// equals D_kk EXACTLY (the LDL^H factorisation of a Hermitian matrix is unique),
// so planting a negative D_kk pins the failure column with no reference
// implementation involved. That is what makes InfoIndexIsExact an exact,
// self-referential oracle rather than a cross-implementation comparison -- and
// exactness matters, because the API's own position (options_api_tests.cc:463)
// is that vendors do not agree on WHICH index they name first.
//
// THE ROW NORMALISATION IS THE WHOLE TEST, AND IT WAS ADDED AFTER A MEASUREMENT
// PROVED THE TEST BLIND WITHOUT IT.
//
// With plain small random L0 entries, the ORIGINAL diagonal at the failure
// column is A_cc = D_c + sum_{p<c} |L0(c,p)|^2 = -1 + (something small), i.e.
// still negative -- so a kernel reading the STALE pivot straight out of the tile
// flags the same column and InfoIndexIsExact passes. That was executed, not
// reasoned about: with the stale-pivot defect reintroduced, InfoIndexIsExact and
// InfoReportsTheFirstFailure were among the 24 tests that stayed GREEN. That is
// precisely this repository's recorded failure mode of a guard test that cannot
// fail by construction.
//
// The fix is to scale row c's prefix so that sum_{p<c} |L0(c,p)|^2 == 2, making
// A_cc = -1 + 2 = +1: the ORIGINAL diagonal at the failure column is POSITIVE
// and only the UPDATED Schur diagonal is negative. A stale-pivot reader now sees
// nothing wrong at column c and reports the wrong index (or none), while the
// correct kernel still reports exactly c+1. `negative_cols` must be sorted for
// the reasoning to hold -- every p < c then has D_p == +1.
//
// c == 0 has no prefix to scale and is inherently non-discriminating for this
// defect; it is kept in the sweep because it is a boundary of the panel loop,
// and the caller asserts the positive-diagonal property for c >= 1 so the
// distinction is visible rather than assumed.
template <typename T>
std::vector<T> make_planted_ldl(int n, const std::vector<int>& negative_cols, unsigned seed) {
    using R = RealOf<T>;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<R> d(R(-0.25), R(0.25));

    std::vector<T> L(static_cast<size_t>(n) * n, T{});
    for (int c = 0; c < n; ++c) {
        L[c + static_cast<size_t>(c) * n] = make_scalar<T>(R(1), R(0));
        for (int i = c + 1; i < n; ++i) {
            if constexpr (test_utils::is_complex<T>::value) {
                L[i + static_cast<size_t>(c) * n] = T(d(gen), d(gen));
            } else {
                L[i + static_cast<size_t>(c) * n] = d(gen);
            }
        }
    }
    std::vector<R> D(n, R(1));
    for (int c : negative_cols) {
        if (c >= 0 && c < n) D[c] = R(-1);
    }

    // Row-prefix normalisation: see the note above.
    for (int c : negative_cols) {
        if (c < 1 || c >= n) continue;
        R ss = R(0);
        for (int p = 0; p < c; ++p) {
            const T v = L[c + static_cast<size_t>(p) * n];
            ss += host_real(v) * host_real(v) + host_imag(v) * host_imag(v);
        }
        if (ss <= R(0)) continue;
        const R scale = std::sqrt(R(2) / ss);
        for (int p = 0; p < c; ++p) {
            L[c + static_cast<size_t>(p) * n] =
                make_scalar<T>(host_real(L[c + static_cast<size_t>(p) * n]) * scale,
                               host_imag(L[c + static_cast<size_t>(p) * n]) * scale);
        }
    }

    std::vector<T> A(static_cast<size_t>(n) * n, T{});
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            T acc{};
            for (int k = 0; k <= std::min(i, j); ++k) {
                acc += L[i + static_cast<size_t>(k) * n] * make_scalar<T>(D[k], R(0)) *
                       host_conj(L[j + static_cast<size_t>(k) * n]);
            }
            A[i + static_cast<size_t>(j) * n] = acc;
        }
    }
    for (int i = 0; i < n; ++i) {
        A[i + static_cast<size_t>(i) * n] =
            make_scalar<T>(host_real(A[i + static_cast<size_t>(i) * n]), R(0));
    }
    return A;
}

// ||L L^H - A||_F / ||A||_F, computed here, from the factor the kernel returned
// and the input this file generated. Independent of every other implementation.
template <typename T>
RealOf<T> multiply_back_residual(const std::vector<T>& A, const std::vector<T>& L, int n) {
    using R = RealOf<T>;
    R num = R(0), den = R(0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            T acc{};
            for (int k = 0; k <= std::min(i, j); ++k) {
                acc += L[i + static_cast<size_t>(k) * n] *
                       host_conj(L[j + static_cast<size_t>(k) * n]);
            }
            const T diff = acc - A[i + static_cast<size_t>(j) * n];
            num += host_real(diff) * host_real(diff) + host_imag(diff) * host_imag(diff);
            const T a = A[i + static_cast<size_t>(j) * n];
            den += host_real(a) * host_real(a) + host_imag(a) * host_imag(a);
        }
    }
    if (den == R(0)) return R(0);
    return std::sqrt(num) / std::sqrt(den);
}

// The residual bound. Cholesky's backward error is O(n) * eps * ||A||; the
// constant is slack for the reduction order, which the kernel does not share
// with the host loop above.
//
// IT WAS 40, AND 40 COULD NOT FAIL. The bound was measured by sweeping the
// multiplier over the whole size ladder and both triangles for all four types:
// 40, 8, 4 and 1 all pass; 0.2 turns ResidualBothTriangles red for every type.
// So the kernel's true worst-case relative Frobenius residual is in
// (0.2, 1] * n * eps and the shipped bound carried 40-200x of slack -- enough
// that an accuracy defect would have to be catastrophic to be seen. Concretely,
// swapping sycl::sqrt + R(1)/dkk for rsqrt.approx, which
// potrf_cta_device.hh:205 explicitly forbids, is invisible at 40 and no other
// test in this file looks at accuracy at all.
//
// 4 is the measured worst case rounded up by one binary order, i.e. 4-20x of
// margin. A future BLOCKED driver has a different error constant and should get
// its own bound rather than slackening this one.
template <typename T>
RealOf<T> residual_tol(int n) {
    using R = RealOf<T>;
    return R(4) * R(n) * std::numeric_limits<R>::epsilon();
}

template <typename T, Backend B>
struct PotrfConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

using PotrfTestTypes = typename test_utils::backend_types<PotrfConfig>::type;

template <typename Config>
class PotrfCtaTest : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    using R = RealOf<T>;
    static constexpr Backend BackendType = Config::BackendVal;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx) GTEST_SKIP() << "no queue";
        // The native POTRF route is GPU-only and needs sub-group 32 -- these are
        // supports()' own correctness gates, not a convenience.
        if (this->ctx->device().type != DeviceType::GPU) {
            GTEST_SKIP() << "potrf_cta is a GPU kernel";
        }
        if (!this->ctx->device().supports_sub_group_size(32)) {
            GTEST_SKIP() << "device does not offer sub-group size 32";
        }
    }

    // THE DEVICE'S ceiling, not the reference budget's.
    //
    // This was sycl_potrf::potrf_cta_max_n<T>(), which is hardcoded to
    // kPotrfReferenceSlmBudget = 97,280 (potrf_cta.cc). But supports()
    // (potrf_route.hh) and potrf_cta_dispatch (potrf_cta.cc) both use the
    // RUNTIME budget, LOCAL_MEM_SIZE - 4096. The two coincide only on a box
    // reporting local_mem_size == 101,376, i.e. this one.
    //
    // On any other GPU the binary broke in two ways that look like kernel bugs
    // and are not: with a LARGER local memory, JustPastTheCeilingHasNoCtaRoute's
    // anti-vacuity guard fails because it asks supports() about a frozen cap
    // while supports() answers for the device; with a SMALLER one,
    // ResidualBothTriangles pushes `cap` into its size list and
    // potrf_cta_dispatch throws for an order the device genuinely cannot hold.
    // MeasuredFitCeilings keeps the pin on the fixed 97,280 formula, which is
    // where a budget-independent assertion belongs.
    int ceiling() const {
        const std::size_t local_mem =
            this->ctx->device().get_property(DeviceProperty::LOCAL_MEM_SIZE);
        const std::size_t budget = (local_mem > 4096) ? (local_mem - 4096) : 0;
        return sycl_potrf::potrf_cta_max_n_for_slm<T>(budget);
    }

    // Load the `uplo` triangle of `src` (dense column-major, n x n) into batch
    // item `b`, and POISON the other triangle. The poison is load-bearing: the
    // contract says the other triangle is neither read nor written, and
    // ortho.cc:156-161 depends on the stronger "not read" because it forms only
    // half of its Gram matrix and leaves the rest uninitialised workspace.
    void load_triangle(Matrix<T, MatrixFormat::Dense>& A, int b, int n,
                       const std::vector<T>& src, Uplo uplo, T poison) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                const bool in_tri = (uplo == Uplo::Lower) ? (i >= j) : (i <= j);
                A(i, j, b) = in_tri ? src[i + static_cast<size_t>(j) * n] : poison;
            }
        }
    }

    // Extract the lower-triangular L for batch item b, whichever triangle the
    // factor was written into. For Upper the stored object is U with A = U^H U,
    // and L = U^H -- so the multiply-back oracle is the same one function.
    std::vector<T> extract_L(const Matrix<T, MatrixFormat::Dense>& A, int b, int n, Uplo uplo) {
        std::vector<T> L(static_cast<size_t>(n) * n, T{});
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                L[i + static_cast<size_t>(j) * n] =
                    (uplo == Uplo::Lower) ? A(i, j, b) : host_conj(A(j, i, b));
            }
        }
        return L;
    }

    // Run the CTA kernel directly. Returns info.
    std::vector<int32_t> run_cta(Matrix<T, MatrixFormat::Dense>& A, Uplo uplo,
                                 bool pass_info_span = true) {
        const int batch = A.batch_size();
        UnifiedVector<std::byte> ws(sycl_potrf::potrf_cta_buffer_size<T>(*this->ctx, A.view()));
        UnifiedVector<int32_t> info(batch, int32_t(-7));
        if (pass_info_span) {
            sycl_potrf::potrf_cta_dispatch<T>(*this->ctx, A.view(), uplo, ws.to_span(),
                                              info.to_span());
        } else {
            sycl_potrf::potrf_cta_dispatch<T>(*this->ctx, A.view(), uplo, ws.to_span(),
                                              Span<int32_t>{});
        }
        this->ctx->wait();
        return std::vector<int32_t>(info.begin(), info.end());
    }
};

TYPED_TEST_SUITE(PotrfCtaTest, PotrfTestTypes);

// ---------------------------------------------------------------------------
// T1. Residual, both Uplo, across the whole order range including the ceiling.
//
// n = 2 and n = 3 are MANDATORY and not padding: the stale-pivot defect first
// shows at n = 2, because at n = 1 there is no second column to be wrong.
// ---------------------------------------------------------------------------
TYPED_TEST(PotrfCtaTest, ResidualBothTriangles) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;

    const int cap = this->ceiling();
    ASSERT_GT(cap, 0) << "no CTA capacity for this type -- the kernel is not linked";

    std::vector<int> sizes = {1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65};
    // 108..111 straddle THE 48 KB LAUNCH HOLE. Measured cold on this box: a
    // dynamic local-memory request in (49152 - static_shared, 49152] fails with
    // CUDA_ERROR_INVALID_VALUE -- too big for CUDA's non-opt-in 48 KB limit once
    // the kernel's static shared is added, not big enough for the UR adapter to
    // raise MaxDynamicSharedMemorySize. Unpadded, float n = 110 asks for 49,044 B
    // and lands squarely in it. potrf_cta.cc pads such a request up to 49,920.
    //
    // The order matters and is why these sit AFTER the small sizes and not at
    // the front: the attribute is sticky per CUfunction and one CUfunction serves
    // every n, so any earlier launch above 48 KB masks the hole for the rest of
    // the process. Every size before this point is under 48 KB, so within THIS
    // test the pad is genuinely on trial. Across a whole binary it is not, and no
    // automated test in this suite can make it so.
    for (int n : {108, 109, 110, 111}) sizes.push_back(n);
    sizes.push_back(cap - 1);
    sizes.push_back(cap);                       // exactly at the fit ceiling
    std::sort(sizes.begin(), sizes.end());
    sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());

    for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
        for (int n : sizes) {
            if (n < 1 || n > cap) continue;
            const int batch = (n <= 64) ? 4 : 2;
            Matrix<T, MatrixFormat::Dense> A(n, n, batch);
            std::vector<std::vector<T>> ref(batch);
            for (int b = 0; b < batch; ++b) {
                // EVERY BATCH ITEM IS DIFFERENT. Identical items make a stride
                // bug invisible: the wrong matrix would be the right answer.
                ref[b] = make_spd<T>(n, 1000u + 17u * b + 3u * n);
                this->load_triangle(A, b, n, ref[b], uplo, make_scalar<T>(R(-999), R(777)));
            }
            const auto info = this->run_cta(A, uplo);
            for (int b = 0; b < batch; ++b) {
                ASSERT_EQ(info[b], 0) << "n=" << n << " b=" << b
                                      << " uplo=" << static_cast<int>(uplo);
                const auto L = this->extract_L(A, b, n, uplo);
                const R res = multiply_back_residual<T>(ref[b], L, n);
                EXPECT_LE(res, residual_tol<T>(n))
                    << "n=" << n << " b=" << b << " uplo=" << static_cast<int>(uplo);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// T1b. The ceiling is a HARD capacity: one past it must not launch, and
// supports() must already have said so.
// ---------------------------------------------------------------------------
TYPED_TEST(PotrfCtaTest, JustPastTheCeilingHasNoCtaRoute) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::BackendType;

    const int cap = this->ceiling();
    ASSERT_GT(cap, 0);

    Matrix<T, MatrixFormat::Dense> A(cap + 1, cap + 1, 1);
    A.fill(T{});
    for (int i = 0; i < cap + 1; ++i) A(i, i, 0) = make_scalar<T>(typename TestFixture::R(1),
                                                                    typename TestFixture::R(0));

    const auto shape = backend::potrf_op_shape<B, T>(*this->ctx, A.view(), Uplo::Lower);
    ASSERT_TRUE(shape.has_value());
    // The guard that keeps the next assertion from passing vacuously: at the
    // ceiling itself the CTA arm MUST be supported, or "unsupported one past it"
    // proves nothing.
    auto at_cap = *shape;
    at_cap.m = at_cap.n = at_cap.k = cap;
    EXPECT_TRUE((dispatch::RouteTable<dispatch::Op::potrf, T>::supports(
        dispatch::Route{dispatch::Origin::Native, dispatch::Algorithm::CTA}, at_cap)));
    EXPECT_FALSE((dispatch::RouteTable<dispatch::Op::potrf, T>::supports(
        dispatch::Route{dispatch::Origin::Native, dispatch::Algorithm::CTA}, *shape)));

    // And the direct entry point refuses rather than launching something that
    // cannot fit.
    UnifiedVector<std::byte> ws(sycl_potrf::potrf_cta_buffer_size<T>(*this->ctx, A.view()));
    UnifiedVector<int32_t> info(1, int32_t(0));
    EXPECT_THROW(sycl_potrf::potrf_cta_dispatch<T>(*this->ctx, A.view(), Uplo::Lower,
                                                   ws.to_span(), info.to_span()),
                 std::invalid_argument);
}

// ---------------------------------------------------------------------------
// T2. The other triangle is neither read nor written.
//
// Two passes. The first poisons with a finite sentinel and memcmp's it back
// bit for bit -- that proves NOT WRITTEN. The second poisons with a quiet NaN
// and asserts the produced factor is NaN-free -- that proves NOT READ, which is
// the stronger claim and the one ortho.cc depends on.
// ---------------------------------------------------------------------------
TYPED_TEST(PotrfCtaTest, OtherTriangleIsNeitherReadNorWritten) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;

    const int batch = 3;

    // BOTH PARITIES. n = 37 alone was not enough: lda = n | 1, so an ODD order
    // has no pad row in the SLM tile and an even one does. A store-back that ran
    // to `i < lda` instead of `i < n` would write A(n, c), which in a packed
    // Matrix(n, n, batch) is linear index (c+1)*n -- element (0, c+1), inside
    // the untouched upper triangle. Only an even n can see that.
    for (int n : {36, 37}) {
      if (n > this->ceiling()) continue;
      for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
        // Pass 1: not written.
        {
            Matrix<T, MatrixFormat::Dense> A(n, n, batch);
            const T poison = make_scalar<T>(R(-3.5), R(11.25));
            for (int b = 0; b < batch; ++b) {
                this->load_triangle(A, b, n, make_spd<T>(n, 55u + b), uplo, poison);
            }
            const auto info = this->run_cta(A, uplo);
            for (int b = 0; b < batch; ++b) {
                ASSERT_EQ(info[b], 0);
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        const bool in_tri = (uplo == Uplo::Lower) ? (i >= j) : (i <= j);
                        if (!in_tri) {
                            const T v = A(i, j, b);
                            ASSERT_EQ(host_real(v), host_real(poison))
                                << "wrote outside the " << static_cast<int>(uplo)
                                << " triangle at (" << i << "," << j << ")";
                            ASSERT_EQ(host_imag(v), host_imag(poison));
                        }
                    }
                }
            }
        }
        // Pass 2: not read.
        {
            Matrix<T, MatrixFormat::Dense> A(n, n, batch);
            const R nan = std::numeric_limits<R>::quiet_NaN();
            const T poison = make_scalar<T>(nan, nan);
            for (int b = 0; b < batch; ++b) {
                this->load_triangle(A, b, n, make_spd<T>(n, 55u + b), uplo, poison);
            }
            const auto info = this->run_cta(A, uplo);
            for (int b = 0; b < batch; ++b) {
                ASSERT_EQ(info[b], 0);
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        const bool in_tri = (uplo == Uplo::Lower) ? (i >= j) : (i <= j);
                        if (in_tri) {
                            const T v = A(i, j, b);
                            ASSERT_FALSE(std::isnan(host_real(v)))
                                << "NaN leaked from the untouched triangle into ("
                                << i << "," << j << ")";
                            ASSERT_FALSE(std::isnan(host_imag(v)));
                        }
                    }
                }
            }
        }
    }
      }
}

// ---------------------------------------------------------------------------
// T3. A packed launch (G > 1 matrices per work-group) agrees BIT FOR BIT with
// the same matrices launched one per work-group.
//
// This is the test for the (P1) publish guard `lane < ib`. Without it, lanes
// ib..31 write S(j+ib .. j+31, j+k): into the A21 panel (P2) is about to read on
// every panel where ib < 32, and past the end of the tile on the ragged last
// panel -- which under G > 1 lands in the NEIGHBOURING MATRIX. A batch-1
// residual test cannot see the second half at all, and the comparison is
// bit-exact rather than tolerant because both sides run the same arithmetic in
// the same order; only the SLM neighbourhood differs.
//
// n is chosen small enough that the launch parameters actually pick G > 1, AND
// THE TEST NOW ASKS. It used to say this in a comment and assert it nowhere:
// which n pack is a consequence of kPotrfSlmSoftTarget, the clamp on G and
// sizeof(T), none of which the test can see, so any change to those three would
// have collapsed it to one matrix per work-group and left the neighbouring-
// matrix half of the defect invisible while the test stayed green. That is this
// repository's recorded blind-guard shape. potrf_cta_debug_launch exists for
// this one assertion.
//
// The n values that pack are TYPE-DEPENDENT, so a value that does not pack for
// some type is SKIPPED rather than silently tested at G == 1 -- and the test
// fails outright if no n packed for this type at all, which is the case a bare
// `continue` would hide.
// ---------------------------------------------------------------------------
TYPED_TEST(PotrfCtaTest, PackedBatchMatchesSolo) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;

    int packed_ns = 0;
    for (int n : {9, 15, 17, 31}) {
        if (n > this->ceiling()) continue;
        const int batch = 8;

        const unsigned geom = sycl_potrf::potrf_cta_debug_launch<T>(*this->ctx, n, batch);
        ASSERT_NE(geom, 0u) << "n=" << n << " does not fit, which the ceiling check missed";
        const int G = static_cast<int>(geom & 0xffffu);
        if (G <= 1) continue;   // this type does not pack at this n
        ++packed_ns;

        Matrix<T, MatrixFormat::Dense> packed(n, n, batch);
        std::vector<std::vector<T>> ref(batch);
        for (int b = 0; b < batch; ++b) {
            // Distinct per item, and distinctly SCALED, so a cross-matrix write
            // changes a value rather than swapping in an identical one.
            ref[b] = make_spd<T>(n, 2000u + 31u * b, R(1) + R(b));
            this->load_triangle(packed, b, n, ref[b], Uplo::Lower,
                                make_scalar<T>(R(0), R(0)));
        }
        const auto info_packed = this->run_cta(packed, Uplo::Lower);

        for (int b = 0; b < batch; ++b) {
            ASSERT_EQ(info_packed[b], 0) << "n=" << n << " b=" << b;
            Matrix<T, MatrixFormat::Dense> solo(n, n, 1);
            this->load_triangle(solo, 0, n, ref[b], Uplo::Lower, make_scalar<T>(R(0), R(0)));
            const auto info_solo = this->run_cta(solo, Uplo::Lower);
            ASSERT_EQ(info_solo[0], 0);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j <= i; ++j) {
                    ASSERT_EQ(host_real(packed(i, j, b)), host_real(solo(i, j, 0)))
                        << "packed vs solo differ at n=" << n << " b=" << b
                        << " (" << i << "," << j << ")";
                    ASSERT_EQ(host_imag(packed(i, j, b)), host_imag(solo(i, j, 0)));
                }
            }
        }
    }
    // The whole point of this test is the G > 1 launch. If no n reached one,
    // every assertion above was vacuous and the test must say so rather than
    // report a green it did not earn.
    ASSERT_GT(packed_ns, 0)
        << "no n in the sweep packed more than one matrix per work-group for this type; "
           "this test proved nothing";
}

// ---------------------------------------------------------------------------
// T5. `info` names the EXACT 1-based global column at which the updated Schur
// diagonal was not > 0.
//
// The oracle is the planted L0 D L0^H of make_planted_ldl: Cholesky's k-th pivot
// IS D_kk, so the failure column is known exactly with no reference solve. The
// sweep straddles panel boundaries for both NB ladders in the tree (8 for
// complex<double>, 16 for the rest), which is where the stale-pivot defect and
// any off-by-one in the panel loop live.
// ---------------------------------------------------------------------------
TYPED_TEST(PotrfCtaTest, InfoIndexIsExact) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;

    const int n = std::min(69, this->ceiling());
    ASSERT_GE(n, 34) << "the sweep needs room for several panels";

    // BOTH TRIANGLES. The kernel's own comment calls Upper "a LOAD/STORE
    // TRANSFORM and not a second algorithm", and the route table declines to add
    // an uplo gate to the CTA arm on the strength of that claim -- but every
    // failure-path test was Lower-only, so nothing falsified it for `info`.
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
    for (int c : {0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, n - 1}) {
        if (c < 0 || c >= n) continue;
        Matrix<T, MatrixFormat::Dense> A(n, n, 1);
        const auto ref = make_planted_ldl<T>(n, {c}, 4242u + static_cast<unsigned>(c));
        // THE TEST ASSERTS ITS OWN SENSITIVITY. For c >= 1 the ORIGINAL diagonal
        // at the failure column is positive by construction, so only a kernel
        // that tests the UPDATED Schur diagonal can name this column. Without
        // this line the case would still pass against the stale-pivot defect,
        // which is measured, not hypothetical.
        if (c >= 1) {
            ASSERT_GT(host_real(ref[c + static_cast<size_t>(c) * n]), R(0))
                << "the planted matrix is not discriminating at column " << c;
        }
        this->load_triangle(A, 0, n, ref, uplo, make_scalar<T>(R(0), R(0)));
        const auto info = this->run_cta(A, uplo);
        EXPECT_EQ(info[0], c + 1)
            << "planted a non-positive pivot at global column " << c << " of " << n
            << " uplo=" << static_cast<int>(uplo);
    }
    }
}

// FIRST FAILURE WINS -- the sticky rule in the contract. Two planted failures,
// and info must name the earlier.
TYPED_TEST(PotrfCtaTest, InfoReportsTheFirstFailure) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;

    const int n = std::min(69, this->ceiling());
    for (int c : {3, 17, 20}) {
        const int c2 = c + 11;
        if (c2 >= n) continue;
        Matrix<T, MatrixFormat::Dense> A(n, n, 1);
        const auto ref = make_planted_ldl<T>(n, {c, c2}, 777u + static_cast<unsigned>(c));
        this->load_triangle(A, 0, n, ref, Uplo::Lower, make_scalar<T>(R(0), R(0)));
        const auto info = this->run_cta(A, Uplo::Lower);
        EXPECT_EQ(info[0], c + 1) << "failures planted at " << c << " and " << c2;
    }
}

// ---------------------------------------------------------------------------
// T4/T9. `info` at batch scale, and what a FAILED item's A looks like.
//
// The failed items are at different columns so a single shared flag would be
// visible; the surviving items must both report 0 AND agree bit for bit with the
// same matrix factored alone, which is what catches a failure flag leaking
// across the G matrices packed into one work-group.
//
// The finiteness assertion is not a contract claim -- a failed item's A is
// undefined, exactly as in LAPACK -- but it IS a property of this kernel worth
// pinning: the `!(akk > 0)` test precedes both the sqrt and the reciprocal, so a
// non-PD item executes neither and its tile stays bounded by its input.
// ---------------------------------------------------------------------------
TYPED_TEST(PotrfCtaTest, InfoAtBatchScaleAndFailedItemsStayFinite) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;

    const int n = std::min(33, this->ceiling());
    const int batch = 64;
    const std::vector<int> bad_items = {0, 37, batch - 1};
    const std::vector<int> bad_cols = {0, 12, n - 1};

    Matrix<T, MatrixFormat::Dense> A(n, n, batch);
    std::vector<std::vector<T>> ref(batch);
    for (int b = 0; b < batch; ++b) {
        auto it = std::find(bad_items.begin(), bad_items.end(), b);
        if (it != bad_items.end()) {
            const int c = bad_cols[static_cast<size_t>(it - bad_items.begin())];
            ref[b] = make_planted_ldl<T>(n, {c}, 90u + static_cast<unsigned>(b));
        } else {
            ref[b] = make_spd<T>(n, 300u + 5u * b);
        }
        this->load_triangle(A, b, n, ref[b], Uplo::Lower, make_scalar<T>(R(0), R(0)));
    }
    const auto info = this->run_cta(A, Uplo::Lower);

    for (size_t k = 0; k < bad_items.size(); ++k) {
        EXPECT_EQ(info[bad_items[k]], bad_cols[k] + 1) << "item " << bad_items[k];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                const T v = A(i, j, bad_items[k]);
                ASSERT_TRUE(std::isfinite(host_real(v)))
                    << "failed item " << bad_items[k] << " went non-finite at ("
                    << i << "," << j << ")";
                ASSERT_TRUE(std::isfinite(host_imag(v)));
            }
        }
    }
    for (int b = 0; b < batch; ++b) {
        if (std::find(bad_items.begin(), bad_items.end(), b) != bad_items.end()) continue;
        ASSERT_EQ(info[b], 0) << "healthy item " << b << " reported a failure";
        const auto L = this->extract_L(A, b, n, Uplo::Lower);
        EXPECT_LE((multiply_back_residual<T>(ref[b], L, n)), residual_tol<T>(n))
            << "healthy item " << b << " next to a failed one";
    }
}

// ---------------------------------------------------------------------------
// T7. The complex tests, which a real symmetric matrix cannot perform.
//
//  (a) the input has a genuinely non-trivial imaginary part -- asserted, so the
//      test cannot be blind by construction the way a previous conjugation test
//      in this tree was;
//  (b) imag(diag(L)) is EXACTLY zero, which is the (P3) forced-real-diagonal
//      line; and
//  (c) conjugating the input changes the factor, which is what proves the
//      residual check above is actually sensitive to the conjugation at all.
// ---------------------------------------------------------------------------
TYPED_TEST(PotrfCtaTest, ComplexDiagonalIsExactlyReal) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;
    if constexpr (!test_utils::is_complex<T>::value) {
        GTEST_SKIP() << "real scalar: no imaginary part to check";
    } else {
        const int n = std::min(41, this->ceiling());
        const auto ref = make_spd<T>(n, 31337u);

        // (a) the input must actually be complex, or (b) and (c) prove nothing.
        R max_imag = R(0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                max_imag = std::max(max_imag, std::abs(host_imag(ref[i + static_cast<size_t>(j) * n])));
            }
        }
        ASSERT_GT(max_imag, R(0.01)) << "the generated matrix is effectively real";

        Matrix<T, MatrixFormat::Dense> A(n, n, 1);
        this->load_triangle(A, 0, n, ref, Uplo::Lower, make_scalar<T>(R(0), R(0)));
        ASSERT_EQ(this->run_cta(A, Uplo::Lower)[0], 0);
        for (int i = 0; i < n; ++i) {
            // (b) EXACTLY zero, not near zero.
            ASSERT_EQ(host_imag(A(i, i, 0)), R(0)) << "imag(L(" << i << "," << i << ")) != 0";
        }
        const auto L = this->extract_L(A, 0, n, Uplo::Lower);
        EXPECT_LE((multiply_back_residual<T>(ref, L, n)), residual_tol<T>(n));

        // (c) conj(A) is a different Hermitian matrix, so it must give a
        //     different factor. If the kernel dropped a conjugate somewhere,
        //     these two could agree.
        std::vector<T> refc(ref);
        for (auto& v : refc) v = host_conj(v);
        Matrix<T, MatrixFormat::Dense> Ac(n, n, 1);
        this->load_triangle(Ac, 0, n, refc, Uplo::Lower, make_scalar<T>(R(0), R(0)));
        ASSERT_EQ(this->run_cta(Ac, Uplo::Lower)[0], 0);
        bool differs = false;
        for (int i = 1; i < n && !differs; ++i) {
            for (int j = 0; j < i && !differs; ++j) {
                if (host_imag(A(i, j, 0)) != host_imag(Ac(i, j, 0))) differs = true;
            }
        }
        EXPECT_TRUE(differs) << "conjugating the input did not change the factor";
        const auto Lc = this->extract_L(Ac, 0, n, Uplo::Lower);
        EXPECT_LE((multiply_back_residual<T>(refc, Lc, n)), residual_tol<T>(n));

        // (d) imag(diag(A)) IS IGNORED -- LAPACK's and cuSOLVER's documented
        //     contract ("imaginary parts of the diagonal need not be set and are
        //     assumed zero"), and the test for the load transform's
        //     `if (i == c) v = from_real(real(v))`.
        //
        //     This part exists because of a measurement: removing the SAME
        //     forcing from (P3)'s trailing update turned no test red at all, and
        //     that is structurally correct -- (P1) publishes an exactly real
        //     diagonal, so the output cannot carry a residue. The LOAD is
        //     different: garbage in imag(diag(A)) is caller-supplied and
        //     unbounded, and it enters the very first pivot.
        Matrix<T, MatrixFormat::Dense> Ap(n, n, 1);
        this->load_triangle(Ap, 0, n, ref, Uplo::Lower, make_scalar<T>(R(0), R(0)));
        for (int i = 0; i < n; ++i) {
            Ap(i, i, 0) = make_scalar<T>(host_real(Ap(i, i, 0)), R(0.75) * R(i + 1));
        }
        ASSERT_EQ(this->run_cta(Ap, Uplo::Lower)[0], 0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                ASSERT_EQ(host_real(Ap(i, j, 0)), host_real(A(i, j, 0)))
                    << "imag(diag(A)) was not ignored, at (" << i << "," << j << ")";
                ASSERT_EQ(host_imag(Ap(i, j, 0)), host_imag(A(i, j, 0)));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// T8. An empty `info` span means "not requested" and must not change the answer.
//
// The failure this guards against is the pool-scratch one: info_target's
// fallback returns UNINITIALISED memory, and a driver that reads its own info
// without zeroing it first takes the "already failed" path for every item and
// returns A UNMODIFIED with no error at all. The Phase-1 kernel writes the flag
// from local memory and never reads global info, so this passes by construction
// today -- it is here because Phase 2's blocked driver is where the trap is, and
// the test must exist before the code it guards.
// ---------------------------------------------------------------------------
TYPED_TEST(PotrfCtaTest, EmptyInfoSpanStillFactorises) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;

    const int n = std::min(29, this->ceiling());
    const int batch = 4;
    Matrix<T, MatrixFormat::Dense> A(n, n, batch);
    std::vector<std::vector<T>> ref(batch);
    for (int b = 0; b < batch; ++b) {
        ref[b] = make_spd<T>(n, 606u + b);
        this->load_triangle(A, b, n, ref[b], Uplo::Lower, make_scalar<T>(R(0), R(0)));
    }
    this->run_cta(A, Uplo::Lower, /*pass_info_span=*/false);
    for (int b = 0; b < batch; ++b) {
        const auto L = this->extract_L(A, b, n, Uplo::Lower);
        EXPECT_LE((multiply_back_residual<T>(ref[b], L, n)), residual_tol<T>(n)) << "b=" << b;
    }
}

// ---------------------------------------------------------------------------
// T6. The facade actually reaches the CTA kernel.
//
// The ONLY routing test in this file, and the only one whose subject is routing.
//
// IT USED TO ASSERT ON A RE-RESOLUTION OF THE ROUTE TABLE, AND THAT WAS NOT A
// GUARD -- MEASURED, not argued. The table answering {Native, CTA} says what the
// table would decide; it says nothing about what potrf<B,T> executed. The
// adversarial review removed the facade's CTA arm outright, so that
// `if (is_native(route))` fell straight through to backend::potrf_vendor -- the
// exact linked-but-never-reached defect route_compiled.hh:1-24 names -- rebuilt,
// and this test stayed GREEN across all four scalar types while every number in
// it came from cuSOLVER. That is this repository's FIFTH recorded blind guard,
// and it was found by executing the break rather than by reading the test.
//
// The guard is now a BIT-EXACT comparison against the direct entry point, which
// no vendor can satisfy. Re-running the same break with the comparison in place
// turns it red for all four types ("the facade did not run the CTA kernel").
// ---------------------------------------------------------------------------
TYPED_TEST(PotrfCtaTest, FacadeReachesTheCtaKernel) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;
    static constexpr Backend B = TestFixture::BackendType;

    const int n = std::min(48, this->ceiling());
    const int batch = 3;

    struct EnvGuard {
        std::string saved;
        bool had = false;
        EnvGuard() {
            if (const char* v = std::getenv("BATCHLAS_POTRF_ROUTE")) { saved = v; had = true; }
            ::setenv("BATCHLAS_POTRF_ROUTE", "cta", 1);
        }
        ~EnvGuard() {
            if (had) ::setenv("BATCHLAS_POTRF_ROUTE", saved.c_str(), 1);
            else ::unsetenv("BATCHLAS_POTRF_ROUTE");
        }
    } guard;

    Matrix<T, MatrixFormat::Dense> A(n, n, batch);
    std::vector<std::vector<T>> ref(batch);
    for (int b = 0; b < batch; ++b) {
        ref[b] = make_spd<T>(n, 8080u + b);
        this->load_triangle(A, b, n, ref[b], Uplo::Lower, make_scalar<T>(R(0), R(0)));
    }

    // The route assertion LOCALISES a failure -- it says which of the two links
    // in the chain broke -- but it is NOT the guard. See the header comment on
    // this test for why.
    const auto route = backend::potrf_route<B, T>(*this->ctx, A.view(), Uplo::Lower,
                                                  /*vendor_available=*/true);
    ASSERT_TRUE(dispatch::is_native(route))
        << "BATCHLAS_POTRF_ROUTE=cta did not resolve to a native route";
    ASSERT_EQ(route.algo, dispatch::Algorithm::CTA);

    UnifiedVector<std::byte> ws(potrf_buffer_size<B, T>(*this->ctx, A.view(), Uplo::Lower));
    UnifiedVector<int32_t> info(batch, int32_t(-7));
    potrf<B, T>(*this->ctx, A.view(), Uplo::Lower, ws.to_span(), info.to_span());
    this->ctx->wait();

    // THE GUARD: the same input, through the DIRECT entry point, must agree with
    // what the facade produced BIT FOR BIT.
    //
    // Bit-exactness is what makes this an observation of EXECUTION rather than
    // of the route table. cuSOLVER does not reproduce this kernel's reduction
    // order, so if the facade ran the vendor these two disagree in the low bits
    // even though both are correct factorisations and both pass any residual
    // bound. A residual check here would be satisfied by either.
    Matrix<T, MatrixFormat::Dense> direct(n, n, batch);
    for (int b = 0; b < batch; ++b) {
        this->load_triangle(direct, b, n, ref[b], Uplo::Lower, make_scalar<T>(R(0), R(0)));
    }
    const auto info_direct = this->run_cta(direct, Uplo::Lower);

    for (int b = 0; b < batch; ++b) {
        ASSERT_EQ(info[b], 0);
        ASSERT_EQ(info_direct[b], 0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                ASSERT_EQ(host_real(A(i, j, b)), host_real(direct(i, j, b)))
                    << "the facade did not run the CTA kernel: its answer differs from "
                       "potrf_cta_dispatch's at (" << i << "," << j << ") b=" << b;
                ASSERT_EQ(host_imag(A(i, j, b)), host_imag(direct(i, j, b)));
            }
        }
        const auto L = this->extract_L(A, b, n, Uplo::Lower);
        EXPECT_LE((multiply_back_residual<T>(ref[b], L, n)), residual_tol<T>(n)) << "b=" << b;
    }
}

// ---------------------------------------------------------------------------
// T10. A PADDED LEADING DIMENSION AND A STRIDE THAT IS NOT ld * cols.
//
// Every other test in this file builds Matrix<T>(n, n, batch), for which
// ld == rows == n and stride == ld * cols exactly. The kernel reads A.ld() and
// A.stride() (potrf_cta.cc), and no test could tell either apart from its
// default -- so the two most consequential lines in the launcher were
// structurally unfalsifiable.
//
// This is not a hypothetical failure class in this tree. trsm_native.cc:590-599
// records it happening: the 6-arg MatrixView constructor DEFAULTS stride to
// ld*cols when 0 is passed, after which every batch item but the first reads the
// wrong matrix. There is a standing memory entry for its GEMM twin ("Native GEMM
// collapses on strided ld"), and panel updates in this library routinely pass
// strided sub-views.
//
// MEASURED: with `stride_a = ldg * n` substituted in the launcher, and again
// with `ldg = A.rows()`, the entire shipped suite stayed GREEN. This test goes
// red for all four scalar types under either.
//
// The surrounding buffer is filled with a NON-PD poison, so reading outside the
// intended window is a wrong answer rather than merely a different one.
// ---------------------------------------------------------------------------
TYPED_TEST(PotrfCtaTest, PaddedLeadingDimensionAndNonDefaultStride) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;

    const int n = std::min(33, this->ceiling());
    const int batch = 5;
    const int ld = n + 7;                       // != rows
    const int stride = ld * n + 13;             // != ld * cols

    for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
        UnifiedVector<T> buf(static_cast<size_t>(stride) * batch,
                             make_scalar<T>(R(-11), R(5)));   // non-PD poison
        MatrixView<T, MatrixFormat::Dense> V(buf.data(), n, n, ld, stride, batch);

        std::vector<std::vector<T>> ref(batch);
        for (int b = 0; b < batch; ++b) {
            ref[b] = make_spd<T>(n, 4711u + 13u * b);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    const bool in_tri = (uplo == Uplo::Lower) ? (i >= j) : (i <= j);
                    buf[static_cast<size_t>(b) * stride + i + static_cast<size_t>(j) * ld] =
                        in_tri ? ref[b][i + static_cast<size_t>(j) * n]
                               : make_scalar<T>(R(0), R(0));
                }
            }
        }

        UnifiedVector<std::byte> ws(sycl_potrf::potrf_cta_buffer_size<T>(*this->ctx, V));
        UnifiedVector<int32_t> info(batch, int32_t(-7));
        sycl_potrf::potrf_cta_dispatch<T>(*this->ctx, V, uplo, ws.to_span(), info.to_span());
        this->ctx->wait();

        for (int b = 0; b < batch; ++b) {
            ASSERT_EQ(info[b], 0) << "b=" << b << " uplo=" << static_cast<int>(uplo);
            std::vector<T> L(static_cast<size_t>(n) * n, T{});
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j <= i; ++j) {
                    const size_t base = static_cast<size_t>(b) * stride;
                    L[i + static_cast<size_t>(j) * n] =
                        (uplo == Uplo::Lower)
                            ? buf[base + i + static_cast<size_t>(j) * ld]
                            : host_conj(buf[base + j + static_cast<size_t>(i) * ld]);
                }
            }
            EXPECT_LE((multiply_back_residual<T>(ref[b], L, n)), residual_tol<T>(n))
                << "b=" << b << " uplo=" << static_cast<int>(uplo)
                << " (ld=" << ld << " stride=" << stride << ")";
        }
    }
}

// ---------------------------------------------------------------------------
// T11. The direct entry point's correctness gates throw rather than launch.
//
// potrf_cta_dispatch re-applies every gate RouteTable<Op::potrf,T>::supports()
// applies, because it is reachable WITHOUT the table -- every numerical test in
// this file calls it that way. Only the sixth of those gates (does not fit) had
// a test; the other five were unfalsifiable.
//
// The heterogeneous one is the one that matters. Deleting it does not produce an
// error: it produces a SILENT WRONG ANSWER, because one launch covers the batch
// with a single (order, ld, stride) tuple and reads with the CAPACITY extents,
// so every item after the first is factorised at the wrong order IN PLACE. And
// netlib's batched path honours the per-item extents
// (netlib_lapack.cc:1029), so routing such a view natively disagrees with a path
// in this tree that gets it right -- not with a hypothesis.
//
// The route-table twins of these gates ARE tested, in
// RoutePotrf.CorrectnessGatesAreNotSpeedGates -- but that is the pure table, and
// these tests deliberately bypass it.
// ---------------------------------------------------------------------------
TYPED_TEST(PotrfCtaTest, DirectEntryPointRefusesWhatSupportsRefuses) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;

    UnifiedVector<int32_t> info(8, int32_t(0));

    // (a) not square.
    {
        Matrix<T, MatrixFormat::Dense> A(8, 5, 1);
        A.fill(make_scalar<T>(R(1), R(0)));
        UnifiedVector<std::byte> ws(64);
        EXPECT_THROW(sycl_potrf::potrf_cta_dispatch<T>(*this->ctx, A.view(), Uplo::Lower,
                                                       ws.to_span(), info.to_span()),
                     std::invalid_argument);
    }

    // (b) heterogeneous batch -- the silent-wrong-answer one.
    {
        const int n = std::min(16, this->ceiling());
        Matrix<T, MatrixFormat::Dense> A(n, n, 4);
        A.fill(make_scalar<T>(R(0), R(0)));
        for (int b = 0; b < 4; ++b)
            for (int i = 0; i < n; ++i) A(i, i, b) = make_scalar<T>(R(2), R(0));
        UnifiedVector<int> act_r(4), act_c(4);
        for (int b = 0; b < 4; ++b) { act_r[b] = n - b; act_c[b] = n - b; }
        auto V = A.view().with_active_dims(act_r.to_span(), act_c.to_span());
        ASSERT_TRUE(V.is_heterogeneous())
            << "the view is not actually heterogeneous; this case would prove nothing";
        UnifiedVector<std::byte> ws(sycl_potrf::potrf_cta_buffer_size<T>(*this->ctx, A.view()));
        EXPECT_THROW(sycl_potrf::potrf_cta_dispatch<T>(*this->ctx, V, Uplo::Lower,
                                                       ws.to_span(), info.to_span()),
                     std::invalid_argument);
    }
}

// ---------------------------------------------------------------------------
// The measured fit ceilings, pinned.
//
// These four numbers are what step 0.2 measured on this box at a 97,280 B budget
// (runtime local_mem_size 101,376 minus the standard 4,096 B reserve), and they
// are NOT WP4_POTRF_SPEC.md:273's {105, 74, 74, 52}: that set follows from a
// 45,056 B budget which is refuted -- device_limits.hh's 49152 is hardcoded by
// cmake/BatchLASDetectSYCL.cmake:44-45 for any nvidia_gpu_sm_* pattern and is
// wrong here by 2.06x. Shipping the small numbers leaves float n in 106..155
// with no route at all in a vendor-free build.
//
// The assertion is against the BUDGET-parameterised query, not the device, so it
// is a pin on the SLM formula and holds on any machine.
// ---------------------------------------------------------------------------
TYPED_TEST(PotrfCtaTest, MeasuredFitCeilings) {
    using T = typename TestFixture::T;
    const int expect = std::is_same_v<T, float>                ? 155
                     : std::is_same_v<T, double>               ? 109
                     : std::is_same_v<T, std::complex<float>>  ? 109
                                                               : 77;
    EXPECT_EQ(sycl_potrf::potrf_cta_max_n_for_slm<T>(97280), expect);
    // And the ceiling really is a ceiling of the formula: one more does not fit.
    EXPECT_LT(sycl_potrf::potrf_cta_max_n_for_slm<T>(97280),
              sycl_potrf::potrf_cta_max_n_for_slm<T>(101376));
}

}  // namespace
