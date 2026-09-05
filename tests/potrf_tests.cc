// Native batched POTRF: the CTA leaf kernel and the blocked driver above it.
//
// Tests call sycl_potrf::potrf_{cta,blocked}_dispatch DIRECTLY and check a host
// multiply-back residual computed here: a vendor reference is inert in a vendor-free
// build, and a forced route that supports() rejects silently becomes the vendor.
// evidence: docs/perf/potrf.md#correctness-findings
#include <gtest/gtest.h>

#include <batchlas/blas/linalg.hh>
#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/potrf.hh>
#include <batchlas/blas/functions/trsm.hh>
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
#include <cstring>
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

// Column-major Hermitian PD matrix A = (M M^H)/n + shift*I. The shift pins the
// condition number at O(1), so a residual failure is a bug and not conditioning.
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

// A = L0 D L0^H with L0 UNIT lower triangular and D real: the k-th updated Schur
// diagonal equals D_kk exactly, so a planted negative D_kk pins the failure column.
// The row-prefix normalisation below looks gratuitous and is the point: it makes the
// ORIGINAL diagonal at the failure column positive, so a kernel reading the STALE pivot
// cannot name it. `negative_cols` must be sorted; c == 0 never discriminates.
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

// ||L L^H - A||_F / ||A||_F -- the oracle, independent of every other implementation.
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

// The leaf's residual bound; the constant is slack for the kernel's reduction order.
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
        // GPU-only and sub-group 32 are supports()' own correctness gates.
        if (this->ctx->device().type != DeviceType::GPU) {
            GTEST_SKIP() << "potrf_cta is a GPU kernel";
        }
        if (!this->ctx->device().supports_sub_group_size(32)) {
            GTEST_SKIP() << "device does not offer sub-group size 32";
        }
    }

    // THE DEVICE'S ceiling, not the reference budget's: supports() and
    // potrf_cta_dispatch both gate on the RUNTIME budget LOCAL_MEM_SIZE - 4096.
    // evidence: docs/perf/potrf.md#the-slm-budget-and-the-fit-ceilings
    int ceiling() const {
        const std::size_t local_mem =
            this->ctx->device().get_property(DeviceProperty::LOCAL_MEM_SIZE);
        const std::size_t budget = (local_mem > 4096) ? (local_mem - 4096) : 0;
        return sycl_potrf::potrf_cta_max_n_for_slm<T>(budget);
    }

    // Load the `uplo` triangle of `src` into item `b` and POISON the other: the contract
    // says the other triangle is neither read nor written, and ortho.cc relies on it.
    void load_triangle(Matrix<T, MatrixFormat::Dense>& A, int b, int n,
                       const std::vector<T>& src, Uplo uplo, T poison) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                const bool in_tri = (uplo == Uplo::Lower) ? (i >= j) : (i <= j);
                A(i, j, b) = in_tri ? src[i + static_cast<size_t>(j) * n] : poison;
            }
        }
    }

    // Extract L for item b: for Upper the stored object is U with A = U^H U, so L = U^H.
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

// n = 2 and 3 are mandatory: the stale-pivot defect first shows at n = 2.
TYPED_TEST(PotrfCtaTest, ResidualBothTriangles) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;

    const int cap = this->ceiling();
    ASSERT_GT(cap, 0) << "no CTA capacity for this type -- the kernel is not linked";

    std::vector<int> sizes = {1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 47, 63, 64, 65};
    // 108..111 straddle the 48 KB launch hole. They MUST stay after the smaller sizes:
    // the attribute is sticky per CUfunction, so any earlier launch above 48 KB masks it.
    // evidence: docs/perf/potrf.md#the-48-kb-launch-hole
    for (int n : {108, 109, 110, 111}) sizes.push_back(n);
    sizes.push_back(cap - 1);
    sizes.push_back(cap);
    std::sort(sizes.begin(), sizes.end());
    sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());

    for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
        for (int n : sizes) {
            if (n < 1 || n > cap) continue;
            const int batch = (n <= 64) ? 4 : 2;
            Matrix<T, MatrixFormat::Dense> A(n, n, batch);
            std::vector<std::vector<T>> ref(batch);
            for (int b = 0; b < batch; ++b) {
                // Every batch item differs: identical items hide a stride bug.
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

// The ceiling is a HARD capacity: one past it must not launch, and supports() agrees.
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
    // Anti-vacuity: the CTA arm must be supported AT the ceiling, or this proves nothing.
    auto at_cap = *shape;
    at_cap.m = at_cap.n = at_cap.k = cap;
    EXPECT_TRUE((dispatch::RouteTable<dispatch::Op::potrf, T>::supports(
        dispatch::Route{dispatch::Origin::Native, dispatch::Algorithm::CTA}, at_cap)));
    EXPECT_FALSE((dispatch::RouteTable<dispatch::Op::potrf, T>::supports(
        dispatch::Route{dispatch::Origin::Native, dispatch::Algorithm::CTA}, *shape)));

    UnifiedVector<std::byte> ws(sycl_potrf::potrf_cta_buffer_size<T>(*this->ctx, A.view()));
    UnifiedVector<int32_t> info(1, int32_t(0));
    EXPECT_THROW(sycl_potrf::potrf_cta_dispatch<T>(*this->ctx, A.view(), Uplo::Lower,
                                                   ws.to_span(), info.to_span()),
                 std::invalid_argument);
}

// The other triangle is neither read nor written. A finite sentinel memcmp'd back
// proves NOT WRITTEN; a quiet NaN with the factor asserted NaN-free proves NOT READ.
TYPED_TEST(PotrfCtaTest, OtherTriangleIsNeitherReadNorWritten) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;

    const int batch = 3;

    // BOTH PARITIES: lda = n | 1, so only an even order has a pad row in the SLM tile,
    // where a store-back running to `i < lda` writes into the untouched triangle.
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

// A packed launch (G > 1 matrices per work-group) must agree BIT FOR BIT with the same
// matrices run solo: the guard on the `lane < ib` publish predicate, without which the
// ragged last panel writes into the NEIGHBOURING matrix. Which n packs is type-dependent.
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
            // Distinct AND distinctly scaled, so a cross-matrix write changes a value.
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
    // Anti-vacuity: without a G > 1 launch every assertion above was vacuous.
    ASSERT_GT(packed_ns, 0)
        << "no n in the sweep packed more than one matrix per work-group for this type; "
           "this test proved nothing";
}

// `info` is the EXACT 1-based global column whose updated Schur diagonal was not > 0.
TYPED_TEST(PotrfCtaTest, InfoIndexIsExact) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;

    const int n = std::min(69, this->ceiling());
    ASSERT_GE(n, 34) << "the sweep needs room for several panels";

    // BOTH TRIANGLES: the route table declines an uplo gate on the CTA arm.
    for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
    for (int c : {0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, n - 1}) {
        if (c < 0 || c >= n) continue;
        Matrix<T, MatrixFormat::Dense> A(n, n, 1);
        const auto ref = make_planted_ldl<T>(n, {c}, 4242u + static_cast<unsigned>(c));
        // THE TEST ASSERTS ITS OWN SENSITIVITY: for c >= 1 the ORIGINAL diagonal at the
        // failure column is positive, so only a kernel testing the UPDATED pivot names it.
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

// FIRST FAILURE WINS -- the sticky rule in the contract.
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

// `info` at batch scale, failures at different columns so a shared flag is visible.
// A failed item's A is undefined in LAPACK, but this kernel keeps it finite.
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

// The complex-only checks; each is asserted non-vacuous at its own site below.
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

        // (c) conj(A) is a different Hermitian matrix, so it must give a different factor.
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

        // (d) imag(diag(A)) IS IGNORED, per LAPACK's contract; the load transform drops it.
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

// An empty `info` span means "not requested" and must not change the answer: the
// fallback is UNINITIALISED pool scratch, which a driver reading its own info without
// zeroing takes as "already failed".
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

// The facade actually reaches the CTA kernel. The table answering {Native, CTA} says
// nothing about what ran, so the guard is a BIT-EXACT comparison against the direct call.
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

    // The route assertion LOCALISES a failure -- which link broke -- but is NOT the guard.
    const auto route = backend::potrf_route<B, T>(*this->ctx, A.view(), Uplo::Lower,
                                                  /*vendor_available=*/true);
    ASSERT_TRUE(dispatch::is_native(route))
        << "BATCHLAS_POTRF_ROUTE=cta did not resolve to a native route";
    ASSERT_EQ(route.algo, dispatch::Algorithm::CTA);

    UnifiedVector<std::byte> ws(potrf_buffer_size<B, T>(*this->ctx, A.view(), Uplo::Lower));
    UnifiedVector<int32_t> info(batch, int32_t(-7));
    potrf<B, T>(*this->ctx, A.view(), Uplo::Lower, ws.to_span(), info.to_span());
    this->ctx->wait();

    // THE GUARD: bit-exactness observes EXECUTION; a residual check would accept either.
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

// A PADDED LEADING DIMENSION AND A STRIDE THAT IS NOT ld * cols. Every other test uses
// Matrix<T>(n, n, batch), where the launcher's ld and stride reads are unfalsifiable.
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

// The direct entry point re-applies every gate supports() applies, because it is
// reachable WITHOUT the table. Dropping the heterogeneous gate is a SILENT WRONG ANSWER.
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

// The fit ceilings, pinned against the BUDGET-parameterised query, not against a device.
// evidence: docs/perf/potrf.md#the-slm-budget-and-the-fit-ceilings
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

// ===========================================================================
// THE BLOCKED DRIVER -- the right-looking driver above potrf_cta_max_n<T>().
// ===========================================================================
//
// THE RESIDUAL CANNOT SEE THE FOLD: delete the trailing update's triangular fold and the
// lower triangle still comes out BIT-IDENTICAL, with only the caller's upper triangle
// scribbled on. nb and W are ASKED of potrf_blocked_debug_params, never hardcoded.
// evidence: docs/perf/potrf.md#the-blocked-driver

// The blocked driver's bound, which is NOT the leaf's: it composes n/nb panels.
#ifndef BLKTOL
#define BLKTOL 0.05
#endif
template <typename T>
RealOf<T> blocked_residual_tol(int n) {
    using R = RealOf<T>;
    return R(BLKTOL) * R(n) * std::numeric_limits<R>::epsilon();
}

template <typename Config>
class PotrfBlockedTest : public PotrfCtaTest<Config> {
protected:
    using T = typename Config::ScalarType;
    using R = RealOf<T>;
    static constexpr Backend BackendType = Config::BackendVal;

    struct Blocking { int nb; int W; };

    // The blocking the driver WOULD use; `n` is passed because nb is clamped by it.
    Blocking blocking(int n) const {
        const unsigned p = sycl_potrf::potrf_blocked_debug_params<T>(*this->ctx, n);
        return Blocking{static_cast<int>(p & 0xffffu), static_cast<int>(p >> 16)};
    }

    // Run the blocked driver DIRECTLY. The -12345 info seed is load-bearing: the driver
    // READS info to decide whether an earlier panel failed. info_len < 0 means a full
    // span; 0 the EMPTY span ("not requested"); anything else a SHORT span, likewise not.
    std::vector<int32_t> run_blocked(const MatrixView<T, MatrixFormat::Dense>& V,
                                     Uplo uplo, int info_len = -1) {
        const int batch = static_cast<int>(V.batch_size());
        UnifiedVector<std::byte> ws(
            sycl_potrf::potrf_blocked_buffer_size<T>(*this->ctx, V, uplo));
        const int len = (info_len < 0) ? batch : info_len;
        UnifiedVector<int32_t> info(static_cast<size_t>(std::max(len, 1)), int32_t(-12345));
        if (len > 0) {
            sycl_potrf::potrf_blocked_dispatch<T>(
                *this->ctx, V, uplo, ws.to_span(),
                Span<int32_t>(info.data(), static_cast<size_t>(len)));
        } else {
            sycl_potrf::potrf_blocked_dispatch<T>(*this->ctx, V, uplo, ws.to_span(),
                                                  Span<int32_t>{});
        }
        this->ctx->wait();
        return std::vector<int32_t>(info.begin(), info.end());
    }
};

TYPED_TEST_SUITE(PotrfBlockedTest, PotrfTestTypes);

// Residual above the CTA ceiling at every structurally distinct blocking; each structure
// is ASSERTED of the n the test uses:
//   cap+1        the tier boundary
//   2*nb         an exact multiple of nb: every block is full
//   2*nb+nb/2    a SHORT FINAL BLOCK; the driver relies on `ib < nb implies m2 == 0`
//   nb+2W+6      a SHORT FINAL TRAILING COLUMN PANEL (m2 % W != 0)
//   3*nb+7       several panels AND a short final block together
TYPED_TEST(PotrfBlockedTest, ResidualAboveTheCtaCeiling) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;

    const int cap = this->ceiling();
    ASSERT_GT(cap, 0) << "no CTA capacity for this type -- the leaf is not linked";
    const auto bp = this->blocking(1 << 20);   // the unclamped blocking
    const int nb = bp.nb, W = bp.W;
    ASSERT_GT(nb, 0);
    ASSERT_GT(W, 0);
    ASSERT_LE(nb, cap) << "nb is above the leaf's own capacity: the leaf would throw";

    std::vector<int> sizes = {cap + 1, 2 * nb, 2 * nb + nb / 2, nb + 2 * W + 6, 3 * nb + 7};
    std::sort(sizes.begin(), sizes.end());
    sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());

    int saw_short_block = 0, saw_exact_multiple = 0, saw_short_panel = 0;
    for (int n : sizes) {
        ASSERT_GT(n, cap) << "n=" << n << " is inside the CTA tier; this case proves "
                             "nothing the leaf's own tests do not";
        if (n % nb == 0) ++saw_exact_multiple; else ++saw_short_block;
        const int m2_first = n - nb;
        if (m2_first > W && (m2_first % W) != 0) ++saw_short_panel;

        const int batch = 3;
        Matrix<T, MatrixFormat::Dense> A(n, n, batch);
        std::vector<std::vector<T>> ref(batch);
        for (int b = 0; b < batch; ++b) {
            ref[b] = make_spd<T>(n, 6100u + 29u * b + 7u * static_cast<unsigned>(n));
            this->load_triangle(A, b, n, ref[b], Uplo::Lower,
                                make_scalar<T>(R(-999), R(777)));
        }
        const auto info = this->run_blocked(A.view(), Uplo::Lower);
        for (int b = 0; b < batch; ++b) {
            ASSERT_EQ(info[b], 0) << "n=" << n << " b=" << b;
            const auto L = this->extract_L(A, b, n, Uplo::Lower);
            const R res = multiply_back_residual<T>(ref[b], L, n);
            EXPECT_LE(res, blocked_residual_tol<T>(n))
                << "n=" << n << " b=" << b << " nb=" << nb << " W=" << W
                << " residual/(n*eps)="
                << (res / (R(n) * std::numeric_limits<R>::epsilon()));
        }
    }

    // ANTI-VACUITY: each structure the driver special-cases must actually have occurred.
    ASSERT_GT(saw_exact_multiple, 0) << "no n in the sweep was an exact multiple of nb";
    ASSERT_GT(saw_short_block, 0)    << "no n in the sweep had a short final block";
    ASSERT_GT(saw_short_panel, 0)    << "no n in the sweep had a short trailing column panel";
}

// The other triangle is neither read nor written -- THE FOLD'S ONLY GUARD: with the gemm
// aimed straight at A the lower triangle stays bit-identical while the caller's upper
// triangle fills with the symmetric product, and every other test here stays green.
TYPED_TEST(PotrfBlockedTest, BlockedOtherTriangleIsNeitherReadNorWritten) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;

    const auto bp = this->blocking(1 << 20);
    const int n = bp.nb + 2 * bp.W + 6;
    const int batch = 3;
    ASSERT_GT(n, this->ceiling());
    ASSERT_GT(n - bp.nb, bp.W) << "no full-width trailing panel at this n";
    ASSERT_NE((n - bp.nb) % bp.W, 0) << "no short trailing panel at this n";

    // A SECOND n, AND IT IS NOT DECORATION: only n_big is wide enough to reach the
    // register gemm, whose store path no other guard in this file exercises.
    const int n_big = bp.nb + 3 * bp.W + 134;
    ASSERT_GE(n_big - bp.nb - bp.W, 128)
        << "n_big does not give a rectangle gemm with m >= 128, so it does not "
           "reach float's register ladder and adds nothing over n";
    ASSERT_NE((n_big - bp.nb) % bp.W, 0) << "no short trailing panel at n_big";

    // Pass 1: NOT WRITTEN.
    for (int nn : {n, n_big}) {
        Matrix<T, MatrixFormat::Dense> A(nn, nn, batch);
        const T poison = make_scalar<T>(R(-3.5), R(11.25));
        for (int b = 0; b < batch; ++b) {
            this->load_triangle(A, b, nn, make_spd<T>(nn, 771u + b), Uplo::Lower, poison);
        }
        const auto info = this->run_blocked(A.view(), Uplo::Lower);
        int changed = 0;
        for (int b = 0; b < batch; ++b) {
            ASSERT_EQ(info[b], 0) << "n=" << nn;
            for (int i = 0; i < nn; ++i) {
                for (int j = i + 1; j < nn; ++j) {
                    const T v = A(i, j, b);
                    if (host_real(v) != host_real(poison) ||
                        host_imag(v) != host_imag(poison)) ++changed;
                }
            }
        }
        ASSERT_EQ(changed, 0)
            << changed << " words of the UPPER triangle were written at n=" << nn
            << ". The trailing update's triangular fold is the only thing that "
               "stops this, and no residual test in this file can see it.";
    }

    // Pass 2: NOT READ.
    {
        Matrix<T, MatrixFormat::Dense> A(n, n, batch);
        const R nan = std::numeric_limits<R>::quiet_NaN();
        const T poison = make_scalar<T>(nan, nan);
        std::vector<std::vector<T>> ref(batch);
        for (int b = 0; b < batch; ++b) {
            ref[b] = make_spd<T>(n, 771u + b);
            this->load_triangle(A, b, n, ref[b], Uplo::Lower, poison);
        }
        const auto info = this->run_blocked(A.view(), Uplo::Lower);
        for (int b = 0; b < batch; ++b) {
            ASSERT_EQ(info[b], 0);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j <= i; ++j) {
                    const T v = A(i, j, b);
                    ASSERT_FALSE(std::isnan(host_real(v)))
                        << "NaN leaked out of the untouched upper triangle into ("
                        << i << "," << j << ") b=" << b;
                    ASSERT_FALSE(std::isnan(host_imag(v)));
                }
            }
            const auto L = this->extract_L(A, b, n, Uplo::Lower);
            EXPECT_LE((multiply_back_residual<T>(ref[b], L, n)), blocked_residual_tol<T>(n));
        }
    }
}

// `info` names the 1-based GLOBAL column: the leaf reports an index LOCAL to the
// sub-view it was handed, so the driver adds j. The CTA suite cannot see that addition.
TYPED_TEST(PotrfBlockedTest, BlockedInfoIsTheGlobalColumn) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;

    const int nb = this->blocking(1 << 20).nb;
    const int n = 2 * nb + 8;                // two full blocks plus a short one
    ASSERT_GT(n, this->ceiling());

    int discriminating = 0;
    for (int c : {0, 1, nb - 1, nb, nb + 1, 2 * nb, n - 1}) {
        if (c < 0 || c >= n) continue;
        if (c >= nb) ++discriminating;       // only these can see a missing offset
        Matrix<T, MatrixFormat::Dense> A(n, n, 1);
        const auto ref = make_planted_ldl<T>(n, {c}, 5150u + static_cast<unsigned>(c));
        if (c >= 1) {
            ASSERT_GT(host_real(ref[c + static_cast<size_t>(c) * n]), R(0))
                << "the planted matrix is not discriminating at column " << c;
        }
        this->load_triangle(A, 0, n, ref, Uplo::Lower, make_scalar<T>(R(0), R(0)));
        const auto info = this->run_blocked(A.view(), Uplo::Lower);
        EXPECT_EQ(info[0], c + 1)
            << "planted a non-positive pivot at GLOBAL column " << c << " of " << n
            << " (nb=" << nb << ", i.e. local column " << (c % nb) << " of block "
            << (c / nb) << ")";
    }
    // ANTI-VACUITY: a failure outside block 0, or local->global was never exercised.
    ASSERT_GT(discriminating, 0)
        << "every planted failure was in block 0, where local == global; this test "
           "cannot see a missing info offset";
}

// FIRST FAILURE WINS across panels -- the info MERGE. The leaf writes info
// UNCONDITIONALLY, so aiming every panel's leaf at the caller's info gives
// LAST-PANEL-WINS: SUCCESS reported on a non-PD matrix.
TYPED_TEST(PotrfBlockedTest, BlockedInfoFirstFailureWinsAcrossPanels) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;

    const int nb = this->blocking(1 << 20).nb;
    const int n = 2 * nb + 8;

    struct Case { std::vector<int> cols; int expect; const char* why; };
    const std::vector<Case> cases = {
        {{5}, 6, "one failure in block 0 followed by HEALTHY blocks -- a last-panel-wins "
                 "merge reports SUCCESS here"},
        {{5, nb + 9}, 6, "failures in two different blocks: the earlier wins"},
        {{nb + 3, 2 * nb + 2}, nb + 4, "both failures outside block 0"},
    };

    for (const auto& cs : cases) {
        bool fits = true;
        for (int c : cs.cols) if (c >= n) fits = false;
        if (!fits) continue;
        Matrix<T, MatrixFormat::Dense> A(n, n, 1);
        const auto ref =
            make_planted_ldl<T>(n, cs.cols, 913u + static_cast<unsigned>(cs.expect));
        this->load_triangle(A, 0, n, ref, Uplo::Lower, make_scalar<T>(R(0), R(0)));
        const auto info = this->run_blocked(A.view(), Uplo::Lower);
        EXPECT_EQ(info[0], cs.expect) << cs.why;
    }
}

// `info` at batch scale, and the QUENCH. The quench needs a NaN pivot: a negative one
// divides to a finite number, so a negative-pivot check stays green with it deleted.
TYPED_TEST(PotrfBlockedTest, BlockedInfoAtBatchScaleAndFailedItemsStayFinite) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;

    const int nb = this->blocking(1 << 20).nb;
    const int n = 2 * nb + 8;
    const int batch = 32;
    ASSERT_GT(n, this->ceiling());

    // THE NaN GOES IN THE MIDDLE BLOCK, AND THE PLACEMENT IS LOAD-BEARING: in the FINAL
    // block m2 == 0, so nothing follows to propagate it and the test weakens to one word.
    const int nan_item = 11;
    const int nan_col = nb + 2;              // second of three blocks: local != global
    const std::vector<int> neg_items = {0, 19, batch - 1};
    const std::vector<int> neg_cols  = {0, nb - 1, 2 * nb + 1};

    Matrix<T, MatrixFormat::Dense> A(n, n, batch);
    std::vector<std::vector<T>> ref(batch);
    for (int b = 0; b < batch; ++b) {
        auto it = std::find(neg_items.begin(), neg_items.end(), b);
        if (it != neg_items.end()) {
            ref[b] = make_planted_ldl<T>(
                n, {neg_cols[static_cast<size_t>(it - neg_items.begin())]},
                140u + static_cast<unsigned>(b));
        } else {
            ref[b] = make_spd<T>(n, 400u + 5u * b);
        }
        this->load_triangle(A, b, n, ref[b], Uplo::Lower, make_scalar<T>(R(0), R(0)));
    }
    A(nan_col, nan_col, nan_item) =
        make_scalar<T>(std::numeric_limits<R>::quiet_NaN(), R(0));

    const auto info = this->run_blocked(A.view(), Uplo::Lower);

    for (size_t k = 0; k < neg_items.size(); ++k) {
        EXPECT_EQ(info[neg_items[k]], neg_cols[k] + 1) << "item " << neg_items[k];
    }
    EXPECT_EQ(info[nan_item], nan_col + 1) << "the NaN-pivot item";

    // The quench: every failed item stays FINITE; only the NaN item can falsify it.
    std::vector<int> bad = neg_items;
    bad.push_back(nan_item);
    for (int b : bad) {
        int nonfinite = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                const T v = A(i, j, b);
                if (!std::isfinite(host_real(v)) || !std::isfinite(host_imag(v))) ++nonfinite;
            }
        }
        EXPECT_EQ(nonfinite, 0) << "failed item " << b << " has " << nonfinite
                                << " non-finite entries; the quench did not hold";
    }

    for (int b = 0; b < batch; ++b) {
        if (std::find(bad.begin(), bad.end(), b) != bad.end()) continue;
        ASSERT_EQ(info[b], 0) << "healthy item " << b << " reported a failure";
        const auto L = this->extract_L(A, b, n, Uplo::Lower);
        EXPECT_LE((multiply_back_residual<T>(ref[b], L, n)), blocked_residual_tol<T>(n))
            << "healthy item " << b << " next to failed ones";
    }
}

// The fold does not project the diagonal real, so imag(diag(A22)) carries L21 L21^H's
// rounding into the next panel; the leaf's reload as T(real(A(c,c)), 0) fixes it anyway.
TYPED_TEST(PotrfBlockedTest, BlockedComplexDiagonalIsExactlyReal) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;
    if constexpr (!test_utils::is_complex<T>::value) {
        GTEST_SKIP() << "real scalar: no imaginary part to check";
    } else {
        const int nb = this->blocking(1 << 20).nb;
        const int n = 2 * nb + 8;            // three panels
        ASSERT_GT(n, this->ceiling());
        const auto ref = make_spd<T>(n, 24601u);

        // (a) the input must actually be complex, or nothing below proves anything.
        R max_imag = R(0);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < i; ++j)
                max_imag = std::max(
                    max_imag, std::abs(host_imag(ref[i + static_cast<size_t>(j) * n])));
        ASSERT_GT(max_imag, R(0.01)) << "the generated matrix is effectively real";

        Matrix<T, MatrixFormat::Dense> A(n, n, 1);
        this->load_triangle(A, 0, n, ref, Uplo::Lower, make_scalar<T>(R(0), R(0)));
        ASSERT_EQ(this->run_blocked(A.view(), Uplo::Lower)[0], 0);

        // (b) EXACTLY zero, past every panel boundary, not merely small.
        for (int i = 0; i < n; ++i) {
            ASSERT_EQ(host_imag(A(i, i, 0)), R(0))
                << "imag(L(" << i << "," << i << ")) != 0 at nb=" << nb;
        }
        const auto L = this->extract_L(A, 0, n, Uplo::Lower);
        EXPECT_LE((multiply_back_residual<T>(ref, L, n)), blocked_residual_tol<T>(n));

        // (c) THE SENSITIVITY: conj(A) must give a different factor. Here the conjugate
        //     that matters is the trailing update's transB (Trans would give L21 L21^T).
        std::vector<T> refc(ref);
        for (auto& v : refc) v = host_conj(v);
        Matrix<T, MatrixFormat::Dense> Ac(n, n, 1);
        this->load_triangle(Ac, 0, n, refc, Uplo::Lower, make_scalar<T>(R(0), R(0)));
        ASSERT_EQ(this->run_blocked(Ac.view(), Uplo::Lower)[0], 0);
        bool differs = false;
        for (int i = 1; i < n && !differs; ++i)
            for (int j = 0; j < i && !differs; ++j)
                if (host_imag(A(i, j, 0)) != host_imag(Ac(i, j, 0))) differs = true;
        EXPECT_TRUE(differs) << "conjugating the input did not change the factor";
        const auto Lc = this->extract_L(Ac, 0, n, Uplo::Lower);
        EXPECT_LE((multiply_back_residual<T>(refc, Lc, n)), blocked_residual_tol<T>(n));

        // (d) imag(diag(A)) IS IGNORED end to end: the trailing update ADDS to A22's
        //     diagonal, so caller garbage reaches the NEXT panel's leaf load transform.
        Matrix<T, MatrixFormat::Dense> Ap(n, n, 1);
        this->load_triangle(Ap, 0, n, ref, Uplo::Lower, make_scalar<T>(R(0), R(0)));
        for (int i = 0; i < n; ++i)
            Ap(i, i, 0) = make_scalar<T>(host_real(Ap(i, i, 0)), R(0.75) * R(i + 1));
        ASSERT_EQ(this->run_blocked(Ap.view(), Uplo::Lower)[0], 0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                ASSERT_EQ(host_real(Ap(i, j, 0)), host_real(A(i, j, 0)))
                    << "imag(diag(A)) was not ignored, at (" << i << "," << j << ")";
                ASSERT_EQ(host_imag(Ap(i, j, 0)), host_imag(A(i, j, 0)));
            }
        }
    }
}

// THE PARENT LEADING DIMENSION AND A STRIDE THAT IS NOT ld * cols. Every sub-view the
// driver hands the leaf, the trsm and the gemm must carry the PARENT's ld and stride:
// MatrixView otherwise defaults stride to ld*cols and later items read the wrong matrix.
TYPED_TEST(PotrfBlockedTest, BlockedPaddedLeadingDimensionAndNonDefaultStride) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;

    const int nb = this->blocking(1 << 20).nb;
    const int n = 2 * nb + 8;
    const int batch = 4;
    const int ld = n + 7;                    // != rows
    const int stride = ld * n + 13;          // != ld * cols
    ASSERT_GT(n, this->ceiling());

    UnifiedVector<T> buf(static_cast<size_t>(stride) * batch,
                         make_scalar<T>(R(-11), R(5)));    // non-PD poison
    MatrixView<T, MatrixFormat::Dense> V(buf.data(), n, n, ld, stride, batch);

    std::vector<std::vector<T>> ref(batch);
    for (int b = 0; b < batch; ++b) {
        ref[b] = make_spd<T>(n, 8291u + 13u * b);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j <= i; ++j)
                buf[static_cast<size_t>(b) * stride + i + static_cast<size_t>(j) * ld] =
                    ref[b][i + static_cast<size_t>(j) * n];
    }

    const auto info = this->run_blocked(V, Uplo::Lower);
    for (int b = 0; b < batch; ++b) {
        ASSERT_EQ(info[b], 0) << "b=" << b << " (ld=" << ld << " stride=" << stride << ")";
        std::vector<T> L(static_cast<size_t>(n) * n, T{});
        for (int i = 0; i < n; ++i)
            for (int j = 0; j <= i; ++j)
                L[i + static_cast<size_t>(j) * n] =
                    buf[static_cast<size_t>(b) * stride + i + static_cast<size_t>(j) * ld];
        EXPECT_LE((multiply_back_residual<T>(ref[b], L, n)), blocked_residual_tol<T>(n))
            << "b=" << b << " (ld=" << ld << " stride=" << stride << ")";
    }
}

// THE INFO SPAN in all three of its states, and the ZERO PRE-PASS. The driver READS info
// to decide whether an earlier panel already failed, so a FULL span of caller garbage
// without the pre-pass is a SILENT WRONG ANSWER; an EMPTY span and a SHORT one (size <
// batch) both mean "not requested" and fall back to UNINITIALISED pool scratch.
TYPED_TEST(PotrfBlockedTest, BlockedInfoSpanStatesAndTheZeroPrePass) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;

    const int nb = this->blocking(1 << 20).nb;
    const int n = nb + nb / 2;
    const int batch = 4;
    ASSERT_GT(n, this->ceiling());

    std::vector<std::vector<T>> ref(batch);
    for (int b = 0; b < batch; ++b) ref[b] = make_spd<T>(n, 3300u + b);

    struct Mode { int len; const char* why; };
    const std::vector<Mode> modes = {
        {-1,        "a FULL span seeded with caller garbage"},
        {0,         "the EMPTY span (not requested)"},
        {batch - 1, "a SHORT span (also not requested)"},
    };

    for (const auto& m : modes) {
        Matrix<T, MatrixFormat::Dense> A(n, n, batch);
        for (int b = 0; b < batch; ++b)
            this->load_triangle(A, b, n, ref[b], Uplo::Lower, make_scalar<T>(R(0), R(0)));
        const auto info = this->run_blocked(A.view(), Uplo::Lower, m.len);
        if (m.len < 0) {
            for (int b = 0; b < batch; ++b)
                ASSERT_EQ(info[b], 0)
                    << m.why << ": info[" << b << "] came back as the caller's garbage, "
                                "so the zero pre-pass did not run";
        }
        for (int b = 0; b < batch; ++b) {
            const auto L = this->extract_L(A, b, n, Uplo::Lower);
            EXPECT_LE((multiply_back_residual<T>(ref[b], L, n)), blocked_residual_tol<T>(n))
                << m.why << " b=" << b;
        }
    }
}

// THE TIER OVERLAP, and the single-block path. supports()'s Blocked arm deliberately
// carries NO LOWER BOUND on order: a forced `blocked` below one would become the VENDOR.
//   n == nb   ONE block, m2 == 0, and the W x W x batch scratch is NOT DRAWN
//   n == cap  the CTA ceiling itself, where both tiers are supported
TYPED_TEST(PotrfBlockedTest, BlockedIsCorrectInsideTheCtaTierAndDrawsNoScratchAtOneBlock) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;

    const int cap = this->ceiling();
    const int nb = this->blocking(1 << 20).nb;
    ASSERT_LE(nb, cap);

    // The single-block branch really is a branch: one more column draws the scratch.
    Matrix<T, MatrixFormat::Dense> One(nb, nb, 8);
    Matrix<T, MatrixFormat::Dense> Two(nb + 1, nb + 1, 8);
    const std::size_t s1 =
        sycl_potrf::potrf_blocked_buffer_size<T>(*this->ctx, One.view(), Uplo::Lower);
    const std::size_t s2 =
        sycl_potrf::potrf_blocked_buffer_size<T>(*this->ctx, Two.view(), Uplo::Lower);
    EXPECT_LT(s1, s2)
        << "the n <= nb single-block case draws the same workspace as the multi-block "
           "case, so the W x W x batch trailing scratch is being allocated for callers "
           "that never reach the trailing update (ortho.cc:78 is one)";

    for (int n : {nb, cap}) {
        if (n < 1) continue;
        const int batch = 3;
        Matrix<T, MatrixFormat::Dense> A(n, n, batch);
        std::vector<std::vector<T>> ref(batch);
        for (int b = 0; b < batch; ++b) {
            ref[b] = make_spd<T>(n, 1717u + 11u * b + static_cast<unsigned>(n));
            this->load_triangle(A, b, n, ref[b], Uplo::Lower,
                                make_scalar<T>(R(-999), R(777)));
        }
        const auto info = this->run_blocked(A.view(), Uplo::Lower);
        for (int b = 0; b < batch; ++b) {
            ASSERT_EQ(info[b], 0) << "n=" << n << " b=" << b;
            const auto L = this->extract_L(A, b, n, Uplo::Lower);
            EXPECT_LE((multiply_back_residual<T>(ref[b], L, n)), blocked_residual_tol<T>(n))
                << "n=" << n << " b=" << b << " (nb=" << nb << ", cap=" << cap << ")";
        }
    }
}

// The direct entry point's correctness gates throw rather than launch. This cannot go
// through the facade: supports() REJECTS Upper, so a forced `blocked` becomes the vendor.
TYPED_TEST(PotrfBlockedTest, BlockedDirectEntryPointRefusesWhatSupportsRefuses) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;

    const int nb = this->blocking(1 << 20).nb;
    const int n = nb + nb / 2;
    UnifiedVector<int32_t> info(8, int32_t(0));

    // (a) Uplo::Upper -- CORRECTNESS, not fit. The schedule is Lower-shaped.
    {
        Matrix<T, MatrixFormat::Dense> A(n, n, 2);
        A.fill(make_scalar<T>(R(0), R(0)));
        for (int b = 0; b < 2; ++b)
            for (int i = 0; i < n; ++i) A(i, i, b) = make_scalar<T>(R(2), R(0));
        UnifiedVector<std::byte> ws(
            sycl_potrf::potrf_blocked_buffer_size<T>(*this->ctx, A.view(), Uplo::Lower));
        EXPECT_THROW(sycl_potrf::potrf_blocked_dispatch<T>(*this->ctx, A.view(), Uplo::Upper,
                                                           ws.to_span(), info.to_span()),
                     std::invalid_argument);
    }

    // (b) not square.
    {
        Matrix<T, MatrixFormat::Dense> A(n, n - 3, 1);
        A.fill(make_scalar<T>(R(1), R(0)));
        UnifiedVector<std::byte> ws(64);
        EXPECT_THROW(sycl_potrf::potrf_blocked_dispatch<T>(*this->ctx, A.view(), Uplo::Lower,
                                                           ws.to_span(), info.to_span()),
                     std::invalid_argument);
    }

    // (c) heterogeneous batch -- the silent-wrong-answer one.
    {
        Matrix<T, MatrixFormat::Dense> A(n, n, 4);
        A.fill(make_scalar<T>(R(0), R(0)));
        for (int b = 0; b < 4; ++b)
            for (int i = 0; i < n; ++i) A(i, i, b) = make_scalar<T>(R(2), R(0));
        UnifiedVector<int> act_r(4), act_c(4);
        for (int b = 0; b < 4; ++b) { act_r[b] = n - b; act_c[b] = n - b; }
        auto V = A.view().with_active_dims(act_r.to_span(), act_c.to_span());
        ASSERT_TRUE(V.is_heterogeneous())
            << "the view is not actually heterogeneous; this case would prove nothing";
        UnifiedVector<std::byte> ws(
            sycl_potrf::potrf_blocked_buffer_size<T>(*this->ctx, A.view(), Uplo::Lower));
        EXPECT_THROW(sycl_potrf::potrf_blocked_dispatch<T>(*this->ctx, V, Uplo::Lower,
                                                           ws.to_span(), info.to_span()),
                     std::invalid_argument);
    }
}

// The route table above the CTA ceiling, including the VENDOR-FREE FALLBACK.
// evidence: docs/perf/potrf.md#route-arms-and-the-supports-gates
TYPED_TEST(PotrfBlockedTest, BlockedRouteTableAndTheVendorFreeFallback) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;
    static constexpr Backend B = TestFixture::BackendType;
    using Tbl = dispatch::RouteTable<dispatch::Op::potrf, T>;
    const dispatch::Route cta{dispatch::Origin::Native, dispatch::Algorithm::CTA};
    const dispatch::Route blk{dispatch::Origin::Native, dispatch::Algorithm::Blocked};

    const int n = this->ceiling() + 1;
    Matrix<T, MatrixFormat::Dense> A(n, n, 4);
    A.fill(T{});
    for (int b = 0; b < 4; ++b)
        for (int i = 0; i < n; ++i) A(i, i, b) = make_scalar<T>(R(1), R(0));

    const auto lower = backend::potrf_op_shape<B, T>(*this->ctx, A.view(), Uplo::Lower);
    const auto upper = backend::potrf_op_shape<B, T>(*this->ctx, A.view(), Uplo::Upper);
    ASSERT_TRUE(lower.has_value());
    ASSERT_TRUE(upper.has_value());

    // (1) Above the ceiling the CTA tier is gone and the Blocked tier is there.
    EXPECT_FALSE(Tbl::supports(cta, *lower));
    EXPECT_TRUE(Tbl::supports(blk, *lower));

    // (2) Uplo::Upper is a CORRECTNESS gate on the Blocked arm and must stay one until
    //     the driver mirrors, or a forced `blocked` there silently becomes cuSOLVER.
    EXPECT_FALSE(Tbl::supports(blk, *upper));

    // (3) preferred() is still ALL FALSE, so a vendor-present build takes the vendor for
    //     this shape. evidence: docs/perf/potrf.md#preferred-is-false-everywhere
    EXPECT_FALSE(Tbl::preferred(blk, *lower));
    EXPECT_FALSE(Tbl::preferred(cta, *lower));

    // (4) With no vendor in the build resolve_route hands over any SUPPORTED native
    //     route; above the CTA ceiling the blocked driver is the only one.
    const auto free_route = backend::potrf_route<B, T>(*this->ctx, A.view(), Uplo::Lower,
                                                       /*vendor_available=*/false);
    EXPECT_TRUE(dispatch::is_native(free_route));
    EXPECT_EQ(free_route.algo, dispatch::Algorithm::Blocked)
        << "a vendor-free build does not reach the blocked driver above the CTA ceiling";
}

// potrf_buffer_size SURVIVES THE ROUTE CHANGING BETWEEN QUERY AND CALL: the route is
// resolved twice, so a query that sizes only the route it resolved under-allocates.
// evidence: docs/perf/potrf.md#workspace-sizing
TYPED_TEST(PotrfBlockedTest, BufferSizeCoversEverySupportedNativeTier) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;
    static constexpr Backend B = TestFixture::BackendType;

    const int cap = this->ceiling();
    const int nb = this->blocking(1 << 20).nb;
    const int n = cap;                       // inside the CTA tier
    const int batch = 16;
    ASSERT_GT(n, nb) << "at this order the blocked driver is a single block and draws no "
                        "trailing scratch, so the two tiers cost the same and this test "
                        "would be vacuous";

    Matrix<T, MatrixFormat::Dense> A(n, n, batch);
    std::vector<std::vector<T>> ref(batch);
    for (int b = 0; b < batch; ++b) {
        ref[b] = make_spd<T>(n, 5959u + b);
        this->load_triangle(A, b, n, ref[b], Uplo::Lower, make_scalar<T>(R(0), R(0)));
    }

    // ANTI-VACUITY: the two tiers must actually want different amounts.
    const std::size_t cta_need = sycl_potrf::potrf_cta_buffer_size<T>(*this->ctx, A.view());
    const std::size_t blk_need =
        sycl_potrf::potrf_blocked_buffer_size<T>(*this->ctx, A.view(), Uplo::Lower);
    ASSERT_GT(blk_need, cta_need)
        << "the blocked tier does not need more workspace than the CTA tier here, so a "
           "chosen-route-only query would pass this test by accident";

    struct EnvGuard {
        std::string saved; bool had = false;
        EnvGuard() {
            if (const char* v = std::getenv("BATCHLAS_POTRF_ROUTE")) { saved = v; had = true; }
        }
        void set(const char* v) { ::setenv("BATCHLAS_POTRF_ROUTE", v, 1); }
        ~EnvGuard() {
            if (had) ::setenv("BATCHLAS_POTRF_ROUTE", saved.c_str(), 1);
            else ::unsetenv("BATCHLAS_POTRF_ROUTE");
        }
    } guard;

    guard.set("cta");
    const std::size_t queried = potrf_buffer_size<B, T>(*this->ctx, A.view(), Uplo::Lower);
    ASSERT_GE(queried, blk_need)
        << "potrf_buffer_size resolved `cta` and sized only that tier; a caller whose "
           "environment changes between the query and the call (options.hh:546-552 reads "
           "getenv twice) under-allocates by " << (blk_need - queried) << " bytes";

    guard.set("blocked");
    UnifiedVector<std::byte> ws(queried);
    UnifiedVector<int32_t> info(batch, int32_t(-12345));
    ASSERT_NO_THROW(
        (potrf<B, T>(*this->ctx, A.view(), Uplo::Lower, ws.to_span(), info.to_span())));
    this->ctx->wait();
    for (int b = 0; b < batch; ++b) {
        ASSERT_EQ(info[b], 0) << "b=" << b;
        const auto L = this->extract_L(A, b, n, Uplo::Lower);
        EXPECT_LE((multiply_back_residual<T>(ref[b], L, n)), blocked_residual_tol<T>(n))
            << "b=" << b;
    }
}

// The facade actually reaches the BLOCKED driver, guarded BIT-EXACTLY against the direct
// entry point with the same routed gemm and trsm injected -- the INJECTION SEAM too.
TYPED_TEST(PotrfBlockedTest, FacadeReachesTheBlockedDriver) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;
    static constexpr Backend B = TestFixture::BackendType;

    const int nb = this->blocking(1 << 20).nb;
    const int n = 2 * nb;
    const int batch = 3;
    ASSERT_GT(n, this->ceiling());

    struct EnvGuard {
        std::string saved; bool had = false;
        EnvGuard() {
            if (const char* v = std::getenv("BATCHLAS_POTRF_ROUTE")) { saved = v; had = true; }
            ::setenv("BATCHLAS_POTRF_ROUTE", "blocked", 1);
        }
        ~EnvGuard() {
            if (had) ::setenv("BATCHLAS_POTRF_ROUTE", saved.c_str(), 1);
            else ::unsetenv("BATCHLAS_POTRF_ROUTE");
        }
    } guard;

    std::vector<std::vector<T>> ref(batch);
    for (int b = 0; b < batch; ++b) ref[b] = make_spd<T>(n, 4040u + b);

    Matrix<T, MatrixFormat::Dense> A(n, n, batch);
    for (int b = 0; b < batch; ++b)
        this->load_triangle(A, b, n, ref[b], Uplo::Lower, make_scalar<T>(R(0), R(0)));

    // The route assertion LOCALISES a failure but is NOT the guard.
    const auto route = backend::potrf_route<B, T>(*this->ctx, A.view(), Uplo::Lower,
                                                  /*vendor_available=*/true);
    ASSERT_TRUE(dispatch::is_native(route))
        << "BATCHLAS_POTRF_ROUTE=blocked did not resolve to a native route";
    ASSERT_EQ(route.algo, dispatch::Algorithm::Blocked);

    UnifiedVector<std::byte> ws(potrf_buffer_size<B, T>(*this->ctx, A.view(), Uplo::Lower));
    UnifiedVector<int32_t> info(batch, int32_t(-7));
    potrf<B, T>(*this->ctx, A.view(), Uplo::Lower, ws.to_span(), info.to_span());
    this->ctx->wait();

    // THE GUARD.
    Matrix<T, MatrixFormat::Dense> direct(n, n, batch);
    for (int b = 0; b < batch; ++b)
        this->load_triangle(direct, b, n, ref[b], Uplo::Lower, make_scalar<T>(R(0), R(0)));
    UnifiedVector<std::byte> dws(
        sycl_potrf::potrf_blocked_buffer_size<T>(*this->ctx, direct.view(), Uplo::Lower));
    UnifiedVector<int32_t> dinfo(batch, int32_t(-7));
    sycl_potrf::potrf_blocked_dispatch<T>(
        *this->ctx, direct.view(), Uplo::Lower, dws.to_span(), dinfo.to_span(),
        [](Queue& c, const MatrixView<T, MatrixFormat::Dense>& ga,
           const MatrixView<T, MatrixFormat::Dense>& gb,
           const MatrixView<T, MatrixFormat::Dense>& gc,
           T galpha, T gbeta, Transpose gta, Transpose gtb, ComputePrecision gp) {
            return gemm<B, T>(c, ga, gb, gc, galpha, gbeta, gta, gtb, gp);
        },
        [](Queue& c, const MatrixView<T, MatrixFormat::Dense>& ta,
           const MatrixView<T, MatrixFormat::Dense>& tb,
           T talpha, Side tside, Uplo tuplo, Transpose ttrans, Diag tdiag) {
            return trsm<B, T>(c, ta, tb, talpha, tside, tuplo, ttrans, tdiag);
        });
    this->ctx->wait();

    for (int b = 0; b < batch; ++b) {
        ASSERT_EQ(info[b], 0);
        ASSERT_EQ(dinfo[b], 0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                ASSERT_EQ(host_real(A(i, j, b)), host_real(direct(i, j, b)))
                    << "the facade did not run the blocked driver with the routed seams: "
                       "its answer differs from potrf_blocked_dispatch's at ("
                    << i << "," << j << ") b=" << b;
                ASSERT_EQ(host_imag(A(i, j, b)), host_imag(direct(i, j, b)));
            }
        }
        const auto L = this->extract_L(A, b, n, Uplo::Lower);
        EXPECT_LE((multiply_back_residual<T>(ref[b], L, n)), blocked_residual_tol<T>(n))
            << "b=" << b;
    }
}

// A POISONED WORKSPACE: the driver must not read scratch it has not written. Fresh
// UnifiedVector pages come back ZEROED, so every other test here is blind to this. The
// trap: the native gemm's epilogue reads `prior` unconditionally and 0 * NaN = NaN.
TYPED_TEST(PotrfBlockedTest, BlockedDoesNotReadUninitialisedWorkspace) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;

    const auto bp = this->blocking(1 << 20);
    const int n = 2 * bp.nb;          // > nb, so the W x W product IS drawn
    const int batch = 4;
    ASSERT_GT(n, bp.nb) << "at n <= nb the scratch is not allocated at all "
                           "(potrf_blocked_layout), so this test would be vacuous";

    Matrix<T, MatrixFormat::Dense> A(n, n, batch);
    std::vector<std::vector<T>> ref(batch);
    for (int b = 0; b < batch; ++b) {
        ref[b] = make_spd<T>(n, 9001u + b);
        this->load_triangle(A, b, n, ref[b], Uplo::Lower, make_scalar<T>(R(0), R(0)));
    }

    const std::size_t bytes =
        sycl_potrf::potrf_blocked_buffer_size<T>(*this->ctx, A.view(), Uplo::Lower);
    ASSERT_GT(bytes, 0u);
    UnifiedVector<std::byte> ws(bytes);
    std::memset(ws.data(), 0xFF, bytes);

    UnifiedVector<int32_t> info(batch, int32_t(-12345));
    sycl_potrf::potrf_blocked_dispatch<T>(*this->ctx, A.view(), Uplo::Lower,
                                          ws.to_span(), info.to_span());
    this->ctx->wait();

    for (int b = 0; b < batch; ++b) {
        ASSERT_EQ(info[b], 0)
            << "a positive-definite matrix was reported not positive definite from a "
               "POISONED workspace, b=" << b;
        const auto L = this->extract_L(A, b, n, Uplo::Lower);
        EXPECT_LE((multiply_back_residual<T>(ref[b], L, n)), blocked_residual_tol<T>(n))
            << "b=" << b;
    }
}


}  // namespace
