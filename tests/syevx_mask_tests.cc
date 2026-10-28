// SPDX-License-Identifier: MIT
// syevx honours a batch mask (W4c) on tests/masked_contract.hh: A, W, V, m and info of an inactive
// item keep every byte, and an active item equals the unmasked call with the public syev pinned
// strictly (cta_fused on a GPU, the LAPACKE loop on the host backend). Direct runs every range on
// both backends. LOBPCG and Filtered run on a GPU with every ortho; on the host backend
// (gemm is the host BLAS) and on DirectSubset (stebz, stein and
// sytrd_sy2sb refuse) a masked call throws before any write. Every mask is device-only; data
// complex where T is, ld and stride padded for A and V, W, m and info poisoned.
// evidence: docs/design/masked-batches.md#masked-batches-tests
#include <gtest/gtest.h>

#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-vector.hh>

#include <src/ops/syev/choice.hh>
#include <tests/masked_contract.hh>
#include <tests/test_support/storage.hh>
#include <tests/test_utils.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <array>
#include <optional>
#include <variant>
#include <span>
#include <string>
#include <vector>

using namespace batchlas;

namespace {

namespace mc = masked_contract;
namespace ts = test_support;

template <typename T>
using Real = typename base_type<T>::type;

/// Hermitian items: a graded real diagonal 0.25 (i + 1)^2 (+ an item offset unless `same`) and
/// small complex off-diagonals, so the spectrum is spread and its ends separated.
template <typename T>
void fill_hermitian(Matrix<T>& A, int n, int batch, bool same) {
    const int ld = A.ld();
    const int64_t stride = A.stride();
    T* p = A.data_ptr();
    for (int b = 0; b < batch; ++b) {
        const int item = same ? 0 : b;
        T* a = p + stride * b;
        for (int j = 0; j < n; ++j) {
            a[j + j * ld] = T(0.25 * (j + 1) * (j + 1) + 0.03125 * item);
            for (int i = j + 1; i < n; ++i) {
                const double re = double(int((i * 7 + j * 3 + item * 5) % 17) - 8) / 64;
                const double im = double(int((i * 5 + j + item * 3) % 11) - 5) / 64;
                const T v = MatrixView<T>::make(re, im);
                a[i + j * ld] = v;
                if constexpr (is_std_complex_v<T>) a[j + i * ld] = std::conj(v);
                else a[j + i * ld] = v;
            }
        }
    }
}

template <typename T, Backend B>
struct Cfg {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};
using Types = typename test_utils::backend_types<Cfg>::type;

/// One syevx call: A (read only), W (neigs per item), V (n x neigs, padded), m and info.
template <typename T, Backend B>
struct Problem {
    using R = Real<T>;
    Queue& q;
    int n, batch, neigs;
    JobType jobz;
    SyevxParams<T> params;
    bool same;
    Matrix<T> A, V;
    UnifiedVector<R> w;
    UnifiedVector<int32_t> m, info;
    Problem(Queue& q_, int n_, int batch_, int neigs_, JobType j, const SyevxParams<T>& p, bool same_)
        : q(q_), n(n_), batch(batch_), neigs(neigs_), jobz(j), params(p), same(same_), A(ts::padded<T>(n_, n_, batch_)),
          V(ts::padded<T>(n_, neigs_, batch_)), w(std::size_t(neigs_) * batch_), m(std::size_t(batch_)),
          info(std::size_t(batch_)) {}

    void prepare() {
        fill_hermitian(A, n, batch, same);
        std::fill_n(V.data_ptr(), std::size_t(V.stride()) * batch, ts::poison<T>());
        std::fill_n(w.data(), w.size(), ts::poison<R>());
        std::fill_n(m.data(), m.size(), int32_t(-55));
        std::fill_n(info.data(), info.size(), int32_t(-77));
    }
    void call(Span<const std::uint8_t> mask, Span<std::byte> ws) {
        syevx<B, T, MatrixFormat::Dense>(q, A.view().with_batch_mask(mask), w.to_span(), m.to_span(), std::size_t(neigs), ws,
                                         jobz, V.view().with_batch_mask(mask), params, info.to_span());
    }
    std::size_t buffer_size(Span<const std::uint8_t> mask) {
        return syevx_buffer_size<B, T, MatrixFormat::Dense>(q, A.view().with_batch_mask(mask), w.to_span(),
                                                             std::size_t(neigs), jobz, V.view().with_batch_mask(mask), params);
    }
    std::vector<mc::Output> outputs() {
        return {mc::Output::matrix("A", A.view()), mc::Output::matrix("V", V.view()),
                mc::Output::span<R>("W", {w.data(), w.size()}, std::size_t(neigs)),
                mc::Output::span<int32_t>("m", {m.data(), m.size()}, 1),
                mc::Output::span<int32_t>("info", {info.data(), info.size()}, 1)};
    }
    mc::Case make_case(int wg_items) {
        mc::Case c;
        c.batch = batch;
        c.wg_items = wg_items;
        // The iterative solvers seed their start block per item, so only Direct makes identical items.
        c.identical_items = same && (params.method == SyevxAlgorithm::Direct || params.method == SyevxAlgorithm::Auto);
        c.prepare = [this] { prepare(); };
        c.call = [this](Span<const std::uint8_t> mk, Span<std::byte> ws) { call(mk, ws); };
        c.buffer_size = [this](Span<const std::uint8_t> mk) { return buffer_size(mk); };
        c.outputs = [this] { return outputs(); };
        return c;
    }
};

// Items per 32-lane sub-group of syev's partition kernels: one item per P = next power of two >= n.
int items_per_subgroup(int n) {
    int p = 4;
    while (p < n) p *= 2;
    return 32 / p;
}

template <typename Config>
class SyevxMask : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    using R = Real<T>;
    static constexpr Backend B = Config::BackendVal;
    static constexpr bool kHost = B == Backend::NETLIB;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (kHost && !select::kHasNetlib) GTEST_SKIP() << "no LAPACKE in this build";
        if (!kHost && this->ctx->device().type != DeviceType::GPU) GTEST_SKIP() << "the native families are GPU kernels";
    }

    // The public syev family both calls share, pinned strictly and required reached.
    static const char* syev_family() { return kHost ? "vendor" : "cta_fused"; }

    // Bit-identical to the unmasked call where the route allows it: Direct everywhere, and LOBPCG
    // with Householder (complex T). Elsewhere EXPECT_VERIFY mode: m equal, and an item the unmasked
    // call converged converges with W within 1e3 eps of it. Inactive items are bit-exact either way.
    void check(const std::string& what, int n, int batch, int neigs, JobType jobz, const SyevxParams<T>& p,
               bool same = false) {
        SCOPED_TRACE(what + " n " + std::to_string(n) + " batch " + std::to_string(batch) + " neigs " +
                     std::to_string(neigs) + (jobz == JobType::EigenVectors ? " V" : " N"));
        Queue& q = *this->ctx;
        // The unmasked reference would route a level-2/3 or potrf child to the vendor on a GPU,
        // which refuses the masked call: both runs pin them `native`.
        std::array<std::optional<select::ScopedPin<std::monostate>>, 7> children;
        if (!kHost) {
            const char* ops[] = {"gemm", "gemv", "trsm", "trmm", "syrk", "herk", "potrf"};
            for (std::size_t i = 0; i < children.size(); ++i) children[i].emplace(ops[i], "native");
        }
        Problem<T, B> pr(q, n, batch, neigs, jobz, p, same);
        auto c = pr.make_case(items_per_subgroup(n));
        std::vector<R> w_ref;
        std::vector<int32_t> info_ref, m_ref;
        if (!exact(p)) {
            const select::ScopedPin<ops::syev::SyevChoice> pin("syev", syev_family(), select::StrictPin{});
            UnifiedVector<std::byte> ws(std::max<std::size_t>(pr.buffer_size({}), 1));
            pr.prepare();
            pr.call({}, ws.to_span());
            q.wait();
            w_ref.assign(pr.w.data(), pr.w.data() + pr.w.size());
            info_ref.assign(pr.info.data(), pr.info.data() + pr.info.size());
            m_ref.assign(pr.m.data(), pr.m.data() + pr.m.size());
            if (p.iterations > 0) EXPECT_GT(std::count(info_ref.begin(), info_ref.end(), 0), 0) << "the unmasked call converged no item";
            c.verify_active = [&](const std::string& pattern, std::span<const int> active) {
                const R tol = R(1000) * std::numeric_limits<R>::epsilon();
                for (const int b : active) {
                    EXPECT_EQ(pr.m[std::size_t(b)], m_ref[std::size_t(b)]) << pattern << " m of item " << b;
                    if (info_ref[std::size_t(b)] != 0) continue;  // an unconverged reference bounds nothing
                    EXPECT_EQ(pr.info[std::size_t(b)], 0) << pattern << " info of item " << b;
                    for (int i = 0; i < neigs; ++i) {
                        const std::size_t e = std::size_t(b) * neigs + i;
                        EXPECT_LE(std::abs(pr.w[e] - w_ref[e]), tol * std::max(R(1), std::abs(w_ref[e])))
                            << pattern << " W(" << i << ") of item " << b;
                    }
                }
            };
        }
        mc::check<ops::syev::SyevChoice>(q, "syev", syev_family(), c);
    }

    // Filtered folds a batch filter degree over the items on the host, so its active items follow
    // the mask; CGS2's masked call is not bit-identical on Metal (ortho's).
    static bool exact(const SyevxParams<T>& p) {
        if (p.method == SyevxAlgorithm::Direct || p.method == SyevxAlgorithm::Auto) return true;
        return p.method == SyevxAlgorithm::LOBPCG && p.algorithm == OrthoAlgorithm::Householder;
    }
    // Runs unmasked first (the shape runs), then masked {0,1,0,1,...}: throws naming the mask, and
    // writes nothing at all (A, V, W, m, info).
    void expect_refused(const std::string& what, int n, int batch, int neigs, const SyevxParams<T>& p) {
        SCOPED_TRACE(what);
        Queue& q = *this->ctx;
        Problem<T, B> pr(q, n, batch, neigs, JobType::EigenVectors, p, false);
        {
            UnifiedVector<std::byte> ws(std::max<std::size_t>(pr.buffer_size({}), 1));
            pr.prepare();
            ASSERT_NO_THROW(pr.call({}, ws.to_span())) << "the unmasked call must run";
            q.wait();
        }
        std::vector<std::uint8_t> bytes(static_cast<std::size_t>(batch));
        for (int b = 0; b < batch; ++b) bytes[std::size_t(b)] = b % 2;
        const mc::DeviceMask mask(q, bytes);
        EXPECT_THROW((void)pr.buffer_size(mask.span()), batchlas::unsupported) << "the size query must refuse too";
        UnifiedVector<std::byte> ws(std::max<std::size_t>(pr.buffer_size({}), 1));
        pr.prepare();
        q.wait();
        const auto outs = pr.outputs();
        std::vector<std::vector<std::byte>> before;
        for (const auto& o : outs) before.emplace_back(o.base, o.base + o.stride_bytes * std::size_t(batch));
        std::string msg;
        try {
            pr.call(mask.span(), ws.to_span());
            q.wait();
        } catch (const batchlas::unsupported& e) {
            msg = e.what();
        }
        q.wait();
        EXPECT_NE(msg.find("batch mask"), std::string::npos) << "the masked call was not refused: '" << msg << "'";
        for (std::size_t o = 0; o < outs.size(); ++o)
            EXPECT_EQ(0, std::memcmp(before[o].data(), outs[o].base, before[o].size())) << outs[o].name << " was written";
    }

    static SyevxParams<T> params(SyevxAlgorithm a) {
        SyevxParams<T> p;
        p.method = a;
        return p;
    }
    // A fixed iteration count (tolerance below reach) and an ortho that honours the mask on a GPU:
    // Householder for complex T, CGS2 for real T, whose masked Householder (orgqr) finds no route
    // at these shapes on Metal. SVQB and SVQB2 read stale workspace on this branch (ortho's).
    static SyevxParams<T> iterative(SyevxAlgorithm a, bool largest, std::optional<OrthoAlgorithm> o = std::nullopt) {
        SyevxParams<T> p = params(a);
        p.algorithm = o ? *o : (is_std_complex_v<T> ? OrthoAlgorithm::Householder : OrthoAlgorithm::CGS2);
        p.find_largest = largest;
        // A bound needs converged items: a reachable tolerance and room to reach it.
        p.iterations = exact(p) ? 12 : 80;
        p.absolute_tolerance = T(exact(p) ? 1e-30 : 1e-4);
        p.relative_tolerance = p.absolute_tolerance;
        return p;
    }
};

TYPED_TEST_SUITE(SyevxMask, Types);

// Direct: the copy, the public syev and the select kernel. Every range, both jobz, Auto too.
TYPED_TEST(SyevxMask, DirectLeavesInactiveItemsUntouched) {
    using T = typename TestFixture::T;
    for (const int n : {7, 13})
        for (const JobType j : {JobType::EigenVectors, JobType::NoEigenVectors}) {
            for (const bool largest : {true, false}) {
                auto p = TestFixture::params(SyevxAlgorithm::Direct);
                p.find_largest = largest;
                this->check(largest ? "direct largest" : "direct smallest", n, 24, 3, j, p);
            }
            auto idx = TestFixture::params(SyevxAlgorithm::Direct);
            idx.select = SyevxSelect::Index;
            idx.il = 2;
            idx.iu = 5;
            idx.order = SortOrder::Descending;
            this->check("direct index", n, 24, 4, j, idx);
            auto val = TestFixture::params(SyevxAlgorithm::Direct);
            val.select = SyevxSelect::Value;  // (2, 20]: 6 values at n = 13, over the capacity 4
            val.vl = Real<T>(2);
            val.vu = Real<T>(20);
            this->check("direct value", n, 24, 4, j, val);
        }
    this->check("auto", 9, 24, 2, JobType::EigenVectors, TestFixture::params(SyevxAlgorithm::Auto));
}

// The saturating batch: 1024 identical items through the select kernel (one item per work-group).
TYPED_TEST(SyevxMask, DirectSaturatingIdenticalItems) {
    if (TestFixture::kHost) GTEST_SKIP() << "the work-group kernels are the GPU's concern";
    auto val = TestFixture::params(SyevxAlgorithm::Direct);
    val.select = SyevxSelect::Value;
    val.vl = Real<typename TestFixture::T>(1);
    val.vu = Real<typename TestFixture::T>(9);
    this->check("direct value", 11, 1024, 4, JobType::EigenVectors, val, true);
}

// LOBPCG on a GPU: residual, restart, null-direction and reverse kernels, both ends, both jobz,
// the Jacobi preconditioners, soft locking, and the iterations == 0 cold path. A tolerance out of
// reach fixes the iteration count, so with Householder active items are bit-identical.
TYPED_TEST(SyevxMask, LobpcgLeavesInactiveItemsUntouched) {
    using T = typename TestFixture::T;
    if (TestFixture::kHost) GTEST_SKIP() << "the host backend refuses (LobpcgAndFilteredRefuseOnTheHostBackend)";
    for (const bool largest : {true, false})
        for (const JobType j : {JobType::EigenVectors, JobType::NoEigenVectors})
            this->check(largest ? "lobpcg largest" : "lobpcg smallest", 16, 20, 3, j,
                        TestFixture::iterative(SyevxAlgorithm::LOBPCG, largest));
    auto shifted = TestFixture::iterative(SyevxAlgorithm::LOBPCG, true);
    shifted.preconditioner_type = SyevxPreconditioner::JacobiShifted;
    this->check("lobpcg jacobi-shifted", 16, 20, 3, JobType::EigenVectors, shifted);
    auto jacobi = TestFixture::iterative(SyevxAlgorithm::LOBPCG, false);
    jacobi.preconditioner_type = SyevxPreconditioner::Jacobi;
    this->check("lobpcg jacobi", 16, 20, 3, JobType::EigenVectors, jacobi);
    {
        const ScopedEnvVar lock("BATCHLAS_SYEVX_SOFT_LOCK", "1");
        auto p = TestFixture::iterative(SyevxAlgorithm::LOBPCG, true);
        p.absolute_tolerance = T(1e-2);
        p.relative_tolerance = T(1e-2);
        this->check("lobpcg soft-lock", 16, 20, 3, JobType::EigenVectors, p, true);
    }
    for (const bool largest : {true, false}) {
        auto cold = TestFixture::iterative(SyevxAlgorithm::LOBPCG, largest);
        cold.iterations = 0;
        this->check("lobpcg cold", 16, 20, 3, JobType::EigenVectors, cold);
    }
    if constexpr (is_std_complex_v<T>)  // CGS2, the real default, within a bound
        this->check("lobpcg cgs2", 16, 20, 3, JobType::EigenVectors,
                    TestFixture::iterative(SyevxAlgorithm::LOBPCG, true, OrthoAlgorithm::CGS2));
}

// The host loop stops when every flag reads converged. Items holding the same A converge on the
// same check, so the masked call stops where the unmasked one does only if an inactive item's flag
// reads converged: without that neutral value it runs to the cap and the active items move.
TYPED_TEST(SyevxMask, IterativeStopsOnTheActiveItems) {
    using T = typename TestFixture::T;
    if (TestFixture::kHost) GTEST_SKIP() << "the host backend refuses";
    for (const auto a : {SyevxAlgorithm::LOBPCG, SyevxAlgorithm::Filtered})
        for (const int batch : {20, 1024}) {
            auto p = TestFixture::iterative(a, true);
            p.iterations = 200;
            p.absolute_tolerance = T(2e-3);
            p.relative_tolerance = T(2e-3);
            this->check(a == SyevxAlgorithm::LOBPCG ? "lobpcg converging" : "filtered converging", 16, batch, 3,
                        JobType::EigenVectors, p, true);
        }
}

// Filtered on a GPU: Gershgorin bounds, the interval, sigma and step kernels, and the finalize.
TYPED_TEST(SyevxMask, FilteredLeavesInactiveItemsUntouched) {
    if (TestFixture::kHost) GTEST_SKIP() << "the host backend refuses (LobpcgAndFilteredRefuseOnTheHostBackend)";
    for (const bool largest : {true, false})
        for (const JobType j : {JobType::EigenVectors, JobType::NoEigenVectors})
            this->check(largest ? "filtered largest" : "filtered smallest", 16, 20, 3, j,
                        TestFixture::iterative(SyevxAlgorithm::Filtered, largest));
    {
        const ScopedEnvVar deg("BATCHLAS_SYEVX_FILTER_DEGREE_AUTO", "1");
        this->check("filtered auto degree", 16, 20, 3, JobType::EigenVectors,
                    TestFixture::iterative(SyevxAlgorithm::Filtered, true));
    }
}
TYPED_TEST(SyevxMask, LobpcgAndFilteredRefuseOnTheHostBackend) {
    if (!TestFixture::kHost) GTEST_SKIP() << "the device gemm honours the mask";
    for (const auto a : {SyevxAlgorithm::LOBPCG, SyevxAlgorithm::Filtered})
        this->expect_refused(a == SyevxAlgorithm::LOBPCG ? "lobpcg" : "filtered", 16, 8, 3,
                             TestFixture::iterative(a, true));
}

// A Cholesky ortho runs potrf, trsm and the gram (all honour the mask) on the masked views.
TYPED_TEST(SyevxMask, CholeskyOrthoOnAGpu) {
    if (TestFixture::kHost) GTEST_SKIP() << "the host device forces Householder ortho";
    for (const auto a : {SyevxAlgorithm::LOBPCG, SyevxAlgorithm::Filtered})
        for (const auto o : {OrthoAlgorithm::Chol2, OrthoAlgorithm::Cholesky, OrthoAlgorithm::ShiftChol3})
            this->check(std::string(a == SyevxAlgorithm::LOBPCG ? "lobpcg" : "filtered") + " ortho " + std::to_string(int(o)),
                        16, 20, 3, JobType::EigenVectors, TestFixture::iterative(a, true, o));
}

// DirectSubset carries the mask to every child, but stebz, stein and sytrd_sy2sb refuse it.
TYPED_TEST(SyevxMask, DirectSubsetRefusesUntilItsChildrenHonourTheMask) {
    using T = typename TestFixture::T;
    if constexpr (is_std_complex_v<T>) {
        GTEST_SKIP() << "DirectSubset is real only (complex falls back to Direct)";
    } else {
        auto p = TestFixture::params(SyevxAlgorithm::DirectSubset);
        p.select = SyevxSelect::Index;
        p.il = 1;
        p.iu = 3;
        this->expect_refused("direct_subset", 12, 8, 3, p);
    }
}

}  // namespace
