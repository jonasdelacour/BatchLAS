// SPDX-License-Identifier: MIT
// The LU consumers honour a batch mask (W2b), each family pinned and reached, on
// tests/masked_contract.hh: getrs (device: cta in every transpose form; NETLIB: the host loop),
// gesv (device: tiny; both: blocked, which runs the masked getrf and getrs), getri and inv (NETLIB:
// the host loop). Every pattern runs on a device-only mask, complex data, padded ld and stride; an
// inactive item's B, X, pivots, info and C are bit-identical to the poisoned input. getri's
// blocked family and inv on a device keep refusing until trsm honours the mask.
// evidence: docs/design/masked-batches.md#masked-batches-tests
#include <gtest/gtest.h>

#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/functions/gesv.hh>
#include <batchlas/blas/functions/getrf.hh>
#include <batchlas/blas/functions/getri.hh>
#include <batchlas/blas/functions/getrs.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-vector.hh>

#include <src/kernels/common/solve_native.hh>
#include <src/ops/getrf/getrf_native.hh>
#include <src/ops/gesv/choice.hh>
#include <src/ops/getri/choice.hh>
#include <src/ops/getrs/choice.hh>
#include <src/select/vendor.hh>
#include <src/util/resident_capacity.hh>
#include <tests/masked_contract.hh>
#include <tests/test_support/storage.hh>
#include <tests/test_utils.hh>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

using namespace batchlas;

namespace {

namespace mc = masked_contract;

struct Shape {
    std::string spelling;
    int n, nrhs, batch;
    Transpose trans = Transpose::NoTrans;
    int wg_items = 0;
    bool identical = false;
    bool singular = true;  ///< item 5 has a zero third column (info = 3)
};

std::string label(const Shape& s) {
    const char* t[3] = {"N", "T", "C"};
    return s.spelling + " " + t[int(s.trans)] + " n " + std::to_string(s.n) + " nrhs " + std::to_string(s.nrhs) +
           " batch " + std::to_string(s.batch);
}

inline double spread(std::size_t item, std::size_t e, int salt) {
    std::uint64_t h = (item * 0x9E3779B97F4A7C15ull) ^ (e * 0xC2B2AE3D27D4EB4Full) ^ std::uint64_t(salt * 7919);
    h ^= h >> 29;
    h *= 0xBF58476D1CE4E5B9ull;
    h ^= h >> 32;
    return double(int(h % 2001) - 1000) / 1000.0;
}

template <typename T, Backend B>
struct Problem {
    Queue& q;
    Shape s;
    Matrix<T> A, Bm, C;
    UnifiedVector<std::int64_t> piv;
    UnifiedVector<std::int32_t> info;
    std::vector<T> lu;                 ///< A's LU factors (getrs, getri inputs), one unmasked getrf
    std::vector<std::int64_t> lu_piv;  ///< and their pivots, in this backend's format

    Problem(Queue& queue, const Shape& shape)
        : q(queue), s(shape), A(test_support::padded<T>(s.n, s.n, s.batch)),
          Bm(test_support::padded<T>(s.n, s.nrhs, s.batch)), C(test_support::padded<T>(s.n, s.n, s.batch)),
          piv(std::size_t(s.n) * s.batch), info(std::size_t(s.batch)) {}

    static constexpr bool kInt64Pivots = B == Backend::NETLIB;

    void fill(Matrix<T>& M, int cols, int salt, bool singular) {
        T* p = M.data_ptr();
        const std::size_t per = std::size_t(M.stride()), ld = std::size_t(M.ld());
        std::fill_n(p, per * s.batch, test_support::poison<T>());
        for (std::size_t b = 0; b < std::size_t(s.batch); ++b)
            for (std::size_t j = 0; j < std::size_t(cols); ++j)
                for (std::size_t i = 0; i < std::size_t(s.n); ++i) {
                    const std::size_t item = s.identical ? 0 : b, e = i + j * ld;
                    const bool zero = singular && s.singular && !s.identical && b == 5 && j == 2;
                    p[b * per + e] = zero ? T(0) : MatrixView<T>::make(spread(item, e, salt), spread(item, e, salt + 1));
                }
    }
    void poison_outputs() {
        std::fill(piv.begin(), piv.end(), std::int64_t(0x5A5A5A5A5A5A5A5A));
        std::fill(info.begin(), info.end(), std::int32_t(-77));
    }
    // The factors every getrs / getri pattern starts from: one unmasked getrf on this backend.
    void factor() {
        fill(A, s.n, 1, true);
        poison_outputs();
        UnifiedVector<std::byte> ws(std::max<std::size_t>(getrf_buffer_size<B, T>(q, A.view()), 1));
        getrf<B, T>(q, A.view(), piv.to_span(), ws.to_span(), info.to_span());
        q.wait();
        lu.assign(A.data_ptr(), A.data_ptr() + std::size_t(A.stride()) * s.batch);
        lu_piv.assign(piv.begin(), piv.end());
    }
    void restore_factors() {
        std::copy(lu.begin(), lu.end(), A.data_ptr());
        std::copy(lu_piv.begin(), lu_piv.end(), piv.begin());
    }
    mc::Output pivots() const {
        const std::size_t n = std::size_t(s.n);
        if constexpr (kInt64Pivots) return mc::Output::span<std::int64_t>("pivots", {piv.data(), piv.size()}, n);
        return mc::Output::span<std::int32_t>("pivots", {reinterpret_cast<const std::int32_t*>(piv.data()), n * s.batch},
                                              n);
    }
    mc::Output infos() const { return mc::Output::span<std::int32_t>("info", {info.data(), info.size()}, 1); }
    mc::Case base_case() {
        mc::Case c;
        c.batch = s.batch;
        c.wg_items = s.wg_items;
        c.identical_items = s.identical;
        return c;
    }

    mc::Case getrs_case() {
        factor();
        mc::Case c = base_case();
        c.prepare = [this] {
            restore_factors();
            fill(Bm, s.nrhs, 5, false);
        };
        c.call = [this](Span<const std::uint8_t> m, Span<std::byte> ws) {
            getrs<B, T>(q, A.view().with_batch_mask(m), Bm.view().with_batch_mask(m), s.trans, piv.to_span(), ws);
        };
        c.buffer_size = [this](Span<const std::uint8_t> m) {
            return getrs_buffer_size<B, T>(q, A.view().with_batch_mask(m), Bm.view().with_batch_mask(m), s.trans);
        };
        c.outputs = [this] {
            return std::vector<mc::Output>{mc::Output::matrix("B", Bm.view()), mc::Output::matrix("A", A.view()),
                                           pivots()};
        };
        return c;
    }
    mc::Case gesv_case() {
        mc::Case c = base_case();
        c.prepare = [this] {
            fill(A, s.n, 1, true);
            fill(Bm, s.nrhs, 5, false);
            poison_outputs();
        };
        c.call = [this](Span<const std::uint8_t> m, Span<std::byte> ws) {
            gesv<B, T>(q, A.view().with_batch_mask(m), Bm.view().with_batch_mask(m), piv.to_span(), ws, info.to_span());
        };
        c.buffer_size = [this](Span<const std::uint8_t> m) {
            return gesv_buffer_size<B, T>(q, A.view().with_batch_mask(m), Bm.view().with_batch_mask(m));
        };
        c.outputs = [this] {
            return std::vector<mc::Output>{mc::Output::matrix("A", A.view()), mc::Output::matrix("B", Bm.view()),
                                           pivots(), infos()};
        };
        return c;
    }
    mc::Case getri_case() {
        factor();
        mc::Case c = base_case();
        c.prepare = [this] {
            restore_factors();
            std::fill_n(C.data_ptr(), std::size_t(C.stride()) * s.batch, test_support::poison<T>());
            std::fill(info.begin(), info.end(), std::int32_t(-77));
        };
        c.call = [this](Span<const std::uint8_t> m, Span<std::byte> ws) {
            getri<B, T>(q, A.view().with_batch_mask(m), C.view().with_batch_mask(m), piv.to_span(), ws, info.to_span());
        };
        c.buffer_size = [this](Span<const std::uint8_t> m) {
            return getri_buffer_size<B, T>(q, A.view().with_batch_mask(m));
        };
        c.outputs = [this] {
            return std::vector<mc::Output>{mc::Output::matrix("C", C.view()), mc::Output::matrix("A", A.view()),
                                           infos()};
        };
        return c;
    }
    mc::Case inv_case() {
        mc::Case c = base_case();
        c.prepare = [this] {
            fill(A, s.n, 1, false);
            std::fill_n(C.data_ptr(), std::size_t(C.stride()) * s.batch, test_support::poison<T>());
        };
        c.call = [this](Span<const std::uint8_t> m, Span<std::byte> ws) {
            inv<B, T>(q, A.view().with_batch_mask(m), C.view().with_batch_mask(m), ws);
        };
        c.buffer_size = [this](Span<const std::uint8_t> m) { return inv_buffer_size<B, T>(q, A.view().with_batch_mask(m)); };
        c.outputs = [this] {
            return std::vector<mc::Output>{mc::Output::matrix("Ainv", C.view()), mc::Output::matrix("A", A.view())};
        };
        return c;
    }
};

template <typename Config>
class LuSolveMask : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    static constexpr Backend B = Config::BackendVal;

    bool device() const { return this->ctx->device().type == DeviceType::GPU && B != Backend::NETLIB; }
    bool host_loop() const { return B == Backend::NETLIB && select::kHasNetlib; }

    void getrs_check(const Shape& s) {
        SCOPED_TRACE("getrs " + label(s));
        Problem<T, B> p(*this->ctx, s);
        mc::check<ops::getrs::GetrsChoice>(*this->ctx, "getrs", s.spelling, p.getrs_case());
    }
    void gesv_check(const Shape& s) {
        SCOPED_TRACE("gesv " + label(s));
        Problem<T, B> p(*this->ctx, s);
        mc::check<ops::gesv::GesvChoice>(*this->ctx, "gesv", s.spelling, p.gesv_case());
    }
    static int tiny_items(int n) { return 64 / (n <= 8 ? 8 : n <= 16 ? 16 : 32); }
};

using Types = typename test_utils::backend_types<test_support::Config>::type;
TYPED_TEST_SUITE(LuSolveMask, Types);

TYPED_TEST(LuSolveMask, GetrsEveryFamilyLeavesInactiveItemsUntouched) {
    std::string fam;
    if (this->device()) fam = "cta";
    else if (this->host_loop()) fam = "vendor";
    else GTEST_SKIP() << "no masked getrs family here";
    for (Transpose t : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
        this->getrs_check({fam, 9, 3, 12, t});
        this->getrs_check({fam, 33, 1, 8, t});
    }
}

TYPED_TEST(LuSolveMask, GesvEveryFamilyLeavesInactiveItemsUntouched) {
    using T = typename TestFixture::T;
    if (!this->device() && !this->host_loop()) GTEST_SKIP() << "no masked gesv family here";
    if (this->device()) {
        for (int n : {5, 13, 27}) {
            if (n > sycl_gesv::gesv_tiny_max_n<T>()) continue;
            this->gesv_check({"tiny", n, n == 13 ? 4 : 2, 24, Transpose::NoTrans, TestFixture::tiny_items(n)});
        }
    }
    // blocked = the masked getrf, then the masked getrs (each on its own Auto). On a device the
    // larger order is cta's ceiling: above it only getrf's blocked family runs, which waits for trsm.
    this->gesv_check({"blocked", 11, 3, 12});
    int big = 40;
    if (this->device()) {
        const std::size_t budget =
            resident::device_slm_budget(this->ctx->device().get_property(DeviceProperty::LOCAL_MEM_SIZE));
        big = std::min(big, sycl_getrf::getrf_cta_max_n_for_slm<T>(budget));
    }
    this->gesv_check({"blocked", big, 2, 8});
}

// The saturating batch (>= 1024, identical items): gesv tiny's packed partitions, getrs cta's tile.
TYPED_TEST(LuSolveMask, SaturatingBatchIsBitIdentical) {
    if (!this->device()) GTEST_SKIP() << "tiny and cta are device families";
    this->gesv_check({"tiny", 8, 2, 1024, Transpose::NoTrans, 8, true});
    this->gesv_check({"tiny", 20, 1, 1024, Transpose::NoTrans, 2, true});
    this->getrs_check({"cta", 24, 2, 1024, Transpose::NoTrans, 0, true});
    this->getrs_check({"cta", 24, 2, 1024, Transpose::ConjTrans, 0, true});
}

TYPED_TEST(LuSolveMask, GetriAndInvOnTheHostLoop) {
    if (!this->host_loop()) GTEST_SKIP() << "getri honours the mask on NETLIB only until trsm does";
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::B;
    Queue& q = *this->ctx;
    for (const Shape& s : {Shape{"vendor", 7, 0, 12}, Shape{"vendor", 18, 0, 8}}) {
        SCOPED_TRACE("getri " + label(s));
        Problem<T, B> p(q, s);
        mc::check<ops::getri::GetriChoice>(q, "getri", s.spelling, p.getri_case());
    }
    for (const Shape& s : {Shape{"inv", 6, 0, 12}, Shape{"inv", 15, 0, 8}}) {
        SCOPED_TRACE(label(s));
        Problem<T, B> p(q, s);
        mc::check(q, p.inv_case());
    }
}

// Families that keep refusing: getrs blocked and getri blocked (until trsm), a strided vendor
// (permanently). Each pin runs unmasked first, then the masked call throws and writes nothing.
TYPED_TEST(LuSolveMask, WaitingAndStridedFamiliesRefuseBeforeWriting) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::B;
    Queue& q = *this->ctx;
    struct Refusal {
        const char* op;
        std::string spelling;
    };
    std::vector<Refusal> v;
    if (this->device()) {
        v.push_back({"getrs", "blocked"});
        v.push_back({"getri", "blocked"});
    }
    if constexpr (B != Backend::NETLIB)
        if (select::factorization_vendor_available<B>) {
            v.push_back({"getrs", "vendor"});
            v.push_back({"getri", "vendor"});
        }
    if (v.empty()) GTEST_SKIP() << "nothing refuses on this backend";
    for (const Refusal& r : v) {
        SCOPED_TRACE(std::string(r.op) + " " + r.spelling);
        const Shape s{r.spelling, 12, 2, 4};
        Problem<T, B> p(q, s);
        const bool is_getrs = std::string(r.op) == "getrs";
        const mc::Case c = is_getrs ? p.getrs_case() : p.getri_case();
        auto pinned = [&](auto choice_tag, Span<const std::uint8_t> m, Span<std::byte> ws) {
            const select::ScopedPin<decltype(choice_tag)> pin(r.op, r.spelling, select::StrictPin{});
            c.call(m, ws);
            q.wait();
        };
        auto run = [&](Span<const std::uint8_t> m, Span<std::byte> ws) {
            if (is_getrs) pinned(ops::getrs::GetrsChoice{}, m, ws);
            else pinned(ops::getri::GetriChoice{}, m, ws);
        };
        UnifiedVector<std::byte> ws(1 << 20);
        c.prepare();
        ASSERT_NO_THROW(run({}, ws.to_span()));
        c.prepare();
        q.wait();
        const auto before = mc::detail::snapshot(c.outputs(), s.batch);
        const mc::DeviceMask mask(q, mc::patterns(s.batch)[2].bytes);
        EXPECT_ANY_THROW(run(mask.span(), ws.to_span()));
        EXPECT_TRUE(mc::detail::snapshot(c.outputs(), s.batch) == before) << "a refused pin wrote an output";
    }
}

}  // namespace
