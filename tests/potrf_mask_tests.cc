// SPDX-License-Identifier: MIT
// The Cholesky group honours a batch mask (W2a, and the lift pass), each family pinned and reached,
// on tests/masked_contract.hh: potrf's tiny, cta, lpanel and blocked, posv's tiny, cta and blocked,
// potrs' fused and trsm, potri's cta (either backend) and blocked, and lascl. Every pattern (packed
// work-group holes for the packed kernels, part edges and a wholly inactive part for the chunked
// drivers), a device-only mask, complex data, padded ld and stride, failing items (info, potri's
// untouched zero-diagonal item), both triangles; the heavy tier adds the saturating identical-item
// batch. The vendor refuses a masked call and writes nothing. evidence: docs/design/masked-batches.md#masked-batches-tests
#include <gtest/gtest.h>

#include <batchlas/blas/functions/posv.hh>
#include <batchlas/blas/functions/potrf.hh>
#include <batchlas/blas/functions/potri.hh>
#include <batchlas/blas/functions/potrs.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/error.hh>
#include <batchlas/util/sycl-vector.hh>

#include <src/kernels/common/solve_native.hh>
#include <src/ops/posv/choice.hh>
#include <src/ops/potrf/choice.hh>
#include <src/ops/potrf/potrf_native.hh>
#include <src/ops/potri/choice.hh>
#include <src/ops/potrs/choice.hh>
#include <src/select/vendor.hh>
#include <tests/masked_contract.hh>
#include <tests/test_support/inputs.hh>
#include <tests/test_support/storage.hh>
#include <tests/test_utils.hh>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <variant>
#include <vector>

using namespace batchlas;

namespace {

namespace mc = masked_contract;
namespace ts = test_support;

enum class Op { potrf, posv, potri, potrs };

struct Shape {
    Op op;
    std::string spelling;  ///< the pinned choice
    int n, nrhs, batch;
    Uplo uplo = Uplo::Lower;
    int wg_items = 0;        ///< items per packed work-group
    bool identical = false;  ///< every item the same input (the saturating case)
    int chunk = 0;           ///< forced part size (BATCHLAS_TUNE_L2_CHUNK_ITEMS) for the chunked drivers
    bool only_b = false;     ///< the mask rides on B alone: A adopts it (posv's potrf child sees A first)
};

const char* op_name(Op o) {
    return o == Op::potrf ? "potrf" : o == Op::posv ? "posv" : o == Op::potri ? "potri" : "potrs";
}

std::string label(const Shape& s) {
    return std::string(op_name(s.op)) + " " + s.spelling + " " + ts::uplo_name(s.uplo) + " n " + std::to_string(s.n) +
           " nrhs " + std::to_string(s.nrhs) + " batch " + std::to_string(s.batch);
}

// An item that fails: not positive definite (potrf, posv), or a factor with a zero pivot (potri).
bool failing(const Shape& s, int b) { return !s.identical && b % 5 == 3; }

template <typename T, Backend B>
struct Problem {
    Queue& q;
    Shape s;
    Matrix<T> A, Bm;
    std::vector<T> a_image, b_image;
    UnifiedVector<int32_t> info;

    Problem(Queue& queue, const Shape& shape)
        : q(queue), s(shape), A(ts::padded<T>(s.n, s.n, s.batch)), Bm(ts::padded<T>(s.n, std::max(s.nrhs, 1), s.batch)),
          info(static_cast<std::size_t>(s.batch)) {
        Matrix<T> src = ts::hpd<T>(q, s.n, s.identical ? 1 : s.batch, 0, 17);
        // potri and potrs take a factor: the unmasked, unpinned potrf of the same items.
        if (s.op == Op::potri || s.op == Op::potrs) {
            UnifiedVector<int32_t> fi(static_cast<std::size_t>(src.batch_size()));
            ts::potrf_in_place<B, T>(q, src.view(), s.uplo, fi.to_span());
        }
        ts::store(A.view(), src.view());
        T* a = A.data_ptr();
        for (int b = 0; b < s.batch; ++b)
            if (failing(s, b)) {
                const int k = s.n / 2;
                a[std::size_t(b) * A.stride() + std::size_t(k) * A.ld() + k] = s.op == Op::potri ? T(0) : T(-1);
            }
        a_image.assign(a, a + std::size_t(A.stride()) * s.batch);
        T* bp = Bm.data_ptr();
        for (int b = 0; b < s.batch; ++b)
            for (int j = 0; j < Bm.cols(); ++j)
                for (int i = 0; i < s.n; ++i) {
                    const int item = s.identical ? 0 : b;
                    bp[std::size_t(b) * Bm.stride() + std::size_t(j) * Bm.ld() + i] =
                        MatrixView<T>::make(double((i * 7 + j * 3 + item) % 11) / 4 - 1, double((i + 2 * j + item) % 5) / 8);
                }
        b_image.assign(bp, bp + std::size_t(Bm.stride()) * s.batch);
    }

    MatrixView<T, MatrixFormat::Dense> a(Span<const std::uint8_t> m) {
        return A.view().with_batch_mask(s.only_b ? Span<const std::uint8_t>{} : m);
    }
    MatrixView<T, MatrixFormat::Dense> b(Span<const std::uint8_t> m) { return Bm.view().with_batch_mask(m); }

    mc::Case make_case() {
        mc::Case c;
        c.batch = s.batch;
        c.wg_items = s.wg_items;
        c.identical_items = s.identical;
        c.chunk_items = s.chunk;
        c.prepare = [this] {
            std::copy(a_image.begin(), a_image.end(), A.data_ptr());
            std::copy(b_image.begin(), b_image.end(), Bm.data_ptr());
            std::fill_n(info.data(), info.size(), int32_t(-7));
        };
        c.call = [this](Span<const std::uint8_t> m, Span<std::byte> ws) {
            switch (s.op) {
                case Op::potrf: potrf<B, T>(q, a(m), s.uplo, ws, info.to_span()); break;
                case Op::posv: posv<B, T>(q, a(m), b(m), s.uplo, ws, info.to_span()); break;
                case Op::potri: potri<B, T>(q, a(m), s.uplo, s.spelling != "cta:full=0", ws, info.to_span()); break;
                case Op::potrs: potrs<B, T>(q, a(m), b(m), s.uplo); break;
            }
        };
        if (s.op != Op::potrs)
            c.buffer_size = [this](Span<const std::uint8_t> m) -> std::size_t {
                switch (s.op) {
                    case Op::potrf: return potrf_buffer_size<B, T>(q, a(m), s.uplo);
                    case Op::posv: return posv_buffer_size<B, T>(q, a(m), b(m), s.uplo);
                    default: return potri_buffer_size<B, T>(q, a(m), s.uplo);
                }
            };
        c.outputs = [this] {
            std::vector<mc::Output> o{mc::Output::matrix("A", A.view())};
            if (s.op == Op::posv || s.op == Op::potrs) o.push_back(mc::Output::matrix("B", Bm.view()));
            if (s.op != Op::potrs) o.push_back(mc::Output::span("info", std::span<const int32_t>(info.data(), info.size()), 1));
            return o;
        };
        return c;
    }
};

template <typename T, Backend B>
struct Cfg {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};
using Types = typename test_utils::backend_types<Cfg>::type;

template <typename Config>
class PotrfMask : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    static constexpr Backend B = Config::BackendVal;

    bool gpu() const { return this->ctx && this->ctx->device().type == DeviceType::GPU; }

    // The pinned spelling: potri's full flag rides in the shape's spelling ("cta:full=0").
    static std::string pin_of(const Shape& s) { return s.spelling.starts_with("cta:full") ? "cta" : s.spelling; }

    template <typename Choice>
    void check(const Shape& s, bool reach = true) {
        SCOPED_TRACE(label(s));
        Queue& q = *this->ctx;
        Problem<T, B> p(q, s);
        // A masked call never reaches the vendor, so the composites' children take a native family
        // in the unmasked reference too: the same route, bit for bit.
        using Pin = select::ScopedPin<std::monostate>;
        const Pin c0("potrf", "native"), c1("trsm", "native"), c2("gemm", "native"), c3("trmm", "native"),
            c4("syrk", "native"), c5("herk", "native");
        if (reach) {
            mc::check<Choice>(q, op_name(s.op), pin_of(s), p.make_case());
        } else {
            const select::ScopedPin<Choice> pin(op_name(s.op), pin_of(s), select::StrictPin{});
            mc::check(q, p.make_case());
        }
    }

    // A family that refuses a masked call under a strict pin: the unmasked call ran, the masked
    // one throws (a refusal) and writes none of the outputs.
    template <typename Choice>
    void refuses(const Shape& s) {
        SCOPED_TRACE(label(s) + " refuses");
        Queue& q = *this->ctx;
        Problem<T, B> p(q, s);
        const mc::Case c = p.make_case();
        const select::ScopedPin<Choice> pin(op_name(s.op), pin_of(s), select::StrictPin{});
        UnifiedVector<std::byte> ws(std::size_t(1) << 22);
        c.prepare();
        ASSERT_NO_THROW({
            c.call({}, ws.to_span());
            q.wait();
        }) << "the unmasked call threw, so a masked throw proves nothing";
        c.prepare();
        q.wait();
        const auto before = mc::detail::snapshot(c.outputs(), s.batch);
        const mc::DeviceMask mask(q, mc::patterns(s.batch)[2].bytes);
        try {
            c.call(mask.span(), ws.to_span());
            q.wait();
            ADD_FAILURE() << "a masked call ran on a refusing family";
        } catch (const batchlas::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("run this shape"), std::string::npos) << e.what();
        }
        EXPECT_TRUE(mc::detail::snapshot(c.outputs(), s.batch) == before) << "the refused call wrote an output";
    }

    // potrf tiny's matrices per work-group (G | S << 16), cta's and lpanel's G.
    int tiny_wg(int n) {
        const unsigned v = sycl_potrf::potrf_tiny_debug_launch<T>(*this->ctx, n);
        return int(v & 0xffffu) * int(v >> 16);
    }
    int cta_wg(int n, int batch) { return int(sycl_potrf::potrf_cta_debug_launch<T>(*this->ctx, n, batch) & 0xffffu); }
    int lpanel_wg(int n, int batch, int nb) {
        return int(sycl_potrf::potrf_lpanel_debug_launch<T>(*this->ctx, n, batch, resident::kMinBlocksPerSm, nb) & 0xfu);
    }
    static int posv_wg(int n) { return sycl_posv::kPosvTinyWgSize / (n <= 8 ? 8 : n <= 16 ? 16 : 32); }

    std::vector<Shape> potrf_shapes(int batch, bool identical) {
        std::vector<Shape> v;
        for (const auto& [n, u] : {std::pair{5, Uplo::Lower}, {13, Uplo::Upper}, {27, Uplo::Lower}})
            if (n <= sycl_potrf::potrf_tiny_max_n<T>())
                v.push_back({Op::potrf, "tiny", n, 0, batch, u, tiny_wg(n), identical});
        v.push_back({Op::potrf, "cta", 11, 0, batch, Uplo::Lower, cta_wg(11, batch), identical});
        const int big = cta_wg(45, batch) ? 45 : 30;  // complex<double>'s local-memory ceiling is lower
        v.push_back({Op::potrf, "cta", big, 0, batch, Uplo::Upper, cta_wg(big, batch), identical});
        v.push_back({Op::potrf, "lpanel:panel=8", 19, 0, batch, Uplo::Lower, lpanel_wg(19, batch, 8), identical});
        if constexpr (std::is_same_v<T, float>)
            v.push_back({Op::potrf, "lpanel:panel=16", 40, 0, batch, Uplo::Lower, lpanel_wg(40, batch, 16), identical});
        return v;
    }
    std::vector<Shape> posv_shapes(int batch, bool identical) {
        std::vector<Shape> v;
        for (const auto& [n, r, u] : {std::tuple{6, 1, Uplo::Lower}, {14, 3, Uplo::Upper}, {30, 4, Uplo::Lower}})
            if (n <= sycl_posv::posv_tiny_max_n<T>()) v.push_back({Op::posv, "tiny", n, r, batch, u, posv_wg(n), identical});
        return v;
    }
};

TYPED_TEST_SUITE(PotrfMask, Types);

// Every potrf and posv native family's kernels, both triangles and every register bucket.
TYPED_TEST(PotrfMask, PackedAndCtaFamiliesLeaveInactiveItemsUntouched) {
    if (!this->gpu()) GTEST_SKIP() << "potrf's and posv's natives are GPU kernels";
    for (const Shape& s : this->potrf_shapes(24, false)) this->template check<ops::potrf::PotrfChoice>(s);
    for (const Shape& s : this->posv_shapes(24, false)) this->template check<ops::posv::PosvChoice>(s);
}

// potri's cta family (one work-group per item) on either backend, with and without the full mirror.
TYPED_TEST(PotrfMask, PotriCtaLeavesInactiveItemsUntouched) {
    this->template check<ops::potri::PotriChoice>({Op::potri, "cta", 9, 0, 12, Uplo::Lower});
    this->template check<ops::potri::PotriChoice>({Op::potri, "cta:full=0", 21, 0, 12, Uplo::Upper});
}

// lascl is scale's loop: every pattern on either backend (real scalars: lascl's instantiations).
TYPED_TEST(PotrfMask, LasclLeavesInactiveItemsUntouched) {
    using T = typename TestFixture::T;
    if constexpr (test_utils::is_complex<T>::value) {
        GTEST_SKIP() << "lascl is real only";
    } else {
        Queue& q = *this->ctx;
        Matrix<T> A = ts::padded<T>(7, 5, 10);
        std::vector<T> image(std::size_t(A.stride()) * 10);
        for (std::size_t e = 0; e < image.size(); ++e) image[e] = T(int(e % 17) - 8) / T(4);
        mc::Case c;
        c.batch = 10;
        c.prepare = [&] { std::copy(image.begin(), image.end(), A.data_ptr()); };
        c.call = [&](Span<const std::uint8_t> m, Span<std::byte>) {
            lascl<MatrixFormat::Dense, T>(q, A.view().with_batch_mask(m), T(3), T(-0.5));
        };
        c.outputs = [&] { return std::vector<mc::Output>{mc::Output::matrix("A", A.view())}; };
        mc::check(q, c);
    }
}

// potrf blocked and potri blocked chunk the batch: every part edge, a wholly inactive part, both
// triangles, a trailing update past one fold block (n > nb + W).
TYPED_TEST(PotrfMask, BlockedFamiliesLeaveInactiveItemsUntouched) {
    if (!this->gpu()) GTEST_SKIP() << "the blocked families are GPU drivers";
    this->template check<ops::potrf::PotrfChoice>({Op::potrf, "blocked", 100, 0, 24, Uplo::Lower, 0, false, 8});
    this->template check<ops::potrf::PotrfChoice>({Op::potrf, "blocked", 72, 0, 21, Uplo::Upper, 0, false, 5});
    this->template check<ops::potri::PotriChoice>({Op::potri, "blocked:nb=64", 150, 0, 24, Uplo::Lower, 0, false, 8});
    this->template check<ops::potri::PotriChoice>({Op::potri, "blocked:nb=64", 100, 0, 21, Uplo::Upper, 0, false, 5});
}

// The solves: posv's cta (potrf, then the fused solve) and blocked (potrf, then two trsm), and
// potrs' fused kernel and trsm pair, both triangles and every fused nrhs bucket.
TYPED_TEST(PotrfMask, SolveFamiliesLeaveInactiveItemsUntouched) {
    if (!this->gpu()) GTEST_SKIP() << "posv's and potrs' natives are GPU families";
    this->template check<ops::posv::PosvChoice>({Op::posv, "cta", 12, 2, 16, Uplo::Lower});
    this->template check<ops::posv::PosvChoice>({Op::posv, "cta", 37, 5, 16, Uplo::Upper});
    this->template check<ops::posv::PosvChoice>({Op::posv, "blocked", 70, 3, 16, Uplo::Upper});
    this->template check<ops::posv::PosvChoice>({Op::posv, "blocked", 9, 1, 16, Uplo::Lower});
    for (const auto& [n, r, u] : {std::tuple{6, 1, Uplo::Lower}, {19, 2, Uplo::Upper}, {33, 4, Uplo::Lower},
                                  {40, 7, Uplo::Upper}})
        this->template check<ops::potrs::PotrsChoice>({Op::potrs, "fused", n, r, 16, u});
    this->template check<ops::potrs::PotrsChoice>({Op::potrs, "trsm", 6, 2, 16, Uplo::Upper});
    this->template check<ops::potrs::PotrsChoice>({Op::potrs, "trsm", 50, 3, 16, Uplo::Lower});
    // The mask on B alone still covers A and info.
    for (const char* sp : {"cta", "blocked"})
        this->template check<ops::posv::PosvChoice>({Op::posv, sp, 21, 2, 16, Uplo::Lower, 0, false, 0, true});
    this->template check<ops::potrs::PotrsChoice>({Op::potrs, "fused", 21, 2, 16, Uplo::Upper, 0, false, 0, true});
    this->template check<ops::potrs::PotrsChoice>({Op::potrs, "trsm", 21, 2, 16, Uplo::Lower, 0, false, 0, true});
}

// The vendor's batched call cannot skip an item: a strict pin refuses the masked call before it
// writes (Auto walks on to a native family instead).
TYPED_TEST(PotrfMask, VendorRefuses) {
    constexpr Backend B = TestFixture::B;
    if constexpr (select::has_library<B>(ops::potrf::spec.vendor)) {
        if (!this->gpu()) GTEST_SKIP() << "the vendor's potrf is a GPU library";
        this->template refuses<ops::potrf::PotrfChoice>({Op::potrf, "vendor", 12, 0, 8, Uplo::Lower});
    } else {
        GTEST_SKIP() << "no vendor potrf on this backend";
    }
}

// Auto with a mask never picks the vendor: inactive items untouched, active ones a Cholesky factor.
TYPED_TEST(PotrfMask, AutoRunsAMaskedPotrfNatively) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::B;
    using MV = MatrixView<T, MatrixFormat::Dense>;
    if (!this->gpu()) GTEST_SKIP() << "potrf's natives are GPU kernels";
    Queue& q = *this->ctx;
    const Shape s{Op::potrf, "auto", 24, 0, 32, Uplo::Lower, 0, true};
    Problem<T, B> p(q, s);
    mc::Case c = p.make_case();
    const Matrix<T> A0 = ts::hpd<T>(q, s.n, 1, 0, 17);
    Matrix<T> ref(s.n, s.n, s.batch);
    ts::store(ref.view(), A0.view());
    c.verify_active = [&](const std::string&, std::span<const int> active) {
        if (active.empty()) return;
        for (int b : active) EXPECT_EQ(p.info.data()[b], 0) << "item " << b;
        const double err = MV::potrf_residual(q, ref.view(), p.A.view(), s.uplo, active);
        EXPECT_VERIFY(T, verify::Check::factorization, s.n, err);
    };
    mc::check(q, c);
}

// The saturating batch (>= 1024, identical items) for every packed and SLM kernel: holes in
// every packed work-group, active items bit-identical to each other and to the unmasked call.
TYPED_TEST(PotrfMask, SaturatingIdenticalItems) {
    if (!this->gpu()) GTEST_SKIP() << "potrf's and posv's natives are GPU kernels";
    for (const Shape& s : this->potrf_shapes(1024, true)) this->template check<ops::potrf::PotrfChoice>(s, false);
    for (const Shape& s : this->posv_shapes(1024, true)) this->template check<ops::posv::PosvChoice>(s, false);
    this->template check<ops::posv::PosvChoice>({Op::posv, "cta", 20, 3, 1024, Uplo::Lower, 0, true}, false);
    this->template check<ops::potrs::PotrsChoice>({Op::potrs, "fused", 24, 4, 1024, Uplo::Upper, 0, true}, false);
}

}  // namespace
