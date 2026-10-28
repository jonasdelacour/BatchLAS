// SPDX-License-Identifier: MIT
// spmm honours a batch mask in its native family (W1b), pinned and reached, on
// tests/masked_contract.hh: a device-only mask, complex data, padded ld and stride, per-item CSR
// patterns with different nnz, beta != 0 and beta = 0. The gather body (NoTrans) owns every output
// element, so its active items are bit-identical to the unmasked call; the scale-and-scatter body
// (Trans, ConjTrans) accumulates with atomics, so its active items are judged by the backward error.
// evidence: docs/design/masked-batches.md#masked-batches-bit-identity
#include <gtest/gtest.h>

#include <batchlas/blas/functions/spmm.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/error.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/tolerance.hh>

#include <src/ops/spmm/choice.hh>
#include <src/select/vendor.hh>
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

template <typename T>
using Dense = MatrixView<T, MatrixFormat::Dense>;

struct Shape {
    int m, k, nrhs, batch;
    Transpose ta = Transpose::NoTrans, tb = Transpose::NoTrans;
    bool identical = false;
};

std::string label(const Shape& s) {
    const char* w[3] = {"N", "T", "C"};
    return std::string(w[int(s.ta)]) + w[int(s.tb)] + " " + std::to_string(s.m) + "x" + std::to_string(s.k) + " nrhs " +
           std::to_string(s.nrhs) + " batch " + std::to_string(s.batch);
}

template <typename T, Backend B>
struct Problem {
    Queue& q;
    Shape s;
    T alpha, beta;
    int cap, out_rows, red_rows;
    UnifiedVector<T> val;
    UnifiedVector<int> ro, ci;
    Matrix<T> Bm, C, C0;

    Problem(Queue& queue, const Shape& shape, T beta_)
        : q(queue), s(shape), alpha(Dense<T>::make(0.75, -0.5)), beta(beta_), cap(3 * shape.m),
          out_rows(shape.ta == Transpose::NoTrans ? shape.m : shape.k),
          red_rows(shape.ta == Transpose::NoTrans ? shape.k : shape.m), val(std::size_t(cap) * shape.batch),
          ro(std::size_t(shape.m + 1) * shape.batch), ci(std::size_t(cap) * shape.batch),
          Bm(shape.tb == Transpose::NoTrans ? test_support::padded<T>(red_rows, shape.nrhs, shape.batch)
                                            : test_support::padded<T>(shape.nrhs, red_rows, shape.batch)),
          C(test_support::padded<T>(out_rows, shape.nrhs, shape.batch)),
          C0(test_support::padded<T>(out_rows, shape.nrhs, shape.batch)) {
        // Row i of item b holds 1 + (i + b) % 3 entries (identical items: b = 0), columns spread over k.
        for (int b = 0; b < s.batch; ++b) {
            const int pb = s.identical ? 0 : b;
            int* r = ro.data() + std::size_t(b) * (s.m + 1);
            r[0] = 0;
            for (int i = 0; i < s.m; ++i) {
                const int cnt = 1 + (i + pb) % 3;
                for (int j = 0; j < cnt; ++j) ci[std::size_t(b) * cap + r[i] + j] = (i + 5 * j + pb) % s.k;
                r[i + 1] = r[i] + cnt;
            }
        }
    }
    MatrixView<T, MatrixFormat::CSR> A() {
        return MatrixView<T, MatrixFormat::CSR>(val.data(), ro.data(), ci.data(), s.m, s.k, NonZeros{cap}, cap, s.m + 1,
                                                s.batch);
    }
    void fill(T* p, std::size_t per, int salt) {
        for (std::size_t e = 0; e < per * std::size_t(s.batch); ++e) {
            const std::size_t item = s.identical ? 0 : e / per, off = e % per;
            const double re = double(int((off * 7 + item * 3 + salt) % 23) - 11) / 8;
            const double im = double(int((off * 5 + item + salt) % 13) - 6) / 8;
            p[e] = Dense<T>::make(re, im);
        }
    }
    mc::Case make_case() {
        mc::Case c;
        c.batch = s.batch;
        c.identical_items = s.identical;
        c.prepare = [this] {
            fill(val.data(), std::size_t(cap), 1);
            fill(Bm.data_ptr(), std::size_t(Bm.stride()), 5);
            std::fill_n(C.data_ptr(), std::size_t(C.stride()) * s.batch, test_support::poison<T>());
            if (beta != T(0)) fill(C.data_ptr(), std::size_t(C.stride()), 9);
            std::copy_n(C.data_ptr(), std::size_t(C.stride()) * s.batch, C0.data_ptr());
        };
        c.call = [this](Span<const std::uint8_t> mask, Span<std::byte> ws) {
            spmm<B, T>(q, A().with_batch_mask(mask), Bm.view().with_batch_mask(mask), C.view().with_batch_mask(mask),
                       alpha, beta, s.ta, s.tb, ws);
        };
        c.buffer_size = [this](Span<const std::uint8_t> mask) {
            return spmm_buffer_size<B, T>(q, A().with_batch_mask(mask), Bm.view().with_batch_mask(mask),
                                          C.view().with_batch_mask(mask), alpha, beta, s.ta, s.tb);
        };
        c.outputs = [this] { return std::vector<mc::Output>{mc::Output::matrix("C", C.view())}; };
        if (s.ta != Transpose::NoTrans) {
            // Atomics: the active items against the library's componentwise backward error.
            c.verify_active = [this](const std::string&, std::span<const int> active) {
                if (active.empty()) return;
                const double err = Dense<T>::spmm_backward_error(q, A(), s.ta, Bm.view(), s.tb, C0.view(), C.view(),
                                                                 Dense<T>::up(alpha), Dense<T>::up(beta), active);
                EXPECT_TRUE(verify::pass<T>(verify::Check::blas, red_rows, err))
                    << "backward error " << err << " exceeds " << verify::bound<T>(verify::Check::blas, red_rows);
            };
        }
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
class SpmmMask : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    static constexpr Backend B = Config::BackendVal;

    // Pinned to direct and reached (unless `reach` is off), at beta != 0 and beta = 0.
    void check(const Shape& s, bool reach = true) {
        SCOPED_TRACE(label(s));
        Queue& q = *this->ctx;
        {
            Problem<T, B> p(q, s, Dense<T>::make(-0.25, 0.5));
            if (reach) {
                mc::check<ops::spmm::SpmmChoice>(q, "spmm", "direct", p.make_case());
            } else {
                const select::ScopedPin<ops::spmm::SpmmChoice> pin("spmm", "direct", select::StrictPin{});
                mc::check(q, p.make_case());
            }
        }
        const select::ScopedPin<ops::spmm::SpmmChoice> pin("spmm", "direct", select::StrictPin{});
        Problem<T, B> p(q, s, T(0));
        mc::check(q, p.make_case());
    }

    static std::vector<Shape> shapes(int batch, bool identical) {
        const auto N = Transpose::NoTrans, Tr = Transpose::Trans, C = Transpose::ConjTrans;
        std::vector<Shape> v{{11, 9, 2, batch, N, N},  {11, 9, 7, batch, N, C},  {10, 13, 5, batch, Tr, N},
                             {9, 12, 3, batch, C, Tr}, {9, 12, 1, batch, C, N}};
        for (Shape& s : v) s.identical = identical;
        return v;
    }
};

TYPED_TEST_SUITE(SpmmMask, Types);

// Both bodies: the gather at the pair block (nrhs <= 2) and the full block, the scale plus scatter
// under Trans and ConjTrans, every op(B).
TYPED_TEST(SpmmMask, DirectLeavesInactiveItemsUntouched) {
    for (const Shape& s : TestFixture::shapes(12, false)) this->check(s);
}

TYPED_TEST(SpmmMask, SaturatingIdenticalItems) {
    for (const Shape& s : TestFixture::shapes(1024, true)) this->check(s, false);
}

// A masked call never reaches the vendor: a strict vendor pin finds no candidate and throws
// before writing C.
TYPED_TEST(SpmmMask, VendorRefusesAMaskedCall) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::B;
    if constexpr (!select::has_library<B>(ops::spmm::spec.vendor)) {
        GTEST_SKIP() << "no vendor spmm in this build";
    } else {
        Queue& q = *this->ctx;
        const Shape s{11, 9, 4, 6};
        Problem<T, B> p(q, s, T(0));
        const mc::Case c = p.make_case();
        c.prepare();
        q.wait();
        const auto before = mc::detail::snapshot(c.outputs(), s.batch);
        const mc::DeviceMask mask(q, mc::patterns(s.batch)[2].bytes);
        const select::ScopedPin<ops::spmm::SpmmChoice> pin("spmm", "vendor", select::StrictPin{});
        try {
            c.call(mask.span(), {});
            q.wait();
            ADD_FAILURE() << "a vendor pin ran a masked call";
        } catch (const batchlas::invalid_argument& e) {
            EXPECT_NE(std::string(e.what()).find("run this shape"), std::string::npos) << e.what();
        }
        EXPECT_TRUE(mc::detail::snapshot(c.outputs(), s.batch) == before) << "the refused vendor pin wrote C";
    }
}

}  // namespace
