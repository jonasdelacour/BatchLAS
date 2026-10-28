// SPDX-License-Identifier: MIT
// steqr honours a batch mask (W3a) on tests/masked_contract.hh: the CTA kernel (n <= 32 where the
// device has 32-wide sub-groups, an I2 fold) and the work-group kernel (n > 32, or no such
// sub-groups), through steqr() and steqr_cta(). Every pattern on a device-only mask, padded d, e, w
// and eigenvector storage, jobz V and N, sorted and unsorted, info; the mask on every view and on
// d alone (the others adopt it). The heavy tier adds the saturating identical-item batch.
// evidence: docs/design/masked-batches.md#masked-batches-tests
#include <gtest/gtest.h>

#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-vector.hh>

#include <tests/masked_contract.hh>
#include <tests/test_support/storage.hh>
#include <tests/test_utils.hh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

using namespace batchlas;

namespace {

namespace mc = masked_contract;

struct Shape {
    int n, batch;
    JobType jobz = JobType::EigenVectors;
    bool sort = true;
    bool cta = false;          ///< steqr_cta() rather than steqr()
    bool d_only = false;       ///< the mask on d alone
    bool identical = false;    ///< every item the same input (the saturating case)
    int wg_items = 0;          ///< items per CTA work-group: the packed-hole patterns
};

std::string label(const Shape& s) {
    return std::string(s.cta ? "steqr_cta" : "steqr") + " n " + std::to_string(s.n) + " batch " +
           std::to_string(s.batch) + (s.jobz == JobType::EigenVectors ? " V" : " N") + (s.sort ? " sorted" : "") +
           (s.d_only ? " d-only" : "");
}

// A padded strided vector: n values per item at inc 1, `pad` slots apart beyond n.
template <typename T>
struct Strided {
    UnifiedVector<T> store;
    int n, batch, stride;
    Strided(int n_, int batch_, int pad) : store(std::size_t(std::max(1, n_ + pad)) * batch_), n(n_), batch(batch_), stride(n_ + pad) {}
    VectorView<T> view() const { return VectorView<T>(store.data(), n, batch, 1, stride); }
    mc::Output output(std::string name) const {
        return {std::move(name), reinterpret_cast<const std::byte*>(store.data()), sizeof(T) * std::size_t(n),
                sizeof(T) * std::size_t(stride)};
    }
};

template <typename T, Backend B>
struct Problem {
    Queue& q;
    Shape s;
    SteqrParams<T> params;
    Strided<T> d, e, w;
    Matrix<T> Z;
    UnifiedVector<int32_t> info;

    Problem(Queue& queue, const Shape& shape)
        : q(queue), s(shape), d(s.n, s.batch, 3), e(s.n - 1, s.batch, 2), w(s.n, s.batch, 5),
          Z(test_support::padded<T>(s.n, s.n, s.batch)), info(std::size_t(s.batch)) {
        params.sort = s.sort;
    }

    // Graded diagonal, distinct per item, every off-diagonal nonzero: a full sweep count.
    void prepare() {
        for (int b = 0; b < s.batch; ++b) {
            const int item = s.identical ? 0 : b;
            for (int i = 0; i < s.n; ++i)
                d.store[std::size_t(b) * d.stride + i] = T(std::ldexp(1.0 + 0.1 * ((i * 7 + item * 3) % 11), -(i % 6)));
            for (int i = 0; i + 1 < s.n; ++i)
                e.store[std::size_t(b) * e.stride + i] = T(0.25 + 0.05 * ((i * 5 + item) % 7));
        }
        std::fill_n(w.store.data(), w.store.size(), test_support::poison<T>());
        std::fill_n(Z.data_ptr(), std::size_t(Z.stride()) * s.batch, test_support::poison<T>());
        std::fill_n(info.data(), info.size(), int32_t(-7));
    }

    size_t buffer(Span<const std::uint8_t> m) {
        return steqr_buffer_size<T>(q, d.view().with_batch_mask(m), e.view().with_batch_mask(m), w.view().with_batch_mask(m),
                                    s.jobz, params);
    }

    void run(Span<const std::uint8_t> m, Span<std::byte> ws) {
        const Span<const std::uint8_t> rest = s.d_only ? Span<const std::uint8_t>{} : m;
        const auto Zv = Z.view().with_batch_mask(rest);
        const Span<int32_t> inf(info.data(), info.size());
        if (s.cta)
            (void)steqr_cta<B, T>(q, d.view().with_batch_mask(m), e.view().with_batch_mask(rest), w.view().with_batch_mask(rest),
                                  ws, s.jobz, params, Zv, inf);
        else
            (void)steqr<B, T>(q, d.view().with_batch_mask(m), e.view().with_batch_mask(rest), w.view().with_batch_mask(rest),
                              ws, s.jobz, params, Zv, inf);
    }

    mc::Case make_case() {
        mc::Case c;
        c.batch = s.batch;
        c.wg_items = s.wg_items;
        c.identical_items = s.identical;
        c.prepare = [this] { prepare(); };
        c.call = [this](Span<const std::uint8_t> m, Span<std::byte> ws) { run(m, ws); };
        c.buffer_size = [this](Span<const std::uint8_t> m) { return buffer(m); };
        c.outputs = [this] {
            std::vector<mc::Output> o{w.output("w"), mc::Output::span<int32_t>("info", {info.data(), info.size()}, 1),
                                      d.output("d"), e.output("e")};
            if (s.jobz == JobType::EigenVectors) o.push_back(mc::Output::matrix("Z", Z.view()));
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
using Types = typename test_utils::backend_types_filtered<Cfg, false>::type;

template <typename Config>
class SteqrMask : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    static constexpr Backend B = Config::BackendVal;

    void check(const Shape& s) {
        SCOPED_TRACE(label(s));
        Problem<T, B> p(*this->ctx, s);
        mc::check(*this->ctx, p.make_case());
    }
    // Items per CTA work-group at n (partition P, 32 or 64 lanes): the packed holes.
    static int cta_items(int n) {
        const int P = n <= 4 ? 4 : n <= 8 ? 8 : n <= 16 ? 16 : 32;
        const int mul = (std::is_same_v<T, float> && (P == 8 || P == 16)) ? 2 : 1;
        return std::max(32, P) * mul / P;
    }
};

TYPED_TEST_SUITE(SteqrMask, Types);

TYPED_TEST(SteqrMask, CtaEveryPattern) {
    for (int n : {5, 8, 16, 32})
        for (JobType jobz : {JobType::EigenVectors, JobType::NoEigenVectors})
            this->check({n, 37, jobz, true, false, false, false, this->cta_items(n)});
    this->check({12, 37, JobType::EigenVectors, false, true, false, false, this->cta_items(12)});
    this->check({12, 37, JobType::EigenVectors, true, false, true, false, this->cta_items(12)});
}

TYPED_TEST(SteqrMask, WorkGroupEveryPattern) {
    for (JobType jobz : {JobType::EigenVectors, JobType::NoEigenVectors})
        for (bool sort : {true, false}) this->check({40, 9, jobz, sort});
    this->check({40, 9, JobType::EigenVectors, true, false, true});
    this->check({1, 6, JobType::EigenVectors, true});
}

TYPED_TEST(SteqrMask, SaturatingIdenticalItems) {
    this->check({16, 1024, JobType::EigenVectors, true, false, false, true, this->cta_items(16)});
    this->check({33, 1024, JobType::NoEigenVectors, true, false, false, true});
}

}  // namespace
