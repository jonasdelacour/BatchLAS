// SPDX-License-Identifier: MIT
// stebz, stein and tridiagonal_solver honour a batch mask (W3a, one work-group per item: an I1
// return before the first collective) on tests/masked_contract.hh. Every pattern on a device-only
// mask, padded strided d, e and w, padded eigenvector storage; stebz on the All, Index and Value
// ranges with its count m; stein with and without per-item counts; the mask on every view and on
// one view alone (the others adopt it). The heavy tier adds the saturating identical-item batch.
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
#include <string>
#include <vector>

using namespace batchlas;

namespace {

namespace mc = masked_contract;

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
    T& at(int i, int b) { return store[std::size_t(b) * stride + i]; }
};

// Graded diagonal, distinct per item unless `identical`, every off-diagonal nonzero.
template <typename T>
void fill_tridiagonal(Strided<T>& d, Strided<T>& e, bool identical) {
    for (int b = 0; b < d.batch; ++b) {
        const int item = identical ? 0 : b;
        for (int i = 0; i < d.n; ++i) d.at(i, b) = T(1 + i + 0.125 * ((i * 7 + item * 3) % 11));
        for (int i = 0; i + 1 < d.n; ++i) e.at(i, b) = T(0.25 + 0.0625 * ((i * 5 + item) % 7));
    }
}

template <typename T, Backend B>
struct Stebz {
    Queue& q;
    int n, batch;
    StebzParams<T> params;
    bool identical, d_only;
    Strided<T> d, e, w;
    UnifiedVector<int32_t> m;

    Stebz(Queue& queue, int n_, int batch_, StebzParams<T> p, bool ident = false, bool d_only_ = false)
        : q(queue), n(n_), batch(batch_), params(p), identical(ident), d_only(d_only_), d(n_, batch_, 3), e(n_ - 1, batch_, 2),
          w(n_, batch_, 4), m(std::size_t(batch_)) {}

    mc::Case make_case() {
        mc::Case c;
        c.batch = batch;
        c.identical_items = identical;
        c.prepare = [this] {
            fill_tridiagonal(d, e, identical);
            std::fill_n(w.store.data(), w.store.size(), test_support::poison<T>());
            std::fill_n(m.data(), m.size(), int32_t(-7));
        };
        c.call = [this](Span<const std::uint8_t> mask, Span<std::byte> ws) {
            const Span<const std::uint8_t> rest = d_only ? Span<const std::uint8_t>{} : mask;
            (void)stebz<B, T>(q, d.view().with_batch_mask(mask), e.view().with_batch_mask(rest), w.view().with_batch_mask(rest),
                              Span<int32_t>(m.data(), m.size()), ws, params);
        };
        c.buffer_size = [this](Span<const std::uint8_t>) { return stebz_buffer_size<B, T>(q, n, batch, params); };
        c.outputs = [this] {
            return std::vector<mc::Output>{w.output("w"), mc::Output::span<int32_t>("m", {m.data(), m.size()}, 1),
                                           d.output("d"), e.output("e")};
        };
        return c;
    }
};

template <typename T, Backend B>
struct Stein {
    Queue& q;
    int n, k, batch;
    bool counts_on, identical, z_only;
    Strided<T> d, e, w;
    Matrix<T> Z;
    UnifiedVector<int32_t> counts;

    Stein(Queue& queue, int n_, int k_, int batch_, bool counts_, bool ident = false, bool z_only_ = false)
        : q(queue), n(n_), k(k_), batch(batch_), counts_on(counts_), identical(ident), z_only(z_only_), d(n_, batch_, 3),
          e(n_ - 1, batch_, 2), w(k_, batch_, 4), Z(test_support::padded<T>(n_, k_, batch_)), counts(std::size_t(batch_)) {}

    mc::Case make_case() {
        mc::Case c;
        c.batch = batch;
        c.identical_items = identical;
        c.prepare = [this] {
            fill_tridiagonal(d, e, identical);
            // Ascending shifts near the diagonal, a close pair among them: the reorthogonalisation runs.
            for (int b = 0; b < batch; ++b) {
                for (int j = 0; j < k; ++j) w.at(j, b) = d.at(j, b) + T(j == 1 ? -0.5 : 0.01 * j);
                std::sort(&w.at(0, b), &w.at(0, b) + k);
                counts[std::size_t(b)] = identical ? k : k - (b % 3);
            }
            std::fill_n(Z.data_ptr(), std::size_t(Z.stride()) * batch, test_support::poison<T>());
        };
        c.call = [this](Span<const std::uint8_t> mask, Span<std::byte> ws) {
            const Span<const std::uint8_t> in = z_only ? Span<const std::uint8_t>{} : mask;
            const auto Zv = Z.view().with_batch_mask(mask);
            const Span<const int32_t> cnt = counts_on ? Span<const int32_t>(counts.data(), counts.size()) : stein_all_counts;
            (void)stein<B, T>(q, d.view().with_batch_mask(in), e.view().with_batch_mask(in), w.view().with_batch_mask(in), k, cnt,
                              Zv, ws, SteinParams<T>{});
        };
        c.buffer_size = [this](Span<const std::uint8_t>) { return stein_buffer_size<B, T>(q, n, k, batch, SteinParams<T>{}); };
        c.outputs = [this] {
            return std::vector<mc::Output>{mc::Output::matrix("Z", Z.view()), d.output("d"), e.output("e"), w.output("w")};
        };
        return c;
    }
};

template <typename T, Backend B>
struct TridiagSolver {
    Queue& q;
    int n, batch;
    JobType jobz;
    bool identical;
    UnifiedVector<T> alphas, betas, W;
    Matrix<T> Q;

    TridiagSolver(Queue& queue, int n_, int batch_, JobType jz, bool ident = false)
        : q(queue), n(n_), batch(batch_), jobz(jz), identical(ident), alphas(std::size_t(n_) * batch_),
          betas(std::size_t(n_) * batch_), W(std::size_t(n_) * batch_), Q(test_support::padded<T>(n_, n_, batch_)) {}

    mc::Case make_case() {
        mc::Case c;
        c.batch = batch;
        c.identical_items = identical;
        c.prepare = [this] {
            for (int b = 0; b < batch; ++b) {
                const int item = identical ? 0 : b;
                for (int i = 0; i < n; ++i) {
                    alphas[std::size_t(b) * n + i] = T(1 + i + 0.125 * ((i * 7 + item * 3) % 11));
                    betas[std::size_t(b) * n + i] = i + 1 < n ? T(0.25 + 0.0625 * ((i * 5 + item) % 7)) : T(0);
                }
            }
            std::fill_n(W.data(), W.size(), test_support::poison<T>());
            std::fill_n(Q.data_ptr(), std::size_t(Q.stride()) * batch, test_support::poison<T>());
        };
        c.call = [this](Span<const std::uint8_t> mask, Span<std::byte> ws) {
            (void)tridiagonal_solver<B, T>(q, Span<T>(alphas.data(), alphas.size()), Span<T>(betas.data(), betas.size()),
                                           Span<T>(W.data(), W.size()), ws, jobz, Q.view().with_batch_mask(mask), n, batch);
        };
        c.buffer_size = [this](Span<const std::uint8_t>) { return tridiagonal_solver_buffer_size<B, T>(q, n, batch, jobz); };
        c.outputs = [this] {
            std::vector<mc::Output> o{mc::Output::span<T>("W", {W.data(), W.size()}, std::size_t(n)),
                                      mc::Output::span<T>("alphas", {alphas.data(), alphas.size()}, std::size_t(n)),
                                      mc::Output::span<T>("betas", {betas.data(), betas.size()}, std::size_t(n))};
            if (jobz == JobType::EigenVectors) o.push_back(mc::Output::matrix("Q", Q.view()));
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
class TridiagMask : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    static constexpr Backend B = Config::BackendVal;
};

TYPED_TEST_SUITE(TridiagMask, Types);

TYPED_TEST(TridiagMask, StebzEveryPattern) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::B;
    Queue& q = *this->ctx;
    StebzParams<T> all, index, value, desc;
    index.range = EigenRangeType::Index, index.il = 2, index.iu = 6;
    value.range = EigenRangeType::Value, value.vl = T(3), value.vu = T(9);
    desc.order = SortOrder::Descending;
    for (const auto& p : {all, index, value, desc}) {
        SCOPED_TRACE("stebz range " + std::to_string(int(p.range)) + " order " + std::to_string(int(p.order)));
        Stebz<T, B> s(q, 12, 9, p);
        mc::check(q, s.make_case());
    }
    SCOPED_TRACE("stebz d-only mask");
    Stebz<T, B> s(q, 12, 9, value, false, true);
    mc::check(q, s.make_case());
}

TYPED_TEST(TridiagMask, SteinEveryPattern) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::B;
    Queue& q = *this->ctx;
    for (bool counts : {false, true})
        for (bool z_only : {false, true}) {
            SCOPED_TRACE(std::string("stein") + (counts ? " counts" : "") + (z_only ? " Z-only" : ""));
            Stein<T, B> s(q, 12, 5, 9, counts, false, z_only);
            mc::check(q, s.make_case());
        }
}

TYPED_TEST(TridiagMask, TridiagonalSolverEveryPattern) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::B;
    Queue& q = *this->ctx;
    for (JobType jobz : {JobType::EigenVectors, JobType::NoEigenVectors}) {
        SCOPED_TRACE(jobz == JobType::EigenVectors ? "tridiagonal_solver V" : "tridiagonal_solver N");
        TridiagSolver<T, B> s(q, 10, 9, jobz);
        mc::check(q, s.make_case());
    }
}

TYPED_TEST(TridiagMask, SaturatingIdenticalItems) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::B;
    Queue& q = *this->ctx;
    {
        Stebz<T, B> s(q, 16, 1024, StebzParams<T>{}, true);
        mc::check(q, s.make_case());
    }
    {
        // stein seeds each item's start vector by its index, so equal items are not equal outputs:
        // the batch runs, without the identical-item check.
        Stein<T, B> s(q, 16, 4, 1024, true, false);
        mc::check(q, s.make_case());
    }
    TridiagSolver<T, B> s(q, 16, 1024, JobType::EigenVectors, true);
    mc::check(q, s.make_case());
}

}  // namespace
