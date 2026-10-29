// SPDX-License-Identifier: MIT
// iluk honours a batch mask (W4c) on tests/masked_contract.hh: a device-only mask, complex data, a
// five-point pattern with level-1 fill, padded CSR strides, padded rhs and out. One call is
// factorize plus apply; its outputs are out, the factor values the workspace overload leaves in a
// poisoned workspace (an inactive item's keep their bytes), and the inputs A and rhs. Each path is
// pinned with BATCHLAS_ILUK_DEVICE. The host path reads the mask on the host, so only the host
// backend's CPU queue runs it masked; a device backend's host path refuses. The factor's pattern is
// the union over the active items: where drops differ per item it shrinks, and the active items are
// checked entry by entry against the unmasked call. evidence: docs/design/masked-batches.md#masked-batches-tests
#include <gtest/gtest.h>

#include <batchlas/blas/functions/iluk.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/error.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-vector.hh>

#include <tests/masked_contract.hh>
#include <tests/test_support/storage.hh>
#include <tests/test_utils.hh>

#include <complex>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <vector>

using namespace batchlas;

namespace {

namespace mc = masked_contract;

template <typename T>
using Dense = MatrixView<T, MatrixFormat::Dense>;
template <typename T>
using Csr = MatrixView<T, MatrixFormat::CSR>;

enum class Path { host, device };

struct Shape {
    int grid, batch, nrhs;
    Path path;
    bool owned = false;      ///< the ILUKPreconditioner overload (else the workspace one)
    bool identical = false;  ///< every item the same input
    bool drops = false;      ///< odd items drop entries even items keep: the pattern follows the mask
    bool singular = false;   ///< even items have a zero pivot and no shift: the unmasked call throws
};

std::string label(const Shape& s) {
    return std::string(s.path == Path::host ? "host" : "device") + (s.owned ? " owned" : " workspace") + " n " +
           std::to_string(s.grid * s.grid) + " batch " + std::to_string(s.batch) + " nrhs " + std::to_string(s.nrhs) +
           (s.drops ? " drops" : "") + (s.singular ? " singular" : "");
}

template <typename T, Backend B>
struct Problem {
    Queue& q;
    Shape s;
    int n, nnz = 0, cap = 0, ostride = 0, fstride = 0;
    UnifiedVector<T> val;
    UnifiedVector<int> ro, ci;
    Matrix<T> rhs, out;
    ILUKParams<T> params;
    UnifiedVector<T> factor;  // the workspace factor's values, copied out for every item
    std::vector<T> dense;     // the factor densified per item (n * n), for verify_active
    int last_nnz = 0;

    Problem(Queue& queue, const Shape& shape)
        : q(queue), s(shape), n(shape.grid * shape.grid), rhs(test_support::padded<T>(n, shape.nrhs, shape.batch)),
          out(test_support::padded<T>(n, shape.nrhs, shape.batch)) {
        // Five-point stencil on a grid x grid mesh, sorted columns, every item the same pattern.
        std::vector<int> rows{0}, cols;
        for (int y = 0; y < s.grid; ++y)
            for (int x = 0; x < s.grid; ++x) {
                const int i = y * s.grid + x;
                if (y > 0) cols.push_back(i - s.grid);
                if (x > 0) cols.push_back(i - 1);
                cols.push_back(i);
                if (x + 1 < s.grid) cols.push_back(i + 1);
                if (y + 1 < s.grid) cols.push_back(i + s.grid);
                rows.push_back(int(cols.size()));
            }
        nnz = int(cols.size());
        cap = nnz + 3;
        ostride = n + 3;
        val = UnifiedVector<T>(std::size_t(cap) * s.batch);
        ro = UnifiedVector<int>(std::size_t(ostride) * s.batch, -7);
        ci = UnifiedVector<int>(std::size_t(cap) * s.batch, -7);
        for (int b = 0; b < s.batch; ++b) {
            std::copy(rows.begin(), rows.end(), ro.data() + std::size_t(b) * ostride);
            std::copy(cols.begin(), cols.end(), ci.data() + std::size_t(b) * cap);
        }
        params.levels_of_fill = s.drops ? 0 : 1;  // ILU(0) on this stencil never updates an off-diagonal
        params.drop_tolerance = s.drops ? 1e-4 : 0;
        if (s.singular) {
            params.diagonal_shift = T(0);
            params.diag_pivot_threshold = 0;
        }
        fstride = 9 * n;  // >= the level-1 symbolic nnz of this stencil
        factor = UnifiedVector<T>(std::size_t(fstride) * s.batch);
        dense.resize(std::size_t(n) * n * s.batch);
    }

    Csr<T> A() {
        return Csr<T>(val.data(), ro.data(), ci.data(), n, n, NonZeros{nnz}, cap, ostride, s.batch);
    }

    void fill_values(bool regular_only) {
        for (int b = 0; b < s.batch; ++b) {
            const int pb = s.identical ? 0 : b;
            T* v = val.data() + std::size_t(b) * cap;
            for (int i = 0; i < n; ++i)
                for (int p = ro[std::size_t(b) * ostride + i]; p < ro[std::size_t(b) * ostride + i + 1]; ++p) {
                    const int j = ci[std::size_t(b) * cap + p];
                    if (j == i) {
                        v[p] = Dense<T>::make(6.0 + 0.125 * ((i + 3 * pb) % 5), 0.25 * ((i + pb) % 3 - 1));
                        if (s.singular && !regular_only && b % 2 == 0 && i == 0) v[p] = T(0);
                    } else {
                        v[p] = Dense<T>::make(-1.0 + 0.0625 * ((p * 7 + pb * 3) % 11 - 5),
                                              0.125 * ((p * 5 + pb) % 7 - 3));
                        if (s.drops && b % 2 == 1 && p % 5 == 1) v[p] = Dense<T>::make(1e-6, -1e-6);
                    }
                }
            for (int p = nnz; p < cap; ++p) v[p] = test_support::poison<T>();
        }
    }

    void prepare(bool regular_only = false) {
        fill_values(regular_only);
        T* r = rhs.data_ptr();
        for (std::size_t e = 0; e < std::size_t(rhs.stride()) * s.batch; ++e) {
            const std::size_t item = s.identical ? 0 : e / rhs.stride(), off = e % rhs.stride();
            r[e] = Dense<T>::make(double(int((off * 7 + item * 3) % 23) - 11) / 8, double(int((off * 5 + item) % 13) - 6) / 8);
        }
        std::fill_n(out.data_ptr(), std::size_t(out.stride()) * s.batch, test_support::poison<T>());
        std::memset(static_cast<void*>(factor.data()), 0x5A, factor.size() * sizeof(T));
    }

    template <typename View>
    void densify(const View& lu) {
        std::fill(dense.begin(), dense.end(), T(0));
        for (int b = 0; b < s.batch; ++b)
            for (int i = 0; i < n; ++i)
                for (int p = lu.row_offsets()[b * lu.offset_stride() + i]; p < lu.row_offsets()[b * lu.offset_stride() + i + 1]; ++p)
                    dense[(std::size_t(b) * n + i) * n + lu.col_indices()[b * lu.matrix_stride() + p]] =
                        lu.data()[b * lu.matrix_stride() + p];
    }

    void call(Span<const std::uint8_t> mask, Span<std::byte> ws) {
        const auto r = rhs.view().with_batch_mask(mask);
        const auto o = out.view().with_batch_mask(mask);
        if (s.owned) {
            const auto M = iluk_factorize<B, T>(q, A().with_batch_mask(mask), params);
            q.wait();
            (void)iluk_apply<B, T>(q, M, r, o);
            q.wait();
            last_nnz = M.lu.nnz();
            densify(M.lu.view());
            return;
        }
        // The poison the factor copy starts from: an inactive item's values must keep it.
        std::memset(static_cast<void*>(ws.data()), 0x5A, ws.size());
        const auto M = iluk_factorize<B, T>(q, A().with_batch_mask(mask), ws, params, nullptr);
        q.wait();
        EXPECT_FALSE(M.lu.has_batch_mask()) << "the factor view carries the call's mask";
        ASSERT_LE(M.lu.nnz(), fstride);
        for (int b = 0; b < s.batch; ++b)
            std::memcpy(static_cast<void*>(factor.data() + std::size_t(b) * fstride),
                        M.lu.data_ptr() + std::size_t(b) * M.lu.matrix_stride(), std::size_t(M.lu.nnz()) * sizeof(T));
        (void)iluk_apply<B, T>(q, M, r, o);
        q.wait();
        last_nnz = M.lu.nnz();
        densify(M.lu);
    }

    mc::Case make_case() {
        mc::Case c;
        c.batch = s.batch;
        c.identical_items = s.identical;
        c.prepare = [this] { prepare(); };
        c.call = [this](Span<const std::uint8_t> mask, Span<std::byte> ws) { call(mask, ws); };
        if (!s.owned)
            c.buffer_size = [this](Span<const std::uint8_t> mask) {
                return iluk_buffer_size<B, T>(q, A().with_batch_mask(mask), params);
            };
        c.outputs = [this] {
            const std::size_t abytes = std::size_t(cap) * sizeof(T);
            std::vector<mc::Output> o{mc::Output::matrix("out", out.view()), mc::Output::matrix("rhs", rhs.view()),
                                      {"A", reinterpret_cast<const std::byte*>(val.data()), abytes, abytes}};
            if (!s.owned) {
                const std::size_t fbytes = std::size_t(fstride) * sizeof(T);
                o.push_back({"factor", reinterpret_cast<const std::byte*>(factor.data()), fbytes, fbytes});
            }
            return o;
        };
        return c;
    }
};

// One item's bytes of a run, for the comparisons the harness's bitwise check cannot make.
template <typename T>
bool same_item(const std::vector<T>& a, const std::vector<T>& b, std::size_t per, int item) {
    return std::memcmp(a.data() + per * item, b.data() + per * item, per * sizeof(T)) == 0;
}

template <typename T, Backend B>
struct Cfg {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};
using Types = typename test_utils::backend_types<Cfg>::type;

template <typename Config>
class IlukMask : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    static constexpr Backend B = Config::BackendVal;

    static ScopedEnvVar pin(Path p) { return ScopedEnvVar("BATCHLAS_ILUK_DEVICE", p == Path::device ? "1" : "0"); }

    // The host path of a device backend refuses a masked call before writing anything.
    void expect_host_refusal(const Shape& s) {
        Queue& q = *this->ctx;
        Problem<T, B> p(q, s);
        const mc::Case c = p.make_case();
        c.prepare();
        q.wait();
        const auto before = mc::detail::snapshot(c.outputs(), s.batch);
        const mc::DeviceMask mask(q, mc::patterns(s.batch)[2].bytes);
        UnifiedVector<std::byte> ws(std::size_t(1) << 22);
        try {
            c.call(mask.span(), {ws.data(), ws.size()});
            ADD_FAILURE() << "a device backend's host path ran a masked call";
        } catch (const batchlas::unsupported& e) {
            EXPECT_NE(std::string(e.what()).find("batch mask"), std::string::npos) << e.what();
        }
        EXPECT_TRUE(mc::detail::snapshot(c.outputs(), s.batch) == before) << "the refused call wrote an operand";
    }

    void check(const Shape& s) {
        SCOPED_TRACE(label(s));
        const auto env = pin(s.path);
        if (s.path == Path::host && B != Backend::NETLIB) return expect_host_refusal(s);
        Queue& q = *this->ctx;
        Problem<T, B> p(q, s);
        mc::check(q, p.make_case());
    }

    // Odd items drop entries the even ones keep, so a mask with item 0 off shrinks the pattern:
    // the active items match the unmasked call entry by entry (densified factor) and in out.
    void check_drops(const Shape& s) {
        SCOPED_TRACE(label(s));
        const auto env = pin(s.path);
        if (s.path == Path::host && B != Backend::NETLIB) return;
        Queue& q = *this->ctx;
        Problem<T, B> p(q, s);
        mc::Case c = p.make_case();
        UnifiedVector<std::byte> ws(std::max<std::size_t>(c.buffer_size ? c.buffer_size({}) : 0, 1));
        c.prepare();
        q.wait();
        c.call({}, {ws.data(), ws.size()});
        const int ref_nnz = p.last_nnz;
        const std::vector<T> ref_dense = p.dense;
        const std::size_t per_out = std::size_t(p.out.stride());
        const std::vector<T> ref_out(p.out.data_ptr(), p.out.data_ptr() + per_out * s.batch);
        c.verify_active = [&](const std::string& pattern, std::span<const int> active) {
            if (pattern == "alternating")
                EXPECT_LT(p.last_nnz, ref_nnz) << "the pattern did not shrink: the union ignores the mask";
            const std::vector<T> got_out(p.out.data_ptr(), p.out.data_ptr() + per_out * s.batch);
            for (const int b : active) {
                EXPECT_TRUE(same_item(p.dense, ref_dense, std::size_t(p.n) * p.n, b))
                    << "factor of active item " << b << " differs from the unmasked call";
                EXPECT_TRUE(same_item(got_out, ref_out, per_out, b)) << "out of active item " << b << " differs";
            }
        };
        mc::check(q, c);
    }

    // Even items cannot be factored: unmasked the call throws; with them inactive it runs, and the
    // active items equal the unmasked call on a batch whose even items are regular.
    void check_singular(const Shape& s) {
        SCOPED_TRACE(label(s));
        const auto env = pin(s.path);
        if (s.path == Path::host && B != Backend::NETLIB) return;
        Queue& q = *this->ctx;
        Problem<T, B> p(q, s);
        UnifiedVector<std::byte> ws(iluk_buffer_size<B, T>(q, p.A(), p.params));
        const Span<std::byte> w(ws.data(), ws.size());
        p.prepare();
        EXPECT_THROW(p.call({}, w), batchlas::convergence_error) << "the even items are not singular";
        p.prepare(/*regular_only=*/true);
        p.call({}, w);
        const std::vector<T> ref_dense = p.dense;
        const std::size_t per_out = std::size_t(p.out.stride());
        const std::vector<T> ref_out(p.out.data_ptr(), p.out.data_ptr() + per_out * s.batch);
        const int ref_nnz = p.last_nnz;
        p.prepare();
        const std::vector<T> poisoned(p.out.data_ptr(), p.out.data_ptr() + per_out * s.batch);
        const mc::DeviceMask mask(q, mc::patterns(s.batch)[2].bytes);  // alternating: the even items off
        ASSERT_NO_THROW(p.call(mask.span(), w));
        EXPECT_EQ(p.last_nnz, ref_nnz);
        const std::vector<T> got(p.out.data_ptr(), p.out.data_ptr() + per_out * s.batch);
        for (int b = 0; b < s.batch; ++b) {
            if (b % 2 == 0) {
                EXPECT_TRUE(same_item(got, poisoned, per_out, b)) << "out of inactive item " << b << " was written";
            } else {
                EXPECT_TRUE(same_item(p.dense, ref_dense, std::size_t(p.n) * p.n, b)) << "factor of item " << b;
                EXPECT_TRUE(same_item(got, ref_out, per_out, b)) << "out of item " << b;
            }
        }
    }
};

TYPED_TEST_SUITE(IlukMask, Types);

// Both overloads on both paths: batch 6 takes the host path, 40 the device one.
TYPED_TEST(IlukMask, InactiveItemsKeepEveryByte) {
    for (const Path path : {Path::host, Path::device})
        for (const bool owned : {false, true}) {
            this->check({5, 6, 3, path, owned});
            this->check({4, 40, 2, path, owned});
        }
}

TYPED_TEST(IlukMask, SaturatingIdenticalItems) {
    this->check({4, 1024, 1, Path::device, false, true});
}

TYPED_TEST(IlukMask, PatternIsTheUnionOverActiveItems) {
    for (const Path path : {Path::host, Path::device}) this->check_drops({5, 40, 2, path, false, false, true});
}

TYPED_TEST(IlukMask, InactiveSingularItemDoesNotFailTheCall) {
    for (const Path path : {Path::host, Path::device})
        for (const bool owned : {false, true}) this->check_singular({4, 36, 2, path, owned, false, false, true});
}

}  // namespace
