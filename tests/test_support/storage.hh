// SPDX-License-Identifier: MIT
#pragma once
// Padded, poisoned batched storage for op tests (agent guide section 8, rules 3-5): items at a
// non-natural ld and batch stride, one poison value in every element an op may not touch, and
// bit-exact checks that nothing outside the live region was written and that items agree.
// Everything here reads and writes on the host: wait on the queue first.
#include <batchlas/blas/matrix.hh>

#include <gtest/gtest.h>

#include <tests/usm_bulk_copy.hh>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <span>
#include <vector>

namespace test_support {

template <typename T>
using MV = batchlas::MatrixView<T, batchlas::MatrixFormat::Dense>;

/// The typed-test configuration test_utils::backend_types and BatchLASTest expect.
template <typename T, batchlas::Backend B>
struct Config {
    using ScalarType = T;
    static constexpr batchlas::Backend BackendVal = B;
};

template <typename T>
bool same_bits(const T& a, const T& b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

/// The one poison: in range, large and finite (a kernel consumes it rather than discarding it),
/// with a nonzero imaginary part for complex T.
template <typename T>
T poison() {
    return MV<T>::make(9.0e3, -7.0e3);
}

/// rows x cols items at ld = rows + ld_pad and stride ld * cols + gap, every element poison.
template <typename T>
batchlas::Matrix<T> padded(int rows, int cols, int batch, int ld_pad = 3, int gap = 5) {
    const int ld = rows + ld_pad;
    batchlas::Matrix<T> m(rows, cols, batch, ld, ld * cols + gap);
    std::fill_n(m.data_ptr(), static_cast<std::size_t>(m.stride()) * batch, poison<T>());
    return m;
}

/// Where an op may write in each item: nothing, the rows x cols block (or a window of it, a hole),
/// or one triangle of it.
struct Live {
    enum class Kind { nothing, block, lower, upper } kind = Kind::block;
    int r0 = 0, c0 = 0, rows = -1, cols = -1;  // the window; -1 runs to the view's edge

    static Live nothing() { return {Kind::nothing}; }
    static Live block() { return {}; }
    static Live triangle(batchlas::Uplo u) { return {u == batchlas::Uplo::Lower ? Kind::lower : Kind::upper}; }
    static Live hole(int r0, int c0, int rows, int cols) { return {Kind::block, r0, c0, rows, cols}; }

    bool contains(int i, int j, int m, int n) const {
        const int re = std::min(m, rows < 0 ? m : r0 + rows), ce = std::min(n, cols < 0 ? n : c0 + cols);
        if (kind == Kind::nothing || i < r0 || j < c0 || i >= re || j >= ce) return false;
        return kind == Kind::block || (kind == Kind::lower ? i >= j : i <= j);
    }
};

/// Copies src's elements inside `live` into dst; item b from src's item b, or from item 0 for
/// every b when src holds one item (the saturating-batch shape). The rest of dst is untouched.
template <typename T>
void store(const MV<T>& dst, const MV<T>& src, Live live = Live::block()) {
    T* p = dst.data_ptr();
    for (int b = 0; b < dst.batch_size(); ++b)
        for (int j = 0; j < dst.cols(); ++j)
            for (int i = 0; i < dst.rows(); ++i)
                if (live.contains(i, j, dst.rows(), dst.cols()))
                    p[static_cast<std::size_t>(b) * dst.stride() + static_cast<std::size_t>(j) * dst.ld() + i] =
                        src(i, j, src.batch_size() == 1 ? 0 : b);
}

/// Writes the poison everywhere in dst's storage outside `live`: the other triangle, the ld pads
/// and the item gaps (after an upstream call that may have written them).
template <typename T>
void poison_outside(const MV<T>& dst, Live live) {
    T* p = dst.data_ptr();
    for (int b = 0; b < dst.batch_size(); ++b)
        for (int o = 0; o < dst.stride(); ++o)
            if (!live.contains(o % dst.ld(), o / dst.ld(), dst.rows(), dst.cols()))
                p[static_cast<std::size_t>(b) * dst.stride() + o] = poison<T>();
}

/// The whole Hermitian (real: symmetric) matrix of v's uplo triangle per item, packed. The
/// diagonal is taken as stored, so a non-real one still reaches a residual.
template <typename T>
batchlas::Matrix<T> hermitian(const MV<T>& v, batchlas::Uplo uplo) {
    batchlas::Matrix<T> h(v.rows(), v.cols(), v.batch_size());
    const Live own = Live::triangle(uplo);
    for (int b = 0; b < v.batch_size(); ++b)
        for (int j = 0; j < v.cols(); ++j)
            for (int i = 0; i < v.rows(); ++i)
                h(i, j, b) = own.contains(i, j, v.rows(), v.cols()) ? T(v(i, j, b)) : MV<T>::conj(T(v(j, i, b)));
    return h;
}

namespace detail {
inline std::vector<int> items_or_all(std::span<const int> items, int batch) {
    std::vector<int> all(items.begin(), items.end());
    if (items.empty())
        for (int b = 0; b < batch; ++b) all.push_back(b);
    return all;
}

template <typename T>
::testing::AssertionResult first_write(const MV<T>& got, Live live, const T* ref, std::span<const int> items) {
    const T* p = got.data_ptr();
    const T bad = poison<T>();
    for (int b : items_or_all(items, got.batch_size()))
        for (int o = 0; o < got.stride(); ++o) {
            const int i = o % got.ld(), j = o / got.ld();
            if (live.contains(i, j, got.rows(), got.cols())) continue;
            const std::size_t e = static_cast<std::size_t>(b) * got.stride() + o;
            if (!same_bits(p[e], ref ? ref[e] : bad))
                return ::testing::AssertionFailure() << "written outside the live region: item " << b << " offset " << o
                                                     << " (row " << i << ", col " << j << ")";
        }
    return ::testing::AssertionSuccess();
}
}  // namespace detail

/// Every element of got's storage (stride() per item) outside `live` still holds the poison, bit
/// for bit, over `items` (empty: all); the failure names the first write.
template <typename T>
::testing::AssertionResult unwritten_outside(const MV<T>& got, Live live, std::span<const int> items = {}) {
    return detail::first_write(got, live, static_cast<const T*>(nullptr), items);
}

/// The same against a snapshot `before` in got's layout (a clone() taken before the call).
template <typename T>
::testing::AssertionResult unwritten_outside(const MV<T>& got, Live live, const MV<T>& before, std::span<const int> items = {}) {
    if (before.ld() != got.ld() || before.stride() != got.stride() || before.batch_size() != got.batch_size())
        return ::testing::AssertionFailure() << "the snapshot's layout differs from the storage's";
    return detail::first_write(got, live, before.data_ptr(), items);
}

/// Item b of got bit-identical to item b of want over the whole stride, for each of `items`
/// (empty: all): the item-against-a-clean-batch comparison.
template <typename T>
::testing::AssertionResult items_bit_identical_to(const MV<T>& got, const MV<T>& want, std::span<const int> items = {}) {
    if (want.ld() != got.ld() || want.stride() != got.stride() || want.batch_size() < got.batch_size())
        return ::testing::AssertionFailure() << "the reference's layout differs from the storage's";
    const std::size_t s = static_cast<std::size_t>(got.stride());
    for (int b : detail::items_or_all(items, got.batch_size()))
        if (auto d = test_utils::usm_bulk::first_diff(got.data_ptr() + b * s, want.data_ptr() + b * s, s))
            return ::testing::AssertionFailure() << "item " << b << " differs at offset " << *d << " (row "
                                                 << *d % got.ld() << ", col " << *d / got.ld() << ")";
    return ::testing::AssertionSuccess();
}

/// Every item of v bit-identical to item `ref` over the whole stride (a saturating batch of
/// identical inputs: a race shows as a differing item).
template <typename T>
::testing::AssertionResult items_bit_identical(const MV<T>& v, int ref = 0) {
    const std::size_t s = static_cast<std::size_t>(v.stride());
    for (int b = 0; b < v.batch_size(); ++b)
        if (auto d = test_utils::usm_bulk::first_diff(v.data_ptr() + b * s, v.data_ptr() + ref * s, s))
            return ::testing::AssertionFailure() << "item " << b << " differs from item " << ref << " at offset " << *d
                                                 << " (row " << *d % v.ld() << ", col " << *d / v.ld() << ")";
    return ::testing::AssertionSuccess();
}

}  // namespace test_support
