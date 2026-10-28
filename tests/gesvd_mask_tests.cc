// SPDX-License-Identifier: MIT
// gesvd and its stages honour a batch mask (W4a, lift 2) on tests/masked_contract.hh: an inactive
// item's A, singular values, U, V^H, info, d, e and taus are bit-for-bit untouched, an active one
// equals the unmasked call on the same pin. Device-only masks, complex data, padded ld and stride,
// gaps in the per-item vectors. gesvd pinned jacobi (packed partitions: holes inside a sub-group),
// blocked over each bidiagonal solver (BATCHLAS_GESVD_BIDIAG = bdsdc, normal, bdsqr), cta, the
// Hermitian blocked and cta, and NETLIB's host loop (I4); gebrd's three kernels, bdsqr (its failure
// fold), bdsdc, ormbr 'P' and 'Q' directly. On a vendor backend the composites compare their active
// items with the all-active masked call: an unmasked child may take the vendor.
// evidence: docs/design/masked-batches.md#masked-batches-tests
#include <gtest/gtest.h>

#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/functions/gesvd.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/error.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-vector.hh>

#include <src/ops/gesvd/choice.hh>
#include <src/ops/gesvd/gesvd_native.hh>
#include <src/ops/ormqr/wy_spelling.hh>
#include <src/select/vendor.hh>
#include <tests/masked_contract.hh>
#include <tests/test_support/storage.hh>
#include <tests/test_utils.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

using namespace batchlas;

namespace {

namespace mc = masked_contract;

// A well-spread value in [-1, 1] per (item, element, salt): distinct singular values in every item.
inline double spread(std::size_t item, std::size_t e, int salt) {
    std::uint64_t h = (item * 0x9E3779B97F4A7C15ull) ^ (e * 0xC2B2AE3D27D4EB4Full) ^ std::uint64_t(salt * 7919);
    h ^= h >> 29;
    h *= 0xBF58476D1CE4E5B9ull;
    h ^= h >> 32;
    return double(int(h % 2001) - 1000) / 1000.0;
}

template <typename T>
void fill(T* p, std::size_t elems, std::size_t per, int salt, bool identical, double scale = 1.0) {
    for (std::size_t e = 0; e < elems; ++e) {
        const std::size_t item = identical ? 0 : e / per;
        p[e] = MatrixView<T>::make(spread(item, e % per, salt) * scale, spread(item, e % per, salt + 1) * scale);
    }
}

template <typename T>
void poison_all(T* p, std::size_t elems) {
    std::fill_n(p, elems, test_support::poison<T>());
}

// A per-item output `per` elements long at a batch stride of `stride` elements.
template <typename T>
mc::Output strided(std::string name, const T* base, std::size_t per, std::size_t stride) {
    return {std::move(name), reinterpret_cast<const std::byte*>(base), per * sizeof(T), stride * sizeof(T)};
}

struct Shape {
    std::string spelling;  ///< the pinned gesvd family
    int m, n, batch;
    SvdVectors jobu = SvdVectors::All, jobvh = SvdVectors::All;
    int wg_items = 0;        ///< items sharing a packed sub-group: the hole patterns
    int chunk_items = 0;     ///< the chunk-edge patterns
    bool identical = false;  ///< every item the same input (the saturating case)
    std::optional<Uplo> herm{};  ///< the Hermitian gesvd on this triangle (square)
};

std::string label(const Shape& s) {
    auto job = [](SvdVectors j) { return j == SvdVectors::All ? "A" : j == SvdVectors::Thin ? "S" : "N"; };
    const std::string h = !s.herm ? "" : *s.herm == Uplo::Lower ? " herm L" : " herm U";
    return s.spelling + " " + std::to_string(s.m) + "x" + std::to_string(s.n) + " " + job(s.jobu) + job(s.jobvh) +
           " batch " + std::to_string(s.batch) + h;
}

// Each item's n x n block made Hermitian (real diagonal), from its upper triangle.
template <typename T>
void make_hermitian(Matrix<T>& A, int batch) {
    const std::int64_t n = A.rows(), ld = A.ld(), stride = A.stride();
    for (int b = 0; b < batch; ++b) {
        T* p = A.data_ptr() + b * stride;
        for (std::int64_t j = 0; j < n; ++j) {
            p[j + j * ld] = T(std::real(p[j + j * ld]));
            for (std::int64_t i = 0; i < j; ++i) {
                if constexpr (std::is_same_v<T, typename base_type<T>::type>) p[j + i * ld] = p[i + j * ld];
                else p[j + i * ld] = std::conj(p[i + j * ld]);
            }
        }
    }
}

template <typename T, Backend B>
struct GesvdProblem {
    using Real = typename base_type<T>::type;
    Queue& q;
    Shape s;
    int k;
    Matrix<T> A, U, Vh;
    UnifiedVector<Real> S;
    UnifiedVector<int32_t> info;

    static int rows_or_1(SvdVectors j, int64_t v) { return j == SvdVectors::None ? 1 : int(v); }
    GesvdProblem(Queue& queue, const Shape& shape)
        : q(queue), s(shape), k(std::min(shape.m, shape.n)), A(test_support::padded<T>(shape.m, shape.n, shape.batch)),
          U(test_support::padded<T>(rows_or_1(shape.jobu, shape.m), rows_or_1(shape.jobu, svd_u_cols(shape.jobu, shape.m, k)),
                                    shape.batch)),
          Vh(test_support::padded<T>(rows_or_1(shape.jobvh, svd_vh_rows(shape.jobvh, shape.n, k)),
                                     rows_or_1(shape.jobvh, shape.n), shape.batch)),
          S(std::size_t(k) * shape.batch), info(std::size_t(shape.batch)) {}

    mc::Case make_case(bool masked_reference) {
        mc::Case c;
        c.batch = s.batch;
        c.wg_items = s.wg_items;
        c.chunk_items = s.chunk_items;
        c.identical_items = s.identical;
        c.masked_reference = masked_reference;
        c.prepare = [this] {
            fill(A.data_ptr(), std::size_t(A.stride()) * s.batch, std::size_t(A.stride()), 3, s.identical);
            if (s.herm) make_hermitian(A, s.batch);
            poison_all(U.data_ptr(), std::size_t(U.stride()) * s.batch);
            poison_all(Vh.data_ptr(), std::size_t(Vh.stride()) * s.batch);
            poison_all(S.data(), S.size());
            std::fill(info.begin(), info.end(), -77);
        };
        c.call = [this](Span<const std::uint8_t> mask, Span<std::byte> ws) {
            const auto a = A.view().with_batch_mask(mask), u = U.view().with_batch_mask(mask),
                       vh = Vh.view().with_batch_mask(mask);
            if (s.herm) (void)gesvd<B, T>(q, a, S.to_span(), u, vh, s.jobu, s.jobvh, *s.herm, ws, info.to_span());
            else (void)gesvd<B, T>(q, a, S.to_span(), u, vh, s.jobu, s.jobvh, ws, info.to_span());
        };
        c.buffer_size = [this](Span<const std::uint8_t> mask) {
            const auto a = A.view().with_batch_mask(mask), u = U.view().with_batch_mask(mask),
                       vh = Vh.view().with_batch_mask(mask);
            return s.herm ? gesvd_buffer_size<B, T>(q, a, S.to_span(), u, vh, s.jobu, s.jobvh, *s.herm)
                          : gesvd_buffer_size<B, T>(q, a, S.to_span(), u, vh, s.jobu, s.jobvh);
        };
        c.outputs = [this] {
            return std::vector<mc::Output>{mc::Output::matrix("A", A.view()), mc::Output::matrix("U", U.view()),
                                           mc::Output::matrix("Vh", Vh.view()),
                                           mc::Output::span("s", std::span<const Real>(S.data(), S.size()), k),
                                           mc::Output::span("info", std::span<const int32_t>(info.data(), info.size()), 1)};
        };
        return c;
    }
};

// gebrd (m >= n): A, then d, e, tauq, taup at batch strides with a gap of 2.
template <typename T, Backend B>
struct GebrdProblem {
    using Real = typename base_type<T>::type;
    enum class Kernel { Unblocked, Cta, Blocked };
    Queue& q;
    Kernel kernel;
    int m, n, batch;
    bool identical;
    Matrix<T> A;
    UnifiedVector<Real> d, e;
    UnifiedVector<T> tq, tp;

    GebrdProblem(Queue& queue, Kernel which, int rows, int cols, int b, bool same)
        : q(queue), kernel(which), m(rows), n(cols), batch(b), identical(same), A(test_support::padded<T>(rows, cols, b)),
          d(std::size_t(cols + 2) * b), e(std::size_t(cols + 1) * b), tq(std::size_t(cols + 2) * b),
          tp(std::size_t(cols + 2) * b) {}

    VectorView<Real> dv(Span<const std::uint8_t> mk) { return VectorView<Real>(d.data(), n, batch, 1, n + 2).with_batch_mask(mk); }
    VectorView<Real> ev(Span<const std::uint8_t> mk) {
        return VectorView<Real>(e.data(), n - 1, batch, 1, n + 1).with_batch_mask(mk);
    }
    VectorView<T> tv(UnifiedVector<T>& t, Span<const std::uint8_t> mk) {
        return VectorView<T>(t.data(), n, batch, 1, n + 2).with_batch_mask(mk);
    }

    mc::Case make_case() {
        mc::Case c;
        c.batch = batch;
        c.identical_items = identical;
        // Blocked's unmasked gemm may take the vendor, which a masked one refuses.
        c.masked_reference = kernel == Kernel::Blocked && (B == Backend::CUDA || B == Backend::ROCM);
        c.wg_items = kernel == Kernel::Cta ? 32 / (n <= 4 ? 4 : n <= 8 ? 8 : n <= 16 ? 16 : 32) : 0;
        c.prepare = [this] {
            fill(A.data_ptr(), std::size_t(A.stride()) * batch, std::size_t(A.stride()), 5, identical);
            poison_all(d.data(), d.size());
            poison_all(e.data(), e.size());
            poison_all(tq.data(), tq.size());
            poison_all(tp.data(), tp.size());
        };
        c.call = [this](Span<const std::uint8_t> mk, Span<std::byte> ws) {
            const auto a = A.view().with_batch_mask(mk);
            if (kernel == Kernel::Unblocked) (void)gebrd_unblocked<B, T>(q, a, dv(mk), ev(mk), tv(tq, mk), tv(tp, mk));
            if (kernel == Kernel::Cta) (void)gebrd_cta<B, T>(q, a, dv(mk), ev(mk), tv(tq, mk), tv(tp, mk));
            if (kernel == Kernel::Blocked) (void)gebrd_blocked<B, T>(q, a, dv(mk), ev(mk), tv(tq, mk), tv(tp, mk), ws, 8);
        };
        if (kernel == Kernel::Blocked)
            c.buffer_size = [this](Span<const std::uint8_t> mk) {
                return gebrd_blocked_buffer_size<B, T>(q, A.view().with_batch_mask(mk), dv(mk), ev(mk), tv(tq, mk),
                                                       tv(tp, mk), 8);
            };
        c.outputs = [this] {
            return std::vector<mc::Output>{mc::Output::matrix("A", A.view()), strided("d", d.data(), n, n + 2),
                                           strided("e", e.data(), n - 1, n + 1), strided("tauq", tq.data(), n, n + 2),
                                           strided("taup", tp.data(), n, n + 2)};
        };
        return c;
    }
};

// bdsqr on an upper bidiagonal (d, e), with or without U (rows x n) and V^H (n x cols) to accumulate into.
template <typename Real, Backend B>
struct BdsqrProblem {
    Queue& q;
    int n, batch;
    bool vectors, with_info;
    Matrix<Real> U, Vh;
    UnifiedVector<Real> d, e, S;
    UnifiedVector<int32_t> info;
    int nan_item = -1;  ///< an item whose d holds a NaN: bdsqr flags it as failed

    BdsqrProblem(Queue& queue, int order, int b, bool vecs, bool info_wanted = true)
        : q(queue), n(order), batch(b), vectors(vecs), with_info(info_wanted),
          U(test_support::padded<Real>(order + 2, order, b)), Vh(test_support::padded<Real>(order, order + 1, b)),
          d(std::size_t(order + 2) * b), e(std::size_t(order + 1) * b), S(std::size_t(order) * b), info(std::size_t(b)) {}

    VectorView<Real> dv(Span<const std::uint8_t> mk) { return VectorView<Real>(d.data(), n, batch, 1, n + 2).with_batch_mask(mk); }
    VectorView<Real> ev(Span<const std::uint8_t> mk) {
        return VectorView<Real>(e.data(), n - 1, batch, 1, n + 1).with_batch_mask(mk);
    }
    Span<int32_t> info_span() { return with_info ? info.to_span() : Span<int32_t>(); }

    mc::Case make_case() {
        mc::Case c;
        c.batch = batch;
        c.prepare = [this] {
            fill(d.data(), d.size(), std::size_t(n + 2), 7, false);
            fill(e.data(), e.size(), std::size_t(n + 1), 8, false);
            fill(U.data_ptr(), std::size_t(U.stride()) * batch, std::size_t(U.stride()), 9, false);
            fill(Vh.data_ptr(), std::size_t(Vh.stride()) * batch, std::size_t(Vh.stride()), 10, false);
            if (nan_item >= 0) d[std::size_t(nan_item) * (n + 2)] = std::numeric_limits<Real>::quiet_NaN();
            poison_all(S.data(), S.size());
            std::fill(info.begin(), info.end(), -77);
        };
        c.call = [this](Span<const std::uint8_t> mk, Span<std::byte> ws) {
            if (vectors)
                (void)bdsqr<B, Real>(q, dv(mk), ev(mk), S.to_span(), ws, U.view().with_batch_mask(mk), Vh.view().with_batch_mask(mk),
                               true, info_span());
            else
                (void)bdsqr<B, Real>(q, dv(mk), ev(mk), S.to_span(), ws, true, info_span());
        };
        c.buffer_size = [this](Span<const std::uint8_t> mk) { return bdsqr_buffer_size<Real>(q, dv(mk), ev(mk), S.to_span()); };
        c.outputs = [this] {
            std::vector<mc::Output> o{strided("d", d.data(), n, n + 2), strided("e", e.data(), n - 1, n + 1),
                                      mc::Output::span("s", std::span<const Real>(S.data(), S.size()), n),
                                      mc::Output::span("info", std::span<const int32_t>(info.data(), info.size()), 1)};
            if (vectors) {
                o.push_back(mc::Output::matrix("U", U.view()));
                o.push_back(mc::Output::matrix("Vh", Vh.view()));
            }
            return o;
        };
        return c;
    }
};

// bdsdc on an upper bidiagonal (d, e): it writes s and, with vectors, the n x n U and V^H.
template <typename Real, Backend B>
struct BdsdcProblem {
    Queue& q;
    int n, batch;
    bool vectors;
    Matrix<Real> U, Vh;
    UnifiedVector<Real> d, e, S;
    UnifiedVector<int32_t> info;

    BdsdcProblem(Queue& queue, int order, int b, bool vecs)
        : q(queue), n(order), batch(b), vectors(vecs), U(test_support::padded<Real>(order, order, b)),
          Vh(test_support::padded<Real>(order, order, b)), d(std::size_t(order + 2) * b),
          e(std::size_t(order + 1) * b), S(std::size_t(order) * b), info(std::size_t(b)) {}

    VectorView<Real> dv(Span<const std::uint8_t> mk) { return VectorView<Real>(d.data(), n, batch, 1, n + 2).with_batch_mask(mk); }
    VectorView<Real> ev(Span<const std::uint8_t> mk) {
        return VectorView<Real>(e.data(), n - 1, batch, 1, n + 1).with_batch_mask(mk);
    }

    mc::Case make_case() {
        mc::Case c;
        c.batch = batch;
        c.prepare = [this] {
            fill(d.data(), d.size(), std::size_t(n + 2), 14, false);
            fill(e.data(), e.size(), std::size_t(n + 1), 15, false);
            poison_all(U.data_ptr(), std::size_t(U.stride()) * batch);
            poison_all(Vh.data_ptr(), std::size_t(Vh.stride()) * batch);
            poison_all(S.data(), S.size());
            std::fill(info.begin(), info.end(), -77);
        };
        c.call = [this](Span<const std::uint8_t> mk, Span<std::byte> ws) {
            if (vectors)
                (void)bdsdc<B, Real>(q, dv(mk), ev(mk), S.to_span(), ws, U.view().with_batch_mask(mk),
                                     Vh.view().with_batch_mask(mk), true, info.to_span());
            else
                (void)bdsdc<B, Real>(q, dv(mk), ev(mk), S.to_span(), ws, true, info.to_span());
        };
        c.buffer_size = [this](Span<const std::uint8_t> mk) {
            return bdsdc_buffer_size<B, Real>(q, dv(mk), ev(mk), S.to_span(), vectors);
        };
        c.outputs = [this] {
            std::vector<mc::Output> o{strided("d", d.data(), n, n + 2), strided("e", e.data(), n - 1, n + 1),
                                      mc::Output::span("s", std::span<const Real>(S.data(), S.size()), n),
                                      mc::Output::span("info", std::span<const int32_t>(info.data(), info.size()), 1)};
            if (vectors) {
                o.push_back(mc::Output::matrix("U", U.view()));
                o.push_back(mc::Output::matrix("Vh", Vh.view()));
            }
            return o;
        };
        return c;
    }
};

// ormbr: A holds gebrd-shaped reflectors (|tau| <= 1/8), C is the operand. 'Q' is Left, order
// a.rows(); 'P' is Right, order a.cols(). tau is packed by batch (ormbr's contract).
template <typename T, Backend B>
struct OrmbrProblem {
    Queue& q;
    char vect;
    int am, an, cm, cn, batch, block;
    Transpose trans;
    Matrix<T> A, C;
    UnifiedVector<T> tau;

    int ntau() const { return std::min(am, an); }
    OrmbrProblem(Queue& queue, char v, int a_rows, int a_cols, int c_rows, int c_cols, int b, Transpose t, int nb)
        : q(queue), vect(v), am(a_rows), an(a_cols), cm(c_rows), cn(c_cols), batch(b), block(nb), trans(t),
          A(test_support::padded<T>(a_rows, a_cols, b)), C(test_support::padded<T>(c_rows, c_cols, b)),
          tau(std::size_t(std::min(a_rows, a_cols)) * b) {}

    VectorView<T> tv(Span<const std::uint8_t> mk) {
        return VectorView<T>(tau.data(), ntau(), batch, 1, ntau()).with_batch_mask(mk);
    }
    Side side() const { return vect == 'Q' ? Side::Left : Side::Right; }

    mc::Case make_case() {
        mc::Case c;
        c.batch = batch;
        // Several blocks: the WY apply's unmasked gemm may take the vendor (refused when masked).
        c.masked_reference = block < ntau() && (B == Backend::CUDA || B == Backend::ROCM);
        c.wg_items = am == an && am <= 32 ? 32 / (am <= 4 ? 4 : am <= 8 ? 8 : am <= 16 ? 16 : 32) : 0;
        c.prepare = [this] {
            fill(A.data_ptr(), std::size_t(A.stride()) * batch, std::size_t(A.stride()), 11, false);
            fill(C.data_ptr(), std::size_t(C.stride()) * batch, std::size_t(C.stride()), 12, false);
            fill(tau.data(), tau.size(), std::size_t(ntau()), 13, false, 1.0 / 8);
        };
        c.call = [this](Span<const std::uint8_t> mk, Span<std::byte> ws) {
            (void)ormbr<B, T>(q, A.view().with_batch_mask(mk), tv(mk), C.view().with_batch_mask(mk), vect, side(), trans, ws, block);
        };
        c.buffer_size = [this](Span<const std::uint8_t> mk) {
            return ormbr_buffer_size<B, T>(q, A.view().with_batch_mask(mk), tv(mk), C.view().with_batch_mask(mk), vect,
                                           side(), trans, block);
        };
        c.outputs = [this] { return std::vector<mc::Output>{mc::Output::matrix("C", C.view())}; };
        return c;
    }
};

template <typename Config>
class GesvdMask : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    using Real = typename base_type<T>::type;
    static constexpr Backend B = Config::BackendVal;
    static constexpr bool kComplex = !std::is_same_v<T, Real>;

    bool device() const { return this->ctx->device().type == DeviceType::GPU && B != Backend::NETLIB; }
    bool sg32() const { return this->ctx->device().supports_sub_group_size(32); }
    static constexpr bool vendor() { return B == Backend::CUDA || B == Backend::ROCM; }

    void check(const Shape& s) {
        SCOPED_TRACE(label(s));
        GesvdProblem<T, B> p(*this->ctx, s);
        const bool composite = s.spelling == "blocked" || s.spelling == "cta";
        mc::check<ops::gesvd::GesvdChoice>(*this->ctx, "gesvd", s.spelling, p.make_case(composite && vendor()));
    }
    void check_gebrd(typename GebrdProblem<T, B>::Kernel kind, int m, int n, int batch, bool identical = false) {
        SCOPED_TRACE("gebrd " + std::to_string(int(kind)) + " " + std::to_string(m) + "x" + std::to_string(n) +
                     " batch " + std::to_string(batch));
        GebrdProblem<T, B> p(*this->ctx, kind, m, n, batch, identical);
        mc::check(*this->ctx, p.make_case());
    }
    void check_ormbr(char v, int am, int an, int cm, int cn, int batch, Transpose t, int nb) {
        SCOPED_TRACE(std::string("ormbr ") + v + " A " + std::to_string(am) + "x" + std::to_string(an) + " C " +
                     std::to_string(cm) + "x" + std::to_string(cn));
        OrmbrProblem<T, B> p(*this->ctx, v, am, an, cm, cn, batch, t, nb);
        mc::check(*this->ctx, p.make_case());
    }
    // Partitions sharing a sub-group in gesvdj_cta: 32 / P, P the partition width for max(m, n).
    static int jacobi_items(int m, int n) {
        const int md = std::max(m, n);
        return 32 / (md <= 4 ? 4 : md <= 8 ? 8 : md <= 16 ? 16 : 32);
    }
};

using Types = typename test_utils::backend_types<test_support::Config>::type;
TYPED_TEST_SUITE(GesvdMask, Types);

// jacobi folds the mask into its packed partitions (I2): square, tall, wide (solved transposed) and
// thin, holes inside a sub-group at every partition width.
TYPED_TEST(GesvdMask, JacobiLeavesInactiveItemsUntouched) {
    if (!this->device() || !this->sg32()) GTEST_SKIP() << "jacobi is a device family";
    const SvdVectors A = SvdVectors::All, S = SvdVectors::Thin, N = SvdVectors::None;
    for (const Shape& s : {Shape{"jacobi", 3, 3, 40, A, A}, Shape{"jacobi", 7, 7, 40, A, A}, Shape{"jacobi", 13, 9, 24, A, A},
                           Shape{"jacobi", 5, 11, 24, A, A}, Shape{"jacobi", 20, 20, 12, N, N},
                           Shape{"jacobi", 24, 6, 12, S, A}}) {
        Shape t = s;
        t.wg_items = TestFixture::jacobi_items(s.m, s.n);
        this->check(t);
    }
}

// The saturating batch (>= 1024, identical items) for the packed kernel.
TYPED_TEST(GesvdMask, SaturatingJacobiIsBitIdentical) {
    if (!this->device() || !this->sg32()) GTEST_SKIP() << "jacobi is a device family";
    this->check({"jacobi", 8, 8, 1024, SvdVectors::All, SvdVectors::All, 4, 0, true});
    this->check({"jacobi", 29, 17, 1024, SvdVectors::All, SvdVectors::All, 1, 0, true});
}

// blocked over each bidiagonal solver: gebrd, bdsdc (stedc), the normal equations' tridiagonal
// solve or bdsqr, and both ormbr back-transforms skip an inactive item, per part of the L2 chunk
// loop too (the chunk applies to real types; complex runs one part).
TYPED_TEST(GesvdMask, BlockedOverEachBidiagSolverLeavesInactiveItemsUntouched) {
    if (!this->device()) GTEST_SKIP() << "blocked is a device family";
    const SvdVectors A = SvdVectors::All, S = SvdVectors::Thin, N = SvdVectors::None;
    for (const char* solver : {"bdsdc", "normal", "bdsqr"}) {
        SCOPED_TRACE(std::string("BATCHLAS_GESVD_BIDIAG=") + solver);
        const ScopedEnvVar env("BATCHLAS_GESVD_BIDIAG", solver);
        for (const Shape& s : {Shape{"blocked", 12, 12, 24, A, A, 0, 8}, Shape{"blocked", 20, 9, 24, A, A, 0, 8},
                               Shape{"blocked", 9, 20, 24, A, A, 0, 8}, Shape{"blocked", 20, 9, 12, N, N},
                               Shape{"blocked", 20, 9, 12, N, A}, Shape{"blocked", 9, 20, 12, A, N},
                               Shape{"blocked", 20, 9, 12, S, A}, Shape{"blocked", 40, 40, 24, A, A, 0, 8}})
            this->check(s);
    }
}

// cta (real general, max(m, n) <= 32, never thin): the normal-equations solve on one sub-group
// per matrix, holes inside a packed work-group.
TYPED_TEST(GesvdMask, CtaLeavesInactiveItemsUntouched) {
    if (!this->device() || !this->sg32()) GTEST_SKIP() << "cta is a device family";
    if constexpr (TestFixture::kComplex) {
        GTEST_SKIP() << "cta's general path is real-only (complex runs only Hermitian)";
    } else {
        const SvdVectors A = SvdVectors::All, N = SvdVectors::None;
        for (const Shape& s : {Shape{"cta", 7, 7, 40, A, A}, Shape{"cta", 13, 13, 24, A, A}, Shape{"cta", 20, 9, 24, A, A},
                               Shape{"cta", 9, 20, 24, A, A}, Shape{"cta", 32, 32, 12, A, A}, Shape{"cta", 20, 9, 24, N, N},
                               Shape{"cta", 20, 9, 24, A, N}, Shape{"cta", 9, 20, 24, N, A}}) {
            Shape t = s;
            t.wg_items = TestFixture::jacobi_items(s.m, s.n);
            this->check(t);
        }
    }
}

// The Hermitian gesvd on blocked (syev_blocked) and cta (syev_cta), both triangles, complex too.
TYPED_TEST(GesvdMask, HermitianLeavesInactiveItemsUntouched) {
    if (!this->device()) GTEST_SKIP() << "blocked and cta are device families";
    const SvdVectors A = SvdVectors::All, N = SvdVectors::None;
    for (const Uplo u : {Uplo::Lower, Uplo::Upper}) {
        std::vector<Shape> shapes{{"blocked", 12, 12, 24, A, A, 0, 8}, {"blocked", 12, 12, 24, N, N, 0, 8},
                                  {"blocked", 12, 12, 24, A, N, 0, 8}, {"blocked", 45, 45, 24, A, A, 0, 8}};
        if (this->sg32())
            for (const auto& [n, jobu, jobvh] : {std::tuple{7, A, A}, std::tuple{13, N, N}, std::tuple{13, A, N},
                                                 std::tuple{32, A, A}})
                shapes.push_back({"cta", n, n, 24, jobu, jobvh, TestFixture::jacobi_items(n, n)});
        for (Shape& s : shapes) {
            s.herm = u;
            this->check(s);
        }
    }
}

// bdsdc directly: values only and with vectors, through stedc on the 2n Golub-Kahan tridiagonal.
TYPED_TEST(GesvdMask, BdsdcLeavesInactiveItemsUntouched) {
    using Real = typename TestFixture::Real;
    if constexpr (TestFixture::kComplex) {
        GTEST_SKIP() << "bdsdc is real-only";
    } else {
        for (const bool vectors : {false, true})
            for (const int n : {1, 6, 11, 40}) {
                SCOPED_TRACE("bdsdc n " + std::to_string(n) + (vectors ? " vectors" : " values"));
                BdsdcProblem<Real, TestFixture::B> p(*this->ctx, n, 12, vectors);
                mc::check(*this->ctx, p.make_case());
            }
    }
}

TYPED_TEST(GesvdMask, NetlibHostLoopSkipsInactiveItems) {
    if constexpr (TestFixture::B != Backend::NETLIB) {
        GTEST_SKIP() << "the host loop is NETLIB's vendor family";
    } else {
        if (!select::kHasNetlib) GTEST_SKIP() << "no LAPACKE in this build";
        const SvdVectors A = SvdVectors::All, S = SvdVectors::Thin, N = SvdVectors::None;
        for (const Shape& s : {Shape{"vendor", 7, 7, 12, A, A}, Shape{"vendor", 11, 6, 9, S, A},
                               Shape{"vendor", 5, 9, 9, A, S}, Shape{"vendor", 8, 8, 7, N, N}})
            this->check(s);
    }
}

TYPED_TEST(GesvdMask, GebrdLeavesInactiveItemsUntouched) {
    using K = typename GebrdProblem<typename TestFixture::T, TestFixture::B>::Kernel;
    this->check_gebrd(K::Unblocked, 9, 6, 12);
    this->check_gebrd(K::Unblocked, 7, 7, 9);
    if (!this->device()) return;  // cta and blocked are device kernels (blocked's gemm is the host BLAS here)
    this->check_gebrd(K::Blocked, 40, 21, 12);
    this->check_gebrd(K::Blocked, 17, 17, 9);
    if constexpr (TestFixture::kComplex) {
        GTEST_SKIP() << "gebrd_cta is real-only";
    } else {
        if (!this->sg32()) GTEST_SKIP() << "gebrd_cta needs sub-group size 32";
        for (int n : {3, 7, 13, 29}) this->check_gebrd(K::Cta, n, n, 40);
    }
}

// The saturating batch for gebrd_cta's packed partitions.
TYPED_TEST(GesvdMask, SaturatingGebrdCtaIsBitIdentical) {
    using K = typename GebrdProblem<typename TestFixture::T, TestFixture::B>::Kernel;
    if (!this->device() || !this->sg32()) GTEST_SKIP() << "gebrd_cta is a device kernel";
    if constexpr (TestFixture::kComplex) {
        GTEST_SKIP() << "gebrd_cta is real-only";
    } else {
        this->check_gebrd(K::Cta, 7, 7, 1024, true);
        this->check_gebrd(K::Cta, 29, 29, 1024, true);
    }
}

TYPED_TEST(GesvdMask, BdsqrLeavesInactiveItemsUntouched) {
    using Real = typename TestFixture::Real;
    if constexpr (TestFixture::kComplex) {
        GTEST_SKIP() << "bdsqr is real-only";
    } else {
        for (const bool vectors : {false, true})
            for (const int n : {1, 6, 11}) {
                SCOPED_TRACE("bdsqr n " + std::to_string(n) + (vectors ? " vectors" : " values"));
                BdsqrProblem<Real, TestFixture::B> p(*this->ctx, n, 12, vectors);
                mc::check(*this->ctx, p.make_case());
            }
    }
}

// The bdsqr failure fold (a cross-item host read): an inactive item writes the neutral
// "converged", so a NaN in an inactive item neither throws nor sets any info, while the same NaN
// in an active item throws when no info is requested.
TYPED_TEST(GesvdMask, BdsqrFailureFoldIgnoresInactiveItems) {
    using Real = typename TestFixture::Real;
    if constexpr (TestFixture::kComplex) {
        GTEST_SKIP() << "bdsqr is real-only";
    } else {
        Queue& q = *this->ctx;
        constexpr int kBatch = 8;
        for (const bool with_info : {false, true}) {
            SCOPED_TRACE(with_info ? "info" : "no info");
            BdsqrProblem<Real, TestFixture::B> p(q, 6, kBatch, true, with_info);
            p.nan_item = 0;  // inactive under "alternating"
            const mc::Case c = p.make_case();
            UnifiedVector<std::byte> ws(std::max<std::size_t>(c.buffer_size({}), 1));
            const mc::DeviceMask alternating(q, mc::patterns(kBatch)[2].bytes);
            c.prepare();
            q.wait();
            const auto before = mc::detail::snapshot(c.outputs(), kBatch);
            EXPECT_NO_THROW((c.call(alternating.span(), {ws.data(), ws.size()}), q.wait()));
            const auto got = mc::detail::snapshot(c.outputs(), kBatch);
            const auto outs = c.outputs();
            for (std::size_t o = 0; o < outs.size(); ++o)
                EXPECT_TRUE(mc::detail::item_equal(got[o], before[o], outs[o], 0)) << outs[o].name << ": NaN item written";
            if (with_info)
                for (int b = 1; b < kBatch; b += 2) EXPECT_EQ(p.info[std::size_t(b)], 0) << "active item " << b;
            const mc::DeviceMask all(q, mc::patterns(kBatch)[1].bytes);
            c.prepare();
            q.wait();
            if (with_info) {
                EXPECT_NO_THROW((c.call(all.span(), {ws.data(), ws.size()}), q.wait()));
                EXPECT_EQ(p.info[0], 1) << "the active NaN item is not flagged";
            } else {
                EXPECT_THROW((c.call(all.span(), {ws.data(), ws.size()}), q.wait()), batchlas::convergence_error);
            }
        }
    }
}

// ormbr: 'P' Right through its cta glue (square <= 32) and the blocked WY apply (several blocks),
// 'Q' Left through ormqx_cta and through ormqr_blocked.
TYPED_TEST(GesvdMask, OrmbrLeavesInactiveItemsUntouched) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::B;
    if (!this->device() || !this->sg32()) GTEST_SKIP() << "ormbr runs masked on a device (NETLIB's gemm refuses)";
    const Transpose C = Transpose::ConjTrans, N = Transpose::NoTrans;
    this->check_ormbr('P', 13, 13, 13, 13, 40, C, 32);
    this->check_ormbr('P', 7, 7, 7, 7, 40, N, 32);
    this->check_ormbr('P', 20, 14, 9, 14, 12, C, 4);
    this->check_ormbr('P', 20, 14, 9, 14, 12, N, 4);
    this->check_ormbr('Q', 13, 13, 13, 13, 40, N, 32);
    this->check_ormbr('Q', 20, 14, 20, 9, 12, N, 4);
}

#ifdef BATCHLAS_GEBRD_CTA_CC_PATH
// The first line of `path` whose code (comments stripped) contains `token`, or "".
std::string first_code_line(const std::string& path, const char* token, const char* from = nullptr) {
    std::FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return "<could not open " + path + ">";
    char line[8192];
    bool started = from == nullptr;
    std::string found;
    while (std::fgets(line, sizeof(line), f)) {
        const std::string text(line);
        const std::size_t cut = text.find("//");
        const std::string code = cut == std::string::npos ? text : text.substr(0, cut);
        if (!started) {
            started = code.find(from) != std::string::npos;
            continue;
        }
        if (code.find(token) != std::string::npos) {
            found = text;
            break;
        }
    }
    std::fclose(f);
    return found;
}

// gebrd_cta's work-group barriers are reached by every partition only when no partition returns
// early (I2). Metal tolerates a partial barrier (W2c's revert of ormqx_cta stayed green there), so
// this source scan, not a numerical case, holds the property: from the kernel's launch to the end
// of its anonymous namespace no code line returns, and `live` reads the mask.
TEST(GebrdCtaSource, NoEarlyReturnInThePackedKernelAndLiveReadsTheMask) {
    const std::string path = BATCHLAS_GEBRD_CTA_CC_PATH;
    const char* launch = "parallel_for<GebrdCTAKernel";
    // Positive controls first: every assertion below is an absence.
    ASSERT_FALSE(first_code_line(path, launch).empty()) << path << " does not hold gebrd_cta's kernel launch";
    ASSERT_FALSE(first_code_line(path, "it.barrier(", launch).empty()) << "no work-group barrier after the launch";
    const std::string live = first_code_line(path, "const bool live", launch);
    ASSERT_FALSE(live.empty()) << "the packed kernel has no live predicate";
    EXPECT_FALSE(first_code_line(path, "mask[prob_id]", launch).empty()) << "live does not read the mask: " << live;
    std::FILE* f = std::fopen(path.c_str(), "r");
    ASSERT_NE(f, nullptr) << path;
    char line[8192];
    bool in_kernel = false;
    while (std::fgets(line, sizeof(line), f)) {
        const std::string text(line);
        const std::size_t cut = text.find("//");
        const std::string code = cut == std::string::npos ? text : text.substr(0, cut);
        if (code.find(launch) != std::string::npos) in_kernel = true;
        if (!in_kernel) continue;
        if (text.rfind("} // namespace", 0) == 0 || text.rfind("}  // namespace", 0) == 0) break;  // the kernel's TU scope ends
        EXPECT_EQ(code.find("return"), std::string::npos) << "a return inside gebrd_cta's packed kernel: " << text;
    }
    std::fclose(f);
}
#endif  // BATCHLAS_GEBRD_CTA_CC_PATH

}  // namespace
