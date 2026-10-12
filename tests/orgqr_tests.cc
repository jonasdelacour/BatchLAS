#include <gtest/gtest.h>
#include <batchlas/blas/linalg.hh>
#include <batchlas/util/sycl-vector.hh>
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <vector>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/env.hh>

using namespace batchlas;

template <typename T, Backend B>
struct OrgqrConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

#include "test_utils.hh"
#include <batchlas/verify/tolerance.hh>
using OrgqrTestTypes = typename test_utils::backend_types<OrgqrConfig>::type;

template <typename Config>
class OrgqrTest : public test_utils::BatchLASTest<Config> {
protected:
    Transpose trans_op = test_utils::is_complex<typename Config::ScalarType>() ? Transpose::ConjTrans : Transpose::Trans;
};

TYPED_TEST_SUITE(OrgqrTest, OrgqrTestTypes);

TYPED_TEST(OrgqrTest, SingleMatrix) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 4;

    Matrix<T, MatrixFormat::Dense> A = Matrix<T, MatrixFormat::Dense>::Random(n, n);
    UnifiedVector<T> tau(n);
    UnifiedVector<std::byte> ws_geqrf(geqrf_buffer_size(*this->ctx, A.view(), tau.to_span()));
    (void)geqrf(*this->ctx, A.view(), tau.to_span(), ws_geqrf.to_span());
    this->ctx->wait();

    UnifiedVector<std::byte> ws_orgqr(orgqr_buffer_size(*this->ctx, A.view(), tau.to_span()));
    (void)orgqr(*this->ctx, A.view(), tau.to_span(), ws_orgqr.to_span());
    this->ctx->wait();

    using MV = MatrixView<T, MatrixFormat::Dense>;
    EXPECT_VERIFY(T, verify::Check::orthogonality, A.rows(), MV::orthogonality(*this->ctx, A.view()));
}

TYPED_TEST(OrgqrTest, BatchedMatrices) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 4;
    const int batch = 3;

    Matrix<T, MatrixFormat::Dense> A = Matrix<T, MatrixFormat::Dense>::Random(n, n, false, batch);
    UnifiedVector<T> tau(n * batch);
    UnifiedVector<std::byte> ws_geqrf(geqrf_buffer_size(*this->ctx, A.view(), tau.to_span()));
    (void)geqrf(*this->ctx, A.view(), tau.to_span(), ws_geqrf.to_span());
    this->ctx->wait();

    UnifiedVector<std::byte> ws_orgqr(orgqr_buffer_size(*this->ctx, A.view(), tau.to_span()));
    (void)orgqr(*this->ctx, A.view(), tau.to_span(), ws_orgqr.to_span());
    this->ctx->wait();

    using MV = MatrixView<T, MatrixFormat::Dense>;
    EXPECT_VERIFY(T, verify::Check::orthogonality, A.rows(), MV::orthogonality(*this->ctx, A.view(), A.view().all_items()));
}


// ===========================================================================
// WP5 -- ORGQR AGAINST A HOST REFERENCE.
//
// The two tests above are the pre-WP5 coverage: n = 4, Q^H Q == I through a
// device gemm, item 0 and its two neighbours. Three things they cannot see, and
// each has a recorded precedent in this repository:
//
//   * Q^H Q == I ALONE DOES NOT SAY Q IS *THIS* A's Q. Any orthonormal matrix
//     passes it -- a driver that dropped a reflector, or applied the reflectors
//     in the wrong order, still returns something orthonormal. So the test below
//     also checks || Q R - A ||_F / ||A||_F with R read out of the geqrf factor.
//
//   * ld == rows AND stride == ld*cols. Matrix<T>(n, n, batch) has both, so the
//     two lines of every launcher that read A.ld() and A.stride() were
//     structurally unfalsifiable here. trsm_native.cc records that exact
//     failure: the 6-arg MatrixView constructor defaults stride to ld*cols, after
//     which every batch item but the first reads the wrong matrix.
//   * m == n ONLY. orgqr's whole point is the first n columns of an m x n Q.
// The oracle is MatrixView's promoted-precision check on the input this file generated,
// never the vendor: a vendor reference is inert in the vendor-free build.
// ===========================================================================
namespace orgqr_wp5 {

inline verify::Slack recon_slack(int m, int n) {
    return {double(m + n) / (16.0 * m), "kept from this file's 0.5 (m+n) eps tolerance"};
}
inline verify::Slack orth_slack(int m, int n) {
    return {std::min(1.0, double(m + n) * std::sqrt(double(n)) / (32.0 * m)),
            "kept from this file's 0.5 (m+n) eps / sqrt(n) orthonormality tolerance, clamped at the library bound"};
}

}  // namespace orgqr_wp5

TYPED_TEST(OrgqrTest, QIsOrthonormalAndReconstructsAAtEveryBatchItem) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    using namespace orgqr_wp5;

    struct S { int m, n, batch; };
    // m == n and m > n; one n that is a multiple of every block width this tree
    // uses and one that is not; and batches > 1 so a stride defect has somewhere
    // to hide.
    const S shapes[] = {{32, 32, 3}, {40, 24, 3}, {64, 33, 2}, {96, 96, 2}};
    for (const S& s : shapes) {
        const int ld = s.m + 5;
        const int stride = ld * s.n + 11;
        using MV = MatrixView<T, MatrixFormat::Dense>;
        UnifiedVector<T> buf(static_cast<size_t>(stride) * s.batch, MV::make(-9.75e3, 4.5e3));
        UnifiedVector<T*> ptrs(static_cast<size_t>(s.batch), nullptr);
        MV V(buf.data(), s.m, s.n, ld, stride, s.batch, ptrs.data());
        const auto R = Matrix<T, MatrixFormat::Dense>::Random(s.m, s.n, false, s.batch, 4242u + 13u * unsigned(s.m) + unsigned(s.n));
        MV::copy(*this->ctx, V, R.view()).wait();
        UnifiedVector<T> A0(buf);
        UnifiedVector<T> tau(static_cast<size_t>(std::min(s.m, s.n)) * s.batch, MV::make(-12345.0, -12345.0));

        UnifiedVector<std::byte> wg(std::max<size_t>(
            1, geqrf_buffer_size<B, T>(*this->ctx, V, tau.to_span())));
        (void)geqrf<B, T>(*this->ctx, V, tau.to_span(), wg.to_span());
        this->ctx->wait();
        UnifiedVector<T> F(buf);   // orgqr overwrites it

        UnifiedVector<std::byte> wo(std::max<size_t>(
            1, orgqr_buffer_size<B, T>(*this->ctx, V, tau.to_span())));
        (void)orgqr<B, T>(*this->ctx, V, tau.to_span(), wo.to_span());
        this->ctx->wait();

        const MV A0v(A0.data(), s.m, s.n, ld, stride, s.batch), Fv(F.data(), s.m, s.n, ld, stride, s.batch);
        EXPECT_VERIFY_SLACK(T, verify::Check::orthogonality, s.m, MV::orthogonality(*this->ctx, V, V.all_items()),
                            orth_slack(s.m, s.n))
            << "Q is not orthonormal (m=" << s.m << " n=" << s.n << ")";
        EXPECT_VERIFY_SLACK(T, verify::Check::factorization, s.m,
                            MV::qr_reconstruction(*this->ctx, A0v, V, Fv, V.all_items()), recon_slack(s.m, s.n))
            << "Q R != A (m=" << s.m << " n=" << s.n << ") -- Q is orthonormal but it is not THIS A's Q";
        // A broadcast of item 0 over the batch would pass every check above.
        if (s.batch > 1) {
            bool differ = false;
            for (int j = 0; j < s.n && !differ; ++j)
                for (int i = 0; i < s.m && !differ; ++i)
                    if (std::abs(MV::up(buf[static_cast<size_t>(j) * ld + i]) -
                              MV::up(buf[static_cast<size_t>(s.batch - 1) * stride +
                                      static_cast<size_t>(j) * ld + i])) > 0.0)
                        differ = true;
            EXPECT_TRUE(differ) << "the first and last batch items' Q are identical, so this "
                                   "shape cannot see a batch-stride defect";
        }
        if (this->HasFailure()) return;
    }
}

// geqrf then orgqr back to back on an out-of-order Queue, no wait between them, vendor pinned:
// the vendor orgqr must order its per-item calls after the geqrf still in flight. The factors
// it is checked against come from the same input on the in-order Queue.
TYPED_TEST(OrgqrTest, OutOfOrderQueueGeqrfThenOrgqrVendor) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    using MV = MatrixView<T, MatrixFormat::Dense>;
    const ScopedEnvVar geqrf_route("BATCHLAS_GEQRF_ROUTE", "vendor");
    const ScopedEnvVar orgqr_route("BATCHLAS_ORGQR_ROUTE", "vendor");
    const int m = 96, n = 96, batch = 16, ld = m + 3, stride = ld * n + 7;
    const auto R = Matrix<T, MatrixFormat::Dense>::Random(m, n, false, batch, 777u);
    UnifiedVector<T> a0(static_cast<size_t>(stride) * batch, T(0)), f(a0.size()), q(a0.size());
    UnifiedVector<T*> fptrs(batch, nullptr), qptrs(batch, nullptr);
    const MV A0v(a0.data(), m, n, ld, stride, batch), Fv(f.data(), m, n, ld, stride, batch, fptrs.data()),
        Qv(q.data(), m, n, ld, stride, batch, qptrs.data());
    MV::copy(*this->ctx, A0v, R.view()).wait();
    MV::copy(*this->ctx, Fv, A0v).wait();
    UnifiedVector<T> tau_ref(static_cast<size_t>(n) * batch), tau(static_cast<size_t>(n) * batch);
    UnifiedVector<std::byte> wg(std::max<size_t>(1, geqrf_buffer_size<B, T>(*this->ctx, Fv, tau_ref.to_span())));
    (void)geqrf<B, T>(*this->ctx, Fv, tau_ref.to_span(), wg.to_span());
    this->ctx->wait();

    Queue ooo(*this->ctx, false);
    MV::copy(*this->ctx, Qv, A0v).wait();
    UnifiedVector<std::byte> wg2(std::max<size_t>(1, geqrf_buffer_size<B, T>(ooo, Qv, tau.to_span())));
    UnifiedVector<std::byte> wo(std::max<size_t>(1, orgqr_buffer_size<B, T>(ooo, Qv, tau.to_span())));
    (void)geqrf<B, T>(ooo, Qv, tau.to_span(), wg2.to_span());
    (void)orgqr<B, T>(ooo, Qv, tau.to_span(), wo.to_span());
    ooo.wait();

    EXPECT_VERIFY(T, verify::Check::orthogonality, m, MV::orthogonality(*this->ctx, Qv, Qv.all_items()));
    EXPECT_VERIFY(T, verify::Check::factorization, m, MV::qr_reconstruction(*this->ctx, A0v, Qv, Fv, Qv.all_items()));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

