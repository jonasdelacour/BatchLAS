#include <gtest/gtest.h>

#include <batchlas/backend_config.h>
#include <blas/functions.hh>
#include <blas/matrix.hh>
#include <util/mempool.hh>
#include <util/sycl-device-queue.hh>

#include <cmath>
#include <sstream>
#include <vector>

using namespace batchlas;

namespace {

// The backend a default Queue resolves to on this build/device. Every
// comparison below is against this, so the test says the same thing whether it
// runs on a CUDA box or a host-only one.
Backend resolved_backend(const Queue& q) { return q.backend(); }

Matrix<float, MatrixFormat::Dense> make_matrix(int n, int batch, float scale) {
    Matrix<float, MatrixFormat::Dense> m(n, n, batch);
    auto v = m.view();
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                v.data_ptr()[b * v.stride() + j * v.ld() + i] =
                    scale * static_cast<float>((i * 7 + j * 3 + b) % 11) / 11.0f;
            }
        }
    }
    return m;
}

}  // namespace

TEST(BackendDispatch, AutoResolvesToACompiledBackendAndIsStable) {
    Queue q;
    EXPECT_EQ(q.requested_backend(), Backend::AUTO);

    const Backend b = q.backend();
    EXPECT_NE(b, Backend::AUTO) << "backend() must never hand back the request";
    EXPECT_TRUE(Queue::backend_available(b));
    EXPECT_EQ(q.backend(), b) << "resolution must be stable across queries";
}

// Streaming an enum has to work from a TU shaped like this one -- both header
// families included, `using namespace batchlas;` at the top -- which is how
// every in-tree TU and the documented consumer spelling are written. The three
// enums in <util/sycl-device-queue.hh> live in the global namespace and get
// their own operator<< there; batchlas' generic enum operator<< is a viable
// candidate for them too (its `to_string(e)` constraint is satisfied by ADL),
// so this compiles only as long as the global overloads stay strictly more
// specialised. It is a compile-time test wearing an assertion.
TEST(BackendDispatch, EnumsStreamFromInsideTheNamespace) {
    std::ostringstream os;
    os << Vendor::NVIDIA << ' ' << DeviceType::GPU << ' ' << Policy::SYNC << ' '
       << Backend::CUDA << ' ' << Uplo::Lower;
    EXPECT_EQ(os.str(), "NVIDIA GPU SYNC CUDA Lower");
}

TEST(BackendDispatch, AvailabilityMatchesBuildConfiguration) {
    EXPECT_EQ(Queue::backend_available(Backend::CUDA), bool(BATCHLAS_HAS_CUDA_BACKEND));
    EXPECT_EQ(Queue::backend_available(Backend::ROCM), bool(BATCHLAS_HAS_ROCM_BACKEND));
    EXPECT_EQ(Queue::backend_available(Backend::MKL), bool(BATCHLAS_HAS_MKL_BACKEND));
    EXPECT_EQ(Queue::backend_available(Backend::NETLIB), bool(BATCHLAS_HAS_HOST_BACKEND));

    // AUTO is a request, not a target; MAGMA and SYCL have nothing behind them.
    EXPECT_FALSE(Queue::backend_available(Backend::AUTO));
    EXPECT_FALSE(Queue::backend_available(Backend::MAGMA));
    EXPECT_FALSE(Queue::backend_available(Backend::SYCL));
}

// Pinning an absent backend must fail where the mistake is, not later at the
// first call with no context about what went wrong.
TEST(BackendDispatch, PinningAnAbsentBackendThrowsImmediately) {
    Queue q;
    EXPECT_THROW(q.set_backend(Backend::MAGMA), std::runtime_error);
    EXPECT_THROW(q.set_backend(Backend::SYCL), std::runtime_error);
    // The failed attempts left the queue alone.
    EXPECT_EQ(q.requested_backend(), Backend::AUTO);
    EXPECT_TRUE(Queue::backend_available(q.backend()));
}

TEST(BackendDispatch, PinningAndReturningToAuto) {
    Queue q;
    const Backend automatic = q.backend();

    q.set_backend(automatic);
    EXPECT_EQ(q.requested_backend(), automatic);
    EXPECT_EQ(q.backend(), automatic);

    q.set_backend(Backend::AUTO);
    EXPECT_EQ(q.requested_backend(), Backend::AUTO);
    EXPECT_EQ(q.backend(), automatic) << "AUTO must re-resolve, not go stale";
}

TEST(BackendDispatch, BackendSurvivesMoveAssignment) {
    Queue a;
    const Backend pinned = a.backend();
    a.set_backend(pinned);

    Queue b;
    b = std::move(a);
    EXPECT_EQ(b.requested_backend(), pinned);
    EXPECT_EQ(b.backend(), pinned);
}

// The point of the whole phase: `gemm(ctx, ...)` must produce exactly what
// `gemm<Backend::X>(ctx, ...)` produces, for the X the queue resolved to.
TEST(BackendDispatch, DeducedCallMatchesExplicitBackendCall) {
    Queue q;
    const int n = 16, batch = 3;

    auto A = make_matrix(n, batch, 1.0f);
    auto B = make_matrix(n, batch, 0.5f);
    Matrix<float, MatrixFormat::Dense> C_deduced(n, n, batch);
    Matrix<float, MatrixFormat::Dense> C_explicit(n, n, batch);
    C_deduced.view().fill_zeros(q);
    C_explicit.view().fill_zeros(q);
    q.wait();

    // Deduced from the queue.
    gemm(q, A.view(), B.view(), C_deduced.view(), 1.0f, 0.0f,
         Transpose::NoTrans, Transpose::NoTrans);
    q.wait();

    // Same call, backend named explicitly.
    switch (resolved_backend(q)) {
#if BATCHLAS_HAS_CUDA_BACKEND
        case Backend::CUDA:
            gemm<Backend::CUDA>(q, A.view(), B.view(), C_explicit.view(), 1.0f, 0.0f,
                                Transpose::NoTrans, Transpose::NoTrans);
            break;
#endif
#if BATCHLAS_HAS_HOST_BACKEND
        case Backend::NETLIB:
            gemm<Backend::NETLIB>(q, A.view(), B.view(), C_explicit.view(), 1.0f, 0.0f,
                                  Transpose::NoTrans, Transpose::NoTrans);
            break;
#endif
        default:
            GTEST_SKIP() << "no explicit-call arm compiled for this backend";
    }
    q.wait();

    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                const auto idx = b * C_deduced.view().stride() + j * C_deduced.view().ld() + i;
                ASSERT_FLOAT_EQ(C_deduced.view().data_ptr()[idx],
                                C_explicit.view().data_ptr()[idx])
                    << "mismatch at (" << i << "," << j << ") batch " << b;
            }
        }
    }
}

// A *_buffer_size going through dispatch must report the same figure as the
// explicitly-backended one -- otherwise the concise API would size workspaces
// for a backend other than the one that runs.
TEST(BackendDispatch, DeducedBufferSizeMatchesExplicit) {
    Queue q;
    auto A = make_matrix(24, 2, 1.0f);
    UnifiedVector<float> tau(24 * 2);

    const size_t deduced = geqrf_buffer_size(q, A.view(), tau.to_span());

    size_t explicit_size = 0;
    switch (resolved_backend(q)) {
#if BATCHLAS_HAS_CUDA_BACKEND
        case Backend::CUDA:
            explicit_size = geqrf_buffer_size<Backend::CUDA>(q, A.view(), tau.to_span());
            break;
#endif
#if BATCHLAS_HAS_HOST_BACKEND
        case Backend::NETLIB:
            explicit_size = geqrf_buffer_size<Backend::NETLIB>(q, A.view(), tau.to_span());
            break;
#endif
        default:
            GTEST_SKIP() << "no explicit-call arm compiled for this backend";
    }

    EXPECT_EQ(deduced, explicit_size);
}

// Dispatch forwards into the whole overload set, not just the MatrixView
// primary, so owning Matrix arguments still bind to the Matrix-taking
// forwarders and give the same answer.
TEST(BackendDispatch, OwningMatricesBindThroughDispatch) {
    Queue q;
    const int n = 12, batch = 2;
    auto A = make_matrix(n, batch, 1.0f);
    auto B = make_matrix(n, batch, 0.75f);
    Matrix<float, MatrixFormat::Dense> C_owning(n, n, batch);
    Matrix<float, MatrixFormat::Dense> C_view(n, n, batch);
    C_owning.view().fill_zeros(q);
    C_view.view().fill_zeros(q);
    q.wait();

    gemm(q, A, B, C_owning, 1.0f, 0.0f, Transpose::NoTrans, Transpose::NoTrans);
    gemm(q, A.view(), B.view(), C_view.view(), 1.0f, 0.0f, Transpose::NoTrans, Transpose::NoTrans);
    q.wait();

    for (int i = 0; i < n * n * batch; ++i) {
        ASSERT_FLOAT_EQ(C_owning.view().data_ptr()[i], C_view.view().data_ptr()[i]) << "at " << i;
    }
}

// Defaulted parameters must survive the variadic forwarding -- this is the part
// that would silently break if the macro restated the signature instead.
TEST(BackendDispatch, DefaultArgumentsSurviveForwarding) {
    Queue q;
    const int n = 8, batch = 1;
    auto A = make_matrix(n, batch, 1.0f);
    auto B = make_matrix(n, batch, 0.25f);
    Matrix<float, MatrixFormat::Dense> C_short(n, n, batch);
    Matrix<float, MatrixFormat::Dense> C_full(n, n, batch);
    C_short.view().fill_zeros(q);
    C_full.view().fill_zeros(q);
    q.wait();

    // gemm's last parameter, ComputePrecision, is defaulted in the declaration.
    gemm(q, A.view(), B.view(), C_short.view(), 1.0f, 0.0f,
         Transpose::NoTrans, Transpose::NoTrans);
    gemm(q, A.view(), B.view(), C_full.view(), 1.0f, 0.0f,
         Transpose::NoTrans, Transpose::NoTrans, ComputePrecision::Default);
    q.wait();

    for (int i = 0; i < n * n; ++i) {
        ASSERT_FLOAT_EQ(C_short.view().data_ptr()[i], C_full.view().data_ptr()[i]) << "at " << i;
    }
}

// ---------------------------------------------------------------------------
// Interop surface. This is the only TU in the tree that includes
// <batchlas/sycl_interop.hh>, which is deliberately unreachable from
// <batchlas.hh>; compiling it here is half the point, since a header nothing
// includes is a header nothing notices breaking. The include sits at the bottom
// rather than the top so the rest of the file keeps proving that the public API
// needs no SYCL types. See docs/cpp-api.md "Interop with CUDA and with your own
// SYCL".
// ---------------------------------------------------------------------------
#include <batchlas/sycl_interop.hh>

TEST(InteropTest, NativeHandleMatchesBackend) {
    Queue q;
    if (q.device().type == DeviceType::GPU && q.backend() == Backend::CUDA) {
        EXPECT_NE(q.native_handle(), nullptr) << "a CUDA Queue must expose its CUstream";
    }
    // A CPU queue has no native stream to hand out.
    Queue cpu(Device("cpu"));
    EXPECT_EQ(cpu.native_handle(), nullptr);
}

TEST(InteropTest, EventRoundTripsThroughSycl) {
    Queue q;
    q.wait();
    sycl::event se = batchlas::sycl_event(q.get_event());
    Event back = batchlas::event_from_sycl(se);
    q.enqueue(back);
    EXPECT_NO_THROW(q.wait());
}

TEST(InteropTest, SyclQueueIsTheSameQueue) {
    Queue q;
    sycl::queue& sq = batchlas::sycl_queue(q);
    EXPECT_EQ(&sq, &batchlas::sycl_queue(q));
}
