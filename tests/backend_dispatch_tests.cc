#include <gtest/gtest.h>

#include <batchlas/backend_config.h>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/mempool.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include <cmath>
#include <cstdlib>
#include <complex>
#include <memory>
#include <string>
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
// every in-tree TU and the documented consumer spelling are written.
//
// The three enums in <batchlas/util/sycl-device-queue.hh> and their per-enum
// operator<< now live in namespace batchlas, alongside enums.hh's constrained
// enum-streaming template. Both are exact matches for `os << policy`, and
// partial ordering picks the per-enum overloads because their parameter is a
// concrete enum type rather than a deduced `E`. So this still compiles only as
// long as those overloads stay strictly more specialised -- what changed with
// the WP1 namespace move is that both candidates are now reachable by ADL on
// the enum argument from ANY TU, not only from one carrying the using-directive.
// It is a compile-time test wearing an assertion.
TEST(BackendDispatch, EnumsStreamFromInsideTheNamespace) {
    std::ostringstream os;
    os << Vendor::NVIDIA << ' ' << DeviceType::GPU << ' ' << Policy::SYNC << ' '
       << Backend::CUDA << ' ' << Uplo::Lower;
    EXPECT_EQ(os.str(), "NVIDIA GPU SYNC CUDA Lower");
}

TEST(BackendDispatch, AvailabilityMatchesBuildConfiguration) {
    EXPECT_EQ(Queue::backend_available(Backend::CUDA), bool(BATCHLAS_HAS_CUDA_BACKEND));
    EXPECT_EQ(Queue::backend_available(Backend::ROCM), bool(BATCHLAS_HAS_ROCM_BACKEND));
    EXPECT_FALSE(Queue::backend_available(Backend::MKL));
    EXPECT_EQ(Queue::backend_available(Backend::NETLIB), bool(BATCHLAS_HAS_HOST_BACKEND));
    // SYCL is the vendor-free native GPU backend (AdaptiveCpp Metal).
    EXPECT_EQ(Queue::backend_available(Backend::SYCL), bool(BATCHLAS_HAS_SYCL_BACKEND));

    // AUTO is a request, not a target; MAGMA has nothing behind it.
    EXPECT_FALSE(Queue::backend_available(Backend::AUTO));
    EXPECT_FALSE(Queue::backend_available(Backend::MAGMA));
}

// Pinning an absent backend must fail where the mistake is, not later at the
// first call with no context about what went wrong.
TEST(BackendDispatch, PinningAnAbsentBackendThrowsImmediately) {
    Queue q;
    EXPECT_THROW(q.set_backend(Backend::MAGMA), std::runtime_error);
#if !BATCHLAS_HAS_SYCL_BACKEND
    EXPECT_THROW(q.set_backend(Backend::SYCL), std::runtime_error);
#endif
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
    (void)C_deduced.view().fill_zeros(q);
    (void)C_explicit.view().fill_zeros(q);
    q.wait();

    // Deduced from the queue.
    (void)gemm(q, A.view(), B.view(), C_deduced.view(), 1.0f, 0.0f,
         Transpose::NoTrans, Transpose::NoTrans);
    q.wait();

    // Same call, backend named explicitly.
    switch (resolved_backend(q)) {
#if BATCHLAS_HAS_CUDA_BACKEND
        case Backend::CUDA:
            (void)gemm<Backend::CUDA>(q, A.view(), B.view(), C_explicit.view(), 1.0f, 0.0f,
                                Transpose::NoTrans, Transpose::NoTrans);
            break;
#endif
#if BATCHLAS_HAS_SYCL_BACKEND
        case Backend::SYCL:
            (void)gemm<Backend::SYCL>(q, A.view(), B.view(), C_explicit.view(), 1.0f, 0.0f,
                                Transpose::NoTrans, Transpose::NoTrans);
            break;
#endif
#if BATCHLAS_HAS_HOST_BACKEND
        case Backend::NETLIB:
            (void)gemm<Backend::NETLIB>(q, A.view(), B.view(), C_explicit.view(), 1.0f, 0.0f,
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
#if BATCHLAS_HAS_SYCL_BACKEND
        case Backend::SYCL:
            explicit_size = geqrf_buffer_size<Backend::SYCL>(q, A.view(), tau.to_span());
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
    (void)C_owning.view().fill_zeros(q);
    (void)C_view.view().fill_zeros(q);
    q.wait();

    (void)gemm(q, A, B, C_owning, 1.0f, 0.0f, Transpose::NoTrans, Transpose::NoTrans);
    (void)gemm(q, A.view(), B.view(), C_view.view(), 1.0f, 0.0f, Transpose::NoTrans, Transpose::NoTrans);
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
    (void)C_short.view().fill_zeros(q);
    (void)C_full.view().fill_zeros(q);
    q.wait();

    // gemm's last parameter, ComputePrecision, is defaulted in the declaration.
    (void)gemm(q, A.view(), B.view(), C_short.view(), 1.0f, 0.0f,
         Transpose::NoTrans, Transpose::NoTrans);
    (void)gemm(q, A.view(), B.view(), C_full.view(), 1.0f, 0.0f,
         Transpose::NoTrans, Transpose::NoTrans, ComputePrecision::Default);
    q.wait();

    for (int i = 0; i < n * n; ++i) {
        ASSERT_FLOAT_EQ(C_short.view().data_ptr()[i], C_full.view().data_ptr()[i]) << "at " << i;
    }
}

// ---------------------------------------------------------------------------
// A GPU without fp64 (AdaptiveCpp Metal): Backend::SYCL compiles float and
// complex<float> only, so double data on such a queue must throw a named
// batchlas::unsupported before any launch, through the ORDINARY spellings (the
// variadic dispatch overload and the option structs; agent guide "API traps").
// Removing the gate does not compile-fail quietly: the <SYCL,double> symbols do
// not exist, so the link breaks.
// ---------------------------------------------------------------------------
#if BATCHLAS_HAS_SYCL_BACKEND && !BATCHLAS_HAS_DEVICE_FP64
namespace {

std::unique_ptr<Queue> fp64less_gpu_queue() {
    try {
        auto q = std::make_unique<Queue>(Device("gpu"));
        if (q->device().type == DeviceType::GPU && !q->device().supports_fp64() &&
            q->backend() == Backend::SYCL) {
            return q;
        }
    } catch (const std::exception&) {
    }
    return nullptr;
}

// Host-filled SPD matrix (diagonally dominant), with a poison value in the
// unreferenced triangle so a launch that did run would show up.
template <typename T>
Matrix<T, MatrixFormat::Dense> spd_matrix(int n, int batch) {
    Matrix<T, MatrixFormat::Dense> m(n, n, batch);
    auto v = m.view();
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                v.data_ptr()[b * v.stride() + j * v.ld() + i] =
                    i == j ? T(n + 2) : T(0.25 / (1 + i + j));
    return m;
}

template <typename T>
bool unchanged(const Matrix<T, MatrixFormat::Dense>& a, const Matrix<T, MatrixFormat::Dense>& b) {
    const auto va = a.view(), vb = b.view();
    for (int k = 0; k < va.stride() * va.batch_size(); ++k)
        if (va.data_ptr()[k] != vb.data_ptr()[k]) return false;
    return true;
}

// The refusal names the op and the way out, so the caller can act on it.
template <typename F>
void expect_fp64_refusal(const char* op, F&& call) {
    try {
        call();
        ADD_FAILURE() << op << " with double data on an fp64-less GPU queue did not throw";
    } catch (const batchlas::unsupported& e) {
        const std::string what = e.what();
        EXPECT_NE(what.find(op), std::string::npos) << what;
        EXPECT_NE(what.find("fp64"), std::string::npos) << what;
        EXPECT_NE(what.find("CPU Queue"), std::string::npos) << what;
    }
}

}  // namespace

TEST(NoDeviceFp64, GemmDoubleThrowsAndFloatRuns) {
    auto q = fp64less_gpu_queue();
    if (!q) GTEST_SKIP() << "no GPU without fp64 resolving to Backend::SYCL";
    const int n = 8, batch = 2;
    auto A = spd_matrix<double>(n, batch), B = spd_matrix<double>(n, batch);
    auto C = spd_matrix<double>(n, batch), C0 = spd_matrix<double>(n, batch);

    expect_fp64_refusal("gemm", [&] {
        (void)gemm(*q, A.view(), B.view(), C.view(), 1.0, 1.0, Transpose::NoTrans, Transpose::NoTrans);
    });
    expect_fp64_refusal("gemm", [&] { (void)gemm(*q, A, B, C, 1.0, 1.0, Transpose::NoTrans, Transpose::NoTrans); });
    expect_fp64_refusal("gemm", [&] { (void)gemm(*q, A, B, C, {.alpha = 2.0, .beta = 1.0}); });
    q->wait();
    EXPECT_TRUE(unchanged(C, C0)) << "a refused call must not have launched";

    // The same float call on the same queue runs: the refusal is by type, not by queue.
    auto Af = spd_matrix<float>(n, batch), Bf = spd_matrix<float>(n, batch);
    Matrix<float, MatrixFormat::Dense> Cf(n, n, batch);
    (void)Cf.view().fill_zeros(*q);
    EXPECT_NO_THROW((void)gemm(*q, Af, Bf, Cf, {.alpha = 1.0f}));
    q->wait();
    double c00 = double(n + 2) * double(n + 2);
    for (int k = 1; k < n; ++k) c00 += double(float(0.25 / (1 + k))) * double(float(0.25 / (1 + k)));
    EXPECT_NEAR(Cf.view().data_ptr()[0], c00, 1e-5 * c00);
}

TEST(NoDeviceFp64, PotrfDoubleThrowsOnEverySpelling) {
    auto q = fp64less_gpu_queue();
    if (!q) GTEST_SKIP() << "no GPU without fp64 resolving to Backend::SYCL";
    const int n = 8, batch = 2;
    auto A = spd_matrix<double>(n, batch), A0 = spd_matrix<double>(n, batch);
    UnifiedVector<std::byte> ws(1 << 16);

    expect_fp64_refusal("potrf", [&] { (void)potrf(*q, A, {.uplo = Uplo::Lower}); });
    expect_fp64_refusal("potrf", [&] { (void)potrf(*q, A, PotrfOptions{}, ws.to_span()); });
    expect_fp64_refusal("potrf", [&] { (void)potrf(*q, A.view(), Uplo::Lower, ws.to_span()); });
    expect_fp64_refusal("potrf_buffer_size", [&] { (void)potrf_buffer_size(*q, A.view(), Uplo::Lower); });
    q->wait();
    EXPECT_TRUE(unchanged(A, A0));

    // The advice in the message works: the same data factorises on a CPU queue.
    Queue cpu(Device("cpu"));
    ASSERT_EQ(cpu.backend(), Backend::NETLIB);
    (void)potrf(cpu, A, {.uplo = Uplo::Lower});
    cpu.wait();
    EXPECT_NEAR(A.view().data_ptr()[0], std::sqrt(double(n + 2)), 1e-12);
}

TEST(NoDeviceFp64, SyevDoubleThrowsOnEverySpelling) {
    auto q = fp64less_gpu_queue();
    if (!q) GTEST_SKIP() << "no GPU without fp64 resolving to Backend::SYCL";
    const int n = 8, batch = 2;
    auto A = spd_matrix<double>(n, batch), A0 = spd_matrix<double>(n, batch);
    UnifiedVector<double> W(n * batch, -7.0);
    UnifiedVector<std::byte> ws(1 << 20);

    expect_fp64_refusal("syev", [&] { (void)syev(*q, A, W.to_span(), {.jobz = JobType::EigenVectors}); });
    expect_fp64_refusal("syev", [&] {
        (void)syev(*q, A.view(), W.to_span(), JobType::EigenVectors, Uplo::Lower, ws.to_span());
    });
    expect_fp64_refusal("syev_buffer_size", [&] {
        (void)syev_buffer_size(*q, A.view(), W.to_span(), JobType::EigenVectors, Uplo::Lower);
    });
    q->wait();
    EXPECT_TRUE(unchanged(A, A0));
    EXPECT_EQ(W[0], -7.0);

    // complex<double> is refused the same way.
    Matrix<std::complex<double>, MatrixFormat::Dense> Z(n, n, batch);
    expect_fp64_refusal("syev", [&] { (void)syev(*q, Z, W.to_span(), {.jobz = JobType::NoEigenVectors}); });
}
#endif

// At ACPP_ADAPTIVITY_LEVEL=0 AdaptiveCpp JITs every kernel of a translation unit together,
// double ones included, which Metal cannot compile: a Queue on the GPU must refuse up front
// rather than fail every float launch later. ctest runs this suite a second time with the
// variable set (backend_dispatch_tests_adaptivity0); unset, the same Queue must construct.
TEST(MetalAdaptivityGuard, LevelZeroRefusesAGpuQueue) {
    Device gpu = Device::default_device();
    try {
        gpu = Device("gpu");
    } catch (const std::exception&) {
        GTEST_SKIP() << "no GPU device";
    }
    if (gpu.type != DeviceType::GPU || gpu.supports_fp64()) GTEST_SKIP() << "not a GPU without fp64";
    const char* level = std::getenv("ACPP_ADAPTIVITY_LEVEL");
    if (level == nullptr || std::string(level) != "0") {
        EXPECT_NO_THROW(Queue{gpu});
        return;
    }
    try {
        Queue q(gpu);
        ADD_FAILURE() << "a GPU Queue at ACPP_ADAPTIVITY_LEVEL=0 constructed";
    } catch (const batchlas::unsupported& e) {
        EXPECT_NE(std::string(e.what()).find("ACPP_ADAPTIVITY_LEVEL=0"), std::string::npos) << e.what();
    }
    // The guard is the GPU's: a CPU Queue (the double path) still constructs.
    EXPECT_NO_THROW(Queue{Device("cpu")});
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
