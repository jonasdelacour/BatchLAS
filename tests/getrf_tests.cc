// Native batched LU -- getrf, getrs and getri: both getrf tiers (CTA and
// blocked), both getrs arms (composed and fused narrow-RHS), and getri.
// Every numerical test drives the native dispatch entry points DIRECTLY against a
// HOST reference; the vendor is never a pivot oracle, because
// cublas{C,Z}getrfBatched pivots on the modulus where this library and LAPACK
// pivot on cabs1 = |Re| + |Im|.
// evidence: docs/perf/lu.md
#include <gtest/gtest.h>

#include <batchlas/blas/functions/getrf.hh>
#include <batchlas/blas/functions/getrs.hh>
#include <batchlas/blas/functions/getri.hh>
#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/trsm.hh>
#include "../src/select/vendor.hh"
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/settings.hh>

#include "test_utils.hh"

#include <batchlas/verify/residuals.hh>
// The only non-BatchLAS oracle here (verify::getrf_pivots): both tiers share lu_cabs1, so only
// TinyPivotsMatchLapackeOnUnstructuredData sees a defect in the pivot METRIC itself.
#include <batchlas/verify/reference.hh>

#include "../src/extensions/getrf_native.hh"
#include "../src/extensions/getrs_native.hh"
#include "../src/extensions/getri_native.hh"
#include "../src/ops/getrs/choice.hh"
#include "../src/ops/getri/choice.hh"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <vector>

using namespace batchlas;

namespace {

template <typename T>
using RealOf = typename batchlas::base_type<T>::type;


// Scale by a real factor without naming .real()/.imag() on a type without them.
template <class T> inline T scale(T v, double f) { return T(v * static_cast<RealOf<T>>(f)); }
template <class R> inline std::complex<R> scale(std::complex<R> v, double f) {
    return std::complex<R>(v.real() * static_cast<R>(f), v.imag() * static_cast<R>(f));
}

using verify::Check;
using verify::make;
using verify::nanmax;
using verify::up;
using verify::Rng;

// The fixtures hold raw pointers at padded ld and odd strides; these hand one item to the library.
template <class T>
VectorView<int32_t> pivots_of(const int* ipiv, int count) {
    return VectorView<int32_t>(const_cast<int32_t*>(reinterpret_cast<const int32_t*>(ipiv)), count, 1);
}
template <class T>
double factor_residual(const T* A0, const T* F, const int* ipiv, int m, int n, int ld) {
    return verify::getrf_residual(verify::view(A0, m, n, ld), verify::view(F, m, n, ld),
                                  pivots_of<T>(ipiv, std::min(m, n)));
}
template <class T>
double solve_residual(const T* A0, const T* X, const T* B0, int n, int nrhs, int lda, int ldb, Transpose op) {
    return verify::solve_residual(verify::view(A0, n, n, lda), verify::Shape::general, op,
                                  verify::view(X, n, nrhs, ldb), verify::view(B0, n, nrhs, ldb));
}
template <class T>
double inverse_residual(const T* A0, const T* C, int n, int lda, int ldc) {
    return verify::solve_residual(verify::view(A0, n, n, lda), verify::view(C, n, n, ldc),
                                  verify::view(static_cast<const T*>(nullptr), 0, 0, 1));
}
template <class T>
double lu_solve_residual(const T* F, const int* ipiv, const T* X, const T* B0, int n, int nrhs, int ldf, int ldb,
                         Transpose op) {
    return verify::lu_solve_residual(verify::view(F, n, n, ldf), pivots_of<T>(ipiv, n), op,
                                     verify::view(X, n, nrhs, ldb), verify::view(B0, n, nrhs, ldb));
}

bool verbose() {
    static const bool v = (std::getenv("BATCHLAS_TEST_VERBOSE") != nullptr);
    return v;
}

template <typename T, Backend B>
struct LuConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

// A batch of DISTINCT matrices with a PADDED ld and a stride that is NOT ld*cols,
// with the pad POISONED, so a launcher that lets MatrixView default the stride to
// ld*cols is falsifiable by default. The pointer array is not optional either:
// every vendor batched call dereferences data_ptrs_ and throws when it is empty.
template <typename T>
struct Lu {
    int n = 0, batch = 0, ld = 0, stride = 0;
    UnifiedVector<T> buf;        // working copy, overwritten by getrf
    std::vector<T> a0;           // the pristine input, same ld/stride
    UnifiedVector<T*> ptrs;
    UnifiedVector<int64_t> piv;  // the PUBLIC int64 span; CUDA/ROCm pack int32 into it
    UnifiedVector<int32_t> info;
    // The interchange list this matrix MUST produce, when it is known exactly
    // (the dominant-permuted construction). Empty when it is not.
    std::vector<int> expect_piv;
};

template <typename T>
void poison(Lu<T>& p) {
    std::copy(p.a0.begin(), p.a0.end(), p.buf.begin());
    std::fill(p.piv.begin(), p.piv.end(), int64_t(0x0BADBEEF0BADBEEFLL));
    std::fill(p.info.begin(), p.info.end(), int32_t(-12345));
}

template <typename T>
void alloc(Lu<T>& p, int n, int batch, int ld_pad, int stride_pad) {
    p.n = n; p.batch = batch;
    p.ld = n + ld_pad;
    p.stride = p.ld * n + stride_pad;
    p.buf = UnifiedVector<T>(static_cast<size_t>(p.stride) * batch, make<T>(-9.75e3, 4.5e3));
    p.ptrs = UnifiedVector<T*>(static_cast<size_t>(batch), nullptr);
    p.piv = UnifiedVector<int64_t>(static_cast<size_t>(n) * batch, int64_t(0));
    p.info = UnifiedVector<int32_t>(static_cast<size_t>(batch), int32_t(0));
}

// A RANDOM matrix: the pivot sequence is data-dependent, so only the residual and
// pivot-ratio oracles apply.
template <typename T>
Lu<T> make_random(int n, int batch, unsigned seed, int ld_pad = 5, int stride_pad = 11) {
    Lu<T> p;
    alloc(p, n, batch, ld_pad, stride_pad);
    Rng rg(seed);
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                p.buf[size_t(b) * p.stride + size_t(j) * p.ld + i] = make<T>(rg.next(), rg.next());
    p.a0.assign(p.buf.begin(), p.buf.end());
    poison(p);
    return p;
}

// THE MATRIX WITH A KNOWN-EXACT PIVOT SEQUENCE. A strictly column-diagonally-
// dominant B (|B(k,k)| = 4n, |B(i,k)| <= 1) whose rows are then permuted: column
// dominance survives elimination, so the winner at step k is known exactly and the
// expected interchange list is pure integer bookkeeping. cond(A) is O(1), which is
// why every getrs and getri residual runs on it too.
template <typename T>
Lu<T> make_dominant_permuted(int n, int batch, unsigned seed,
                             int ld_pad = 5, int stride_pad = 11) {
    Lu<T> p;
    alloc(p, n, batch, ld_pad, stride_pad);
    Rng rg(seed);

    // sigma: B's row r ends up at position sigma[r]. A CYCLIC SHIFT and not a
    // reversal: a reversal is its own inverse, so every test of a permutation
    // DIRECTION would be satisfied by the wrong direction too.
    std::vector<int> sigma(n);
    for (int r = 0; r < n; ++r) sigma[r] = (r + 1) % n;

    for (int b = 0; b < batch; ++b) {
        for (int r = 0; r < n; ++r) {
            const int dst = sigma[r];
            for (int j = 0; j < n; ++j) {
                const double re = rg.next();
                const double im = rg.next();
                T v = make<T>(re, im);
                if (j == r) v = make<T>(4.0 * double(n) * (re >= 0 ? 1.0 : -1.0), 0.0);
                p.buf[size_t(b) * p.stride + size_t(j) * p.ld + dst] = v;
            }
        }
    }
    p.a0.assign(p.buf.begin(), p.buf.end());

    // The expected interchange list, by simulating the SELECTION only.
    // home[i] = which B-row currently sits at position i.
    std::vector<int> home(n);
    for (int r = 0; r < n; ++r) home[sigma[r]] = r;
    p.expect_piv.resize(n);
    for (int k = 0; k < n; ++k) {
        int q = -1;
        for (int i = k; i < n; ++i) if (home[i] == k) { q = i; break; }
        p.expect_piv[k] = q + 1;              // 1-BASED, LAPACK ipiv
        std::swap(home[k], home[q]);
    }
    poison(p);
    return p;
}

template <typename T>
MatrixView<T, MatrixFormat::Dense> view_of(Lu<T>& p) {
    return MatrixView<T, MatrixFormat::Dense>(p.buf.data(), p.n, p.n, p.ld, p.stride, p.batch,
                                              p.ptrs.data());
}

// A right-hand-side / output block, same padding and poison discipline.
template <typename T>
struct Rhs {
    int n = 0, nrhs = 0, batch = 0, ld = 0, stride = 0;
    UnifiedVector<T> buf;
    std::vector<T> b0;
    UnifiedVector<T*> ptrs;
};

template <typename T>
Rhs<T> make_rhs(int n, int nrhs, int batch, unsigned seed,
                int ld_pad = 3, int stride_pad = 7) {
    Rhs<T> r;
    r.n = n; r.nrhs = nrhs; r.batch = batch;
    r.ld = n + ld_pad;
    r.stride = r.ld * nrhs + stride_pad;
    r.buf = UnifiedVector<T>(size_t(r.stride) * batch, make<T>(-9.75e3, 4.5e3));
    r.ptrs = UnifiedVector<T*>(size_t(batch), nullptr);
    Rng rg(seed);
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < nrhs; ++j)
            for (int i = 0; i < n; ++i)
                r.buf[size_t(b) * r.stride + size_t(j) * r.ld + i] = make<T>(rg.next(), rg.next());
    r.b0.assign(r.buf.begin(), r.buf.end());
    return r;
}

template <typename T>
void reset_rhs(Rhs<T>& r) { std::copy(r.b0.begin(), r.b0.end(), r.buf.begin()); }

template <typename T>
MatrixView<T, MatrixFormat::Dense> view_of(Rhs<T>& r) {
    return MatrixView<T, MatrixFormat::Dense>(r.buf.data(), r.n, r.nrhs, r.ld, r.stride, r.batch,
                                              r.ptrs.data());
}

// The PACKED int32 view of the public int64 pivot span, for ONE batch item. This
// spelling -- not a widening read -- IS the pivot contract on CUDA and ROCm.
template <typename T>
const int* piv_item(const Lu<T>& p, int b) {
    return reinterpret_cast<const int*>(p.piv.data()) + size_t(b) * p.n;
}

// The fixture.
template <typename Config>
class LuTest : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    static constexpr Backend BackendType = Config::BackendVal;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx) GTEST_SKIP() << "no queue";
        if (this->ctx->device().type != DeviceType::GPU)
            GTEST_SKIP() << "the native LU kernels are GPU-only (getrf.cc can_run)";
        if (!this->ctx->device().supports_sub_group_size(32))
            GTEST_SKIP() << "device does not offer sub-group size 32 (getrf.cc can_run)";
    }

    // The DEVICE's local-memory budget, spelled exactly as
    // select::Device::slm_budget spells it, NOT device_limits.hh's hardcoded 49152.
    std::size_t budget() const {
        const std::size_t lm = static_cast<std::size_t>(
            this->ctx->device().get_property(DeviceProperty::LOCAL_MEM_SIZE));
        return lm > 4096 ? lm - 4096 : std::size_t(0);
    }
    int cta_max_n() const { return sycl_getrf::getrf_cta_max_n_for_slm<T>(budget()); }
    bool leaf_fits(int m, int n) const { return sycl_getrf::getrf_leaf_fits<T>(m, n, budget()); }

    // The blocked driver's OWN block width and leading-panel leaf choice, QUERIED and
    // never hardcoded: a straddle test that cannot see the boundary tests nothing.
    int nb(int n) const {
        return int(sycl_getrf::getrf_blocked_debug_params<T>(*this->ctx, n) & 0xffffu);
    }
    unsigned leaf(int n) const {
        return (sycl_getrf::getrf_blocked_debug_params<T>(*this->ctx, n) >> 16) & 0xffu;
    }
    // Bits 24+: the deferred left-hand interchange spelling (0 in-loop, 1 deferred
    // walk, 2 deferred gather). MASKED, so a field added above cannot move leaf().
    unsigned left_mode(int n) const {
        return (sycl_getrf::getrf_blocked_debug_params<T>(*this->ctx, n) >> 24) & 0xffu;
    }

    // The ROUTED gemm and trsm, exactly as the factorization entry points inject them.
    // A direct caller MUST inject them: the blocked driver throws on an empty seam.
    sycl_getrf::GetrfTrailingGemm<T> gemm_seam() const {
        return [](Queue& c, const MatrixView<T, MatrixFormat::Dense>& ga,
                  const MatrixView<T, MatrixFormat::Dense>& gb,
                  const MatrixView<T, MatrixFormat::Dense>& gc,
                  T al, T be, Transpose ta, Transpose tb, ComputePrecision pr) {
            return gemm<BackendType, T>(c, ga, gb, gc, al, be, ta, tb, pr);
        };
    }
    sycl_getrf::GetrfPanelSolveTrsm<T> trsm_seam() const {
        return [](Queue& c, const MatrixView<T, MatrixFormat::Dense>& ta,
                  const MatrixView<T, MatrixFormat::Dense>& tb,
                  T al, Side sd, Uplo ul, Transpose tr, Diag dg) {
            return trsm<BackendType, T>(c, ta, tb, al, sd, ul, tr, dg);
        };
    }
    sycl_getrs::GetrsSolveTrsm<T> getrs_seam() const {
        return [](Queue& c, const MatrixView<T, MatrixFormat::Dense>& ta,
                  const MatrixView<T, MatrixFormat::Dense>& tb,
                  T al, Side sd, Uplo ul, Transpose tr, Diag dg) {
            return trsm<BackendType, T>(c, ta, tb, al, sd, ul, tr, dg);
        };
    }
    // getrs selects in src/ops/getrs/: a pin its can_run refuses throws from the sizing call.
    bool getrs_pin_accepted(const ops::getrs::GetrsChoice& c, const MatrixView<T, MatrixFormat::Dense>& A,
                            const MatrixView<T, MatrixFormat::Dense>& Bv, Transpose op) {
        const select::ScopedPin<ops::getrs::GetrsChoice> pin("getrs", c);
        try {
            (void)getrs_buffer_size<BackendType, T>(*this->ctx, A, Bv, op);
            return true;
        } catch (const std::invalid_argument& e) {
            if (std::string(e.what()).find("cannot run this shape") == std::string::npos) throw;
            return false;
        }
    }
    sycl_getri::GetriSolveTrsm<T> getri_seam() const {
        return [](Queue& c, const MatrixView<T, MatrixFormat::Dense>& ta,
                  const MatrixView<T, MatrixFormat::Dense>& tb,
                  T al, Side sd, Uplo ul, Transpose tr, Diag dg) {
            return trsm<BackendType, T>(c, ta, tb, al, sd, ul, tr, dg);
        };
    }

    // Run one tier DIRECTLY. `pass_info` false exercises the info_target
    // fallback, of which src/extensions/inv.cc:48 is a real instance.
    void run_cta(Lu<T>& p, bool pass_info = true) {
        auto V = view_of(p);
        UnifiedVector<std::byte> ws(std::max<std::size_t>(
            1, sycl_getrf::getrf_cta_buffer_size<T>(*this->ctx, V)));
        (void)sycl_getrf::getrf_cta_dispatch<T>(*this->ctx, V, p.piv.to_span(), ws.to_span(),
                                          pass_info ? p.info.to_span() : Span<int32_t>{});
        this->ctx->wait();
    }
    // THE REGISTER-RESIDENT TIER. Its ceiling is a compile-time property of the
    // kernel, not of the device: no local memory enters it.
    int tiny_max_n() const { return sycl_getrf::getrf_tiny_max_n<T>(); }
    void run_tiny(Lu<T>& p, bool pass_info = true) {
        auto V = view_of(p);
        UnifiedVector<std::byte> ws(std::max<std::size_t>(
            1, sycl_getrf::getrf_tiny_buffer_size<T>(*this->ctx, V)));
        (void)sycl_getrf::getrf_tiny_dispatch<T>(*this->ctx, V, p.piv.to_span(), ws.to_span(),
                                           pass_info ? p.info.to_span() : Span<int32_t>{});
        this->ctx->wait();
    }
    void run_blocked(Lu<T>& p, bool pass_info = true) {
        auto V = view_of(p);
        UnifiedVector<std::byte> ws(std::max<std::size_t>(
            1, sycl_getrf::getrf_blocked_buffer_size<T>(*this->ctx, V)));
        (void)sycl_getrf::getrf_blocked_dispatch<T>(*this->ctx, V, p.piv.to_span(), ws.to_span(),
                                              pass_info ? p.info.to_span() : Span<int32_t>{},
                                              gemm_seam(), trsm_seam());
        this->ctx->wait();
    }
};

// EVERY BATCH ITEM IS CHECKED, NOT ITEM 0: item 0 sits at offset 0, so a wrong
// batch stride cannot move it. The distinctness assertion is what makes "the
// kernel broadcast item 0 over the batch" a failure rather than a pass.
template <typename T>
void check_factor(const Lu<T>& p, const char* what, bool check_L = true) {
    for (int b = 0; b < p.batch; ++b) {
        const T* F = p.buf.data() + size_t(b) * p.stride;
        const T* A0 = p.a0.data() + size_t(b) * p.stride;
        const int* ip = piv_item(p, b);

        for (int k = 0; k < p.n; ++k)
            ASSERT_TRUE(ip[k] >= k + 1 && ip[k] <= p.n)
                << what << ": ipiv[" << k << "] = " << ip[k]
                << " is outside [k+1, n] at b=" << b << " -- not a 1-based interchange list";

        for (int j = 0; j < p.n; ++j)
            for (int i = 0; i < p.n; ++i)
                ASSERT_TRUE(verify::finite(up(F[size_t(j) * p.ld + i])))
                    << what << ": F(" << i << "," << j << ") is not finite at b=" << b;

        const double res = factor_residual<T>(A0, F, ip, p.n, p.n, p.ld);
        if (verbose())
            std::printf("[verbose] %-34s n=%4d b=%d  ||PA-LU||=%.4e  tol=%.4e\n",
                        what, p.n, b, res, verify::bound<T>(Check::factorization, p.n));
        EXPECT_VERIFY(T, Check::factorization, p.n, res)
            << what << ": ||PA - LU||_F / ||A||_F too large at b=" << b << " (n=" << p.n << ")";

        if (check_L) {
            const double ratio = verify::pivot_ratio(verify::view(F, p.n, p.n, p.ld));
            EXPECT_LE(ratio, verify::pivot_ratio_bound<T>())
                << what << ": cabs1(L(i,k) U(k,k)) / cabs1(U(k,k)) reached " << ratio
                << " > 1 at b=" << b
                << " -- a row with a LARGER cabs1 than the chosen pivot was left below it, so "
                   "this is not a cabs1 PARTIAL-pivoting factorization";
        }

        if (!p.expect_piv.empty()) {
            for (int k = 0; k < p.n; ++k)
                ASSERT_EQ(ip[k], p.expect_piv[k])
                    << what << ": ipiv[" << k << "] = " << ip[k] << ", expected "
                    << p.expect_piv[k] << " at b=" << b
                    << " -- the pivot base, direction or metric disagrees with LAPACK";
        }
    }

    if (p.batch > 1) {
        const T* f0 = p.buf.data();
        const T* fl = p.buf.data() + size_t(p.batch - 1) * p.stride;
        bool differ = false;
        for (int j = 0; j < p.n && !differ; ++j)
            for (int i = 0; i < p.n && !differ; ++i)
                if (verify::abs(up(f0[size_t(j) * p.ld + i]) - up(fl[size_t(j) * p.ld + i])) > 0.0)
                    differ = true;
        EXPECT_TRUE(differ) << what << ": the first and last batch items' factors are identical, "
                               "so this shape cannot see a batch-stride defect";
    }
}

// ANTI-VACUITY FOR EVERY TEST OF A PERMUTATION *DIRECTION*: if the permutation the
// interchange list denotes is SELF-INVERSE, a backwards walk and a forwards walk
// produce the same answer and no residual can tell them apart.
inline bool interchange_is_involution(const std::vector<int>& ipiv) {
    const int n = int(ipiv.size());
    std::vector<int> p(n);
    for (int i = 0; i < n; ++i) p[i] = i;
    for (int k = 0; k < n; ++k) std::swap(p[k], p[ipiv[k] - 1]);
    for (int i = 0; i < n; ++i) if (p[p[i]] != i) return false;
    return true;
}

// Anti-vacuity for the pivot oracle: the construction must actually MOVE rows.
// On a plain diagonally dominant matrix partial pivoting picks the diagonal at
// every step and every pivot assertion in this file would be vacuous.
template <typename T>
int non_diagonal_pivots(const Lu<T>& p, int b) {
    const int* ip = piv_item(p, b);
    int c = 0;
    for (int k = 0; k < p.n; ++k) if (ip[k] != k + 1) ++c;
    return c;
}

// THE FUSED NARROW-RHS GETRS TIER -- SHARED SCAFFOLDING. getrs_fused.cc is a
// SECOND native getrs arm: one work-group per matrix, the interchange walk and
// BOTH substitutions in ONE kernel. All three ways in are exercised below.
// evidence: docs/perf/lu.md#the-fused-narrow-rhs-getrs

// The tier's local-memory request, PINNED below against the library's own capacity
// query. A request landing in the 48 KB launch hole -- (47104, 49664] BYTES -- is
// raised to 49920, so the pad is NOT monotone and its inversion is not obvious.
constexpr std::size_t kFusedHoleLo    = 47104;
constexpr std::size_t kFusedHoleHi    = 49664;
constexpr std::size_t kFusedHolePadTo = 49920;
constexpr int kFusedNbMax = 32;

inline std::size_t fused_pad(std::size_t bytes) {
    return (bytes > kFusedHoleLo && bytes <= kFusedHoleHi) ? kFusedHolePadTo : bytes;
}

// What the LAUNCHER will actually ask for, given the block width it will pick.
inline int fused_nb_for(int n) {
    const int nb = (n >= 1024) ? 32 : 16;
    return (nb > n) ? n : nb;
}
inline std::size_t fused_launch_bytes(int n, int nrhs, std::size_t sz) {
    const int nb = fused_nb_for(n);
    return fused_pad((std::size_t(n) * std::size_t(nrhs) +
                      std::size_t(nb) * std::size_t(nb + 1)) * sz);
}
// What the CAPACITY QUERY charges for a given element count (the largest nb).
inline std::size_t fused_capacity_bytes(std::size_t rhs_elems, std::size_t sz) {
    return fused_pad((rhs_elems + std::size_t(kFusedNbMax) * std::size_t(kFusedNbMax + 1)) * sz);
}

// The order n whose RAW (pre-pad) launch request is exactly `want` bytes at this
// nrhs, or -1. Solved rather than tabulated because nb itself depends on n.
inline int fused_order_for_raw_bytes(std::size_t want, int nrhs, std::size_t sz) {
    if (want % sz) return -1;
    const std::size_t total = want / sz;
    for (int nb : {16, 32}) {
        const std::size_t blk = std::size_t(nb) * std::size_t(nb + 1);
        if (total <= blk) continue;
        const std::size_t rhs = total - blk;
        if (rhs % std::size_t(nrhs)) continue;
        const std::size_t n = rhs / std::size_t(nrhs);
        if (n < 1 || n > (std::size_t(1) << 20)) continue;
        if (fused_nb_for(int(n)) != nb) continue;   // the launcher must agree
        return int(n);
    }
    return -1;
}

// A FABRICATED LU FACTOR, built on the host with NO getrf: the 48 KB ladder runs at
// orders of 334-1428, where a ||PA - LU|| oracle is O(n^3), and this makes an exact
// O(n^2) residual available instead. The pivot list is a genuine INTERCHANGE LIST
// -- ipiv[k] in [k+1, n], 1-BASED, PACKED int32 into the public int64 span.
template <typename T>
Lu<T> make_fabricated_factor(int n, int batch, unsigned seed,
                             int ld_pad = 5, int stride_pad = 11) {
    Lu<T> p;
    alloc(p, n, batch, ld_pad, stride_pad);
    Rng rg(seed);
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                T v;
                if (i == j)      v = make<T>(4.0 * double(n), 0.0);
                else if (i < j)  v = make<T>(rg.next(), rg.next());                 // U
                else             v = scale(make<T>(rg.next(), rg.next()), 0.25);    // L
                p.buf[size_t(b) * p.stride + size_t(j) * p.ld + i] = v;
            }
        int* ip = reinterpret_cast<int*>(p.piv.data()) + size_t(b) * n;
        for (int k = 0; k < n; ++k) {
            const int span = n - k;
            const int off = int(std::fabs(rg.next()) * double(span)) % span;
            ip[k] = k + off + 1;                       // 1-BASED, in [k+1, n]
        }
        if (b == 0) p.expect_piv.assign(ip, ip + n);
    }
    p.a0.assign(p.buf.begin(), p.buf.end());
    return p;
}

// The RHS pad and the inter-item gap must come back BIT-IDENTICAL: the fused
// kernel writes B[i + c*ldb] for i < n and c < nrhs and nothing else.
template <typename T>
void check_rhs_pad_intact(const Rhs<T>& r, const char* what) {
    for (int b = 0; b < r.batch; ++b)
        for (int j = 0; j < r.stride; ++j) {
            const int col = j / r.ld, row = j % r.ld;
            const bool live = (col < r.nrhs) && (row < r.n);
            if (live) continue;
            const size_t k = size_t(b) * r.stride + j;
            ASSERT_EQ(verify::abs(up(r.buf[k]) - up(r.b0[k])), 0.0)
                << what << ": the RHS PAD was written at b=" << b << " offset " << j
                << " (ld=" << r.ld << ", n=" << r.n << ", nrhs=" << r.nrhs
                << ", stride=" << r.stride << ")";
        }
}

// Every batch item, and the LAST one especially: item 0 sits at offset 0, so a
// wrong batch stride cannot move it.
template <typename T>
void check_items_differ(const Rhs<T>& r, const char* what) {
    if (r.batch < 2) return;
    bool differ = false;
    const T* x0 = r.buf.data();
    const T* xl = r.buf.data() + size_t(r.batch - 1) * r.stride;
    for (int c = 0; c < r.nrhs && !differ; ++c)
        for (int i = 0; i < r.n && !differ; ++i)
            if (verify::abs(up(x0[size_t(c) * r.ld + i]) - up(xl[size_t(c) * r.ld + i])) > 0.0)
                differ = true;
    EXPECT_TRUE(differ) << what << ": the first and last batch items' solutions are identical, "
                           "so this shape cannot see a batch-stride defect";
}

using LuTestTypes = typename test_utils::backend_types<LuConfig>::type;

}  // namespace

TYPED_TEST_SUITE(LuTest, LuTestTypes);

// L0. THE 48 KB LAUNCH HOLE: a resident-leaf launch asking for a local-memory
// size in (47104, 49664] BYTES is refused, so the launcher pads it to 49,920 B.
//
// DECLARED FIRST ON PURPOSE: the raised cap is STICKY PER CUfunction and one
// GetrfPanelResidentKernel<T> serves every panel shape of a type, so any earlier
// launch of a LARGER panel warms the cap and this test can never fail again. DO
// NOT MOVE IT, and add no resident-leaf launch above it.
// evidence: docs/perf/lu.md#lu-the-48-kb-launch-hole
TYPED_TEST(LuTest, ResidentLeafLaunchHoleAt48KiB) {
    using T = typename TestFixture::T;

    // getrf_cta.cc's getrf_scratch_bytes: 32 argmax slots, each a real plus an
    // int. Restated here and then PINNED against the library's own predicate.
    const std::size_t sz = sizeof(T);
    const std::size_t scratch = 32u * (sizeof(RealOf<T>) + sizeof(int));
    auto raw_bytes = [&](int m, int n) {
        return std::size_t(m | 1) * std::size_t(n) * sz + scratch;
    };
    // The smallest budget at which the LIBRARY says an m x n leaf fits.
    auto min_budget = [&](int m, int n) -> std::size_t {
        std::size_t lo = 0, hi = std::size_t(1) << 24;
        if (!sycl_getrf::getrf_leaf_fits<T>(m, n, hi)) return 0;
        while (lo + 1 < hi) {
            const std::size_t mid = lo + (hi - lo) / 2;
            if (sycl_getrf::getrf_leaf_fits<T>(m, n, mid)) hi = mid; else lo = mid;
        }
        return hi;
    };

    // ANTI-VACUITY 1: the byte formula must be the library's, checked below the band.
    ASSERT_EQ(min_budget(16, 8), raw_bytes(16, 8))
        << "this test's local-memory formula is not the library's; every byte "
           "count below names some other size and the ladder proves nothing";

    // The hole band's endpoints, from getrf_cta.cc:136-138.
    constexpr std::size_t kLo = 47104, kHi = 49664, kPadTo = 49920;

    struct Row { std::size_t bytes; int m, n; };
    std::vector<Row> rows;
    for (std::size_t target : {std::size_t(46080),   // below the band: no pad
                               std::size_t(48896),   // measured failure, double/cdouble
                               std::size_t(49152),   // measured failure, ALL FOUR types
                               std::size_t(49664),   // the band's upper edge
                               std::size_t(50176)}) {// above the band: no pad
        if (target <= scratch) continue;
        const std::size_t tile = target - scratch;
        // (m|1) is ODD by construction (getrf_tile_ld), so search the odd
        // factorisations of tile/sz with m >= n.
        Row found{0, 0, 0};
        for (int n = 1; n <= 1024 && !found.m; ++n) {
            const std::size_t denom = std::size_t(n) * sz;
            if (tile % denom) continue;
            const std::size_t q = tile / denom;
            if ((q & 1u) == 0) continue;                // must be an odd ld
            if (q < std::size_t(n)) continue;           // keep the panel tall
            if (q > 8192) continue;
            found = Row{target, int(q), n};
        }
        if (found.m) rows.push_back(found);
    }

    // ANTI-VACUITY 2: the row that fails for every scalar type must be present.
    bool has_49152 = false;
    for (const Row& r : rows) if (r.bytes == 49152) has_49152 = true;
    ASSERT_TRUE(has_49152) << "no (m, n) with a 49,152 B footprint was constructible for this "
                              "scalar type; the discriminating row is missing";

    for (const Row& r : rows) {
        ASSERT_EQ(raw_bytes(r.m, r.n), r.bytes) << "row does not ask for " << r.bytes << " B";
        const bool in_band = (r.bytes > kLo && r.bytes <= kHi);

        // (a) THE PAD ARITHMETIC. EXPECT and not ASSERT, deliberately: an ASSERT returns
        // from the whole test and would MASK (b), which is an independent claim.
        EXPECT_EQ(min_budget(r.m, r.n), in_band ? kPadTo : r.bytes)
            << "the " << r.bytes << " B leaf (" << r.m << "x" << r.n << ") "
            << (in_band ? "is inside the 48 KB hole band but is not padded over it"
                        : "is outside the band and must not be padded");

        // (b) THE LAUNCH.
        const std::size_t need = in_band ? kPadTo : r.bytes;
        if (this->budget() < need) continue;   // a smaller device; (a) still ran
        ASSERT_TRUE(this->leaf_fits(r.m, r.n));

        const int m = r.m, n = r.n, batch = 2, k = std::min(m, n);
        const int ld = m + 3;
        const int stride = ld * n + 5;
        UnifiedVector<T> buf(size_t(stride) * batch, make<T>(-9.75e3, 4.5e3));
        Rng rg(unsigned(r.bytes % 9973) + 17u);
        for (int b = 0; b < batch; ++b)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i)
                    buf[size_t(b) * stride + size_t(j) * ld + i] = make<T>(rg.next(), rg.next());
        std::vector<T> a0(buf.begin(), buf.end());
        UnifiedVector<int> piv(size_t(k) * batch, -12345);
        UnifiedVector<int32_t> info(size_t(batch), 0);

        bool resident = false;
        ASSERT_NO_THROW(
            (void)sycl_getrf::getrf_panel_factorize<T>(*this->ctx, buf.data(), ld, stride,
                                                 m, n, batch, piv.data(), k, 0,
                                                 info.data(), &resident))
            << "the resident leaf could not be launched with a " << r.bytes << " B tile ("
            << m << "x" << n << ")";
        this->ctx->wait();
        EXPECT_TRUE(resident)
            << "the " << r.bytes << " B panel took the GLOBAL leaf, so this row did not "
               "exercise the local-memory launch at all";
        if (!resident) continue;

        for (int b = 0; b < batch; ++b) {
            const double res = factor_residual<T>(a0.data() + size_t(b) * stride,
                                                 buf.data() + size_t(b) * stride,
                                                 piv.data() + size_t(b) * k, m, n, ld);
            EXPECT_VERIFY(T, Check::factorization, std::max(m, n), res)
                << "the " << r.bytes << " B panel launched but factorised incorrectly at b=" << b;
        }
    }
}

// L0b. THE 48 KB LAUNCH HOLE, FOR THE **FUSED GETRS** KERNELS.
//
// DECLARED AHEAD OF EVERY OTHER FUSED-GETRS TEST, for the sticky-per-CUfunction
// reason L0 states: the tier's kernels are templated on the compile-time
// accumulator width NR, every rung below runs at nrhs = 8, and any earlier getrs
// with 4 < nrhs <= 8 would warm them. DO NOT MOVE IT BELOW THE F-SERIES.
TYPED_TEST(LuTest, FusedGetrsLaunchHoleAt48KiB) {
    using T = typename TestFixture::T;
    const std::size_t sz = sizeof(T);

    // ---- layer (a): the capacity inversion --------------------------------
    // The advertised capacity, once a caller sizes by it, must still be launchable
    // within the budget it was asked about; getrs_hole_padded is NOT monotone.
    for (std::size_t budget = kFusedHoleLo - 2048; budget <= kFusedHolePadTo + 2048; ++budget) {
        const std::size_t cap = sycl_getrs::getrs_fused_max_rhs_elems<T>(budget);
        if (cap == 0) continue;
        ASSERT_LE(fused_capacity_bytes(cap, sz), budget)
            << "getrs_fused_max_rhs_elems advertised " << cap << " elements for a budget of "
            << budget << " B, but that capacity asks the runtime for "
            << fused_capacity_bytes(cap, sz)
            << " B once the 48 KB hole pad is applied -- an UNLAUNCHABLE capacity, which is "
               "the bound getrs's can_run(cta) reads and therefore a pin it accepts that "
               "the driver cannot service";
    }
    // Coarse ladder past the band, including this device's own budget.
    for (std::size_t budget : {std::size_t(4096), std::size_t(16384), std::size_t(32768),
                               std::size_t(65536), std::size_t(98304), std::size_t(163840),
                               std::size_t(232448), this->budget()}) {
        const std::size_t cap = sycl_getrs::getrs_fused_max_rhs_elems<T>(budget);
        if (cap == 0) continue;
        ASSERT_LE(fused_capacity_bytes(cap, sz), budget) << "budget " << budget;
    }
    // ANTI-VACUITY for the sweep: the pad must actually fire somewhere inside it,
    // otherwise the loop above is a tautology over a function that never pads.
    {
        const std::size_t blk = std::size_t(kFusedNbMax) * std::size_t(kFusedNbMax + 1) * sz;
        bool padded_somewhere = false;
        for (std::size_t e = 0; e * sz + blk <= kFusedHolePadTo; ++e)
            if (fused_capacity_bytes(e, sz) == kFusedHolePadTo && e * sz + blk != kFusedHolePadTo)
                padded_somewhere = true;
        ASSERT_TRUE(padded_somewhere)
            << "no element count in range lands inside the (47104, 49664] band, so layer (a) "
               "cannot see a pad regression at all";
    }
    // And the device must advertise a usable capacity, or the tier is dead here.
    ASSERT_GT(sycl_getrs::getrs_fused_max_rhs_elems<T>(this->budget()), std::size_t(0))
        << "the fused tier reports zero capacity on this device";

    // ---- layer (b): the launch, across the band ---------------------------
    const int nrhs = int(sycl_getrs::kGetrsFusedMaxRhs);
    ASSERT_EQ(nrhs, 8) << "the ladder is solved at nrhs = 8; re-derive the orders if this moves";

    const std::size_t cap = sycl_getrs::getrs_fused_max_rhs_elems<T>(this->budget());
    int rungs_run = 0, rungs_over_capacity = 0;
    for (std::size_t want : {kFusedHoleLo, std::size_t(48896), std::size_t(49152),
                             kFusedHoleHi, kFusedHolePadTo}) {
        const int n = fused_order_for_raw_bytes(want, nrhs, sz);
        if (n < 0) {
            ADD_FAILURE() << "no order lands on a raw request of exactly " << want
                          << " B at nrhs=" << nrhs << " for a " << sz << "-byte scalar; the "
                             "ladder cannot cross the band and this test is vacuous";
            continue;
        }
        if (std::size_t(n) * std::size_t(nrhs) > cap) { ++rungs_over_capacity; continue; }

        // The rung must land where this file thinks it does, PAD INCLUDED.
        const std::size_t asked = fused_launch_bytes(n, nrhs, sz);
        const bool in_band = (want > kFusedHoleLo && want <= kFusedHoleHi);
        ASSERT_EQ(asked, in_band ? kFusedHolePadTo : want)
            << "rung " << want << " B (n=" << n << "): the launcher asks for " << asked;

        auto p = make_fabricated_factor<T>(n, 1, 2200u + unsigned(want % 1000u));
        ASSERT_FALSE(interchange_is_involution(p.expect_piv))
            << "rung " << want << ": the fabricated interchange list is SELF-INVERSE, so the "
               "transposed arm's backwards walk is indistinguishable from a forwards one";

        for (Transpose op : {Transpose::NoTrans, Transpose::Trans}) {
            auto rhs = make_rhs<T>(n, nrhs, 1,
                                   3300u + unsigned(want % 1000u) + unsigned(int(op)));
            auto A = view_of(p);
            auto Bv = view_of(rhs);
            UnifiedVector<std::byte> ws(std::max<std::size_t>(
                1, sycl_getrs::getrs_fused_buffer_size<T>(*this->ctx, A, Bv, op)));
            ASSERT_NO_THROW((void)sycl_getrs::getrs_fused_dispatch<T>(
                *this->ctx, A, Bv, op, p.piv.to_span(), ws.to_span()))
                << "rung " << want << " B (n=" << n << ", asked " << asked
                << " B, transA=" << int(op) << ") REFUSED TO LAUNCH";
            this->ctx->wait();

            const double res = lu_solve_residual<T>(
                p.buf.data(), reinterpret_cast<const int*>(p.piv.data()),
                rhs.buf.data(), rhs.b0.data(), n, nrhs, p.ld, rhs.ld, op);
            if (verbose())
                std::printf("[verbose] fused hole rung %6zu B n=%4d op=%d  res=%.4e tol=%.4e\n",
                            want, n, int(op), res, verify::bound<T>(Check::solve, n));
            EXPECT_VERIFY(T, Check::solve, n, res)
                << "rung " << want << " B (n=" << n << ", transA=" << int(op)
                << ") launched but did not solve";
            check_rhs_pad_intact(rhs, "fused/hole");
            if (this->HasFailure()) return;
        }
        ++rungs_run;
    }
    if (rungs_over_capacity > 0) {
        GTEST_SKIP() << rungs_over_capacity << " of 5 rungs are above this device's "
                     << "resident-RHS capacity (" << cap << " elements); the band was crossed "
                     << rungs_run << " times";
    }
    EXPECT_EQ(rungs_run, 5) << "the ladder did not cross the band from both sides";
}

// L1. THE CTA TIER: the residual, the partial-pivoting property, and the
// EXACT interchange list.
TYPED_TEST(LuTest, CtaFactorisesAndPivotsExactly) {
    using T = typename TestFixture::T;
    const int cap = this->cta_max_n();
    ASSERT_GE(cap, 32) << "the CTA tier advertises a capacity of " << cap
                       << ", so this test cannot reach it";

    int ran = 0;
    for (int n : {3, 8, 31, 32, 33, 64}) {
        if (n > cap) continue;
        ++ran;
        auto rnd = make_random<T>(n, 3, 991u + unsigned(n));
        this->run_cta(rnd);
        check_factor(rnd, "cta/random");
        for (int b = 0; b < rnd.batch; ++b) ASSERT_EQ(rnd.info[b], 0) << "n=" << n << " b=" << b;

        auto dom = make_dominant_permuted<T>(n, 3, 7717u + unsigned(n));
        this->run_cta(dom);
        // ANTI-VACUITY: the pivot assertion in check_factor is worthless if the
        // matrix pivots on its own diagonal at every step.
        ASSERT_GE(non_diagonal_pivots(dom, 0), n / 2)
            << "n=" << n << ": the dominant-permuted construction produced a near-identity "
               "pivot list, so the elementwise pivot oracle is vacuous";
        check_factor(dom, "cta/dominant-permuted");
        for (int b = 0; b < dom.batch; ++b) ASSERT_EQ(dom.info[b], 0);
        if (this->HasFailure()) return;
    }
    ASSERT_GT(ran, 0);
}

// L2. THE BLOCKED DRIVER, same two oracles, over orders that straddle its own
// block width in both directions.
TYPED_TEST(LuTest, BlockedFactorisesAndPivotsExactly) {
    using T = typename TestFixture::T;
    for (int n : {5, 31, 32, 33, 64, 96, 100, 129}) {
        auto rnd = make_random<T>(n, 3, 2311u + unsigned(n));
        this->run_blocked(rnd);
        check_factor(rnd, "blocked/random");
        for (int b = 0; b < rnd.batch; ++b) ASSERT_EQ(rnd.info[b], 0) << "n=" << n << " b=" << b;

        auto dom = make_dominant_permuted<T>(n, 3, 5501u + unsigned(n));
        this->run_blocked(dom);
        ASSERT_GE(non_diagonal_pivots(dom, 0), n / 2)
            << "n=" << n << ": the dominant-permuted construction produced a near-identity "
               "pivot list, so the elementwise pivot oracle is vacuous";
        check_factor(dom, "blocked/dominant-permuted");
        for (int b = 0; b < dom.batch; ++b) ASSERT_EQ(dom.info[b], 0);
        if (this->HasFailure()) return;
    }
}

// L2b. THE THREE SPELLINGS OF THE LEFT-HAND INTERCHANGE AGREE BIT FOR BIT -- the
// SAME transposition list in the SAME order, so the assertion is BITWISE and not
// "both residuals are small". n = 129 leaves a ONE-COLUMN final panel, where the
// deferred pass's extents must come from ib and never from nb.
// getrf_blocked.cc latches the knob's PRESENCE in a function-local static (the
// value itself is re-read per call), so the file-scope object below is what makes
// that latch land on "present" before main.
// evidence: docs/perf/lu.md#getrf-deferred-left-gather
namespace {
struct LeftLaswpKnobPresent {
    LeftLaswpKnobPresent() {
        ::setenv("BATCHLAS_GETRF_LASWP", "defer_gather", /*overwrite=*/0);
        // NOT a ScopedEnvVar: this presence has to hold for the WHOLE process and
        // outlive every scope, which is the one shape a restoring guard cannot
        // express. The explicit reload is the guard's other half. Without it the
        // settings() snapshot -- taken before main by the always-linked dispatch
        // coverage TU's own dynamic initialiser, in an order no TU here controls --
        // can be built from an environment that does not yet contain this setenv;
        // the presence latch then lands on "absent" and all three arms below
        // resolve to the SAME DeferGather mode.
        batchlas::detail::reload_settings();
    }
};
const LeftLaswpKnobPresent kLeftLaswpKnobPresent;
}  // namespace

TYPED_TEST(LuTest, LeftInterchangeSpellingsAgreeBitForBit) {
    using T = typename TestFixture::T;
    struct Arm { const char* env; unsigned mode; };
    const Arm arms[] = {{"inloop", 0u}, {"defer_walk", 1u}, {"defer_gather", 2u}};

    for (int n : {33, 64, 96, 129, 160}) {
        std::vector<std::vector<T>> facs;
        std::vector<std::vector<int>> pivs;
        for (const Arm& a : arms) {
            // The guard, never a bare ::setenv: settings() snapshots the environment
            // once, so an unguarded write leaves all three arms reading the SAME
            // value and the bitwise comparisons below compare one arm with itself.
            // On exit it restores what the file-scope object above (or the caller's
            // shell) had pinned, which is the shipping "defer_gather" arm.
            const ScopedEnvVar pin("BATCHLAS_GETRF_LASWP", a.env);
            ASSERT_EQ(this->left_mode(n), a.mode)
                << "n=" << n << ": the driver did not resolve the '" << a.env
                << "' spelling, so every comparison below would be between two copies of the "
                   "SAME arm.";

            auto p = make_random<T>(n, 3, 4441u + unsigned(n));
            this->run_blocked(p);
            check_factor(p, a.env);
            for (int b = 0; b < p.batch; ++b)
                ASSERT_EQ(p.info[b], 0) << a.env << " n=" << n << " b=" << b;

            facs.emplace_back(p.buf.data(), p.buf.data() + p.buf.size());
            std::vector<int> pv;
            for (int b = 0; b < p.batch; ++b) {
                const int* ip = piv_item(p, b);
                pv.insert(pv.end(), ip, ip + p.n);
            }
            pivs.push_back(std::move(pv));
            if (this->HasFailure()) return;  // pin's destructor restores and reloads
        }

        for (std::size_t a = 1; a < facs.size(); ++a) {
            ASSERT_EQ(facs[a].size(), facs[0].size());
            std::size_t diff = 0;
            for (std::size_t i = 0; i < facs[0].size(); ++i)
                if (std::memcmp(&facs[a][i], &facs[0][i], sizeof(T)) != 0) ++diff;
            EXPECT_EQ(diff, std::size_t(0))
                << "n=" << n << ": '" << arms[a].env << "' differs from '" << arms[0].env
                << "' in " << diff << " of " << facs[0].size()
                << " elements -- the deferred pass is not the same composition";
            EXPECT_EQ(pivs[a], pivs[0])
                << "n=" << n << ": '" << arms[a].env << "' produced a different interchange list";
        }
        if (this->HasFailure()) return;
    }
}

// L2b. The right-hand pass, walk against gather, bit for bit. n = 129 and 300 leave a short
// final panel and several column tiles per step; batch 37 is odd so the last (tile, item)
// groups are partial. evidence: docs/perf/lu.md#the-right-hand-gather
// ARMED BREAK (R9): write every row in lu_laswp_right_gather_launch's store loop from
// `tile[... + row]` instead of `tile[... + src]`. EXPECTED: RED at every n.
TYPED_TEST(LuTest, RightInterchangeSpellingsAgreeBitForBit) {
    using T = typename TestFixture::T;
    for (int n : {33, 64, 129, 300}) {
        std::vector<std::vector<T>> facs;
        std::vector<std::vector<int>> pivs;
        for (const char* spelling : {"walk", "gather"}) {
            const ScopedEnvVar pin("BATCHLAS_GETRF_RIGHT_LASWP", spelling);
            auto p = make_random<T>(n, 37, 5557u + unsigned(n));
            this->run_blocked(p);
            check_factor(p, spelling);
            for (int b = 0; b < p.batch; ++b)
                ASSERT_EQ(p.info[b], 0) << spelling << " n=" << n << " b=" << b;
            facs.emplace_back(p.buf.data(), p.buf.data() + p.buf.size());
            std::vector<int> pv;
            for (int b = 0; b < p.batch; ++b) {
                const int* ip = piv_item(p, b);
                pv.insert(pv.end(), ip, ip + p.n);
            }
            pivs.push_back(std::move(pv));
            if (this->HasFailure()) return;
        }
        std::size_t diff = 0;
        for (std::size_t i = 0; i < facs[0].size(); ++i)
            if (std::memcmp(&facs[1][i], &facs[0][i], sizeof(T)) != 0) ++diff;
        EXPECT_EQ(diff, std::size_t(0)) << "n=" << n << ": gather differs from walk in "
                                        << diff << " of " << facs[0].size() << " elements";
        EXPECT_EQ(pivs[1], pivs[0]) << "n=" << n;
        if (this->HasFailure()) return;
    }
}

// L3. THE BLOCK BOUNDARY IS QUERIED, NOT ASSUMED. A straddle test that cannot
// see where the boundary is keeps passing after the width moves while silently
// no longer testing a short final panel.
TYPED_TEST(LuTest, BlockWidthStraddleIsQueriedNotAssumed) {
    using T = typename TestFixture::T;

    const int nb0 = this->nb(256);
    ASSERT_GE(nb0, 1) << "getrf_blocked_debug_params reports no blocking, so the blocked "
                         "driver is absent and this test cannot straddle anything";

    const int n_exact = 4 * nb0;          // an EXACT multiple: no short final panel
    const int n_short = 4 * nb0 + 1;      // one column over: the shortest possible final panel
    const int n_mid   = 3 * nb0 + nb0 / 2 + 1;

    // The straddle is ASSERTED against the driver's own width, at the ORDERS the
    // test actually runs, because nb is clamped to n (getrf_blocked_nb).
    ASSERT_EQ(this->nb(n_exact), nb0);
    ASSERT_EQ(this->nb(n_short), nb0);
    ASSERT_EQ(this->nb(n_mid), nb0);
    ASSERT_EQ(n_exact % nb0, 0) << "the 'exact multiple' order is not one";
    ASSERT_NE(n_short % nb0, 0) << "the 'short final panel' order has none";
    ASSERT_NE(n_mid % nb0, 0);
    ASSERT_EQ(n_short % nb0, 1) << "the short final panel is not the narrowest one available";

    // The leaf choice the query reports must be the one getrf_leaf_fits makes for the
    // leading panel.
    for (int n : {n_exact, n_short, n_mid}) {
        const unsigned lf = this->leaf(n);
        ASSERT_TRUE(lf == 1u || lf == 2u) << "n=" << n << ": leaf tag " << lf;
        EXPECT_EQ(lf == 1u, this->leaf_fits(n, std::min(nb0, n)))
            << "n=" << n << ": getrf_blocked_debug_params and getrf_leaf_fits disagree about "
               "the leading panel's residency";
    }

    for (int n : {n_exact, n_short, n_mid}) {
        auto dom = make_dominant_permuted<T>(n, 2, 313u + unsigned(n));
        this->run_blocked(dom);
        ASSERT_GE(non_diagonal_pivots(dom, 0), n / 2);
        check_factor(dom, "blocked/straddle");
        for (int b = 0; b < dom.batch; ++b) ASSERT_EQ(dom.info[b], 0) << "n=" << n;
        if (this->HasFailure()) return;
    }
}

// L4. BOTH PANEL RESIDENCIES FACTORISE CORRECTLY. getrf_panel_factorize is the ONE
// decision site between the local-memory leaf and the global one; the residency is
// ASSERTED from the launcher's own out-parameter.
TYPED_TEST(LuTest, BothPanelLeavesFactoriseCorrectly) {
    using T = typename TestFixture::T;
    const int nbw = this->nb(4096);
    ASSERT_GE(nbw, 1);

    // A panel that fits and one that provably cannot: grow m until the predicate
    // says no, rather than picking a number that stops being large enough.
    int m_small = 64, m_big = 128;
    while (m_big < (1 << 20) && this->leaf_fits(m_big, nbw)) m_big *= 2;
    ASSERT_FALSE(this->leaf_fits(m_big, nbw))
        << "no panel height was found that overflows local memory";
    ASSERT_TRUE(this->leaf_fits(m_small, nbw));

    for (int pass = 0; pass < 2; ++pass) {
        const int m = pass ? m_big : m_small;
        const int n = nbw, batch = 2, k = std::min(m, n);
        const int ld = m + 3, stride = ld * n + 5;
        UnifiedVector<T> buf(size_t(stride) * batch, make<T>(-9.75e3, 4.5e3));
        Rng rg(4441u + unsigned(pass));
        for (int b = 0; b < batch; ++b)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i)
                    buf[size_t(b) * stride + size_t(j) * ld + i] = make<T>(rg.next(), rg.next());
        std::vector<T> a0(buf.begin(), buf.end());
        UnifiedVector<int> piv(size_t(k) * batch, -12345);
        UnifiedVector<int32_t> info(size_t(batch), 0);

        bool resident = false;
        ASSERT_NO_THROW(
            (void)sycl_getrf::getrf_panel_factorize<T>(*this->ctx, buf.data(), ld, stride,
                                                 m, n, batch, piv.data(), k, 0,
                                                 info.data(), &resident));
        this->ctx->wait();
        ASSERT_EQ(resident, pass == 0)
            << "panel " << m << "x" << n << " took the "
            << (resident ? "resident" : "global") << " leaf, which is not the one under test";

        for (int b = 0; b < batch; ++b) {
            ASSERT_EQ(info[b], 0);
            const int* ip = piv.data() + size_t(b) * k;
            for (int s = 0; s < k; ++s)
                ASSERT_TRUE(ip[s] >= s + 1 && ip[s] <= m) << "ipiv[" << s << "] = " << ip[s];
            EXPECT_VERIFY(T, Check::factorization, m, factor_residual<T>(a0.data() + size_t(b) * stride,
                                        buf.data() + size_t(b) * stride, ip, m, n, ld))
                << (resident ? "resident" : "global") << " leaf, b=" << b;
            EXPECT_LE(verify::pivot_ratio(verify::view(buf.data() + size_t(b) * stride, m, n, ld)),
                      verify::pivot_ratio_bound<T>())
                << (resident ? "resident" : "global") << " leaf, b=" << b;
        }
        if (this->HasFailure()) return;
    }
}


// ===========================================================================
// P4. THE REGISTER-RESIDENT PANEL LEAF (getrf_panel_reg.cc).
//
// A drop-in for getrf_panel_factorize with the identical contract, so the oracle
// that matters is THE OTHER LEAF: the interchange list is an integer sequence and
// two implementations of the same pivot rule must produce it exactly. The factor
// itself is compared only to a tolerance, because the two kernels are free to
// contract a multiply-subtract differently.
//
// ARMED BREAKS for R9: seven, each with the cell it must turn red.
// evidence: docs/perf/lu.md#armed-breaks-p4
// ===========================================================================
namespace {

// An m x ncols PANEL: ld padded, stride NOT ld*ncols, both regions poisoned, so a
// launcher that defaults either extent is falsifiable by construction.
template <typename T>
struct Panel {
    int m = 0, ncols = 0, batch = 0, ld = 0, stride = 0, piv_stride = 0, piv_base = 0;
    UnifiedVector<T> buf;
    std::vector<T> a0;
    UnifiedVector<int> piv;
    UnifiedVector<int32_t> info;

    const int* piv_of(int b) const {
        return piv.data() + size_t(b) * piv_stride + piv_base;
    }
    T* fac_of(int b) { return buf.data() + size_t(b) * stride; }
    const T* src_of(int b) const { return a0.data() + size_t(b) * stride; }
};

template <typename T>
Panel<T> make_panel(int m, int ncols, int batch, unsigned seed, int piv_base = 0) {
    Panel<T> p;
    p.m = m; p.ncols = ncols; p.batch = batch;
    p.ld = m + 3;
    p.stride = p.ld * ncols + 7;
    p.piv_base = piv_base;
    p.piv_stride = piv_base + ncols + 4;
    p.buf = UnifiedVector<T>(size_t(p.stride) * batch, make<T>(-9.75e3, 4.5e3));
    Rng rg(seed);
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < ncols; ++j)
            for (int i = 0; i < m; ++i)
                p.buf[size_t(b) * p.stride + size_t(j) * p.ld + i] = make<T>(rg.next(), rg.next());
    p.a0.assign(p.buf.begin(), p.buf.end());
    p.piv = UnifiedVector<int>(size_t(p.piv_stride) * batch, -12345);
    p.info = UnifiedVector<int32_t>(size_t(batch), 0);
    return p;
}

template <typename T>
void repanel(Panel<T>& p) {
    std::copy(p.a0.begin(), p.a0.end(), p.buf.begin());
    std::fill(p.piv.begin(), p.piv.end(), -12345);
    std::fill(p.info.begin(), p.info.end(), int32_t(0));
}

// The interchange list with piv_base removed, which is what pa_lu_residual's P wants.
std::vector<int> panel_local_piv(const int* ip, int k, int piv_base) {
    std::vector<int> v(static_cast<std::size_t>(k), 0);
    for (int s = 0; s < k; ++s) v[size_t(s)] = ip[s] - piv_base;
    return v;
}

}  // namespace

// P4a. THE TRANSITION ORACLE. Both leaves, same input, same ipiv EXACTLY.
TYPED_TEST(LuTest, RegPanelAgreesWithTheLocalMemoryLeaf) {
    using T = typename TestFixture::T;
    const int max_wg =
        int(this->ctx->device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const int nbw = sycl_getrf::getrf_panel_reg_nb<T>();
    ASSERT_GE(nbw, 1) << "the register panel reports no width for this type, so nothing "
                         "below can run and this test would pass vacuously";

    int ran = 0;
    for (int m : {33, 48, 64, 65, 96, 128, 129, 192, 256, 257, 288, 384, 512}) {
        for (int ncols : {1, 8, 31, 32}) {
            if (ncols > m) continue;
            if (!sycl_getrf::getrf_panel_reg_fits<T>(m, ncols, max_wg)) continue;

            const int batch = 3, k = std::min(m, ncols);
            auto p = make_panel<T>(m, ncols, batch, 7717u + unsigned(m * 37 + ncols));

            // Arm 1: the register leaf.
            ASSERT_NO_THROW((void)sycl_getrf::getrf_panel_reg_factorize<T>(
                *this->ctx, p.buf.data(), p.ld, p.stride, m, ncols, batch,
                p.piv.data(), p.piv_stride, p.piv_base, p.info.data()))
                << "m=" << m << " ncols=" << ncols;
            this->ctx->wait();
            std::vector<T> reg_fac(p.buf.begin(), p.buf.end());
            std::vector<int> reg_piv(p.piv.begin(), p.piv.end());
            std::vector<int32_t> reg_info(p.info.begin(), p.info.end());

            // Arm 2: today's leaf, on the SAME pristine input.
            repanel(p);
            bool resident = false;
            ASSERT_NO_THROW((void)sycl_getrf::getrf_panel_factorize<T>(
                *this->ctx, p.buf.data(), p.ld, p.stride, m, ncols, batch,
                p.piv.data(), p.piv_stride, p.piv_base, p.info.data(), &resident));
            this->ctx->wait();

            const char* what = resident ? "register vs resident" : "register vs global";
            std::size_t bad_slot = reg_piv.size();
            for (std::size_t i = 0; i < reg_piv.size(); ++i)
                if (reg_piv[i] != p.piv[i]) { bad_slot = i; break; }
            EXPECT_EQ(bad_slot, reg_piv.size())
                << what << " at m=" << m << " ncols=" << ncols << ": the two leaves chose "
                   "DIFFERENT pivots (first at slot " << bad_slot << ": register "
                << reg_piv[std::min(bad_slot, reg_piv.size() - 1)] << " vs local-memory "
                << p.piv[std::min(bad_slot, reg_piv.size() - 1)]
                << "), so they do not implement the same cabs1 partial-pivoting rule";
            EXPECT_EQ(reg_info, std::vector<int32_t>(p.info.begin(), p.info.end()))
                << what << " at m=" << m << " ncols=" << ncols;

            // BOTH ENDS OF THE BATCH: item 0 sits at offset 0, so a wrong batch
            // stride cannot move it.
            for (int b : {0, batch - 1}) {
                const auto ip = panel_local_piv(reg_piv.data() + size_t(b) * p.piv_stride + p.piv_base,
                                          k, p.piv_base);
                for (int s = 0; s < k; ++s)
                    ASSERT_TRUE(ip[size_t(s)] >= s + 1 && ip[size_t(s)] <= m)
                        << what << " m=" << m << " ncols=" << ncols << " b=" << b
                        << ": ipiv[" << s << "] = " << ip[size_t(s)]
                        << " is outside [s+1, m] -- not a 1-based interchange list";
                EXPECT_VERIFY(T, Check::factorization, std::max(m, ncols),
                    factor_residual<T>(p.src_of(b), reg_fac.data() + size_t(b) * p.stride,
                                       ip.data(), m, ncols, p.ld))
                    << "register leaf m=" << m << " ncols=" << ncols << " b=" << b;
                EXPECT_LE(verify::pivot_ratio(verify::view(reg_fac.data() + size_t(b) * p.stride, m, ncols, p.ld)),
                          verify::pivot_ratio_bound<T>())
                    << "register leaf m=" << m << " ncols=" << ncols << " b=" << b
                    << ": a row with a LARGER cabs1 than the chosen pivot was left below it";
            }

            // The two items must DIFFER, or "the kernel broadcast item 0" passes.
            bool differ = false;
            for (int j = 0; j < ncols && !differ; ++j)
                for (int i = 0; i < m; ++i)
                    if (verify::abs(up(reg_fac[size_t(0) * p.stride + size_t(j) * p.ld + i]) -
                             up(reg_fac[size_t(batch - 1) * p.stride + size_t(j) * p.ld + i])) >
                        0.0) { differ = true; break; }
            EXPECT_TRUE(differ) << "m=" << m << " ncols=" << ncols
                                << ": items 0 and " << batch - 1 << " are identical, so this "
                                   "cell cannot see a batch-stride defect";
            ++ran;
            if (this->HasFailure()) return;
        }
    }
    ASSERT_GT(ran, 6) << "only " << ran << " panel shapes were eligible; the ladder is not "
                         "exercising the register leaf";
}

// P4b. THE PIVOT CROSSES THE SUB-GROUP BOUNDARY AND THE PANEL'S LOWER HALF. The
// winner for column 3 is planted at row 200, which no single sub-group's butterfly
// can see: it is found only if the cross-sub-group slot scan runs.
TYPED_TEST(LuTest, RegPanelPivotCrossesTheSubGroupBoundary) {
    using T = typename TestFixture::T;
    const int max_wg =
        int(this->ctx->device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const int m = 256, ncols = 32, batch = 2, col = 3, row = 200;
    if (!sycl_getrf::getrf_panel_reg_fits<T>(m, ncols, max_wg))
        GTEST_SKIP() << "a " << m << "x" << ncols << " panel is outside this type's "
                     << "register leaf, so this shape cannot be built";
    ASSERT_GE(row / 32, 6) << "the planted winner is inside the FIRST sub-group, so a kernel "
                              "with no cross-sub-group scan would pass this test";

    auto p = make_panel<T>(m, ncols, batch, 30313u);
    // Every entry of column `col` at or below `col` is small; one row is large. The
    // columns to the LEFT are untouched, so rows 0..col-1 are eliminated normally.
    for (int b = 0; b < batch; ++b)
        for (int i = col; i < m; ++i)
            p.a0[size_t(b) * p.stride + size_t(col) * p.ld + i] = make<T>(1.0 / 1024.0, 0.0);
    for (int b = 0; b < batch; ++b)
        p.a0[size_t(b) * p.stride + size_t(col) * p.ld + row] = make<T>(64.0, -64.0);
    repanel(p);

    ASSERT_NO_THROW((void)sycl_getrf::getrf_panel_reg_factorize<T>(
        *this->ctx, p.buf.data(), p.ld, p.stride, m, ncols, batch,
        p.piv.data(), p.piv_stride, p.piv_base, p.info.data()));
    this->ctx->wait();

    for (int b = 0; b < batch; ++b) {
        const int* ip = p.piv_of(b);
        EXPECT_EQ(ip[col], row + 1)
            << "b=" << b << ": column " << col << " did not pivot to row " << row
            << " (got " << ip[col] << "). The winner is " << (row / 32)
            << " sub-groups down the work-group, so a butterfly-only argmax cannot find it";
        const auto lp = panel_local_piv(ip, ncols, p.piv_base);
        EXPECT_VERIFY(T, Check::factorization, m, factor_residual<T>(p.src_of(b), p.fac_of(b), lp.data(), m, ncols, p.ld))
            << "b=" << b;
    }
}

// P4c. ipiv AND info ARE GLOBAL AT A NON-ZERO piv_base. Dropping the offset is
// invisible to every piv_base == 0 cell, and the blocked driver runs exactly one
// such panel (the first) out of n/nb.
TYPED_TEST(LuTest, RegPanelIpivAndInfoAreGlobalAtANonZeroPivBase) {
    using T = typename TestFixture::T;
    const int max_wg =
        int(this->ctx->device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const int m = 96, ncols = 32, batch = 3, base = 64;
    if (!sycl_getrf::getrf_panel_reg_fits<T>(m, ncols, max_wg)) GTEST_SKIP() << "not eligible";

    auto p0 = make_panel<T>(m, ncols, batch, 5107u, /*piv_base=*/0);
    auto pb = make_panel<T>(m, ncols, batch, 5107u, /*piv_base=*/base);
    ASSERT_EQ(p0.stride, pb.stride);
    ASSERT_EQ(p0.a0, pb.a0) << "the two panels must hold the SAME matrix, or the offset "
                               "comparison below is between two different factorisations";

    for (Panel<T>* p : {&p0, &pb}) {
        ASSERT_NO_THROW((void)sycl_getrf::getrf_panel_reg_factorize<T>(
            *this->ctx, p->buf.data(), p->ld, p->stride, m, ncols, batch,
            p->piv.data(), p->piv_stride, p->piv_base, p->info.data()));
        this->ctx->wait();
    }

    for (int b = 0; b < batch; ++b) {
        const int* i0 = p0.piv_of(b);
        const int* ib = pb.piv_of(b);
        for (int s = 0; s < ncols; ++s)
            ASSERT_EQ(ib[s], i0[s] + base)
                << "b=" << b << " s=" << s
                << ": ipiv at piv_base=" << base << " is not the piv_base=0 list shifted by "
                   "the base, so the panel's rows are not being reported globally";
        // The slots BELOW piv_base and ABOVE the panel must be untouched.
        for (int s = 0; s < base; ++s)
            ASSERT_EQ(pb.piv[size_t(b) * pb.piv_stride + s], -12345)
                << "b=" << b << ": the panel wrote ipiv slot " << s << ", left of its base";
        for (int s = base + ncols; s < pb.piv_stride; ++s)
            ASSERT_EQ(pb.piv[size_t(b) * pb.piv_stride + s], -12345)
                << "b=" << b << ": the panel wrote ipiv slot " << s << ", right of its width";
    }
}

// P4d. A PLANTED ZERO COLUMN: info is EXACT-ZERO, 1-BASED, GLOBAL, per item and
// FIRST-FAILURE-WINS ACROSS CALLS -- the leaf READS info, which is what lets the
// blocked driver report the first failure over all its panels.
TYPED_TEST(LuTest, RegPanelPlantedZeroColumnIsGlobalAndFirstFailureWins) {
    using T = typename TestFixture::T;
    const int max_wg =
        int(this->ctx->device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const int m = 128, ncols = 32, batch = 3, base = 32, c1 = 5, c2 = 19;
    if (!sycl_getrf::getrf_panel_reg_fits<T>(m, ncols, max_wg)) GTEST_SKIP() << "not eligible";

    auto p = make_panel<T>(m, ncols, batch, 9041u, base);
    const int bad = 2;   // NOT item 0: a wrong batch stride cannot move item 0
    for (int j : {c1, c2})
        for (int i = 0; i < m; ++i)
            p.a0[size_t(bad) * p.stride + size_t(j) * p.ld + i] = make<T>(0.0, 0.0);
    repanel(p);

    ASSERT_NO_THROW((void)sycl_getrf::getrf_panel_reg_factorize<T>(
        *this->ctx, p.buf.data(), p.ld, p.stride, m, ncols, batch,
        p.piv.data(), p.piv_stride, p.piv_base, p.info.data()));
    this->ctx->wait();

    for (int b = 0; b < batch; ++b) {
        EXPECT_EQ(p.info[b], (b == bad) ? (base + c1 + 1) : 0)
            << "b=" << b << ": info must be the GLOBAL 1-based column of the FIRST zero "
               "pivot (planted at panel columns " << c1 << " and " << c2
            << ", piv_base=" << base << ")";
        const T* F = p.fac_of(b);
        for (int j = 0; j < ncols; ++j)
            for (int i = 0; i < m; ++i)
                ASSERT_TRUE(verify::finite(up(F[size_t(j) * p.ld + i])))
                    << "b=" << b << " left F(" << i << "," << j << ") non-finite; a failed "
                       "item must stay finite, as LAPACK's and cuBLAS's do";
    }

    // FIRST-FAILURE-WINS ACROSS CALLS: a pre-set info must survive, because the
    // blocked driver's later panels must not overwrite an earlier panel's column.
    repanel(p);
    for (int b = 0; b < batch; ++b) p.info[b] = int32_t(7);
    ASSERT_NO_THROW((void)sycl_getrf::getrf_panel_reg_factorize<T>(
        *this->ctx, p.buf.data(), p.ld, p.stride, m, ncols, batch,
        p.piv.data(), p.piv_stride, p.piv_base, p.info.data()));
    this->ctx->wait();
    for (int b = 0; b < batch; ++b)
        EXPECT_EQ(p.info[b], int32_t(7))
            << "b=" << b << ": the leaf overwrote a non-zero info, so first-failure-wins "
               "does not hold across the blocked driver's panels";
}

// P4e. THE CAPACITY HAS ONE SPELLING. getrf_panel_reg_fits, the debug hook and the
// entry point's refusal must be the same predicate, and the sweep must contain
// cells on BOTH sides or it is a tautology over a function that never refuses.
TYPED_TEST(LuTest, RegPanelCapacityIsOneSpelling) {
    using T = typename TestFixture::T;
    const int max_wg =
        int(this->ctx->device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const int cap = sycl_getrf::getrf_panel_reg_max_m<T>();
    const int nbw = sycl_getrf::getrf_panel_reg_nb<T>();
    ASSERT_GT(cap, 32) << "the register panel must at least cover one nb = 32 block's height";
    ASSERT_EQ(nbw, 32) << "the panel width moved; every ladder above names 32";

    int accepted = 0, refused = 0;
    for (int m : {1, 8, 32, 33, 64, 128, 256, 288, 320, 384, 512, cap, cap + 1, cap + 32,
                  2 * cap}) {
        for (int n : {0, 1, 32, nbw + 1}) {
            const bool fits = sycl_getrf::getrf_panel_reg_fits<T>(m, n, max_wg);
            EXPECT_EQ(fits, sycl_getrf::getrf_panel_reg_debug_launch<T>(*this->ctx, m, n) != 0u)
                << "m=" << m << " n=" << n
                << ": getrf_panel_reg_fits and the debug hook disagree, so a caller that asks "
                   "one of them can be handed a panel the launcher refuses";
            if (fits) ++accepted; else ++refused;
        }
    }
    EXPECT_GT(accepted, 0);
    EXPECT_GT(refused, 0) << "no cell in the sweep was refused, so the agreement above is a "
                             "tautology";

    // And the entry point refuses exactly what the predicate refuses.
    UnifiedVector<T> buf(size_t(64) * 8, make<T>(1.0, 0.0));
    UnifiedVector<int> piv(size_t(64) * 1, -1);
    UnifiedVector<int32_t> info(size_t(1), 0);
    auto call = [&](int m, int n) {
        (void)sycl_getrf::getrf_panel_reg_factorize<T>(*this->ctx, buf.data(), 64, 64 * 8,
                                                 m, n, 1, piv.data(), 64, 0, info.data());
    };
    EXPECT_THROW(call(8, nbw + 1), batchlas::unsupported)
        << "a panel WIDER than the compile-time NB was accepted; it would be factorised as a "
           "leading submatrix and the trailing columns silently left alone";
    EXPECT_THROW(call(cap + 32, 8), batchlas::unsupported)
        << "a panel taller than the register cap was accepted; the launch would exceed this "
           "type's per-sub-partition register ceiling and abort";
    EXPECT_THROW(call(0, 8), batchlas::invalid_argument);

    // AND THE CAP IS LAUNCHABLE. The agreement sweep above compares one arithmetic
    // spelling with another and cannot see a cap that the DRIVER refuses; the first
    // cap this file shipped accepted a 288-lane cdouble panel that aborts with
    // CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES, because regs x work-group <= 65,536 is not
    // the gate -- registers are owned per sub-partition.
    // evidence: docs/perf/lu.md#the-register-panel-leaf-register-probe
    UnifiedVector<T> tall(std::size_t(cap + 3) * 32, make<T>(1.0, 0.0));
    UnifiedVector<int> tall_piv(std::size_t(cap + 32), -1);
    UnifiedVector<int32_t> tall_info(std::size_t(1), 0);
    for (int i = 0; i < 32; ++i) tall[std::size_t(i) * (cap + 3) + i] = make<T>(2.0, 0.5);
    ASSERT_TRUE(sycl_getrf::getrf_panel_reg_fits<T>(cap, 32, max_wg))
        << "the advertised cap is not accepted by the predicate, so the launch below "
           "would be vacuous";
    EXPECT_NO_THROW({
        (void)sycl_getrf::getrf_panel_reg_factorize<T>(
            *this->ctx, tall.data(), cap + 3, (cap + 3) * 32, cap, 32, 1,
            tall_piv.data(), cap + 32, 0, tall_info.data());
        this->ctx->wait();
    }) << "the panel at the ADVERTISED cap m=" << cap << " (work-group " << cap
       << ") does not launch, so getrf_panel_reg_max_m advertises capacity the device "
          "refuses";
}

// P4f. THE BLOCKED DRIVER TAKES THE REGISTER LEAF UNDER THE KNOB AND PRODUCES THE
// SAME INTERCHANGE LIST. The knob is the only way a benchmark can A/B the two, so
// a knob that resolves to the same arm in both passes is the failure mode here.
TYPED_TEST(LuTest, BlockedDriverTakesTheRegisterLeafUnderTheKnob) {
    using T = typename TestFixture::T;

    int saw_reg = 0;
    for (int n : {33, 64, 65, 96, 128, 129, 192, 256, 257, 384, 512}) {
        std::vector<int> pivs[2];
        std::vector<T> facs[2];
        for (int arm = 0; arm < 2; ++arm) {
            // The guard, never a bare ::setenv: settings() snapshots the environment
            // once, so an unguarded write leaves both arms reading the SAME value.
            const ScopedEnvVar pin("BATCHLAS_GETRF_LEAF", arm ? "reg" : "slm");
            const unsigned kind = sycl_getrf::getrf_blocked_debug_leaf<T>(*this->ctx, n);
            ASSERT_GE(kind, 1u) << "n=" << n << ": the driver reports no leaf at all";
            if (arm == 0) {
                ASSERT_NE(kind, 3u) << "n=" << n << ": the DEFAULT arm resolved to the register "
                                       "leaf, so the comparison below is one arm against itself";
            } else if (kind == 3u) {
                ++saw_reg;
            }

            auto p = make_random<T>(n, 3, 2281u + unsigned(n));
            this->run_blocked(p);
            check_factor(p, arm ? "blocked(leaf=reg)" : "blocked(leaf=slm)");
            for (int b = 0; b < p.batch; ++b)
                ASSERT_EQ(p.info[b], 0) << "n=" << n << " arm=" << arm << " b=" << b;

            for (int b = 0; b < p.batch; ++b) {
                const int* ip = piv_item(p, b);
                pivs[arm].insert(pivs[arm].end(), ip, ip + p.n);
            }
            facs[arm].assign(p.buf.data(), p.buf.data() + p.buf.size());
            if (this->HasFailure()) return;   // pin's destructor restores and reloads
        }
        EXPECT_EQ(pivs[1], pivs[0])
            << "n=" << n << ": the register leaf produced a different interchange list from "
               "the local-memory leaf, so the two are not the same pivot rule";
        ASSERT_EQ(facs[1].size(), facs[0].size());
        double worst = 0.0, scale = 0.0;
        for (std::size_t i = 0; i < facs[0].size(); ++i) {
            worst = nanmax(worst, verify::abs(up(facs[1][i]) - up(facs[0][i])));
            scale = nanmax(scale, verify::abs(up(facs[0][i])));
        }
        // RELATIVE, and generous: the two kernels run the same operations in the same
        // order, so the only legitimate difference is a contracted multiply-subtract.
        EXPECT_LE(worst, 1000.0 * double(n) * (2.0 * verify::eps<T>()) * std::max(scale, 1.0))
            << "n=" << n << ": the two leaves' factors differ by " << worst
            << " at the worst element (scale " << scale
            << ") -- far past a contraction difference";
        if (this->HasFailure()) return;
    }
    ASSERT_GT(saw_reg, 4) << "the register leaf was taken at only " << saw_reg
                          << " of the orders above, so this test mostly compared the "
                             "local-memory leaf with itself";

    // AND THE DEFAULT IS THE REGISTER LEAF. Every P4 ratio on the page was measured
    // against a driver that takes it with the variable UNSET, so a silent revert of
    // the default would leave the transcribed getrf rows scored against an arm
    // nothing runs. evidence: docs/perf/lu.md#lu-the-register-leaf-ab
    {
        const ScopedEnvVar unset("BATCHLAS_GETRF_LEAF", nullptr);
        for (int n : {64, 128, 256}) {
            EXPECT_EQ(sycl_getrf::getrf_blocked_debug_leaf<T>(*this->ctx, n), 3u)
                << "n=" << n << ": with BATCHLAS_GETRF_LEAF unset the blocked driver did "
                   "NOT take the register leaf";
        }
    }
}

// L5. A SINGULAR MATRIX: `info` is EXACT-ZERO, 1-BASED, GLOBAL, per item and
// FIRST-FAILURE-WINS, with the other batch items unaffected and the failed item
// still FINITE (?GETF2 records the failure and SKIPS the reciprocal scale). The
// failure is planted as an EXACTLY ZERO COLUMN inside the SECOND AND THIRD PANELS:
// a block-local info offset reports the panel-relative column and passes every
// single-panel test.
TYPED_TEST(LuTest, SingularColumnGivesGlobalOneBasedInfoFirstFailureWins) {
    using T = typename TestFixture::T;
    const int nbw = this->nb(256);
    ASSERT_GE(nbw, 2);

    const int n = 3 * nbw + 7;
    const int c1 = nbw + 3;              // second panel: piv_base = nbw, so a block-local
    const int c2 = 2 * nbw + 5;          // offset would report 4 instead of c1 + 1
    ASSERT_GT(c1, nbw) << "the planted failure is inside the FIRST panel; a block-local info "
                          "offset would be indistinguishable from a global one";
    ASSERT_LT(c1, c2);
    ASSERT_LT(c2, n);

    auto p = make_dominant_permuted<T>(n, 4, 8123u);
    const int bad = 2;                   // NOT item 0: a wrong batch stride cannot move item 0
    for (int j : {c1, c2})
        for (int i = 0; i < n; ++i)
            p.a0[size_t(bad) * p.stride + size_t(j) * p.ld + i] = make<T>(0.0, 0.0);
    p.expect_piv.clear();                // the zero columns change the sequence
    poison(p);

    this->run_blocked(p);

    for (int b = 0; b < p.batch; ++b) {
        if (b == bad) {
            EXPECT_EQ(p.info[b], c1 + 1)
                << "info must be the GLOBAL 1-based column of the FIRST zero pivot "
                   "(planted at global columns " << c1 << " and " << c2 << ", nb=" << nbw << ")";
        } else {
            EXPECT_EQ(p.info[b], 0) << "healthy item " << b << " reported a failure";
        }
        const T* F = p.buf.data() + size_t(b) * p.stride;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                ASSERT_TRUE(verify::finite(up(F[size_t(j) * p.ld + i])))
                    << "item " << b << " left F(" << i << "," << j << ") non-finite; a failed "
                       "item must stay finite, as LAPACK's and cuBLAS's do";
        // A failure in one item must not corrupt the others.
        if (b != bad) {
            const int* ip = piv_item(p, b);
            EXPECT_VERIFY(T, Check::factorization, n, factor_residual<T>(p.a0.data() + size_t(b) * p.stride, F, ip, n, n, p.ld))
                << "healthy item " << b;
        }
    }

    // The CTA tier tells the same story at an order it can hold, where piv_base is
    // 0 throughout -- so this half pins the 1-based-ness alone.
    if (this->cta_max_n() >= 40) {
        const int nc = 40, cc1 = 11, cc2 = 27;
        auto q = make_dominant_permuted<T>(nc, 3, 6611u);
        for (int j : {cc1, cc2})
            for (int i = 0; i < nc; ++i)
                q.a0[size_t(1) * q.stride + size_t(j) * q.ld + i] = make<T>(0.0, 0.0);
        q.expect_piv.clear();
        poison(q);
        this->run_cta(q);
        EXPECT_EQ(q.info[0], 0);
        EXPECT_EQ(q.info[1], cc1 + 1);
        EXPECT_EQ(q.info[2], 0);
    }
}

// L5b. THE `info` ZERO PRE-PASS IS ORDERED AHEAD OF THE PANEL THAT READS IT, ON
// AN OUT-OF-ORDER QUEUE -- the only test here not on the fixture's queue.
//
// getf2_panel_device READS info[b] to implement first-failure-wins across the
// blocked driver's panels, so the fill is a true read-after-write dependence.
// Unordered, the panel loads the caller's pre-call garbage and never records the
// real failure. The batch is large on purpose: this is a race.
TYPED_TEST(LuTest, InfoFillIsOrderedAheadOfThePanelOnAnOutOfOrderQueue) {
    using T = typename TestFixture::T;
    // ONE SCALAR TYPE, DELIBERATELY: what is under test is a HOST-SIDE SUBMISSION
    // ORDER, identical for every scalar type and backend; the sweep's cost is not.
    if constexpr (!std::is_same_v<T, float>) {
        GTEST_SKIP() << "the submission-order defect is type-independent; float carries it";
    } else {
    if (this->cta_max_n() < 32)
        GTEST_SKIP() << "this device's CTA tile cannot hold order 32";

    Queue ooo(*this->ctx, /*in_order=*/false);
    ASSERT_FALSE(ooo.in_order())
        << "ANTI-VACUITY: an in-order queue orders the fill for free, so the whole "
           "test would pass over the defect it exists to catch";

    const int zc = 7;                   // the planted singular column, 0-based

    // THE LOOP SHAPE IS PART OF THE CALIBRATION: re-copying the matrix from the host
    // between repetitions -- the obvious thing to write -- touches managed memory hard
    // enough to serialise the queue and close the window, so the matrix is staged once
    // and re-factorised in place; the planted zero column survives the factorisation.
    auto seed_info = [](Lu<T>& q) {
        std::fill(q.info.begin(), q.info.end(), int32_t(-12345));
    };
    auto count_wrong = [&](Lu<T>& q, long& wrong, long& poisoned) {
        for (int b = 0; b < q.batch; ++b) {
            if (q.info[b] != zc + 1) ++wrong;
            if (q.info[b] == -12345) ++poisoned;
        }
    };

    // --- the CTA tier ---------------------------------------------------
    {
        const int n = 32, batch = 65536, reps = 25;
        auto p = make_random<T>(n, batch, 3u);
        for (int b = 0; b < batch; ++b)
            for (int i = 0; i < n; ++i)
                p.a0[size_t(b) * p.stride + size_t(zc) * p.ld + i] = make<T>(0.0, 0.0);
        p.expect_piv.clear();
        poison(p);                                   // stage the matrix ONCE
        auto V = view_of(p);

        // THE IN-ORDER CONTROL, which makes the sweep's oracle legitimate: a miss is then
        // an ORDERING failure and not the oracle drifting.
        long cw = 0, cp = 0;
        for (int r = 0; r < 5; ++r) {
            seed_info(p);
            UnifiedVector<std::byte> ws(std::max<std::size_t>(
                1, sycl_getrf::getrf_cta_buffer_size<T>(*this->ctx, V)));
            (void)sycl_getrf::getrf_cta_dispatch<T>(*this->ctx, V, p.piv.to_span(), ws.to_span(),
                                              p.info.to_span());
            this->ctx->wait();
            count_wrong(p, cw, cp);
        }
        ASSERT_EQ(cw, 0) << "CONTROL: the in-order queue itself did not report the planted "
                            "column, so the oracle is wrong and the sweep below means nothing";

        long wrong = 0, poisoned = 0;
        for (int r = 0; r < reps; ++r) {
            seed_info(p);
            UnifiedVector<std::byte> ws(std::max<std::size_t>(
                1, sycl_getrf::getrf_cta_buffer_size<T>(ooo, V)));
            (void)sycl_getrf::getrf_cta_dispatch<T>(ooo, V, p.piv.to_span(), ws.to_span(),
                                              p.info.to_span());
            ooo.wait();
            count_wrong(p, wrong, poisoned);
        }
        EXPECT_EQ(wrong, 0)
            << "CTA tier: " << wrong << " of " << long(reps) * batch
            << " items did not report the planted singular column " << (zc + 1)
            << "; " << poisoned << " of them returned the CALLER's own -12345, which is "
               "the signature of the panel reading info before the fill landed";
    }

    // --- the blocked tier, whose panel loop reads info ACROSS panels -----
    {
        const int nbw = this->nb(256);
        ASSERT_GE(nbw, 2);
        const int n = 2 * nbw;              // at least two panels, so the READ matters
        const int batch = 32768, reps = 15;
        auto p = make_random<T>(n, batch, 17u);
        for (int b = 0; b < batch; ++b)
            for (int i = 0; i < n; ++i)
                p.a0[size_t(b) * p.stride + size_t(zc) * p.ld + i] = make<T>(0.0, 0.0);
        p.expect_piv.clear();
        poison(p);
        auto V = view_of(p);

        long cw = 0, cp = 0;
        for (int r = 0; r < 3; ++r) {
            seed_info(p);
            UnifiedVector<std::byte> ws(std::max<std::size_t>(
                1, sycl_getrf::getrf_blocked_buffer_size<T>(*this->ctx, V)));
            (void)sycl_getrf::getrf_blocked_dispatch<T>(*this->ctx, V, p.piv.to_span(), ws.to_span(),
                                                  p.info.to_span(),
                                                  this->gemm_seam(), this->trsm_seam());
            this->ctx->wait();
            count_wrong(p, cw, cp);
        }
        ASSERT_EQ(cw, 0) << "CONTROL (blocked): the in-order queue disagrees with the oracle";

        long wrong = 0, poisoned = 0;
        for (int r = 0; r < reps; ++r) {
            seed_info(p);
            UnifiedVector<std::byte> ws(std::max<std::size_t>(
                1, sycl_getrf::getrf_blocked_buffer_size<T>(ooo, V)));
            (void)sycl_getrf::getrf_blocked_dispatch<T>(ooo, V, p.piv.to_span(), ws.to_span(),
                                                  p.info.to_span(),
                                                  this->gemm_seam(), this->trsm_seam());
            ooo.wait();
            count_wrong(p, wrong, poisoned);
        }
        EXPECT_EQ(wrong, 0)
            << "blocked tier (n=" << n << ", nb=" << nbw << "): " << wrong << " of "
            << long(reps) * batch << " items wrong, " << poisoned
            << " of them the caller's own -12345";
    }
    }
}

// L6. A NEARLY singular matrix is NOT flagged. The `info` predicate is a TRUE
// BINARY ZERO, never a tolerance, and that is a PUBLIC CONTRACT shared with LAPACK
// and cuBLAS: an epsilon floor would report a failure where neither of them does.
TYPED_TEST(LuTest, NearlySingularIsNotFlagged) {
    using T = typename TestFixture::T;
    const int n = 48, c = 19;
    auto p = make_dominant_permuted<T>(n, 2, 4242u);
    // Scale one whole column to ~1e-30: U(c,c) is then tiny but exactly representable
    // and NON-ZERO for all four types, and column scaling does not move an argmax.
    for (int b = 0; b < p.batch; ++b)
        for (int i = 0; i < n; ++i) {
            T& v = p.a0[size_t(b) * p.stride + size_t(c) * p.ld + i];
            v = scale(v, 1e-30);
        }
    poison(p);
    this->run_blocked(p);

    for (int b = 0; b < p.batch; ++b) {
        const T* F = p.buf.data() + size_t(b) * p.stride;
        const double diag = verify::cabs1(up(F[size_t(c) * p.ld + c]));
        // ANTI-VACUITY: the pivot really is tiny, or this is just "info == 0".
        ASSERT_GT(diag, 0.0) << "U(c,c) is exactly zero, so this is the SINGULAR case";
        ASSERT_LT(diag, 1e-20) << "U(c,c) = " << diag << " is not nearly singular at all";
        EXPECT_EQ(p.info[b], 0)
            << "info = " << p.info[b] << " at b=" << b << " with |U(c,c)| = " << diag
            << " -- a tolerance crept into the singularity predicate, which diverges from "
               "LAPACK and cuBLAS invisibly";
    }
    check_factor(p, "blocked/near-singular", /*check_L=*/true);
}

// L7. THE PIVOT METRIC IS cabs1, NOT THE MODULUS. cublas{C,Z}getrfBatched pivots
// on |z| while LAPACK and this kernel pivot on |Re| + |Im|; on the matrix below the
// two rules SELECT DIFFERENT ROWS, which they do not on random or dominant data.
// evidence: docs/perf/lu.md#lu-correctness-findings
TYPED_TEST(LuTest, PivotSelectionUsesCabs1AndNotTheModulus) {
    using T = typename TestFixture::T;
    if constexpr (!test_utils::is_complex_type_v<T>) {
        GTEST_SKIP() << "cabs1 and the modulus coincide for a real scalar type";
    } else {
        const int n = 4, batch = 2;
        auto p = make_dominant_permuted<T>(n, batch, 99u);
        p.expect_piv.clear();
        for (int b = 0; b < batch; ++b) {
            T* A = p.a0.data() + size_t(b) * p.stride;
            // The per-item factor keeps the batch items DISTINCT -- without it check_factor's
            // batch-stride assertion is unsatisfiable -- and leaves column 0's two decisive
            // entries untouched.
            const double f = 1.0 + 0.25 * double(b);
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    A[size_t(j) * p.ld + i] = (i == j) ? make<T>(5.0 * f, 0.0)
                                                       : make<T>(0.25 * f, -0.125 * f);
            // Column 0: cabs1 reads 3 vs 4 (row 1 wins); |z| reads 3 vs 2.828
            // (row 0 wins). Every other row of column 0 is far below both.
            A[0]        = make<T>(3.0, 0.0);
            A[1]        = make<T>(2.0, 2.0);
            for (int i = 2; i < n; ++i) A[size_t(i)] = make<T>(0.1 * f, 0.1 * f);
        }
        poison(p);

        // ANTI-VACUITY: the two functionals must genuinely disagree on this data.
        const auto z0 = up(p.a0[0]);
        const auto z1 = up(p.a0[1]);
        ASSERT_LT(verify::cabs1(z0), verify::cabs1(z1)) << "cabs1 does not prefer row 1 on this matrix";
        ASSERT_GT(verify::abs(z0), verify::abs(z1))     << "the modulus does not prefer row 0 on this matrix";

        for (int tier = 0; tier < 2; ++tier) {
            poison(p);
            if (tier == 0) {
                if (this->cta_max_n() < n) continue;
                this->run_cta(p);
            } else {
                this->run_blocked(p);
            }
            for (int b = 0; b < batch; ++b) {
                EXPECT_EQ(piv_item(p, b)[0], 2)
                    << (tier ? "blocked" : "cta") << ": ipiv[0] = " << piv_item(p, b)[0]
                    << " at b=" << b << ". 2 is cabs1's answer (LAPACK's, netlib's); 1 is the "
                       "MODULUS's answer, which is cuBLAS's and is not this library's contract";
            }
            check_factor(p, tier ? "blocked/pivot-metric" : "cta/pivot-metric");
            if (this->HasFailure()) return;
        }
    }
}

// L8. GETRS, ALL THREE transA MODES. The permutation SIDE changes with the
// transpose, and getting it wrong is a silently wrong answer no NoTrans test can
// see:
//   NoTrans  : apply F to B, then solve L, then U.
//   Trans/CT : solve U^T then L^T, then apply F^{-1} -- the SAME list walked
//              BACKWARDS -- to the OUTPUT.
TYPED_TEST(LuTest, GetrsSolvesAllThreeTransposeModes) {
    using T = typename TestFixture::T;
    const int n = 96, batch = 3;

    for (int nrhs : {1, 5}) {
        auto p = make_dominant_permuted<T>(n, batch, 1777u + unsigned(nrhs));
        this->run_blocked(p);
        ASSERT_GE(non_diagonal_pivots(p, 0), n / 2);
        ASSERT_FALSE(interchange_is_involution(p.expect_piv))
            << "this matrix's permutation is SELF-INVERSE, so the transposed getrs's "
               "backwards walk is indistinguishable from a forwards one and the Trans and "
               "ConjTrans rows below prove nothing";
        check_factor(p, "getrs/factor");
        if (this->HasFailure()) return;

        auto rhs = make_rhs<T>(n, nrhs, batch, 909u + unsigned(nrhs));
        std::vector<std::vector<T>> solutions;

        for (Transpose op : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
            reset_rhs(rhs);
            auto A = view_of(p);
            auto Bv = view_of(rhs);
            UnifiedVector<std::byte> ws(std::max<std::size_t>(
                1, sycl_getrs::getrs_blocked_buffer_size<T>(*this->ctx, A, Bv, op)));
            ASSERT_NO_THROW((void)sycl_getrs::getrs_blocked_dispatch<T>(
                *this->ctx, A, Bv, op, p.piv.to_span(), ws.to_span(), this->getrs_seam()));
            this->ctx->wait();

            for (int b = 0; b < batch; ++b) {
                const double res = solve_residual<T>(p.a0.data() + size_t(b) * p.stride,
                                                     rhs.buf.data() + size_t(b) * rhs.stride,
                                                     rhs.b0.data() + size_t(b) * rhs.stride,
                                                     n, nrhs, p.ld, rhs.ld, op);
                if (verbose())
                    std::printf("[verbose] getrs op=%d nrhs=%d b=%d  res=%.4e tol=%.4e\n",
                                int(op), nrhs, b, res, verify::bound<T>(Check::solve, n));
                EXPECT_VERIFY(T, Check::solve, n, res)
                    << "getrs transA=" << int(op) << " nrhs=" << nrhs << " b=" << b;
            }
            solutions.emplace_back(rhs.buf.begin(), rhs.buf.end());
            if (this->HasFailure()) return;
        }

        // ANTI-VACUITY. NoTrans must differ from Trans, or the mode is not read at all,
        // and for a complex type ConjTrans must differ from Trans, or conj(A) is untested.
        EXPECT_NE(solutions[0], solutions[1])
            << "NoTrans and Trans produced identical solutions; transA is not being read";
        if constexpr (test_utils::is_complex_type_v<T>) {
            EXPECT_NE(solutions[1], solutions[2])
                << "Trans and ConjTrans produced identical solutions; the conjugation is "
                   "not being applied";
        }
    }
}

// L8b. GETRS, THE TWO PERMUTATION SPELLINGS, AGREE BIT FOR BIT. Which one runs is
// a SPEED decision and never a correctness one, so the two arms must agree BIT FOR
// BIT -- strictly stronger than the residual, which both arms pass with the SAME
// wrong permutation. The spelling is READ BACK per arm, because the gather FALLS
// BACK to the walk silently when the tile does not fit local memory.
// evidence: docs/perf/lu.md#getrs-collapsed-permutation
TYPED_TEST(LuTest, GetrsPermutationSpellingsAgreeBitForBit) {
    using T = typename TestFixture::T;
    const int batch = 3;

    for (int n : {96, 257}) {
        for (int nrhs : {5, 70}) {
            auto p = make_dominant_permuted<T>(n, batch, 4242u + unsigned(n + nrhs));
            this->run_blocked(p);
            ASSERT_GE(non_diagonal_pivots(p, 0), n / 4);
            ASSERT_FALSE(interchange_is_involution(p.expect_piv))
                << "this matrix's permutation is SELF-INVERSE, so the gather's REVERSED "
                   "index walk is indistinguishable from its forward one and the Trans and "
                   "ConjTrans rows below prove nothing";
            check_factor(p, "getrs/spellings/factor");
            if (this->HasFailure()) return;

            auto rhs = make_rhs<T>(n, nrhs, batch, 1313u + unsigned(n + nrhs));

            for (Transpose op : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
                std::vector<std::vector<T>> answer;
                for (const char* spelling : {"walk", "gather"}) {
                    // The guard reloads the settings snapshot on both ends; a bare
                    // ::setenv here would be read by nothing and BOTH arms would run
                    // the default spelling, making the bit-identity assertion below
                    // compare one arm with itself. GUARD (1) catches that too, but
                    // only after the fact.
                    const ScopedEnvVar pin("BATCHLAS_GETRS_LASWP", spelling);

                    // GUARD (1). The driver's own resolution, for THIS shape on
                    // THIS queue, so a fallback the caller cannot see is visible.
                    const int got =
                        sycl_getrs::getrs_perm_spelling_debug<T>(*this->ctx, n, nrhs);
                    const int want = (std::strcmp(spelling, "gather") == 0) ? 1 : 0;
                    ASSERT_EQ(got, want)
                        << "BATCHLAS_GETRS_LASWP=" << spelling << " at n=" << n
                        << " nrhs=" << nrhs << " resolved spelling " << got
                        << ". A gather that fell back to the walk would make the "
                           "bit-identity assertion below compare the walk with itself.";

                    reset_rhs(rhs);
                    auto A = view_of(p);
                    auto Bv = view_of(rhs);
                    // The query must stay 0 under BOTH spellings: the gather is in place.
                    UnifiedVector<std::byte> ws(std::max<std::size_t>(
                        1, sycl_getrs::getrs_blocked_buffer_size<T>(*this->ctx, A, Bv, op)));
                    ASSERT_NO_THROW((void)sycl_getrs::getrs_blocked_dispatch<T>(
                        *this->ctx, A, Bv, op, p.piv.to_span(), ws.to_span(),
                        this->getrs_seam()));
                    this->ctx->wait();

                    for (int b = 0; b < batch; ++b) {
                        const double res = solve_residual<T>(
                            p.a0.data() + size_t(b) * p.stride,
                            rhs.buf.data() + size_t(b) * rhs.stride,
                            rhs.b0.data() + size_t(b) * rhs.stride,
                            n, nrhs, p.ld, rhs.ld, op);
                        EXPECT_VERIFY(T, Check::solve, n, res)
                            << "getrs spelling=" << spelling << " transA=" << int(op)
                            << " n=" << n << " nrhs=" << nrhs << " b=" << b;
                    }
                    answer.emplace_back(rhs.buf.begin(), rhs.buf.end());
                    if (this->HasFailure()) return;  // pin's destructor restores
                }

                // THE STRONG ASSERTION: the same permutation and the same two solves, so the
                // answers must be identical to the last bit.
                size_t diff = 0;
                for (size_t i = 0; i < answer[0].size(); ++i)
                    if (std::memcmp(&answer[0][i], &answer[1][i], sizeof(T)) != 0) ++diff;
                EXPECT_EQ(diff, size_t(0))
                    << "the walk and the collapsed gather disagree in " << diff
                    << " of " << answer[0].size() << " elements at transA=" << int(op)
                    << " n=" << n << " nrhs=" << nrhs
                    << ". They apply the SAME permutation to the SAME buffer and then run "
                       "the SAME two trsm calls, so any difference is a defect.";
                if (this->HasFailure()) return;
            }
        }
    }
}

// L8c. THE SPELLING DECISION SURFACE, WITHOUT RUNNING A KERNEL: the default nrhs
// boundary kGetrsPermGatherMinNrhs, and the CAPACITY REFUSAL above which the
// gather enqueues NOTHING and the driver re-schedules the walk. That fallback is
// silent by design and invisible to every other test in this suite.
TYPED_TEST(LuTest, GetrsPermSpellingDecisionSurface) {
    using T = typename TestFixture::T;
    if (this->ctx->device().type != DeviceType::GPU) GTEST_SKIP() << "the gather is GPU-only";

    constexpr int kMin = sycl_getrs::kGetrsPermGatherMinNrhs;
    ASSERT_GE(kMin, 1) << "a boundary below 1 would make the walk unreachable by default";

    {
        // A null value UNSETS for the duration and reloads, which is how the DEFAULT
        // arm is reached: an inherited BATCHLAS_GETRS_LASWP would otherwise decide
        // every assertion in this block. A bare ::unsetenv would not be seen at all.
        const ScopedEnvVar unpinned("BATCHLAS_GETRS_LASWP", nullptr);

        // THE DEFAULT nrhs BOUNDARY, both sides.
        if (kMin > 1) {
            EXPECT_EQ(sycl_getrs::getrs_perm_spelling_debug<T>(*this->ctx, 128, kMin - 1), 0)
                << "nrhs just below kGetrsPermGatherMinNrhs must take the WALK by default";
        }
        EXPECT_EQ(sycl_getrs::getrs_perm_spelling_debug<T>(*this->ctx, 128, kMin), 1)
            << "nrhs at kGetrsPermGatherMinNrhs must take the GATHER by default";
        EXPECT_EQ(sycl_getrs::getrs_perm_spelling_debug<T>(*this->ctx, 128, 4 * kMin), 1);

        // linalg::solve issues getrs at nrhs = 1 and is the only caller in the tree;
        // it must keep the walk.
        EXPECT_EQ(sycl_getrs::getrs_perm_spelling_debug<T>(*this->ctx, 512, 1),
                  kMin <= 1 ? 1 : 0);
    }

    // THE OVERRIDES beat the boundary in both directions.
    {
        const ScopedEnvVar pin("BATCHLAS_GETRS_LASWP", "walk");
        EXPECT_EQ(sycl_getrs::getrs_perm_spelling_debug<T>(*this->ctx, 128, 4 * kMin), 0)
            << "BATCHLAS_GETRS_LASWP=walk must force the walk above the boundary";
    }
    {
        const ScopedEnvVar pin("BATCHLAS_GETRS_LASWP", "gather");
        EXPECT_EQ(sycl_getrs::getrs_perm_spelling_debug<T>(*this->ctx, 128, 1), 1)
            << "BATCHLAS_GETRS_LASWP=gather must force the gather below the boundary";

        // THE CAPACITY REFUSAL, forced on -- the pin above is what "forced" means, and
        // it stays in scope for both rows -- at an order no tile can hold. This is the
        // only assertion in the suite that the fallback branch is reachable at all.
        EXPECT_EQ(sycl_getrs::getrs_perm_spelling_debug<T>(*this->ctx, 1 << 20, 4 * kMin), 0)
            << "the gather must REFUSE (and fall back to the walk) at an order whose column "
               "cannot fit local memory, rather than launching a kernel that cannot run";

        // ...and it must NOT refuse at an order the suite reaches: a capacity that fires
        // early is a lever that never runs.
        EXPECT_EQ(sycl_getrs::getrs_perm_spelling_debug<T>(*this->ctx, 1024, 4 * kMin), 1)
            << "the gather must still fit at n = 1024, the largest order this pass measured";
    }
}

// L8d. THE GATHER BUYS NO WORKSPACE, AT ANY WIDTH. The facade takes the workspace
// maximum over EVERY NATIVE TIER WHOSE can_run ACCEPTS the shape, not over the tier the
// route named, so a gather that bought an out-of-place RHS here would bill every
// narrow call that routes to the FUSED tier and needs nothing.
TYPED_TEST(LuTest, GetrsPermGatherBuysNoWorkspace) {
    using T = typename TestFixture::T;
    const int n = 96, batch = 3;
    auto p = make_dominant_permuted<T>(n, batch, 606u);

    for (int nrhs : {1, 8, 128}) {
        auto rhs = make_rhs<T>(n, nrhs, batch, 707u + unsigned(nrhs));
        auto A = view_of(p);
        auto Bv = view_of(rhs);
        for (const char* spelling : {"walk", "gather"}) {
            // The sizing query resolves the spelling through the same settings
            // snapshot the solve does, so the pin has to be a guard that reloads it
            // or both rows below measure the default spelling twice.
            const ScopedEnvVar pin("BATCHLAS_GETRS_LASWP", spelling);
            for (Transpose op : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
                EXPECT_EQ(sycl_getrs::getrs_blocked_buffer_size<T>(*this->ctx, A, Bv, op),
                          std::size_t(0))
                    << "the composed getrs must stay workspace-free at nrhs=" << nrhs
                    << " spelling=" << spelling << " transA=" << int(op)
                    << ". A buffer bought here is charged to every narrow call that "
                       "routes to the FUSED tier, because the facade maxes over every "
                       "SUPPORTED native tier and not over the routed one.";
            }
        }
    }
}

// L9. GETRI: the inverse, and the promise that A SURVIVES. cublas<t>getriBatched
// takes `const T* const A[]`, so a native arm that wrote through A would be a
// drop-in failure invisible to every residual; the survival is asserted BIT-EXACTLY.
TYPED_TEST(LuTest, GetriInvertsAndLeavesTheFactorUntouched) {
    using T = typename TestFixture::T;
    const int n = 80, batch = 3;

    auto p = make_dominant_permuted<T>(n, batch, 3131u);
    this->run_blocked(p);
    ASSERT_GE(non_diagonal_pivots(p, 0), n / 2);
    ASSERT_FALSE(interchange_is_involution(p.expect_piv))
        << "this matrix's permutation is SELF-INVERSE, so getri's BACKWARD trace through the "
           "interchange list is indistinguishable from a forward one";
    check_factor(p, "getri/factor");
    if (this->HasFailure()) return;
    const std::vector<T> factored(p.buf.begin(), p.buf.end());

    Lu<T> c;
    alloc(c, n, batch, 7, 13);
    std::fill(c.buf.begin(), c.buf.end(), make<T>(-9.75e3, 4.5e3));
    UnifiedVector<int32_t> cinfo(size_t(batch), int32_t(-12345));

    auto A = view_of(p);
    auto C = view_of(c);
    UnifiedVector<std::byte> ws(std::max<std::size_t>(
        1, sycl_getri::getri_blocked_buffer_size<T>(*this->ctx, A)));
    ASSERT_NO_THROW((void)sycl_getri::getri_blocked_dispatch<T>(
        *this->ctx, A, C, p.piv.to_span(), ws.to_span(), cinfo.to_span(), this->getri_seam()));
    this->ctx->wait();

    for (int b = 0; b < batch; ++b) {
        EXPECT_EQ(cinfo[b], 0) << "b=" << b;
        const double res = inverse_residual<T>(p.a0.data() + size_t(b) * p.stride,
                                               c.buf.data() + size_t(b) * c.stride,
                                               n, p.ld, c.ld);
        if (verbose())
            std::printf("[verbose] getri b=%d  ||AC-I||/(||A|| ||C||)=%.4e tol=%.4e\n",
                        b, res, verify::bound<T>(Check::solve, n));
        EXPECT_VERIFY(T, Check::solve, n, res) << "||A C - I||_F / (||A||_F ||C||_F) at b=" << b;
    }

    for (size_t i = 0; i < factored.size(); ++i)
        ASSERT_EQ(verify::abs(up(p.buf[i]) - up(factored[i])), 0.0)
            << "getri wrote through A at element " << i
            << "; cublas<t>getriBatched takes A as const and a caller may reuse it";

    // The LAST batch item differs from the first, so a wrong output stride cannot pass
    // by broadcasting item 0.
    bool differ = false;
    for (int j = 0; j < n && !differ; ++j)
        for (int i = 0; i < n && !differ; ++i)
            if (verify::abs(up(c.buf[size_t(j) * c.ld + i]) -
                     up(c.buf[size_t(batch - 1) * c.stride + size_t(j) * c.ld + i])) > 0.0)
                differ = true;
    EXPECT_TRUE(differ) << "the first and last inverses are identical";
}

// L10 / L11. THE DROP-IN CONTRACT, BOTH DIRECTIONS. getrf, getrs and getri carry
// INDEPENDENT env variables and INDEPENDENT tuned tables, so every mixture
// of native and vendor arms is reachable in a shipped build. The two getrf
// implementations are NOT required to agree on the PIVOTS they choose: cuBLAS
// pivots on the modulus for complex.
TYPED_TEST(LuTest, NativeFactorFeedsTheVendorSolvers) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::BackendType;
    if constexpr (!batchlas::select::factorization_vendor_available<B>) {
        GTEST_SKIP() << "no factorization vendor in this build";
    } else {
        const int n = 72, batch = 3, nrhs = 4;
        auto p = make_dominant_permuted<T>(n, batch, 6767u);
        this->run_blocked(p);
        check_factor(p, "dropin/native-factor");
        if (this->HasFailure()) return;

        auto A = view_of(p);
        {   // vendor getrs on the native factor
            auto rhs = make_rhs<T>(n, nrhs, batch, 4545u);
            auto Bv = view_of(rhs);
            UnifiedVector<std::byte> ws(std::max<std::size_t>(
                1, backend::getrs_vendor_buffer_size<B, T>(*this->ctx, A, Bv,
                                                           Transpose::NoTrans)));
            ASSERT_NO_THROW(((void)backend::getrs_vendor<B, T>(*this->ctx, A, Bv, Transpose::NoTrans,
                                                         p.piv.to_span(), ws.to_span())));
            this->ctx->wait();
            for (int b = 0; b < batch; ++b)
                EXPECT_VERIFY(T, Check::solve, n, solve_residual<T>(p.a0.data() + size_t(b) * p.stride,
                                            rhs.buf.data() + size_t(b) * rhs.stride,
                                            rhs.b0.data() + size_t(b) * rhs.stride,
                                            n, nrhs, p.ld, rhs.ld, Transpose::NoTrans))
                    << "the VENDOR getrs could not consume the NATIVE getrf's factor, b=" << b;
        }
        {   // vendor getri on the native factor
            Lu<T> c; alloc(c, n, batch, 7, 13);
            std::fill(c.buf.begin(), c.buf.end(), make<T>(-9.75e3, 4.5e3));
            UnifiedVector<int32_t> ci(size_t(batch), int32_t(-12345));
            auto C = view_of(c);
            UnifiedVector<std::byte> ws(std::max<std::size_t>(
                1, backend::getri_vendor_buffer_size<B, T>(*this->ctx, A)));
            ASSERT_NO_THROW(((void)backend::getri_vendor<B, T>(*this->ctx, A, C, p.piv.to_span(),
                                                         ws.to_span(), ci.to_span())));
            this->ctx->wait();
            for (int b = 0; b < batch; ++b) {
                EXPECT_EQ(ci[b], 0);
                EXPECT_VERIFY(T, Check::solve, n, inverse_residual<T>(p.a0.data() + size_t(b) * p.stride,
                                              c.buf.data() + size_t(b) * c.stride, n, p.ld, c.ld))
                    << "the VENDOR getri could not consume the NATIVE getrf's factor, b=" << b;
            }
        }
    }
}

TYPED_TEST(LuTest, VendorFactorFeedsTheNativeSolvers) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::BackendType;
    if constexpr (!batchlas::select::factorization_vendor_available<B>) {
        GTEST_SKIP() << "no factorization vendor in this build";
    } else {
        const int n = 72, batch = 3, nrhs = 4;
        auto p = make_dominant_permuted<T>(n, batch, 8989u);
        {
            auto A = view_of(p);
            UnifiedVector<std::byte> ws(std::max<std::size_t>(
                1, backend::getrf_vendor_buffer_size<B, T>(*this->ctx, A)));
            ASSERT_NO_THROW(((void)backend::getrf_vendor<B, T>(*this->ctx, A, p.piv.to_span(),
                                                         ws.to_span(), p.info.to_span())));
            this->ctx->wait();
            for (int b = 0; b < batch; ++b) ASSERT_EQ(p.info[b], 0);
            // The vendor's own factor must satisfy the same host reconstruction, which proves
            // the two agree on the pivot FORMAT even where they differ on the pivot CHOICE.
            check_factor(p, "dropin/vendor-factor", /*check_L=*/false);
            if (this->HasFailure()) return;
        }

        auto A = view_of(p);
        {   // native getrs on the vendor factor
            auto rhs = make_rhs<T>(n, nrhs, batch, 2323u);
            auto Bv = view_of(rhs);
            UnifiedVector<std::byte> ws(std::max<std::size_t>(
                1, sycl_getrs::getrs_blocked_buffer_size<T>(*this->ctx, A, Bv,
                                                            Transpose::NoTrans)));
            ASSERT_NO_THROW((void)sycl_getrs::getrs_blocked_dispatch<T>(
                *this->ctx, A, Bv, Transpose::NoTrans, p.piv.to_span(), ws.to_span(),
                this->getrs_seam()));
            this->ctx->wait();
            for (int b = 0; b < batch; ++b)
                EXPECT_VERIFY(T, Check::solve, n, solve_residual<T>(p.a0.data() + size_t(b) * p.stride,
                                            rhs.buf.data() + size_t(b) * rhs.stride,
                                            rhs.b0.data() + size_t(b) * rhs.stride,
                                            n, nrhs, p.ld, rhs.ld, Transpose::NoTrans))
                    << "the NATIVE getrs could not consume the VENDOR getrf's factor, b=" << b;
        }
        {   // native getri on the vendor factor
            Lu<T> c; alloc(c, n, batch, 7, 13);
            std::fill(c.buf.begin(), c.buf.end(), make<T>(-9.75e3, 4.5e3));
            UnifiedVector<int32_t> ci(size_t(batch), int32_t(-12345));
            auto C = view_of(c);
            UnifiedVector<std::byte> ws(std::max<std::size_t>(
                1, sycl_getri::getri_blocked_buffer_size<T>(*this->ctx, A)));
            ASSERT_NO_THROW((void)sycl_getri::getri_blocked_dispatch<T>(
                *this->ctx, A, C, p.piv.to_span(), ws.to_span(), ci.to_span(),
                this->getri_seam()));
            this->ctx->wait();
            for (int b = 0; b < batch; ++b) {
                EXPECT_EQ(ci[b], 0);
                EXPECT_VERIFY(T, Check::solve, n, inverse_residual<T>(p.a0.data() + size_t(b) * p.stride,
                                              c.buf.data() + size_t(b) * c.stride, n, p.ld, c.ld))
                    << "the NATIVE getri could not consume the VENDOR getrf's factor, b=" << b;
            }
        }
    }
}

// L12. GETRS'S FUSED CAPACITY ON THE REAL DEVICE: whether the device reports a
// capacity at all here. The windows and vendor-free walks are table data, asserted in
// get{rf,rs,ri}_candidates_tests.
TYPED_TEST(LuTest, GetrsFusedCapacityOnTheRealDevice) {
    using T = typename TestFixture::T;

    // This test asserts what the tables do with NO route pinned, so it has to say
    // so: an inherited BATCHLAS_GET*_ROUTE -- exported in a shell, or set by the
    // route-pinned ctest rerun -- otherwise forces the answer and the test reports
    // a window defect that is really just its own environment. Empty reads as
    // unset in select's pin reader.
    ScopedEnvVar clear_getrf("BATCHLAS_GETRF_ROUTE", "");
    ScopedEnvVar clear_getrs("BATCHLAS_GETRS_ROUTE", "");
    ScopedEnvVar clear_getri("BATCHLAS_GETRI_ROUTE", "");

    auto large = make_dominant_permuted<T>(512, 2, 6u);
    auto Vl = view_of(large);

    // getrf, getrs and getri all select from transcribed tables now: their windows and
    // vendor-free walks are asserted in get{rf,rs,ri}_candidates_tests.

    // ---- GETRS'S FUSED TIER ON THIS DEVICE --------------------------------
    // The window itself is the transcribed table's (getrs_candidates_tests). What only a real
    // device shows is whether the fused capacity is there at all: a zero capacity makes every
    // cta row unrunnable and the table's cta entries silently mean the next one.
    {
        auto rhs1 = make_rhs<T>(large.n, 1, large.batch, 5151u);
        auto V1 = view_of(rhs1);
        EXPECT_GE(sycl_getrs::getrs_fused_max_rhs_elems<T>(this->budget()), std::size_t(large.n))
            << "n=" << large.n << " at nrhs=1 must fit the resident-RHS capacity";
        for (Transpose op : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans})
            EXPECT_TRUE(this->getrs_pin_accepted(ops::getrs::Cta{}, Vl, V1, op)) << "transA=" << int(op);
        auto rhsw = make_rhs<T>(large.n, int(sycl_getrs::kGetrsFusedMaxRhs) + 8, large.batch, 5252u);
        auto Vw2 = view_of(rhsw);
        EXPECT_FALSE(this->getrs_pin_accepted(ops::getrs::Cta{}, Vl, Vw2, Transpose::NoTrans))
            << "above the widest instantiated accumulator";
    }

}

// L13. THE FACADE REACHES THE NATIVE KERNELS, ASSERTED BIT-EXACTLY. A route
// assertion plus a residual can stay GREEN while every number in it comes from the
// vendor, so the comparison here is BIT-EXACT against the direct entry point --
// factor AND pivots -- which no vendor can reproduce.
TYPED_TEST(LuTest, FacadeReachesTheNativeKernelsBitExactly) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::BackendType;
    // Derived from the tier's own ceiling, not hardcoded: the occupancy rule moves it,
    // and a pin that cannot be exercised is a red test rather than a lost guard.
    const int n = std::min(64, this->cta_max_n()), batch = 3;
    ASSERT_GE(n, 32) << "the CTA pin cannot be exercised at a useful order on this device";

    for (const char* pin : {"cta", "blocked"}) {
        ScopedEnvVar g("BATCHLAS_GETRF_ROUTE", pin);
        auto direct = make_dominant_permuted<T>(n, batch, 1234u);
        auto viafac = make_dominant_permuted<T>(n, batch, 1234u);

        // A getrf pin that cannot run throws (flat selection, R6) instead of falling to the
        // vendor, so the bit-exact comparison below is against the pinned tier.
        auto Vf = view_of(viafac);

        if (std::strcmp(pin, "cta") == 0) this->run_cta(direct);
        else                              this->run_blocked(direct);

        UnifiedVector<std::byte> ws(std::max<std::size_t>(
            1, getrf_buffer_size<B, T>(*this->ctx, Vf)));
        ASSERT_NO_THROW(((void)getrf<B, T>(*this->ctx, Vf, viafac.piv.to_span(), ws.to_span(),
                                     viafac.info.to_span())));
        this->ctx->wait();

        for (size_t i = 0; i < direct.buf.size(); ++i)
            ASSERT_EQ(verify::abs(up(direct.buf[i]) - up(viafac.buf[i])), 0.0)
                << "pin=" << pin << ": the facade's factor differs from the direct entry "
                   "point's at element " << i << " -- something else served this call";
        for (int b = 0; b < batch; ++b)
            for (int k = 0; k < n; ++k)
                ASSERT_EQ(piv_item(direct, b)[k], piv_item(viafac, b)[k])
                    << "pin=" << pin << ": pivot " << k << " of item " << b << " differs";
        check_factor(viafac, "facade/getrf");
        if (this->HasFailure()) return;
    }

    // getrs and getri through the facade, against their direct entry points.
    {
        auto p = make_dominant_permuted<T>(n, batch, 4321u);
        this->run_blocked(p);
        auto A = view_of(p);

        // A pin its can_run refuses throws, so the comparison below is never vendor vs native.
        const select::ScopedPin<ops::getrs::GetrsChoice> g("getrs", ops::getrs::Blocked{});
        auto r1 = make_rhs<T>(n, 3, batch, 88u);
        auto r2 = make_rhs<T>(n, 3, batch, 88u);
        auto V1 = view_of(r1);
        auto V2 = view_of(r2);

        UnifiedVector<std::byte> w1(std::max<std::size_t>(
            1, sycl_getrs::getrs_blocked_buffer_size<T>(*this->ctx, A, V1, Transpose::Trans)));
        (void)sycl_getrs::getrs_blocked_dispatch<T>(*this->ctx, A, V1, Transpose::Trans,
                                              p.piv.to_span(), w1.to_span(), this->getrs_seam());
        this->ctx->wait();
        UnifiedVector<std::byte> w2(std::max<std::size_t>(
            1, getrs_buffer_size<B, T>(*this->ctx, A, V2, Transpose::Trans)));
        ASSERT_NO_THROW(((void)getrs<B, T>(*this->ctx, A, V2, Transpose::Trans, p.piv.to_span(),
                                     w2.to_span())));
        this->ctx->wait();
        for (size_t i = 0; i < r1.buf.size(); ++i)
            ASSERT_EQ(verify::abs(up(r1.buf[i]) - up(r2.buf[i])), 0.0)
                << "the facade's getrs differs from the direct driver at element " << i;
    }
    {
        auto p = make_dominant_permuted<T>(n, batch, 4321u);
        this->run_blocked(p);
        auto A = view_of(p);

        // A ScopedPin throws when Blocked cannot run this shape, so it needs no readback.
        const select::ScopedPin<ops::getri::GetriChoice> g("getri", ops::getri::Blocked{});
        Lu<T> c1, c2;
        alloc(c1, n, batch, 7, 13);
        alloc(c2, n, batch, 7, 13);
        std::fill(c1.buf.begin(), c1.buf.end(), make<T>(-9.75e3, 4.5e3));
        std::fill(c2.buf.begin(), c2.buf.end(), make<T>(-9.75e3, 4.5e3));
        UnifiedVector<int32_t> i1(size_t(batch), -12345), i2(size_t(batch), -12345);
        auto C1 = view_of(c1);
        auto C2 = view_of(c2);
        UnifiedVector<std::byte> w1(std::max<std::size_t>(
            1, sycl_getri::getri_blocked_buffer_size<T>(*this->ctx, A)));
        (void)sycl_getri::getri_blocked_dispatch<T>(*this->ctx, A, C1, p.piv.to_span(), w1.to_span(),
                                              i1.to_span(), this->getri_seam());
        this->ctx->wait();
        UnifiedVector<std::byte> w2(std::max<std::size_t>(
            1, getri_buffer_size<B, T>(*this->ctx, A)));
        ASSERT_NO_THROW(((void)getri<B, T>(*this->ctx, A, C2, p.piv.to_span(), w2.to_span(),
                                     i2.to_span())));
        this->ctx->wait();
        for (size_t i = 0; i < c1.buf.size(); ++i)
            ASSERT_EQ(verify::abs(up(c1.buf[i]) - up(c2.buf[i])), 0.0)
                << "the facade's getri differs from the direct driver at element " << i;
    }
}

// L14. THE DIRECT ENTRY POINTS REFUSE WHAT can_run REFUSES. They are reachable
// WITHOUT the table, so every gate has to be re-applied there or a pinned-route
// caller walks into an unlaunchable configuration.
TYPED_TEST(LuTest, DirectEntryPointsRefuseWhatSupportsRefuses) {
    using T = typename TestFixture::T;
    const int n = 24, batch = 2;
    auto p = make_dominant_permuted<T>(n, batch, 31u);
    auto A = view_of(p);
    UnifiedVector<std::byte> ws(4096);

    // A non-square view.
    {
        UnifiedVector<T> w(size_t(24) * 32, make<T>(1.0, 0.0));
        UnifiedVector<T*> wp(1, nullptr);
        MatrixView<T, MatrixFormat::Dense> W(w.data(), 24, 32, 24, 24 * 32, 1, wp.data());
        UnifiedVector<int64_t> pv(64, 0);
        EXPECT_THROW((void)sycl_getrf::getrf_blocked_dispatch<T>(*this->ctx, W, pv.to_span(),
                                                           ws.to_span(), Span<int32_t>{},
                                                           this->gemm_seam(), this->trsm_seam()),
                     std::invalid_argument);
    }
    // A pivot span shorter than n * batch.
    {
        UnifiedVector<int64_t> shortpiv(size_t(n) * batch - 1, 0);
        EXPECT_THROW((void)sycl_getrf::getrf_blocked_dispatch<T>(*this->ctx, A, shortpiv.to_span(),
                                                           ws.to_span(), Span<int32_t>{},
                                                           this->gemm_seam(), this->trsm_seam()),
                     std::invalid_argument);
        EXPECT_THROW((void)sycl_getrf::getrf_cta_dispatch<T>(*this->ctx, A, shortpiv.to_span(),
                                                       ws.to_span(), Span<int32_t>{}),
                     std::invalid_argument);
    }
    // An empty trailing-update seam. NOT defaulted to gemm_custom any more.
    EXPECT_THROW((void)sycl_getrf::getrf_blocked_dispatch<T>(*this->ctx, A, p.piv.to_span(),
                                                       ws.to_span(), Span<int32_t>{},
                                                       sycl_getrf::GetrfTrailingGemm<T>{},
                                                       this->trsm_seam()),
                 std::invalid_argument);
    // An empty panel-solve seam. NOT defaulted to a native trsm.
    EXPECT_THROW((void)sycl_getrf::getrf_blocked_dispatch<T>(*this->ctx, A, p.piv.to_span(),
                                                       ws.to_span(), Span<int32_t>{},
                                                       this->gemm_seam(),
                                                       sycl_getrf::GetrfPanelSolveTrsm<T>{}),
                 std::invalid_argument);
    // An order past the CTA tier's advertised capacity.
    {
        const int over = this->cta_max_n() + 1;
        auto big = make_dominant_permuted<T>(over, 1, 41u);
        auto Vb = view_of(big);
        EXPECT_THROW((void)sycl_getrf::getrf_cta_dispatch<T>(*this->ctx, Vb, big.piv.to_span(),
                                                       ws.to_span(), Span<int32_t>{}),
                     std::invalid_argument)
            << "getrf_cta_dispatch accepted order " << over << " with a capacity of "
            << this->cta_max_n() << "; the table would have promised a launch the device refuses";
    }
    // getrs / getri seam and shape refusals.
    {
        this->run_blocked(p);
        auto rhs = make_rhs<T>(n, 2, batch, 12u);
        auto Bv = view_of(rhs);
        EXPECT_THROW((void)sycl_getrs::getrs_blocked_dispatch<T>(*this->ctx, A, Bv, Transpose::NoTrans,
                                                           p.piv.to_span(), ws.to_span(),
                                                           sycl_getrs::GetrsSolveTrsm<T>{}),
                     std::invalid_argument);
        auto mismatched = make_rhs<T>(n + 1, 2, batch, 13u);
        auto Bm = view_of(mismatched);
        EXPECT_THROW((void)sycl_getrs::getrs_blocked_dispatch<T>(*this->ctx, A, Bm, Transpose::NoTrans,
                                                           p.piv.to_span(), ws.to_span(),
                                                           this->getrs_seam()),
                     std::invalid_argument);

        UnifiedVector<int32_t> ci(size_t(batch), 0);
        EXPECT_THROW((void)sycl_getri::getri_blocked_dispatch<T>(*this->ctx, A, A, p.piv.to_span(),
                                                           ws.to_span(), ci.to_span(),
                                                           this->getri_seam()),
                     std::invalid_argument)
            << "getri must refuse C aliasing A: C is zeroed before A's triangles are read";
        // A DISTINCT C, so the alias gate above cannot fire first. Passing A as
        // both operands made this assertion vacuous: it threw on the aliasing
        // check at getri_blocked.cc:193 and never reached the empty-seam refusal
        // at :204, so deleting that refusal entirely left the test green -- and a
        // direct caller that forgot the injection would then call an empty
        // std::function rather than getting a diagnostic.
        auto cbuf = make_dominant_permuted<T>(n, batch, 71u);
        auto Cv = view_of(cbuf);
        EXPECT_THROW((void)sycl_getri::getri_blocked_dispatch<T>(*this->ctx, A, Cv, p.piv.to_span(),
                                                           ws.to_span(), ci.to_span(),
                                                           sycl_getri::GetriSolveTrsm<T>{}),
                     std::invalid_argument)
            << "getri must refuse an empty solve seam rather than call an empty std::function";
    }
}

// L15. THE WORKSPACE QUERY COVERS EVERY SUPPORTED ROUTE, AND DEREFERENCES
// NOTHING: getrf_buffer_size and getri_buffer_size are reached from inside a
// layout function under BumpAllocator::measuring() (src/extensions/inv.cc), where
// A arrives with a NULL data pointer. getrf's figure is exactly the chosen tier's (R5).
TYPED_TEST(LuTest, BufferSizeCoversEveryRouteAndNeverDereferences) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::BackendType;
    const int n = std::min(64, this->cta_max_n()), batch = 3;
    ASSERT_GE(n, 32);

    // NULL data, exactly as a measuring pass presents it.
    {
        MatrixView<T, MatrixFormat::Dense> nullv(nullptr, n, n, n + 4, (n + 4) * n + 3, batch,
                                                 nullptr);
        EXPECT_NO_THROW((void)sycl_getrf::getrf_cta_buffer_size<T>(*this->ctx, nullv));
        EXPECT_NO_THROW((void)sycl_getrf::getrf_blocked_buffer_size<T>(*this->ctx, nullv));
        EXPECT_NO_THROW((void)sycl_getri::getri_blocked_buffer_size<T>(*this->ctx, nullv));
        EXPECT_NO_THROW(((void)getrf_buffer_size<B, T>(*this->ctx, nullv)));
        EXPECT_NO_THROW(((void)getri_buffer_size<B, T>(*this->ctx, nullv)));
    }

    for (const char* pin : {"cta", "blocked"}) {
        ScopedEnvVar g("BATCHLAS_GETRF_ROUTE", pin);
        auto p = make_dominant_permuted<T>(n, batch, 2u);
        auto V = view_of(p);
        const std::size_t need = getrf_buffer_size<B, T>(*this->ctx, V);
        const std::size_t native_need =
            (std::strcmp(pin, "cta") == 0)
                ? sycl_getrf::getrf_cta_buffer_size<T>(*this->ctx, V)
                : sycl_getrf::getrf_blocked_buffer_size<T>(*this->ctx, V);
        EXPECT_EQ(need, native_need)
            << "pin '" << pin << "': the facade's figure is not the pinned tier's own (R5)";

        // Serve EXACTLY that many bytes: a short workspace is a silent heap overflow.
        UnifiedVector<std::byte> ws(std::max<std::size_t>(1, need));
        ASSERT_NO_THROW(((void)getrf<B, T>(*this->ctx, V, p.piv.to_span(), ws.to_span(),
                                     p.info.to_span())));
        this->ctx->wait();
        check_factor(p, "buffer-size/getrf");
        if (this->HasFailure()) return;
    }
}

// F1. THE FUSED TIER SOLVES ALL THREE transA MODES AT EVERY INSTANTIATED WIDTH.
//
// The accumulator width NR is a COMPILE-TIME template parameter chosen by a
// runtime ladder (nrhs <= 1 -> 1, <= 2 -> 2, <= 4 -> 4, else 8), so 1, 2, 4 and 8
// are four different kernels, and 3 and 5 are the shapes where the `if (c < nrhs)`
// guards inside a WIDER accumulator are all that keeps a lane out of a column that
// does not exist. n = 97 is six full nb = 16 blocks plus a FINAL BLOCK OF ONE.
TYPED_TEST(LuTest, FusedGetrsSolvesEveryTransposeAtEveryInstantiatedWidth) {
    using T = typename TestFixture::T;
    const int n = 97, batch = 3;

    auto p = make_dominant_permuted<T>(n, batch, 5150u);
    this->run_blocked(p);
    ASSERT_GE(non_diagonal_pivots(p, 0), n / 2);
    ASSERT_FALSE(interchange_is_involution(p.expect_piv))
        << "this matrix's permutation is SELF-INVERSE, so the transposed arm's backwards walk "
           "is indistinguishable from a forwards one and the Trans/ConjTrans rows prove nothing";
    check_factor(p, "fused/factor");
    if (this->HasFailure()) return;

    for (int nrhs : {1, 2, 3, 4, 5, 8}) {
        ASSERT_LE(int64_t(nrhs), sycl_getrs::kGetrsFusedMaxRhs);
        auto rhs = make_rhs<T>(n, nrhs, batch, 6160u + unsigned(nrhs));
        std::vector<std::vector<T>> solutions;

        for (Transpose op : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
            reset_rhs(rhs);
            auto A = view_of(p);
            auto Bv = view_of(rhs);
            UnifiedVector<std::byte> ws(std::max<std::size_t>(
                1, sycl_getrs::getrs_fused_buffer_size<T>(*this->ctx, A, Bv, op)));
            ASSERT_NO_THROW((void)sycl_getrs::getrs_fused_dispatch<T>(
                *this->ctx, A, Bv, op, p.piv.to_span(), ws.to_span()));
            this->ctx->wait();

            // THE ORACLE IS ||op(A) X - B|| AGAINST THE **ORIGINAL** A, and not the L/U-based
            // one the hole ladder uses: only this form is sensitive to the permutation
            // DIRECTION, because only this form knows what A was before it was factored.
            for (int b = 0; b < batch; ++b) {
                const double res = solve_residual<T>(p.a0.data() + size_t(b) * p.stride,
                                                     rhs.buf.data() + size_t(b) * rhs.stride,
                                                     rhs.b0.data() + size_t(b) * rhs.stride,
                                                     n, nrhs, p.ld, rhs.ld, op);
                if (verbose())
                    std::printf("[verbose] fused getrs op=%d nrhs=%d b=%d  res=%.4e tol=%.4e\n",
                                int(op), nrhs, b, res, verify::bound<T>(Check::solve, n));
                EXPECT_VERIFY(T, Check::solve, n, res)
                    << "fused getrs transA=" << int(op) << " nrhs=" << nrhs << " b=" << b;
            }
            check_rhs_pad_intact(rhs, "fused/window");
            check_items_differ(rhs, "fused/window");
            solutions.emplace_back(rhs.buf.begin(), rhs.buf.end());
            if (this->HasFailure()) return;
        }

        EXPECT_NE(solutions[0], solutions[1])
            << "nrhs=" << nrhs << ": NoTrans and Trans produced identical solutions; transA is "
               "not being read";
        if constexpr (test_utils::is_complex_type_v<T>) {
            EXPECT_NE(solutions[1], solutions[2])
                << "nrhs=" << nrhs << ": Trans and ConjTrans produced identical solutions; the "
                   "conjugation is not being applied";
        }
    }
}

// F2. ORDERS: ONE, THE BLOCK BOUNDARIES, AND THE nb SWITCH AT 1024. nb = 16 below
// order 1024 and 32 at or above it, then clamped to n, so 1023/1024/1025 straddle a
// change of BOTH the block width and the resident block's leading dimension, and
// jb == 1 on the final block disables the unit-diagonal recurrence entirely in two
// of the four substitutions. ORDERS 1 AND 2 SKIP THE INVOLUTION ASSERTION: an
// n-cycle IS self-inverse for n <= 2.
TYPED_TEST(LuTest, FusedGetrsAtBlockBoundariesAndTheNbSwitch) {
    using T = typename TestFixture::T;
    const int batch = 2;

    for (int n : {1, 2, 3, 15, 16, 17, 31, 32, 33, 48, 64, 65, 128, 1023, 1024, 1025}) {
        auto p = make_dominant_permuted<T>(n, batch, 7070u + unsigned(n));
        this->run_blocked(p);
        if (n > 2) {
            ASSERT_FALSE(interchange_is_involution(p.expect_piv)) << "n=" << n;
            ASSERT_GT(non_diagonal_pivots(p, batch - 1), 0) << "n=" << n;
        }
        // ||PA - LU|| is O(n^3); at 1023 and above the end-to-end solve residual
        // against the ORIGINAL A is the oracle, which is O(n^2 nrhs).
        if (n <= 128) {
            check_factor(p, "fused/orders");
            if (this->HasFailure()) return;
        }

        for (int nrhs : {1, 3}) {
            const std::size_t cap = sycl_getrs::getrs_fused_max_rhs_elems<T>(this->budget());
            if (std::size_t(n) * std::size_t(nrhs) > cap) continue;
            auto rhs = make_rhs<T>(n, nrhs, batch, 8080u + unsigned(n) + unsigned(nrhs));
            for (Transpose op : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
                reset_rhs(rhs);
                auto A = view_of(p);
                auto Bv = view_of(rhs);
                UnifiedVector<std::byte> ws(std::max<std::size_t>(
                    1, sycl_getrs::getrs_fused_buffer_size<T>(*this->ctx, A, Bv, op)));
                ASSERT_NO_THROW((void)sycl_getrs::getrs_fused_dispatch<T>(
                    *this->ctx, A, Bv, op, p.piv.to_span(), ws.to_span()))
                    << "n=" << n << " nrhs=" << nrhs << " transA=" << int(op);
                this->ctx->wait();
                for (int b = 0; b < batch; ++b) {
                    const double res = solve_residual<T>(p.a0.data() + size_t(b) * p.stride,
                                                         rhs.buf.data() + size_t(b) * rhs.stride,
                                                         rhs.b0.data() + size_t(b) * rhs.stride,
                                                         n, nrhs, p.ld, rhs.ld, op);
                    EXPECT_VERIFY(T, Check::solve, n, res)
                        << "fused getrs n=" << n << " nrhs=" << nrhs << " transA=" << int(op)
                        << " b=" << b;
                }
                check_rhs_pad_intact(rhs, "fused/orders");
                if (this->HasFailure()) return;
            }
        }
    }
}

// F3. THE TWO CEILINGS: THE WIDTH THE BUILD INSTANTIATED (kGetrsFusedMaxRhs = 8)
// AND THE DEVICE'S RESIDENT-RHS CAPACITY. BOTH MUST HAND BACK, NOT PRODUCE
// GARBAGE. Both live in can_run() and never in a table, because above either the
// kernel does not launch -- and a pinned `cta` past them throws (R6) rather than
// measuring another kernel.
TYPED_TEST(LuTest, FusedGetrsHandsBackAtBothCeilings) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::BackendType;
    const ops::getrs::GetrsChoice cta = ops::getrs::Cta{};
    const int maxr = int(sycl_getrs::kGetrsFusedMaxRhs);
    const int n = 40, batch = 2;

    auto p = make_dominant_permuted<T>(n, batch, 9090u);
    this->run_blocked(p);
    check_factor(p, "fused/ceiling");
    if (this->HasFailure()) return;
    auto A = view_of(p);

    // ---- ceiling 1: the instantiated width --------------------------------
    for (int nrhs : {maxr, maxr + 1}) {
        auto rhs = make_rhs<T>(n, nrhs, batch, 1010u + unsigned(nrhs));
        auto Bv = view_of(rhs);
        EXPECT_EQ(this->getrs_pin_accepted(cta, A, Bv, Transpose::NoTrans), nrhs <= maxr)
            << "can_run(cta) at nrhs=" << nrhs << " with kGetrsFusedMaxRhs=" << maxr;

        UnifiedVector<std::byte> ws(std::max<std::size_t>(
            1, sycl_getrs::getrs_fused_buffer_size<T>(*this->ctx, A, Bv, Transpose::NoTrans)));
        if (nrhs <= maxr) {
            ASSERT_NO_THROW((void)sycl_getrs::getrs_fused_dispatch<T>(
                *this->ctx, A, Bv, Transpose::NoTrans, p.piv.to_span(), ws.to_span()));
        } else {
            EXPECT_THROW((void)sycl_getrs::getrs_fused_dispatch<T>(
                             *this->ctx, A, Bv, Transpose::NoTrans, p.piv.to_span(),
                             ws.to_span()),
                         std::invalid_argument)
                << "the direct entry point accepted nrhs=" << nrhs << " with only " << maxr
                << " instantiated; the table would then promise a route with no kernel behind it";
        }
        this->ctx->wait();
    }

    // ONE PAST THE WIDTH, THROUGH THE FACADE, MUST STILL BE RIGHT: the route has to
    // fall to a tier that can serve it and the answer has to survive the handover.
    {
        const int nrhs = maxr + 1;
        auto rhs = make_rhs<T>(n, nrhs, batch, 1212u);
        auto Bv = view_of(rhs);
        for (Transpose op : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
            reset_rhs(rhs);
            // Auto: the fused driver throws past its width, so a run that returns took another tier.
            UnifiedVector<std::byte> ws(std::max<std::size_t>(
                1, getrs_buffer_size<B, T>(*this->ctx, A, Bv, op)));
            ASSERT_NO_THROW(((void)getrs<B, T>(*this->ctx, A, Bv, op, p.piv.to_span(), ws.to_span())));
            this->ctx->wait();
            for (int b = 0; b < batch; ++b)
                EXPECT_VERIFY(T, Check::solve, n, solve_residual<T>(p.a0.data() + size_t(b) * p.stride,
                                            rhs.buf.data() + size_t(b) * rhs.stride,
                                            rhs.b0.data() + size_t(b) * rhs.stride,
                                            n, nrhs, p.ld, rhs.ld, op))
                    << "one width past the fused tier the facade returned a wrong answer, "
                       "transA=" << int(op) << " b=" << b;
        }
    }

    // ---- ceiling 2: the device's resident-RHS capacity ---------------------
    {
        const std::size_t cap = sycl_getrs::getrs_fused_max_rhs_elems<T>(this->budget());
        ASSERT_GT(cap, std::size_t(maxr));
        const int nrhs = maxr;
        // The largest order that still fits, and the first that does not.
        const int fit  = int(cap / std::size_t(nrhs));
        const int over = fit + 1;
        UnifiedVector<int64_t> piv(size_t(over) * 2, int64_t(1));
        for (int b = 0; b < 2; ++b) {
            int* ip = reinterpret_cast<int*>(piv.data()) + size_t(b) * over;
            for (int k = 0; k < over; ++k) ip[k] = k + 1;
        }
        for (int order : {fit, over}) {
            MatrixView<T, MatrixFormat::Dense> An(nullptr, order, order, order,
                                                  int64_t(order) * order, 2, nullptr);
            MatrixView<T, MatrixFormat::Dense> Bn(nullptr, order, nrhs, order,
                                                  int64_t(order) * nrhs, 2, nullptr);
            EXPECT_EQ(this->getrs_pin_accepted(cta, An, Bn, Transpose::NoTrans), order == fit)
                << "can_run(cta) at n=" << order << " nrhs=" << nrhs << " against a capacity of " << cap;
            if (order == over) {
                UnifiedVector<std::byte> ws(1);
                EXPECT_THROW((void)sycl_getrs::getrs_fused_dispatch<T>(
                                 *this->ctx, An, Bn, Transpose::NoTrans, piv.to_span(),
                                 ws.to_span()),
                             std::invalid_argument)
                    << "the direct entry point accepted n*nrhs = "
                    << std::size_t(order) * std::size_t(nrhs) << " against a capacity of " << cap;
            }
        }
    }
}

// F4. THE DROP-IN CONTRACT FOR THE FUSED TIER, BOTH DIRECTIONS AND BOTH PRODUCERS.
// The fused tier reads the pivot buffer DIRECTLY -- pivots.as_span<int>(), packed
// 1-BASED int32, an INTERCHANGE LIST -- and re-derives the walk in its own kernel
// rather than delegating to the shared laswp, so it is the arm most exposed to a
// format disagreement.
TYPED_TEST(LuTest, FusedGetrsConsumesEveryFactorProducer) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::BackendType;
    const int n = std::min(40, this->cta_max_n()), nrhs = 3, batch = 3;
    ASSERT_GE(n, 32);

    auto solve_and_check = [&](Lu<T>& p, const char* who) {
        auto A = view_of(p);
        for (Transpose op : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
            auto rhs = make_rhs<T>(n, nrhs, batch, 2424u + unsigned(int(op)));
            auto Bv = view_of(rhs);
            UnifiedVector<std::byte> ws(std::max<std::size_t>(
                1, sycl_getrs::getrs_fused_buffer_size<T>(*this->ctx, A, Bv, op)));
            ASSERT_NO_THROW((void)sycl_getrs::getrs_fused_dispatch<T>(
                *this->ctx, A, Bv, op, p.piv.to_span(), ws.to_span()));
            this->ctx->wait();
            for (int b = 0; b < batch; ++b)
                EXPECT_VERIFY(T, Check::solve, n, solve_residual<T>(p.a0.data() + size_t(b) * p.stride,
                                            rhs.buf.data() + size_t(b) * rhs.stride,
                                            rhs.b0.data() + size_t(b) * rhs.stride,
                                            n, nrhs, p.ld, rhs.ld, op))
                    << "the FUSED getrs could not consume the " << who << " factor, transA="
                    << int(op) << " b=" << b;
            check_rhs_pad_intact(rhs, who);
        }
    };

    {   // native getrf, BLOCKED tier
        auto p = make_dominant_permuted<T>(n, batch, 3535u);
        this->run_blocked(p);
        check_factor(p, "dropin/fused/native-blocked");
        if (this->HasFailure()) return;
        solve_and_check(p, "NATIVE BLOCKED getrf's");
    }
    {   // native getrf, CTA tier
        auto p = make_dominant_permuted<T>(n, batch, 3636u);
        this->run_cta(p);
        check_factor(p, "dropin/fused/native-cta");
        if (this->HasFailure()) return;
        solve_and_check(p, "NATIVE CTA getrf's");
    }
    if constexpr (batchlas::select::factorization_vendor_available<B>) {
        // VENDOR getrf. Its pivot CHOICE differs from ours for complex types, which is why
        // the oracle is a residual against the ORIGINAL A and not a factor comparison.
        auto p = make_dominant_permuted<T>(n, batch, 3737u);
        {
            auto A = view_of(p);
            UnifiedVector<std::byte> ws(std::max<std::size_t>(
                1, backend::getrf_vendor_buffer_size<B, T>(*this->ctx, A)));
            ASSERT_NO_THROW(((void)backend::getrf_vendor<B, T>(*this->ctx, A, p.piv.to_span(),
                                                         ws.to_span(), p.info.to_span())));
            this->ctx->wait();
            for (int b = 0; b < batch; ++b) ASSERT_EQ(p.info[b], 0);
            check_factor(p, "dropin/fused/vendor-factor", /*check_L=*/false);
            if (this->HasFailure()) return;
        }
        solve_and_check(p, "VENDOR getrf's");

        // AND THE OTHER DIRECTION, on the same factor: the vendor getrs must still consume
        // it, which is what makes the pivot FORMAT a shared fact and not our convention.
        auto A = view_of(p);
        auto rhs = make_rhs<T>(n, nrhs, batch, 2626u);
        auto Bv = view_of(rhs);
        UnifiedVector<std::byte> ws(std::max<std::size_t>(
            1, backend::getrs_vendor_buffer_size<B, T>(*this->ctx, A, Bv, Transpose::NoTrans)));
        ASSERT_NO_THROW(((void)backend::getrs_vendor<B, T>(*this->ctx, A, Bv, Transpose::NoTrans,
                                                     p.piv.to_span(), ws.to_span())));
        this->ctx->wait();
        for (int b = 0; b < batch; ++b)
            EXPECT_VERIFY(T, Check::solve, n, solve_residual<T>(p.a0.data() + size_t(b) * p.stride,
                                        rhs.buf.data() + size_t(b) * rhs.stride,
                                        rhs.b0.data() + size_t(b) * rhs.stride,
                                        n, nrhs, p.ld, rhs.ld, Transpose::NoTrans))
                << "the VENDOR getrs could not consume the factor the fused tier just read";
    }
}

// F5. A SINGULAR AND A NEARLY-SINGULAR FACTOR. getrs has no info output and no
// singularity contract: ?GETRS divides by U(k,k) unconditionally. So NEARLY
// singular must still be SOLVED to a backward-error bound, and EXACTLY singular
// must PROPAGATE: a finite answer means an EPSILON FLOOR or a SKIPPED division
// returned a plausible-looking wrong number. k = 0 and k = n-1 are where an
// off-by-one in the reverse loop lands.
TYPED_TEST(LuTest, FusedGetrsOnSingularAndNearlySingularFactors) {
    using T = typename TestFixture::T;
    const int n = 64, nrhs = 2, batch = 2;
    const double tiny = std::is_same_v<RealOf<T>, float> ? 1e-6 : 1e-12;

    for (int kz : {0, n / 2, n - 1}) {
        // ---- nearly singular ------------------------------------------------
        {
            auto p = make_dominant_permuted<T>(n, batch, 4747u + unsigned(kz));
            this->run_blocked(p);
            if (this->HasFailure()) return;
            for (int b = 0; b < batch; ++b) {
                T& d = p.buf[size_t(b) * p.stride + size_t(kz) * p.ld + kz];
                d = scale(d, tiny);
            }
            auto A = view_of(p);
            for (Transpose op : {Transpose::NoTrans, Transpose::Trans}) {
                auto rhs = make_rhs<T>(n, nrhs, batch, 4848u + unsigned(kz) + unsigned(int(op)));
                auto Bv = view_of(rhs);
                UnifiedVector<std::byte> ws(std::max<std::size_t>(
                    1, sycl_getrs::getrs_fused_buffer_size<T>(*this->ctx, A, Bv, op)));
                ASSERT_NO_THROW((void)sycl_getrs::getrs_fused_dispatch<T>(
                    *this->ctx, A, Bv, op, p.piv.to_span(), ws.to_span()));
                this->ctx->wait();
                for (int b = 0; b < batch; ++b) {
                    for (int c = 0; c < nrhs; ++c)
                        for (int i = 0; i < n; ++i)
                            ASSERT_TRUE(verify::finite(up(
                                rhs.buf[size_t(b) * rhs.stride + size_t(c) * rhs.ld + i])))
                                << "a NEARLY singular factor produced a non-finite answer at kz="
                                << kz << " transA=" << int(op) << " b=" << b;
                    const double res = lu_solve_residual<T>(
                        p.buf.data() + size_t(b) * p.stride,
                        reinterpret_cast<const int*>(p.piv.data()) + size_t(b) * n,
                        rhs.buf.data() + size_t(b) * rhs.stride,
                        rhs.b0.data() + size_t(b) * rhs.stride, n, nrhs, p.ld, rhs.ld, op);
                    EXPECT_VERIFY(T, Check::solve, n, res)
                        << "a NEARLY singular factor was not solved to a backward-error bound "
                           "at kz=" << kz << " transA=" << int(op) << " b=" << b
                        << " -- an epsilon floor or a skipped division would look like this";
                }
                if (this->HasFailure()) return;
            }
        }
        // ---- exactly singular -----------------------------------------------
        {
            auto p = make_dominant_permuted<T>(n, batch, 4949u + unsigned(kz));
            this->run_blocked(p);
            if (this->HasFailure()) return;
            for (int b = 0; b < batch; ++b)
                p.buf[size_t(b) * p.stride + size_t(kz) * p.ld + kz] = make<T>(0.0, 0.0);
            auto A = view_of(p);
            for (Transpose op : {Transpose::NoTrans, Transpose::Trans}) {
                auto rhs = make_rhs<T>(n, nrhs, batch, 5050u + unsigned(kz) + unsigned(int(op)));
                auto Bv = view_of(rhs);
                UnifiedVector<std::byte> ws(std::max<std::size_t>(
                    1, sycl_getrs::getrs_fused_buffer_size<T>(*this->ctx, A, Bv, op)));
                ASSERT_NO_THROW((void)sycl_getrs::getrs_fused_dispatch<T>(
                    *this->ctx, A, Bv, op, p.piv.to_span(), ws.to_span()))
                    << "an exactly singular factor must not make the launch throw or hang";
                this->ctx->wait();
                for (int b = 0; b < batch; ++b) {
                    bool nonfinite = false;
                    for (int c = 0; c < nrhs && !nonfinite; ++c)
                        for (int i = 0; i < n && !nonfinite; ++i)
                            if (!verify::finite(up(rhs.buf[size_t(b) * rhs.stride +
                                                    size_t(c) * rhs.ld + i])))
                                nonfinite = true;
                    EXPECT_TRUE(nonfinite)
                        << "a factor with U(" << kz << "," << kz << ") == 0 produced an entirely "
                           "FINITE answer at transA=" << int(op) << " b=" << b
                        << " -- the division by the zero pivot was floored or skipped, which is "
                           "the silently-plausible-wrong-answer failure mode";
                }
                check_rhs_pad_intact(rhs, "fused/singular");
                if (this->HasFailure()) return;
            }
        }
    }
}

// F6. THE FACADE REACHES THE FUSED KERNEL, ASSERTED BIT-EXACTLY. The comparison is
// BIT-EXACT against the direct entry point, which no vendor and no other native tier
// can reproduce. evidence: docs/perf/lu.md#getrs-fused-window-evidence
TYPED_TEST(LuTest, FacadeReachesTheFusedGetrsBitExactly) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 64, nrhs = 3, batch = 3;

    // As above: this asserts which tier the facade picks by DEFAULT, so a pinned
    // route in the environment has to be cleared or it decides the answer.
    ScopedEnvVar clear_getrs("BATCHLAS_GETRS_ROUTE", "");
    ScopedEnvVar clear_getrf("BATCHLAS_GETRF_ROUTE", "");

    auto p = make_dominant_permuted<T>(n, batch, 6161u);
    this->run_blocked(p);
    check_factor(p, "facade/fused/factor");
    if (this->HasFailure()) return;
    auto A = view_of(p);

    for (Transpose op : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
        // A ScopedPin that can_run refuses throws, so the pin cannot silently mean another tier.
        const select::ScopedPin<ops::getrs::GetrsChoice> g("getrs", ops::getrs::Cta{});
        auto r1 = make_rhs<T>(n, nrhs, batch, 7171u);
        auto r2 = make_rhs<T>(n, nrhs, batch, 7171u);
        auto V1 = view_of(r1);
        auto V2 = view_of(r2);

        UnifiedVector<std::byte> w1(std::max<std::size_t>(
            1, sycl_getrs::getrs_fused_buffer_size<T>(*this->ctx, A, V1, op)));
        (void)sycl_getrs::getrs_fused_dispatch<T>(*this->ctx, A, V1, op, p.piv.to_span(), w1.to_span());
        this->ctx->wait();

        UnifiedVector<std::byte> w2(std::max<std::size_t>(
            1, getrs_buffer_size<B, T>(*this->ctx, A, V2, op)));
        ASSERT_NO_THROW(((void)getrs<B, T>(*this->ctx, A, V2, op, p.piv.to_span(), w2.to_span())));
        this->ctx->wait();

        for (size_t i = 0; i < r1.buf.size(); ++i)
            ASSERT_EQ(verify::abs(up(r1.buf[i]) - up(r2.buf[i])), 0.0)
                << "transA=" << int(op) << ": the facade's getrs differs from the FUSED direct "
                   "entry point at element " << i << " -- something else served this call";
    }

    // Which tier Auto picks on either side of the old window is getrs_candidates_tests'
    // AutoReadsTheTranscribedTable; the blocked pin's bit-exactness is its
    // PinnedRunIsTheDirectKernelBitForBit.
}

// F7. THE FUSED DIRECT ENTRY POINT REFUSES WHAT can_run REFUSES, AND ITS
// WORKSPACE QUERY DEREFERENCES NOTHING. The workspace is ZERO in every mode for
// this tier by design (the RHS is permuted and solved in LOCAL memory, in place),
// and the facade's figure is a max over BOTH native tiers; the query and the call
// resolve INDEPENDENTLY, so a lease sized for one tier is one the other overruns.
TYPED_TEST(LuTest, FusedGetrsDirectEntryPointRefusesWhatSupportsRefuses) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::BackendType;
    const int n = 24, nrhs = 2, batch = 2;
    auto p = make_dominant_permuted<T>(n, batch, 8181u);
    this->run_blocked(p);
    auto A = view_of(p);
    auto rhs = make_rhs<T>(n, nrhs, batch, 8282u);
    auto Bv = view_of(rhs);
    UnifiedVector<std::byte> ws(4096);

    // A non-square A.
    {
        UnifiedVector<T> w(size_t(24) * 32, make<T>(1.0, 0.0));
        UnifiedVector<T*> wp(1, nullptr);
        MatrixView<T, MatrixFormat::Dense> W(w.data(), 24, 32, 24, 24 * 32, 1, wp.data());
        EXPECT_THROW((void)sycl_getrs::getrs_fused_dispatch<T>(*this->ctx, W, Bv, Transpose::NoTrans,
                                                         p.piv.to_span(), ws.to_span()),
                     std::invalid_argument);
    }
    // B with the wrong number of rows.
    {
        auto mismatched = make_rhs<T>(n + 1, nrhs, batch, 8383u);
        auto Bm = view_of(mismatched);
        EXPECT_THROW((void)sycl_getrs::getrs_fused_dispatch<T>(*this->ctx, A, Bm, Transpose::NoTrans,
                                                         p.piv.to_span(), ws.to_span()),
                     std::invalid_argument);
    }
    // A and B disagreeing on the batch size.
    {
        auto other = make_rhs<T>(n, nrhs, batch + 1, 8484u);
        auto Bo = view_of(other);
        EXPECT_THROW((void)sycl_getrs::getrs_fused_dispatch<T>(*this->ctx, A, Bo, Transpose::NoTrans,
                                                         p.piv.to_span(), ws.to_span()),
                     std::invalid_argument);
    }
    // A pivot span shorter than n * batch.
    {
        UnifiedVector<int64_t> shortpiv(size_t(n) * batch - 1, 0);
        EXPECT_THROW((void)sycl_getrs::getrs_fused_dispatch<T>(*this->ctx, A, Bv, Transpose::NoTrans,
                                                         shortpiv.to_span(), ws.to_span()),
                     std::invalid_argument);
    }

    // ---- the workspace query ----------------------------------------------
    {
        // NULL data, exactly as a measuring pass presents it.
        MatrixView<T, MatrixFormat::Dense> nullA(nullptr, n, n, n + 4, (n + 4) * n + 3, batch,
                                                 nullptr);
        MatrixView<T, MatrixFormat::Dense> nullB(nullptr, n, nrhs, n + 2,
                                                 (n + 2) * nrhs + 5, batch, nullptr);
        for (Transpose op : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
            EXPECT_NO_THROW((void)sycl_getrs::getrs_fused_buffer_size<T>(*this->ctx, nullA,
                                                                        nullB, op));
            EXPECT_EQ(sycl_getrs::getrs_fused_buffer_size<T>(*this->ctx, nullA, nullB, op),
                      std::size_t(0))
                << "the fused tier claims a workspace; it has none, and the facade's max() over "
                   "the two native tiers is what would have to carry it";
            EXPECT_NO_THROW(((void)getrs_buffer_size<B, T>(*this->ctx, nullA, nullB, op)));
        }
    }
    // Serve EXACTLY the facade's figure under the CTA pin. A short workspace is a
    // silent heap overflow, not a throw.
    {
        const select::ScopedPin<ops::getrs::GetrsChoice> g("getrs", ops::getrs::Cta{});
        const std::size_t need = getrs_buffer_size<B, T>(*this->ctx, A, Bv, Transpose::NoTrans);
        EXPECT_GE(need, sycl_getrs::getrs_fused_buffer_size<T>(*this->ctx, A, Bv,
                                                               Transpose::NoTrans));
        UnifiedVector<std::byte> exact(std::max<std::size_t>(1, need));
        ASSERT_NO_THROW(((void)getrs<B, T>(*this->ctx, A, Bv, Transpose::NoTrans, p.piv.to_span(),
                                     exact.to_span())));
        this->ctx->wait();
        for (int b = 0; b < batch; ++b)
            EXPECT_VERIFY(T, Check::solve, n, solve_residual<T>(p.a0.data() + size_t(b) * p.stride,
                                        rhs.buf.data() + size_t(b) * rhs.stride,
                                        rhs.b0.data() + size_t(b) * rhs.stride,
                                        n, nrhs, p.ld, rhs.ld, Transpose::NoTrans));
    }
}

// ===========================================================================
// THE REGISTER-RESIDENT TIER (the `tiny` choice, src/extensions/getrf_tiny.cc).
// One matrix per SubGroupPartition<N>, N in {8, 16, 32}, row r in lane r's
// registers, no local memory and no barriers. These cases guard the three
// properties a functional test otherwise stays green through: a padded row must
// never win a pivot, the pad must never be written, and the partitions packed into
// one work-group must not alias. Every (x) below names a break that was applied,
// observed red and restored. evidence: docs/perf/lu.md#armed-breaks
// ===========================================================================

namespace {

// Is `idx` inside SOME batch item's logical n x n window? Everything else is the
// ld pad, the stride pad, or the gap past the last item -- memory the kernel must
// neither read nor write.
template <typename T>
bool tiny_in_window(const Lu<T>& p, size_t idx) {
    const size_t b = idx / size_t(p.stride);
    if (b >= size_t(p.batch)) return false;
    const size_t off = idx - b * size_t(p.stride);
    const size_t j = off / size_t(p.ld);
    const size_t i = off % size_t(p.ld);
    return j < size_t(p.n) && i < size_t(p.n);
}

// The pad carries alloc()'s large poison, so "unchanged" is a BITWISE claim about
// a value the kernel has no reason to reproduce by accident.
template <typename T>
void expect_pad_untouched(const Lu<T>& p, const char* what) {
    for (size_t i = 0; i < p.buf.size(); ++i) {
        if (tiny_in_window(p, i)) continue;
        ASSERT_EQ(verify::abs(up(p.buf[i]) - up(p.a0[i])), 0.0)
            << what << ": element " << i << " lies outside every item's n x n window "
            << "(n=" << p.n << ", ld=" << p.ld << ", stride=" << p.stride
            << ") and was written -- an off-by-one in the padded store";
    }
}

// A tie in cabs1 that the ORDERING of the argmax must resolve, and which no
// residual and no pivot-growth bound can see: both candidates are equally good
// pivots, so only the INDEX distinguishes LAPACK's answer from the other one.
// Row 0 and row 2 of column 0 carry equal cabs1 and different values.
template <class T>
void tiny_tie_pair(T& a, T& b) {
    if constexpr (std::is_same_v<RealOf<T>, T>) {
        a = make<T>(3.0, 0.0);
        b = make<T>(-3.0, 0.0);      // cabs1 3 == 3
    } else {
        a = make<T>(3.0, 1.0);
        b = make<T>(1.0, -3.0);      // cabs1 4 == 4
    }
}

template <typename T>
Lu<T> make_cabs1_tie_in_column0(int n, int batch, unsigned seed) {
    Lu<T> p;
    alloc(p, n, batch, 5, 11);
    Rng rg(seed);
    T tie_a, tie_b;
    tiny_tie_pair(tie_a, tie_b);
    for (int b = 0; b < batch; ++b) {
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                p.buf[size_t(b) * p.stride + size_t(j) * p.ld + i] =
                    scale(make<T>(rg.next(), rg.next()), 0.2);
        // Columns 1.. are strongly dominant, so the tie is the only decision that
        // is not forced by magnitude and the factor stays well conditioned.
        for (int j = 1; j < n; ++j)
            p.buf[size_t(b) * p.stride + size_t(j) * p.ld + j] =
                make<T>(4.0 * double(n) * (1.0 + 0.01 * double(b)), 0.0);
        p.buf[size_t(b) * p.stride + 0] = tie_a;                 // row 0
        p.buf[size_t(b) * p.stride + 2] = tie_b;                 // row 2
    }
    p.a0.assign(p.buf.begin(), p.buf.end());
    poison(p);
    return p;
}


// LAPACK's ?GETF2 on the host in promoted arithmetic, recording at each step the
// MARGIN by which the winner beat the runner-up: margin[k] = cabs1(runner-up) /
// cabs1(winner), in [0, 1]. An elementwise pivot comparison is well posed only where
// the argmax is decided by more than rounding -- the device contracts a - b*c into an
// FMA and the host does not -- so every caller MUST stop at the first step whose
// margin is within kTinyAmbiguous of 1: past a divergence the two sides are
// factorising different matrices.
// evidence: docs/perf/lu.md#the-pivot-margin-gate-on-elementwise-comparisons
inline constexpr double kTinyAmbiguous = 0.99;

template <typename T>
void host_getf2_with_margins(int n, std::vector<T>& a, std::vector<int>& piv,
                             std::vector<double>& margin) {
    using P = decltype(up(T{}));
    piv.assign(size_t(n), 0);
    margin.assign(size_t(n), 0.0);
    std::vector<P> A(size_t(n) * size_t(n));
    for (size_t i = 0; i < A.size(); ++i) A[i] = up(a[i]);
    for (int k = 0; k < n; ++k) {
        double best = -1.0, second = -1.0;
        int win = k;
        for (int i = k; i < n; ++i) {
            const double m = verify::cabs1(A[size_t(k) * size_t(n) + size_t(i)]);
            // STRICTLY greater, so an exact tie keeps the LOWEST row -- I?AMAX's order
            // and the tier's. A `>=` here would silently make the oracle disagree with
            // LAPACK on exactly the case TinyBreaksAnExactCabs1Tie guards.
            if (m > best) { second = best; best = m; win = i; }
            else if (m > second) { second = m; }
        }
        piv[size_t(k)] = win + 1;
        margin[size_t(k)] = (best > 0.0 && second > 0.0) ? (second / best) : 0.0;
        if (win != k)
            for (int c = 0; c < n; ++c)
                std::swap(A[size_t(c) * size_t(n) + size_t(k)],
                          A[size_t(c) * size_t(n) + size_t(win)]);
        const P p = A[size_t(k) * size_t(n) + size_t(k)];
        if (verify::cabs1(p) == 0.0) continue;                 // MAGMA update = 0: carry on
        for (int i = k + 1; i < n; ++i) A[size_t(k) * size_t(n) + size_t(i)] /= p;
        for (int c = k + 1; c < n; ++c) {
            const P u = A[size_t(c) * size_t(n) + size_t(k)];
            for (int i = k + 1; i < n; ++i)
                A[size_t(c) * size_t(n) + size_t(i)] -=
                    A[size_t(k) * size_t(n) + size_t(i)] * u;
        }
    }
}

// ||P A - L U||_F / ||A||_F for a CONTIGUOUS n x n host factor and its interchange
// list. Used to decide whether the host LAPACKE on THIS machine may be trusted as an
// oracle for a given cell -- see the note in TinyPivotsMatchLapackeOnUnstructuredData.
template <typename T>
double host_factor_residual(int n, const std::vector<T>& a0, const std::vector<T>& f,
                            const std::int32_t* ip) {
    return factor_residual<T>(a0.data(), f.data(), ip, n, n, n);
}

}  // namespace

// T1. RESIDUAL AND ELEMENTWISE PIVOTS, EVERY ORDER THE TIER HOLDS.
//
// n = 1..ceiling covers all three buckets and every padded order inside them;
// ld = n exercises the unpadded leading dimension and ld = n + 5 the padded one;
// batch 19 is past a full work-group at EVERY bucket, so the partial last
// work-group -- the clamp-don't-return tail -- runs in every case. Shrink any of
// those three and the tail stops being exercised.
// Arms break (a) (`act = (rowid > j)` weakened to `>=`) and break (d)
// (tiny_partition_id replaced by part.get_group_linear_id(), which drops the `sg_id *`
// term and makes the sub-groups of one work-group alias the same matrices).
// evidence: docs/perf/lu.md#armed-breaks
TYPED_TEST(LuTest, TinyFactorisesAndPivotsExactlyAtEveryOrder) {
    using T = typename TestFixture::T;
    const int cap = this->tiny_max_n();
    ASSERT_GE(cap, 8) << "the tiny tier reports no capacity for this type";

    for (int n = 1; n <= cap; ++n) {
        for (int ld_pad : {0, 5}) {
            for (int batch : {1, 19}) {
                auto dom = make_dominant_permuted<T>(n, batch, 700u + unsigned(n), ld_pad, 11);
                // ANTI-VACUITY: at n = 1 the permutation is the identity and nothing
                // moves, which is a fact about the fixture, not a lost guard.
                if (n > 1) {
                    ASSERT_GT(non_diagonal_pivots(dom, 0), 0)
                        << "n=" << n << ": the fixture pivots nowhere, so the pivot "
                                       "assertions below are vacuous";
                }
                this->run_tiny(dom);
                check_factor(dom, "tiny/dominant-permuted");
                for (int b = 0; b < batch; ++b)
                    EXPECT_EQ(dom.info[b], 0)
                        << "tiny: a nonsingular item reported info = " << dom.info[b]
                        << " at n=" << n << " ld_pad=" << ld_pad << " b=" << b;
                if (this->HasFailure()) return;

                auto rnd = make_random<T>(n, batch, 900u + unsigned(n), ld_pad, 11);
                this->run_tiny(rnd);
                check_factor(rnd, "tiny/random");
                if (this->HasFailure()) return;
            }
        }
    }
}

// T2. THE PADDING IS INERT: an order inside a wider bucket must factorise as the
// unpadded CTA tier does. The CTA route is the oracle because it carries no
// compile-time N at all, so it cannot share a padding defect.
//
// The PIVOT SEQUENCE is compared EXACTLY -- integer bookkeeping, where a padding
// defect shows first. The ELEMENTS are not: the two bodies are separate translation
// units, so one may contract a multiply-subtract the other does not. That bound must
// stay RELATIVE to the reference element. An ABSOLUTE bound scaled by the fixture's
// 4n diagonal licenses thousands of ulp on exactly the O(1) off-diagonal entries
// where a padding defect shows up.
// evidence: docs/perf/lu.md#why-the-tiny-vs-cta-element-bound-is-relative
// Ulp per elimination step the two tiers may licitly part by: one contracted FMA is
// one ulp of the value it produces, so n steps is n ulp, and the four is margin for a
// toolchain that contracts differently in the two TUs.
constexpr double kTinyVsCtaUlpsPerStep = 4.0;

TYPED_TEST(LuTest, TinyPaddingIsInertAgainstTheCtaRoute) {
    using T = typename TestFixture::T;
    const int cap = this->tiny_max_n();
    const int batch = 5;

    for (int n = 1; n <= cap; ++n) {
        if (this->cta_max_n() < n) continue;
        auto a = make_dominant_permuted<T>(n, batch, 4242u + unsigned(n));
        auto b = make_dominant_permuted<T>(n, batch, 4242u + unsigned(n));
        this->run_tiny(a);
        this->run_cta(b);

        for (int it = 0; it < batch; ++it) {
            for (int k = 0; k < n; ++k)
                ASSERT_EQ(piv_item(a, it)[k], piv_item(b, it)[k])
                    << "n=" << n << " item " << it << ": tiny and cta disagree on pivot "
                    << k << " -- the identity pad changed the selection";
            EXPECT_EQ(a.info[it], b.info[it]) << "n=" << n << " item " << it << ": info differs";
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) {
                    const auto av = up(a.buf[size_t(it) * a.stride + size_t(j) * a.ld + i]);
                    const auto bv = up(b.buf[size_t(it) * b.stride + size_t(j) * b.ld + i]);
                    const double d = verify::abs(av - bv);
                    // Relative to the element itself, floored at 1 so an entry that
                    // underflows towards zero does not demand exact agreement.
                    const double scale = std::max(1.0, verify::abs(bv));
                    const double tol = kTinyVsCtaUlpsPerStep * double(n) * (2.0 * verify::eps<T>()) * scale;
                    ASSERT_LE(d, tol)
                        << "n=" << n << " item " << it << " element (" << i << "," << j
                        << "): tiny and cta differ by " << d << " (" << d / ((2.0 * verify::eps<T>()) * scale)
                        << " ulp of " << verify::abs(bv) << "), tolerance " << tol;
                }
        }
        if (this->HasFailure()) return;
    }
}

// T3. THE PAD IS NEVER WRITTEN. The ld = n + 5 shape is load-bearing: at ld = n the
// pad columns of item b land inside item b+1 and the residual catches the break
// instead, so only a PADDED ld leaves this assertion as the sole guard.
// Arms break (b), dropping `if (k >= n) continue` from the store loop.
// evidence: docs/perf/lu.md#armed-breaks
TYPED_TEST(LuTest, TinyNeverWritesThePad) {
    using T = typename TestFixture::T;
    const int cap = this->tiny_max_n();
    for (int n : {1, 5, 8, 9, 13, 16, 17, 24, 32}) {
        if (n > cap) continue;
        auto p = make_dominant_permuted<T>(n, 7, 55u + unsigned(n));
        this->run_tiny(p);
        expect_pad_untouched(p, "tiny/pad");
        check_factor(p, "tiny/pad-fixture");
        if (this->HasFailure()) return;
    }
}

// T4. A PLANTED ZERO COLUMN: info is 1-based and global, the first failure wins,
// the elimination continues FINITELY past it (MAGMA's `update = 0` semantics), and
// the neighbouring items are untouched.
//
// This case does NOT guard the `rowid < n` candidate mask -- break (e) removed it and
// stayed green, because the argmax tie-break already elects the LOWEST rowid and
// rowid == j is always live. What guards that property is the tie-break DIRECTION, in
// TinyBreaksAnExactCabs1TieTowardsTheLowestRow. This case's live assertions are info
// and the ipiv RANGE, which breaks (d) and (f) do turn red.
// evidence: docs/perf/lu.md#pad-rows-and-the-argmax-corrected
TYPED_TEST(LuTest, TinyPlantedZeroColumnGivesGlobalOneBasedInfo) {
    using T = typename TestFixture::T;
    const int cap = this->tiny_max_n();
    const int batch = 6;

    for (int n : {2, 5, 8, 13, 16, 24, 32}) {
        if (n > cap) continue;
        for (int k = 0; k < n; k += (n > 4 ? (n / 2 + 1) : 1)) {
            auto p = make_dominant_permuted<T>(n, batch, 606u + unsigned(n));
            const int bad = 3;                       // NOT item 0: offset 0 hides a stride bug
            for (int i = 0; i < n; ++i)
                p.buf[size_t(bad) * p.stride + size_t(k) * p.ld + i] = make<T>(0.0, 0.0);
            p.a0.assign(p.buf.begin(), p.buf.end());
            poison(p);
            this->run_tiny(p);

            EXPECT_EQ(p.info[bad], k + 1)
                << "n=" << n << " k=" << k << ": a zero column must give info = k + 1, "
                                              "1-based and global";
            for (int b = 0; b < batch; ++b) {
                if (b != bad)
                    EXPECT_EQ(p.info[b], 0) << "item " << b << " was flagged by item " << bad;
                const int* ip = piv_item(p, b);
                for (int c = 0; c < n; ++c)
                    ASSERT_TRUE(ip[c] >= c + 1 && ip[c] <= n)
                        << "n=" << n << " k=" << k << " b=" << b << ": ipiv[" << c << "] = "
                        << ip[c] << " is outside [c+1, n] -- a padded row won the argmax";
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < n; ++i)
                        ASSERT_TRUE(verify::finite(up(p.buf[size_t(b) * p.stride + size_t(j) * p.ld + i])))
                            << "n=" << n << " k=" << k << " b=" << b << ": F(" << i << "," << j
                            << ") is not finite -- the elimination did not continue finitely";
            }
            if (this->HasFailure()) return;
        }
    }
}

// T5. PACKED LAUNCHES DO NOT BLEED. Everything outside the logical windows -- the
// ld pad, the stride pad, and the gap past the last item -- is NaN, so any read
// past a row or a matrix propagates into the factor and the finiteness assertion
// in check_factor catches it. The batch is deliberately 3 past a full work-group.
TYPED_TEST(LuTest, TinyPackedLaunchesDoNotBleed) {
    using T = typename TestFixture::T;
    const int cap = this->tiny_max_n();
    const double qnan = std::numeric_limits<double>::quiet_NaN();

    for (int n : {3, 8, 11, 16, 21, 32}) {
        if (n > cap) continue;
        auto p = make_dominant_permuted<T>(n, 19, 771u + unsigned(n));
        for (size_t i = 0; i < p.buf.size(); ++i)
            if (!tiny_in_window(p, i)) p.buf[i] = make<T>(qnan, qnan);
        p.a0.assign(p.buf.begin(), p.buf.end());
        // poison() restores a0, which now carries the NaN pad, and re-poisons ipiv.
        poison(p);
        this->run_tiny(p);
        check_factor(p, "tiny/nan-pad");
        for (int b = 0; b < 19; ++b)
            EXPECT_EQ(p.info[b], 0) << "n=" << n << " b=" << b << ": NaN outside the window "
                                                                 "reached the pivot test";
        if (this->HasFailure()) return;
    }
}

// T5b. THE LOCAL-MEMORY PIVOT ROW UNDER A SATURATED LAUNCH, at one order per column
// bucket of both lane buckets. T1's batch of 19 is one or two work-groups; 4096 keeps
// every SM full, which is what a race in the parity double buffer, or a row buffer
// shared by two partitions, needs before it turns into a wrong factor.
// evidence: docs/perf/lu.md#the-local-memory-broadcast
// ARMED BREAK (R9): index the row buffer without `pidx`. EXPECTED: RED here and in T1,
// T3-T6, float and cfloat.
// ARMED BREAK (R9): single-buffer it, `(j & 1)` -> 0. EXPECTED: GREEN everywhere -- a
// BLIND guard on this device: the parity is kept because the memory model needs it,
// not because any failure was seen without it.
// ARMED BREAK (R9): drop the `group_barrier(sg)`. EXPECTED: RED here and in T1, T3-T6,
// float and cfloat, and T11.
TYPED_TEST(LuTest, TinyLocalMemoryRowSurvivesASaturatedLaunch) {
    using T = typename TestFixture::T;
    const int cap = this->tiny_max_n();
    for (int n : {9, 12, 16, 17, 22, 27, 32}) {
        if (n > cap) continue;
        auto p = make_random<T>(n, 4096, 8800u + unsigned(n));
        this->run_tiny(p);
        check_factor(p, "tiny/saturated");
        if (this->HasFailure()) return;
    }
}

// T6. A NaN INSIDE A LIVE COLUMN MUST NOT DECIDE THE PIVOT. The tiny argmax seeds
// every lane from its OWN magnitude, so an unmapped NaN survives every XOR round
// (`ov > NaN` and `ov == NaN` are both false) and the lanes end the butterfly
// disagreeing about the winning LANE: different broadcast sources, a silently wrong
// factor, no crash. The CTA tier cannot share the defect -- it seeds at R(-1) and
// updates through `v > bv` -- which is why it is the oracle here; no host reference in
// this file models "what the butterfly did once its lanes stopped agreeing".
// Arms break (f), removing the `mag == mag` map to the losing sentinel.
// evidence: docs/perf/lu.md#armed-breaks
TYPED_TEST(LuTest, TinyArgmaxIgnoresANaNCandidate) {
    using T = typename TestFixture::T;
    const int cap = this->tiny_max_n();
    const double qnan = std::numeric_limits<double>::quiet_NaN();
    const int batch = 5;

    for (int n : {4, 8, 16, 32}) {
        if (n > cap || this->cta_max_n() < n) continue;
        auto a = make_random<T>(n, batch, 313u + unsigned(n));
        auto b = make_random<T>(n, batch, 313u + unsigned(n));
        // A NON-PIVOT row of column 0 in every item: row n-1 of a random matrix is
        // the argmax with probability 1/n, so the assertion below keeps this honest.
        for (int it = 0; it < batch; ++it) {
            a.buf[size_t(it) * a.stride + size_t(n - 1)] = make<T>(qnan, qnan);
            b.buf[size_t(it) * b.stride + size_t(n - 1)] = make<T>(qnan, qnan);
        }
        a.a0.assign(a.buf.begin(), a.buf.end());
        b.a0.assign(b.buf.begin(), b.buf.end());
        poison(a);
        poison(b);
        this->run_tiny(a);
        this->run_cta(b);

        for (int it = 0; it < batch; ++it) {
            ASSERT_NE(piv_item(b, it)[0], n)
                << "n=" << n << " item " << it << ": the CTA oracle chose the NaN row "
                                                  "itself, so this case tests nothing";
            for (int k = 0; k < n; ++k)
                ASSERT_EQ(piv_item(a, it)[k], piv_item(b, it)[k])
                    << "n=" << n << " item " << it << ": tiny and cta disagree on pivot " << k
                    << " with a NaN in a candidate row -- the butterfly seed was not "
                       "NaN-mapped";
        }
        if (this->HasFailure()) return;
    }
}

// T7. AN EXACT cabs1 TIE GOES TO THE LOWEST ROW, WHICH IS I?AMAX's ORDER.
//
// No residual and no pivot-growth bound can see this: both candidates are equally good
// pivots, and only the index distinguishes LAPACK's answer from the other one.
// make_dominant_permuted cannot produce an exact tie at all, so this fixture is the
// only one that reaches the property -- and the tie-break direction it pins is ALSO
// what keeps a pad row (highest rowid in the partition) from winning an all-zero
// column, which is the guard T4 turned out not to be.
// Arms break (c), `ok < key` -> `ok > key` in tiny_argmax_pair.
// evidence: docs/perf/lu.md#armed-breaks
TYPED_TEST(LuTest, TinyBreaksAnExactCabs1TieTowardsTheLowestRow) {
    using T = typename TestFixture::T;
    const int cap = this->tiny_max_n();
    const int batch = 4;

    for (int n : {4, 12, 20, 32}) {
        if (n > cap) continue;
        auto p = make_cabs1_tie_in_column0<T>(n, batch, 8080u + unsigned(n));
        // ANTI-VACUITY, on the DATA: the two candidates must carry exactly equal
        // cabs1, must be strictly the column maximum, and must be different values.
        for (int b = 0; b < batch; ++b) {
            const T* A0 = p.a0.data() + size_t(b) * p.stride;
            const double c0 = verify::cabs1(up(A0[0]));
            const double c2 = verify::cabs1(up(A0[2]));
            ASSERT_EQ(c0, c2) << "the fixture no longer carries an exact cabs1 tie";
            ASSERT_GT(verify::abs(up(A0[0]) - up(A0[2])), 0.0)
                << "the tied entries are equal, so no tie-break can be observed";
            for (int i = 0; i < n; ++i)
                if (i != 0 && i != 2)
                    ASSERT_LT(verify::cabs1(up(A0[i])), c0)
                        << "row " << i << " outranks the tie, so column 0 is not decided by it";
        }
        this->run_tiny(p);
        for (int b = 0; b < batch; ++b)
            EXPECT_EQ(piv_item(p, b)[0], 1)
                << "n=" << n << " b=" << b << ": an exact cabs1 tie between rows 0 and 2 "
                   "must go to row 0 (ipiv 1), which is I?AMAX's order";
        check_factor(p, "tiny/cabs1-tie", /*check_L=*/true);
        if (this->HasFailure()) return;
    }
}

// T8. THE DIRECT ENTRY POINT REFUSES WHAT can_run REFUSES. The entry point is
// reachable without the selector and nothing else would catch a missing gate, so
// each gate is re-applied inside the dispatch and throws.
TYPED_TEST(LuTest, TinyDirectEntryPointRefusesWhatSupportsRefuses) {
    using T = typename TestFixture::T;
    const int cap = this->tiny_max_n();
    const int n = std::min(8, cap), batch = 2;
    auto p = make_dominant_permuted<T>(n, batch, 17u);
    auto A = view_of(p);
    UnifiedVector<std::byte> ws(4096);

    // An order past the tier's ceiling: `unsupported`, naming the ceiling, and
    // never a silent factorisation of the leading cap x cap submatrix.
    {
        auto big = make_dominant_permuted<T>(cap + 1, batch, 18u);
        auto Vb = view_of(big);
        EXPECT_THROW((void)sycl_getrf::getrf_tiny_dispatch<T>(*this->ctx, Vb, big.piv.to_span(),
                                                        ws.to_span(), Span<int32_t>{}),
                     batchlas::unsupported);
    }
    // A non-square view.
    {
        UnifiedVector<T> w(size_t(8) * 16, make<T>(1.0, 0.0));
        UnifiedVector<T*> wp(1, nullptr);
        MatrixView<T, MatrixFormat::Dense> W(w.data(), 8, 16, 8, 8 * 16, 1, wp.data());
        UnifiedVector<int64_t> pv(64, 0);
        EXPECT_THROW((void)sycl_getrf::getrf_tiny_dispatch<T>(*this->ctx, W, pv.to_span(),
                                                        ws.to_span(), Span<int32_t>{}),
                     std::invalid_argument);
    }
    // A pivot span shorter than n * batch.
    {
        UnifiedVector<int64_t> shortpiv(size_t(n) * batch - 1, 0);
        EXPECT_THROW((void)sycl_getrf::getrf_tiny_dispatch<T>(*this->ctx, A, shortpiv.to_span(),
                                                        ws.to_span(), Span<int32_t>{}),
                     std::invalid_argument);
    }
    // A SHORT info span is "not requested", not an error: the tier draws scratch.
    {
        auto q = make_dominant_permuted<T>(n, batch, 19u);
        EXPECT_NO_THROW(this->run_tiny(q, /*pass_info=*/false));
        check_factor(q, "tiny/empty-info");
    }
}

// T10. THE FACADE REACHES THE TINY KERNEL, ASSERTED BIT-EXACTLY. A route assertion
// plus a residual can stay green while every number comes from the vendor, so the
// comparison is bit-exact against the direct entry point -- factor AND pivots --
// which no vendor can reproduce (cuBLAS pivots on the modulus, this tier on cabs1).
// getrf_buffer_size is exercised on the same shape: without a Tiny arm there, a
// shape only this tier supports throws out of the sizing query.
TYPED_TEST(LuTest, FacadeReachesTheTinyKernelBitExactly) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::BackendType;
    const int n = std::min(16, this->tiny_max_n()), batch = 19;

    ScopedEnvVar g("BATCHLAS_GETRF_ROUTE", "tiny");
    auto direct = make_dominant_permuted<T>(n, batch, 5150u);
    auto viafac = make_dominant_permuted<T>(n, batch, 5150u);

    auto Vf = view_of(viafac);  // a 'tiny' pin that cannot run throws rather than reach the vendor

    this->run_tiny(direct);

    const std::size_t need = getrf_buffer_size<B, T>(*this->ctx, Vf);
    EXPECT_EQ(need, sycl_getrf::getrf_tiny_buffer_size<T>(*this->ctx, Vf))
        << "getrf_buffer_size is not exactly the Tiny tier's info scratch (R5)";
    UnifiedVector<std::byte> ws(std::max<std::size_t>(1, need));
    ASSERT_NO_THROW(((void)getrf<B, T>(*this->ctx, Vf, viafac.piv.to_span(), ws.to_span(),
                                 viafac.info.to_span())));
    this->ctx->wait();

    for (size_t i = 0; i < direct.buf.size(); ++i)
        ASSERT_EQ(verify::abs(up(direct.buf[i]) - up(viafac.buf[i])), 0.0)
            << "the facade's factor differs from getrf_tiny_dispatch's at element " << i
            << " -- something else served this call";
    for (int b = 0; b < batch; ++b)
        for (int k = 0; k < n; ++k)
            ASSERT_EQ(piv_item(direct, b)[k], piv_item(viafac, b)[k])
                << "pivot " << k << " of item " << b << " differs";
    check_factor(viafac, "facade/tiny");
}

// T12. ELEMENTWISE PIVOTS AGAINST AN INDEPENDENT HOST ORACLE, ON UNSTRUCTURED DATA.
//
// The only case in this file whose oracle shares NO line of code with the kernel.
// `expect_piv` is the permutation make_dominant_permuted built, and T2/T6 compare
// against a tier that #includes the SAME getrf_cta_device.hh and calls the SAME
// lu_cabs1 -- so a defect in the shared pivot METRIC moves both tiers together and
// leaves all of those green. make_random is deliberate: on a dominant-permuted matrix
// the argmax is decided by a gap of orders of magnitude, so a merely WRONG metric
// still picks the right row.
//
// TWO ORACLES, because LAPACKE cannot be trusted unconditionally on this box: its
// dgetrf is wrong for every n >= 10, with no BatchLAS code involved. The BLAS-free
// triple loop is the primary oracle and supplies the argmax MARGIN; LAPACKE is
// cross-checked against it ONLY on cells where LAPACKE passes a residual check on its
// OWN factor. The dropped-cell count is asserted rather than silently tolerated -- a
// silent fallback would let the LAPACKE arm quietly stop testing anything.
// Arms break (h), lu_cabs1 -> the modulus in the SHARED getrf_cta_device.hh.
// evidence: docs/perf/lu.md#the-host-dgetrf-oracle-is-broken-on-this-box
//           docs/perf/lu.md#the-pivot-margin-gate-on-elementwise-comparisons
TYPED_TEST(LuTest, TinyPivotsMatchLapackeOnUnstructuredData) {
#if !BATCHLAS_VERIFY_HAVE_LAPACKE
    GTEST_SKIP() << "no host LAPACKE reference in this build";
#else
    using T = typename TestFixture::T;
    const int cap = this->tiny_max_n();
    const int batch = 7;                      // a partial last work-group at every bucket
    // Generous: this only has to separate "a correct factorisation" from the 0.4-1.1
    // this host's dgetrf returns above n = 9.
    const double kLapackeTrustTol = 1e-3;

    std::vector<T> h, f;
    std::vector<std::int32_t> hp;
    std::vector<int> ref;
    std::vector<double> margin;
    int asserted_nondiagonal = 0, lapacke_cells = 0, lapacke_dropped = 0;
    for (int n = 1; n <= cap; ++n) {
        auto p = make_random<T>(n, batch, 6100u + unsigned(n));
        this->run_tiny(p);

        for (int b = 0; b < batch; ++b) {
            // The n x n window out of the PADDED, STRIDED original: a0, not buf,
            // because the factorisation overwrote buf in place.
            h.assign(size_t(n) * size_t(n), make<T>(0.0, 0.0));
            for (int c = 0; c < n; ++c)
                std::memcpy(h.data() + size_t(c) * size_t(n),
                            p.a0.data() + size_t(b) * p.stride + size_t(c) * p.ld,
                            size_t(n) * sizeof(T));
            host_getf2_with_margins<T>(n, h, ref, margin);

            f = h;
            ASSERT_TRUE(verify::getrf_pivots(n, n, f, hp)) << "n=" << n << " b=" << b
                                                           << ": the host reference itself failed";
            const double lres = host_factor_residual<T>(n, h, f, hp.data());
            const bool trust = std::isfinite(lres) && lres <= kLapackeTrustTol;
            ++lapacke_cells;
            if (!trust) ++lapacke_dropped;

            const int* ip = piv_item(p, b);
            for (int k = 0; k < n; ++k) {
                // Above this margin the two implementations may legitimately disagree
                // and every step after would compare different matrices.
                if (margin[size_t(k)] > kTinyAmbiguous) break;
                ASSERT_EQ(ip[k], ref[size_t(k)])
                    << "n=" << n << " b=" << b << ": ipiv[" << k << "] = " << ip[k]
                    << " where the host reference chose " << ref[size_t(k)]
                    << " at a runner-up/winner margin of " << margin[size_t(k)]
                    << " -- the pivot METRIC or its ordering disagrees with LAPACK's, "
                       "and the CTA oracle cannot see it because both tiers share "
                       "lu_cabs1";
                if (trust) {
                    ASSERT_EQ(ref[size_t(k)], static_cast<int>(hp[size_t(k)]))
                        << "n=" << n << " b=" << b << " step " << k
                        << ": the in-file host getf2 disagrees with LAPACKE at a margin "
                           "of " << margin[size_t(k)] << ", and LAPACKE's own factor "
                           "residual was " << lres << " -- one of the two ORACLES is "
                           "wrong, not the kernel";
                }
                if (ref[size_t(k)] != k + 1) ++asserted_nondiagonal;
            }
            EXPECT_EQ(p.info[b], 0) << "n=" << n << " b=" << b << ": info = " << p.info[b];
        }
        check_factor(p, "tiny/host-oracle");
        if (this->HasFailure()) return;
    }
    // ANTI-VACUITY, ACROSS THE WHOLE SWEEP: if every asserted step were the diagonal
    // one, the comparison would be satisfied by a kernel that returns the identity
    // interchange list and never pivots at all.
    EXPECT_GT(asserted_nondiagonal, 0)
        << "every unambiguous step the oracle chose was the diagonal, so this case "
           "asserts nothing about pivoting";
    // AND THE LAPACKE ARM MUST NOT HAVE SILENTLY STOPPED TESTING. If it were dropped
    // everywhere the case would degrade to "the kernel agrees with a triple loop in
    // this same file", which is a weaker claim than the one the name makes.
    EXPECT_LT(lapacke_dropped, lapacke_cells)
        << "LAPACKE failed its own residual check on ALL " << lapacke_cells
        << " cells, so nothing was compared against the reference implementation";
    if (lapacke_dropped > 0) {
        GTEST_LOG_(INFO) << "LAPACKE was untrustworthy on " << lapacke_dropped << " of "
                         << lapacke_cells << " cells (its own factor residual exceeded "
                         << kLapackeTrustTol
                         << "); those cells were compared against the in-file host "
                            "reference only. See "
                            "docs/perf/lu.md#the-host-dgetrf-oracle-is-broken-on-this-box";
    }
#endif
}

// ---------------------------------------------------------------------------
// T11. A SOURCE CHECK, because both properties are invisible to a functional test on
// this device. A WORK-GROUP barrier in a partition kernel is a RACE, not a crash: the
// G partitions sharing a work-group would synchronise with each other and the answer
// stays right until the scheduler makes it wrong. The tier's one barrier is the
// SUB-GROUP barrier of the local-memory pivot row, spelled exactly `group_barrier(sg)`
// (`group_barrier(part)` is a no-op off the native partition path). Its local memory
// is a `local_accessor` the launch accounting sees; a GROUP COLLECTIVE allocates static
// shared it cannot see, and can push a NEIGHBOURING kernel into the (47104, 49664]
// launch hole, whose cap the CUDA adapter raises STICKILY per CUfunction.
// evidence: docs/perf/lu.md#the-local-memory-broadcast
//
// The scan strips `//` comments first, so the prose above is not what is matched.
// ARMED BREAK (R9): spell the pivot-row barrier `group_barrier(it.get_group())`.
// EXPECTED: RED, naming that line, with every numerical case GREEN.
// ---------------------------------------------------------------------------
#ifdef BATCHLAS_GETRF_TINY_CC_PATH
namespace {

// The first line of `path` whose CODE (comments stripped) contains `token`, or "".
std::string FirstCodeLineContaining(const std::string& path, const char* token) {
    std::FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return std::string("<could not open ") + path + ">";
    char line[8192];
    std::string found;
    while (std::fgets(line, sizeof(line), f)) {
        const std::string text(line);
        const size_t comment = text.find("//");
        const std::string code = comment == std::string::npos ? text : text.substr(0, comment);
        if (code.find(token) != std::string::npos) { found = text; break; }
    }
    std::fclose(f);
    return found;
}

}  // namespace

TEST(GetrfTinySource, OnlySubGroupBarriersAndNoGroupCollective) {
    const std::string path = BATCHLAS_GETRF_TINY_CC_PATH;
    // POSITIVE CONTROL FIRST. Every assertion below is an ABSENCE, so a path that
    // resolves to the wrong file -- or to no file at all -- reports a clean bill of
    // health. The scan must first be shown to find something that IS in the tier's
    // source before its silence means anything.
    ASSERT_FALSE(FirstCodeLineContaining(path, "parallel_for").empty())
        << "the source scan found no parallel_for in " << path
        << ": BATCHLAS_GETRF_TINY_CC_PATH does not name the tiny tier's source, and "
           "every absence assertion below would pass vacuously";
    ASSERT_FALSE(FirstCodeLineContaining(path, "group_barrier(sg)").empty())
        << "the scan found no sub-group barrier in " << path
        << ": the local-memory pivot row is gone, or this is the wrong file";

    std::FILE* f = std::fopen(path.c_str(), "r");
    ASSERT_NE(f, nullptr) << path;
    char line[8192];
    while (std::fgets(line, sizeof(line), f)) {
        const std::string text(line);
        const size_t comment = text.find("//");
        const std::string code = comment == std::string::npos ? text : text.substr(0, comment);
        if (code.find("group_barrier(") != std::string::npos) {
            EXPECT_NE(code.find("group_barrier(sg)"), std::string::npos)
                << "a barrier of the wrong scope in the partition kernel: " << text;
        }
        for (const char* token : {"_over_group(", "group_broadcast(", "joint_"}) {
            EXPECT_EQ(code.find(token), std::string::npos)
                << "a group collective allocates static shared the launch accounting "
                   "cannot see: " << text;
        }
    }
    std::fclose(f);
}
#endif  // BATCHLAS_GETRF_TINY_CC_PATH

// The break record for every guarded property, including the breaks that turned
// nothing red: docs/perf/lu.md#blind-guards-and-what-made-them-blind

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
