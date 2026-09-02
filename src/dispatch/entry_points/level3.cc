// The public level-3 entry points, defined once, outside every vendor TU.
//
// WHY THIS FILE EXISTS -- the actual obstacle to vendor independence
//
// `gemm<Backend::CUDA, float>` was DEFINED at src/backends/cublas.cc:1568 and
// explicitly instantiated in the same file. So "build without cuBLAS" did not
// mean "lose the cuBLAS gemm path"; it meant "lose `batchlas::gemm` entirely",
// because the only definition of the public template lived in the file that was
// dropped. The same held in rocblas.cc and netlib_lapack.cc.
//
// That is a *definition ownership* problem, and no amount of enum, CMake or
// routing work addresses it -- which is why S1-S4 could reach a configuration
// that CONFIGURES with `BATCHLAS_HAS_CUDA_BACKEND 1` and `CUBLAS 0` but cannot
// LINK. Moving the definitions here is what makes the public API independent of
// which vendor libraries were compiled in.
//
// What each op looks like now:
//
//     vendor TU   defines and instantiates  backend::<op>_vendor<B, T>
//     this file   defines and instantiates  <op><B, T>, which calls it
//
// The instantiation guards below are per LIBRARY, not per device family, so a
// backend appears here exactly when the TU that defines its vendor entry point
// is compiled. Today every `<op>` still forwards straight to the vendor; S6
// puts the route resolution in between, at which point a backend can be
// instantiated with no vendor at all.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/gemv.hh>
#include <batchlas/blas/functions/trsm.hh>

#include "../../backends/trsm_route.hh"
// WP7: the native GEMV arm. Same shape as the trsm include above -- the
// adapter reaches only public headers plus src/sycl/gemv_native.hh, so the
// vendor-free facade can include it.
#include "../../backends/gemv_route.hh"
#include "../../sycl/gemv_native.hh"
#include <batchlas/blas/functions/symm.hh>
#include <batchlas/blas/functions/hemm.hh>
#include <batchlas/blas/functions/herk.hh>
#include <batchlas/blas/functions/her2k.hh>
#include <batchlas/blas/functions/syrk.hh>
#include <batchlas/blas/functions/syr2k.hh>
#include <batchlas/blas/functions/trmm.hh>

#include <batchlas/blas/dispatch/no_route.hh>
#include <batchlas/blas/dispatch/vendor_available.hh>

// WP1 S5: the native GEMM arm. Both are vendor-free -- gemm_kernels.hh reaches
// only enums/matrix/queue, and gemm_variant.hh stopped including linalg-impl.hh
// (its sole CUDA tie) in the same step.
#include "../../backends/gemm_variant.hh"
#include "../../backends/gemm_heterogeneous.hh"
#include "../../sycl/gemm_kernels.hh"

// WP1 S6: the four level-3 custom-route gates live here now, not in cublas.cc.
//
// Their only callers used to be four sites inside cublas.cc, a TU compiled only
// when cuBLAS exists. So after S4 the tile kernels were compiled in every
// configuration and reachable in none -- linked everywhere, callable nowhere.
// The gate has to run before the vendor-available test, which means it has to
// run here.
//
// These headers are portable as of S3: the dispatchers carry no CUDA include,
// no CUDA symbol and no preprocessor. level3_coverage.hh is the same
// instrumentation the gate-declined path used in cublas.cc.
#include "../../backends/symm_custom_dispatch.hh"
#include "../../backends/syrk_custom_dispatch.hh"
#include "../../backends/syr2k_custom_dispatch.hh"
#include "../../backends/trmm_custom_dispatch.hh"
#include "../../backends/level3_coverage.hh"


#include "../../util/template-instantiations.hh"

#include <complex>

namespace batchlas {

template <Backend Back, typename T>
Event gemm(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& A,
           const MatrixView<T, MatrixFormat::Dense>& B,
           const MatrixView<T, MatrixFormat::Dense>& C,
           T alpha,
           T beta,
           Transpose transA,
           Transpose transB,
           ComputePrecision precision) {
    if constexpr (!dispatch::level3_vendor_available<Back>) {
        // WP1 S5. Without this arm the whole work package delivers nothing
        // vendor-free: S2 pointed the level-3 expansions at this entry point,
        // and this entry point threw. docs/design/vendor-free-status.md claimed gemm
        // "fails only on the shapes outside gemm_custom_problem_supported";
        // it threw on EVERY call, measured at 48 passed / 136 failed where all
        // 48 passes are pure route-resolution tests that never run a kernel.
        //
        // The register-tiled family was linked the whole time -- it lives in
        // the vendor-free batchlas_sycl component -- just unreachable. LINKED
        // is not REACHABLE, and that distinction is what the coverage table's
        // `linked` rows do and do not tell you.
        //
        // The routing is NOT duplicated here. backend::gemm_route is the same
        // adapter cublas.cc consults, over the same RouteTable<Op::gemm, T>;
        // `vendor_available = false` is the one input that differs, and
        // resolve_route already has a branch for it (route_resolve.hh: a
        // requested vendor that does not exist falls back to the ordinary
        // automatic choice rather than being honoured).
        // WP2 C2. A heterogeneous batch is handled BEFORE routing, and has to
        // be: no strided-batched call can serve members of differing shape, so
        // the question is not which kernel but how the batch is walked. The
        // loop, the m==0/n==0 skips and the k==0 -> scale(beta) substitution
        // are shared verbatim with cublas.cc (WP2 C1) rather than restated --
        // this codebase has twice paid for restating one behaviour twice.
        //
        // Recursion is not a hazard here, unlike the level-3 seam in WP1: each
        // batch_item() is HOMOGENEOUS by construction, so the inner call takes
        // the ordinary path below and cannot re-enter this branch.
        if (backend::gemm_has_heterogeneous_batch(A, B, C)) {
            return backend::detail::gemm_heterogeneous_loop<T>(
                ctx, A, B, C, beta, transA, transB,
                [&](const MatrixView<T, MatrixFormat::Dense>& a,
                    const MatrixView<T, MatrixFormat::Dense>& b,
                    const MatrixView<T, MatrixFormat::Dense>& c) {
                    return ::batchlas::gemm<Back, T>(ctx, a, b, c, alpha, beta,
                                                     transA, transB, precision);
                });
        }

        const auto route = backend::gemm_route<T>(ctx, A, B, C, transA, transB,
                                                  precision, /*vendor_available=*/false);
        if (dispatch::is_native(route)) {
            return sycl_gemm::gemm_custom<T>(ctx, A, B, C, alpha, beta, transA, transB, precision);
        }
        // Still honest about the gap: shapes the native kernel does not serve
        // (non-Default precision, degenerate dims) have no route at all without
        // a vendor, and say so by name.
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::gemm, Back, dispatch::kLevel3Library<Back>);
    } else {
        return backend::gemm_vendor<Back, T>(ctx, A, B, C, alpha, beta, transA, transB, precision);
    }
}

template <Backend Back, typename T>
Event gemv(Queue& ctx,
           const MatrixView<T,MatrixFormat::Dense>& A,
           const VectorView<T>& X,
           const VectorView<T>& Y,
           T alpha,
           T beta,
           Transpose transA) {
    // THE GATE RUNS BEFORE THE VENDOR-AVAILABLE TEST, for the reason recorded
    // at the top of this file and restated for trsm: anything below that test
    // is unreachable in the vendor-free build, which is the build WP7 exists
    // for. Putting the route resolution in the `else` branch would leave all
    // 40 vendor-free gemv_tests failures exactly where they are.
    //
    // NO VALIDATION CALL IS HOISTED HERE, unlike trsm. gemv has never had a
    // gemv_validate_params and WP7 deliberately does not add one: the native
    // kernel must accept exactly what the vendor accepts, and a new throw would
    // turn today's silent bugs into crashes in live paths -- ortho.cc:217-224's
    // transA=Trans branch builds A_i as (i x m) and passes a column of length
    // A.rows() as x, which is structurally wrong TODAY under the vendor. The
    // shape builder returns nullopt for it, so it keeps going to the vendor and
    // WP7 changes nothing about it. It is filed, not fixed.
    const dispatch::Route route = backend::gemv_route<Back, T>(
        ctx, A, X, Y, transA,
        /*vendor_available=*/dispatch::level3_vendor_available<Back>);

    if (dispatch::is_native(route)) {
        if (route.algo == dispatch::Algorithm::CTA) {
            return sycl_gemv::gemv_native_cta<T>(ctx, A, X, Y, alpha, beta, transA);
        }
        if (route.algo == dispatch::Algorithm::Direct) {
            return sycl_gemv::gemv_native_direct<T>(ctx, A, X, Y, alpha, beta, transA);
        }
    }

    if constexpr (!dispatch::level3_vendor_available<Back>) {
        // Still honest about the gap. What reaches here in a vendor-free build
        // is exactly what supports() refuses: a heterogeneous A, a negative
        // extent, an empty batch, or a set of views whose lengths do not
        // describe one gemv (the shape builder's nullopt, which resolves to
        // {Vendor, Auto}). Those have no route at all without a vendor, and
        // they say so by name rather than dying downstream.
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::gemv, Back, dispatch::kLevel3Library<Back>);
    } else {
        return backend::gemv_vendor<Back, T>(ctx, A, X, Y, alpha, beta, transA);
    }
}

template <Backend Back, typename T>
Event trsm(Queue& ctx,
           const MatrixView<T,MatrixFormat::Dense>& A,
           const MatrixView<T,MatrixFormat::Dense>& B,
           T alpha,
           Side side,
           Uplo uplo,
           Transpose transA,
           Diag diag) {
    // VALIDATION FIRST, and hoisted to here on purpose. The facade validated
    // nothing before, so cublas.cc:1104 and rocblas.cc:148 each called
    // trsm_validate_params themselves and netlib_lapack.cc never did at all.
    // One call here covers every backend and fixes netlib's long-missing check;
    // the two backend calls become harmless duplicates of a throw-only test.
    // It must precede the shape builder, which reads A.rows()/B.rows()/B.cols()
    // and would otherwise index a non-conforming shape.
    trsm_validate_params(A, B, side, uplo, transA, diag);

    // THE GATE RUNS BEFORE THE VENDOR-AVAILABLE TEST, for the reason recorded
    // at the top of this file: anything below that test is unreachable in the
    // vendor-free build, which is the build WP3 exists for.
    const dispatch::Route route = backend::trsm_route<T>(
        ctx, A, B, side, uplo, transA, diag,
        /*vendor_available=*/dispatch::level3_vendor_available<Back>);

    // All four scalar types now have a native kernel. The guard that used to
    // stand here excluded complex, which had no kernel to link against.
    {
        if (dispatch::is_native(route)) {
            if (route.algo == dispatch::Algorithm::CTA) {
                return sycl_trsm::trsm_native_v1_dispatch<T>(
                    ctx, A, B, alpha, side, uplo, transA, diag);
            }
            if (route.algo == dispatch::Algorithm::Blocked) {
                // THE TRAILING UPDATE GOES THROUGH THE ROUTER, not straight to
                // the native kernel. V2 used to call sycl_gemm::gemm_custom
                // itself, which bypasses RouteTable<Op::gemm> -- so the blocked
                // driver always got the native GEMM even on the shapes WP2 had
                // already measured it losing.
                //
                // It loses them badly, because of a property of the shapes trsm
                // issues that a square-matrix GEMM benchmark never sees: every
                // operand is a SUB-VIEW carrying its parent's leading dimension.
                // Measured on the six shapes V2 issues at order 512 (float,
                // q=1024, batch=512), with those real leading dimensions:
                //
                //   outer  m=128 n=1024 k=128/256/384  native 8.05 ms  vendor 3.89 ms
                //   inner  m=32  n=1024 k=32/64/96     native 7.89 ms  vendor 3.98 ms
                //
                // The same shapes with ld == rows are near parity (0.86-0.98x on
                // the inner three). The native GEMM collapses on a strided ld
                // and cuBLAS does not, and strided is the ONLY case trsm issues.
                //
                // Injection rather than an include: the kernel TU stays free of
                // the dispatch layer, tests keep calling it directly and get the
                // native GEMM, and a VENDOR-FREE build is unaffected because
                // resolve_route falls back to the native GEMM there anyway
                // (route_resolve.hh:60-63). The signatures of gemm_custom and
                // this gemm are identical, so nothing adapts.
                return sycl_trsm::trsm_native_blocked<T>(
                    ctx, A, B, alpha, side, uplo, transA, diag,
                    [](Queue& c,
                       const MatrixView<T, MatrixFormat::Dense>& ga,
                       const MatrixView<T, MatrixFormat::Dense>& gb,
                       const MatrixView<T, MatrixFormat::Dense>& gc,
                       T galpha, T gbeta, Transpose gta, Transpose gtb,
                       ComputePrecision gp) {
                        return gemm<Back, T>(c, ga, gb, gc, galpha, gbeta,
                                             gta, gtb, gp);
                    });
            }
        }
    }

    if constexpr (!dispatch::level3_vendor_available<Back>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::trsm, Back, dispatch::kLevel3Library<Back>);
    } else {
        return backend::trsm_vendor<Back, T>(ctx, A, B, side, uplo, transA, diag, alpha);
    }
}

template <Backend Back, RealScalar T>
Event symm(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& A,
           const MatrixView<T, MatrixFormat::Dense>& B,
           const MatrixView<T, MatrixFormat::Dense>& C,
           T alpha,
           T beta,
           Side side,
           Uplo uplo) {
    // WP1 S6. Relocated verbatim from cublas.cc -- same predicate, same
    // arguments, same order -- so on a vendor-present box this is a pure move.
    // What changes is a vendor-FREE build, where the gate is reachable at all
    // for the first time. Still guarded on CUDA and on float: relocating a
    // decision and widening it are different changes, and only the first is in
    // this step.
    if constexpr (Back == Backend::CUDA && std::is_same_v<T, float>) {
        if (backend::symm_use_cuda_custom(ctx, A, B, C, side, uplo)) {
            return backend::symm_cuda_custom(ctx, A, B, C, alpha, beta, side, uplo);
        }
        // GATE DECLINED -- the half a route diff needs most: a shape
        // moving OFF a native kernel onto the vendor shows up only here.
        backend::detail::record_level3_route(
            dispatch::Op::symm,
            dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto},
            C.rows(), C.cols(), A.rows(), A.batch_size(),
            backend::detail::kNativeUnknown,
            {uplo, side, Diag::NonUnit, Transpose::NoTrans});
    }

    if constexpr (!dispatch::level3_vendor_available<Back>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::symm, Back, dispatch::kLevel3Library<Back>);
    } else {
        return backend::symm_vendor<Back, T>(ctx, A, B, C, alpha, beta, side, uplo);
    }
}

template <Backend Back, ComplexScalar T>
Event hemm(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& A,
           const MatrixView<T, MatrixFormat::Dense>& B,
           const MatrixView<T, MatrixFormat::Dense>& C,
           T alpha,
           T beta,
           Side side,
           Uplo uplo) {
    if constexpr (!dispatch::level3_vendor_available<Back>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::hemm, Back, dispatch::kLevel3Library<Back>);
    } else {
        return backend::hemm_vendor<Back, T>(ctx, A, B, C, alpha, beta, side, uplo);
    }
}

template <Backend Back, ComplexScalar T>
Event herk(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& A,
           const MatrixView<T, MatrixFormat::Dense>& C,
           float_t<T> alpha,
           float_t<T> beta,
           Uplo uplo,
           Transpose transA) {
    if constexpr (!dispatch::level3_vendor_available<Back>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::herk, Back, dispatch::kLevel3Library<Back>);
    } else {
        return backend::herk_vendor<Back, T>(ctx, A, C, alpha, beta, uplo, transA);
    }
}

template <Backend Back, ComplexScalar T>
Event her2k(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            const MatrixView<T, MatrixFormat::Dense>& B,
            const MatrixView<T, MatrixFormat::Dense>& C,
            T alpha,
            float_t<T> beta,
            Uplo uplo,
            Transpose transA) {
    if constexpr (!dispatch::level3_vendor_available<Back>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::her2k, Back, dispatch::kLevel3Library<Back>);
    } else {
        return backend::her2k_vendor<Back, T>(ctx, A, B, C, alpha, beta, uplo, transA);
    }
}

template <Backend Back, RealScalar T>
Event syrk(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& A,
           const MatrixView<T, MatrixFormat::Dense>& C,
           T alpha,
           T beta,
           Uplo uplo,
           Transpose transA) {
    // WP1 S6. Relocated verbatim from cublas.cc -- same predicate, same
    // arguments, same order -- so on a vendor-present box this is a pure move.
    // What changes is a vendor-FREE build, where the gate is reachable at all
    // for the first time. Still guarded on CUDA and on float: relocating a
    // decision and widening it are different changes, and only the first is in
    // this step.
    if constexpr (Back == Backend::CUDA && std::is_same_v<T, float>) {
        if (backend::syrk_use_cuda_custom(ctx, A, C, uplo, transA)) {
            return backend::syrk_cuda_custom(ctx, A, C, alpha, beta, uplo, transA);
        }
        // GATE DECLINED -- the half a route diff needs most: a shape
        // moving OFF a native kernel onto the vendor shows up only here.
        backend::detail::record_level3_route(
            dispatch::Op::syrk,
            dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto},
            C.rows(), C.cols(),
            transA == Transpose::NoTrans ? A.cols() : A.rows(),
            A.batch_size(), backend::detail::kNativeUnknown,
            {uplo, Side::Left, Diag::NonUnit, transA});
    }

    if constexpr (!dispatch::level3_vendor_available<Back>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::syrk, Back, dispatch::kLevel3Library<Back>);
    } else {
        return backend::syrk_vendor<Back, T>(ctx, A, C, alpha, beta, uplo, transA);
    }
}

template <Backend Back, RealScalar T>
Event syr2k(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& A,
            const MatrixView<T, MatrixFormat::Dense>& B,
            const MatrixView<T, MatrixFormat::Dense>& C,
            T alpha,
            T beta,
            Uplo uplo,
            Transpose transA) {
    // WP1 S6. Relocated verbatim from cublas.cc -- same predicate, same
    // arguments, same order -- so on a vendor-present box this is a pure move.
    // What changes is a vendor-FREE build, where the gate is reachable at all
    // for the first time. Still guarded on CUDA and on float: relocating a
    // decision and widening it are different changes, and only the first is in
    // this step.
    if constexpr (Back == Backend::CUDA && std::is_same_v<T, float>) {
        if (backend::syr2k_use_cuda_custom(ctx, A, B, C, uplo, transA)) {
            return backend::syr2k_cuda_custom(ctx, A, B, C, alpha, beta, uplo, transA);
        }
        // GATE DECLINED -- the half a route diff needs most: a shape
        // moving OFF a native kernel onto the vendor shows up only here.
        backend::detail::record_level3_route(
            dispatch::Op::syr2k,
            dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto},
            C.rows(), C.cols(),
            transA == Transpose::NoTrans ? A.cols() : A.rows(),
            A.batch_size(), backend::detail::kNativeUnknown,
            {uplo, Side::Left, Diag::NonUnit, transA});
    }

    if constexpr (!dispatch::level3_vendor_available<Back>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::syr2k, Back, dispatch::kLevel3Library<Back>);
    } else {
        return backend::syr2k_vendor<Back, T>(ctx, A, B, C, alpha, beta, uplo, transA);
    }
}

template <Backend Back, typename T>
Event trmm(Queue& ctx,
           const MatrixView<T, MatrixFormat::Dense>& A,
           const MatrixView<T, MatrixFormat::Dense>& B,
           const MatrixView<T, MatrixFormat::Dense>& C,
           T alpha,
           Side side,
           Uplo uplo,
           Transpose transA,
           Diag diag) {
    // WP1 S6. Relocated verbatim from cublas.cc -- same predicate, same
    // arguments, same order -- so on a vendor-present box this is a pure move.
    // What changes is a vendor-FREE build, where the gate is reachable at all
    // for the first time. Still guarded on CUDA and on float: relocating a
    // decision and widening it are different changes, and only the first is in
    // this step.
    if constexpr (Back == Backend::CUDA && std::is_same_v<T, float>) {
        if (backend::trmm_use_cuda_custom(ctx, A, B, C, side, uplo, transA, diag)) {
            return backend::trmm_cuda_custom(ctx, A, B, C, alpha, side, uplo, transA, diag);
        }
        // GATE DECLINED -- the half a route diff needs most: a shape
        // moving OFF a native kernel onto the vendor shows up only here.
        backend::detail::record_level3_route(
            dispatch::Op::trmm,
            dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto},
            C.rows(), C.cols(), A.rows(), A.batch_size(),
            backend::detail::kNativeUnknown, {uplo, side, diag, transA});
    }

    if constexpr (!dispatch::level3_vendor_available<Back>) {
        dispatch::throw_no_vendor_route<T>(
            dispatch::Op::trmm, Back, dispatch::kLevel3Library<Back>);
    } else {
        return backend::trmm_vendor<Back, T>(ctx, A, B, C, alpha, side, uplo, transA, diag);
    }
}

// ---------------------------------------------------------------------------
// Explicit instantiations, one block per backend whose vendor TU is compiled.
// ---------------------------------------------------------------------------

#define OP_INSTANTIATE(OP, B_, fp) BATCHLAS_INSTANTIATE(sig::OP<fp>, OP, B_, fp)

// Real- and complex-only ops are separated because symm/syrk/syr2k are
// RealScalar-constrained and hemm/herk/her2k ComplexScalar-constrained.
#define REAL_ONLY_OPS(B_)             \
    OP_INSTANTIATE(symm,  B_, float)  \
    OP_INSTANTIATE(symm,  B_, double) \
    OP_INSTANTIATE(syrk,  B_, float)  \
    OP_INSTANTIATE(syrk,  B_, double) \
    OP_INSTANTIATE(syr2k, B_, float)  \
    OP_INSTANTIATE(syr2k, B_, double)

#define COMPLEX_ONLY_OPS(B_)                            \
    OP_INSTANTIATE(hemm,  B_, std::complex<float>)      \
    OP_INSTANTIATE(hemm,  B_, std::complex<double>)     \
    OP_INSTANTIATE(herk,  B_, std::complex<float>)      \
    OP_INSTANTIATE(herk,  B_, std::complex<double>)     \
    OP_INSTANTIATE(her2k, B_, std::complex<float>)      \
    OP_INSTANTIATE(her2k, B_, std::complex<double>)

#define ALL_TYPE_OPS_ONE(B_, fp)  \
    OP_INSTANTIATE(gemm, B_, fp)  \
    OP_INSTANTIATE(gemv, B_, fp)  \
    OP_INSTANTIATE(trsm, B_, fp)  \
    OP_INSTANTIATE(trmm, B_, fp)

#define LEVEL3_INSTANTIATE(B_)                       \
    ALL_TYPE_OPS_ONE(B_, float)                      \
    ALL_TYPE_OPS_ONE(B_, double)                     \
    ALL_TYPE_OPS_ONE(B_, std::complex<float>)        \
    ALL_TYPE_OPS_ONE(B_, std::complex<double>)       \
    REAL_ONLY_OPS(B_)                                \
    COMPLEX_ONLY_OPS(B_)

// Keyed on the DEVICE FAMILY, not on the vendor library. The bodies above
// compile to a throw when the library is absent, so the public entry point
// exists as a symbol in every build that has the device -- which is exactly what
// stopped being true when the definitions lived in the vendor TUs.
#if BATCHLAS_HAS_CUDA_BACKEND
LEVEL3_INSTANTIATE(Backend::CUDA)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
// rocBLAS has no hemm/herk/her2k/symm wrapper in rocblas.cc, so the ROCm
// backend instantiates only the ops it actually implements. That asymmetry
// predates S5 -- rocblas.cc's own instantiation block listed exactly these.
ALL_TYPE_OPS_ONE(Backend::ROCM, float)
ALL_TYPE_OPS_ONE(Backend::ROCM, double)
ALL_TYPE_OPS_ONE(Backend::ROCM, std::complex<float>)
ALL_TYPE_OPS_ONE(Backend::ROCM, std::complex<double>)
OP_INSTANTIATE(syrk,  Backend::ROCM, float)
OP_INSTANTIATE(syrk,  Backend::ROCM, double)
OP_INSTANTIATE(syr2k, Backend::ROCM, float)
OP_INSTANTIATE(syr2k, Backend::ROCM, double)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
LEVEL3_INSTANTIATE(Backend::NETLIB)
#endif

#undef LEVEL3_INSTANTIATE
#undef ALL_TYPE_OPS_ONE
#undef COMPLEX_ONLY_OPS
#undef REAL_ONLY_OPS
#undef OP_INSTANTIATE

}  // namespace batchlas
