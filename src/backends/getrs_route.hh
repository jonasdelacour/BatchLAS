#pragma once

// The GETRS shape builder and route resolution. The route table reads ONLY its arguments,
// so every getenv and device query happens here (route_resolve.hh).
// This header is included by the vendor-free facade: it must NOT gain src/queue.hh or
// <sycl/sycl.hpp> (gemm_variant.hh).

#include <batchlas/blas/dispatch/route_env.hh>
#include <batchlas/blas/dispatch/route_getrs.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include "../extensions/getrs_native.hh"

#include <optional>

namespace batchlas::backend {

// nullopt means "these views do not describe one GETRS" (OpShape holds ONE shape); a
// caller with no shape takes the vendor. The three structural agreements (A square,
// A.rows() == B.rows(), equal batch) deliberately duplicate options.hh's checks, so the
// builder never describes a non-conforming pair (the potrf_route.hh rule).
// NOTHING BELOW MAY DEREFERENCE data_ptr(): this runs in sizing paths.
template <Backend B, typename T>
inline std::optional<dispatch::GetrsShape> getrs_op_shape(
    const Queue& ctx,
    const MatrixView<T, MatrixFormat::Dense>& A,
    const MatrixView<T, MatrixFormat::Dense>& Bmat,
    Transpose transA) {

    if (A.rows() != A.cols()) return std::nullopt;
    if (A.rows() < 0 || Bmat.rows() < 0 || Bmat.cols() < 0) return std::nullopt;
    if (A.rows() != Bmat.rows()) return std::nullopt;
    if (A.batch_size() != Bmat.batch_size()) return std::nullopt;

    dispatch::GetrsShape s;
    s.op = dispatch::Op::getrs;
    s.scalar = dispatch::scalar_kind_of<T>;

    // SET: resolve_route copies it into the coverage table; unset, every row reads AUTO.
    s.backend = B;

    // FIELD MAPPING -- getrs's own. m is the ORDER of the factored matrix, n is
    // nrhs, k is the order again so max_dim()/min_dim() behave sensibly.
    s.m = A.rows();
    s.n = Bmat.cols();
    s.k = A.rows();
    s.batch = A.batch_size();

    // The only field that separates LU coverage rows (coverage.cc's variant_key); dropping
    // it is silent. Also a real algorithm fork: NoTrans applies P first, Trans/ConjTrans
    // applies P^T last, on the output, in reverse (route_getrs.hh).
    s.transA = transA;

    s.is_gpu = (ctx.device().type == DeviceType::GPU);

    // ENUMERATED, not `max_sub_group >= 32`. See GetrfShape::has_sg32 for why
    // MAX_SUB_GROUP_SIZE is wrong in both directions.
    s.has_sg32 = ctx.device().supports_sub_group_size(32);

    // THE GATE AND ITS WRITER LAND TOGETHER (potrf_route.hh). Both views
    // are asked, because either one being heterogeneous breaks the single-tuple
    // launch -- and OpShape has one flag, so the honest reduction is OR.
    s.heterogeneous_batch = A.is_heterogeneous() || Bmat.is_heterogeneous();

    // A capability, TRUE for all four types: supports() admits the native arm everywhere;
    // whether a vendor-present build takes it is preferred()'s question.
    s.blocked_available = sycl_getrs::getrs_blocked_available<T>();

    // The fused tier's capacity: local memory is ASKED OF THE DEVICE, minus the library-wide
    // 4096 B reserve, and the formula lives beside the launcher (getrs_fused.cc) so the
    // advertised ceiling and the allocation cannot disagree. BOTH ZERO when the kernel is
    // absent, which makes CTA unsupported rather than selectable-but-unimplemented.
    // evidence: docs/perf/lu.md#getrs-fused-window-evidence
    if (sycl_getrs::getrs_fused_available<T>()) {
        const std::size_t local_mem = ctx.device().get_property(DeviceProperty::LOCAL_MEM_SIZE);
        const std::size_t budget = (local_mem > 4096) ? (local_mem - 4096) : 0;
        s.fused_max_elems =
            static_cast<int64_t>(sycl_getrs::getrs_fused_max_rhs_elems<T>(budget));
        s.fused_max_nrhs = sycl_getrs::kGetrsFusedMaxRhs;
    }
    return s;
}

// Resolve a route for one call. THE ENV READ IS HERE AND ONLY HERE: BATCHLAS_GETRS_ROUTE,
// with no legacy spelling (adding a legacy_variable_for case would INVENT one).
// Called from exactly two places, getrs and getrs_buffer_size, with the same arguments,
// so sizing and running reach the same route by construction (two getenv reads per call).
template <Backend B, typename T>
inline dispatch::Route getrs_route(
    const Queue& ctx,
    const MatrixView<T, MatrixFormat::Dense>& A,
    const MatrixView<T, MatrixFormat::Dense>& Bmat,
    Transpose transA,
    bool vendor_available) {

    const auto shape = getrs_op_shape<B, T>(ctx, A, Bmat, transA);
    if (!shape) {
        return dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto};
    }
    const auto parsed = dispatch::parse_route_env(dispatch::Op::getrs);
    const dispatch::Route forced =
        parsed.found ? parsed.route : dispatch::legacy_unset_default(dispatch::Op::getrs);
    return dispatch::resolve_getrs_route<T>(forced, *shape, vendor_available);
}

} // namespace batchlas::backend
