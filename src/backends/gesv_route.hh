#pragma once

// The GESV shape builder; its include set must stay free of src/queue.hh and
// <sycl/sycl.hpp>, which the vendor-free facade could not follow.

#include <batchlas/blas/dispatch/route_env.hh>
#include <batchlas/blas/dispatch/route_gesv.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include "../extensions/solve_native.hh"

#include <optional>

namespace batchlas::backend {

// nullopt is a BUG here, not a routing decision: gesv_validate_params already rejected
// every disagreement below. NOTHING HERE MAY DEREFERENCE data_ptr(): a sizing path.
template <Backend B, typename T>
inline std::optional<dispatch::GesvShape> gesv_op_shape(
    const Queue& ctx,
    const MatrixView<T, MatrixFormat::Dense>& A,
    const MatrixView<T, MatrixFormat::Dense>& Bmat) {

    if (A.rows() != A.cols()) return std::nullopt;
    if (A.rows() < 0 || Bmat.rows() < 0 || Bmat.cols() < 0) return std::nullopt;
    if (A.rows() != Bmat.rows()) return std::nullopt;
    if (A.batch_size() != Bmat.batch_size()) return std::nullopt;

    dispatch::GesvShape s;
    s.op = dispatch::Op::gesv;
    s.scalar = dispatch::scalar_kind_of<T>;
    s.backend = B;   // SET, so the coverage rows read a backend and not Backend::AUTO

    s.m = A.rows();
    s.n = Bmat.cols();
    s.k = A.rows();
    s.batch = A.batch_size();

    s.is_gpu = (ctx.device().type == DeviceType::GPU);
    // ENUMERATED, not `max_sub_group >= 32`, which reports entry [0].
    s.has_sg32 = ctx.device().supports_sub_group_size(32);
    // Either view heterogeneous breaks the single-tuple launch; one flag, so OR.
    s.heterogeneous_batch = A.is_heterogeneous() || Bmat.is_heterogeneous();

    // BUILD facts asked of the kernel: the tier owns no local memory, so no budget walk.
    s.tiny_max_n = sycl_gesv::gesv_tiny_max_n<T>();
    s.tiny_max_nrhs = sycl_gesv::kGesvTinyMaxRhs;

    // Two ROUTED public calls, so this is the facade's guarantee, not a kernel's.
    s.composed_available = true;

    return s;
}

// CALLED FROM TWO PLACES WITH THE SAME ARGUMENTS, which is what makes gesv and its
// size query reach the same route by construction.
template <Backend B, typename T>
inline dispatch::Route gesv_route(const Queue& ctx,
                                  const MatrixView<T, MatrixFormat::Dense>& A,
                                  const MatrixView<T, MatrixFormat::Dense>& Bmat) {
    const auto shape = gesv_op_shape<B, T>(ctx, A, Bmat);
    if (!shape) {
        return dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto};
    }
    const auto parsed = dispatch::parse_route_env(dispatch::Op::gesv);
    const dispatch::Route forced =
        parsed.found ? parsed.route : dispatch::legacy_unset_default(dispatch::Op::gesv);
    return dispatch::resolve_gesv_route<T>(forced, *shape);
}

}  // namespace batchlas::backend
