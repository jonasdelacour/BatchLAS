#pragma once

// The POSV shape builder and route resolution; see gesv_route.hh for why this lives
// in src/ and what the include set may not gain.

#include <batchlas/blas/dispatch/route_env.hh>
#include <batchlas/blas/dispatch/route_posv.hh>
#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

#include "../extensions/getrs_native.hh"
#include "../extensions/solve_native.hh"

#include <optional>

namespace batchlas::backend {

// nullopt means "these views do not describe one POSV"; posv_validate_params rejects
// each of these before the builder is reached. Nothing here may dereference data_ptr().
template <Backend B, typename T>
inline std::optional<dispatch::PosvShape> posv_op_shape(
    const Queue& ctx,
    const MatrixView<T, MatrixFormat::Dense>& A,
    const MatrixView<T, MatrixFormat::Dense>& Bmat,
    Uplo uplo) {

    if (A.rows() != A.cols()) return std::nullopt;
    if (A.rows() < 0 || Bmat.rows() < 0 || Bmat.cols() < 0) return std::nullopt;
    if (A.rows() != Bmat.rows()) return std::nullopt;
    if (A.batch_size() != Bmat.batch_size()) return std::nullopt;

    dispatch::PosvShape s;
    s.op = dispatch::Op::posv;
    s.scalar = dispatch::scalar_kind_of<T>;
    s.backend = B;

    s.m = A.rows();
    s.n = Bmat.cols();
    s.k = A.rows();
    s.batch = A.batch_size();

    // A GENUINE ALGORITHM FORK, not just a coverage label: Lower runs L then L^H and
    // Upper runs U^H then U, with different trsm arguments on the composed arm. It is
    // also the only field that separates one posv coverage row from another.
    s.uplo = uplo;

    s.is_gpu = (ctx.device().type == DeviceType::GPU);
    s.has_sg32 = ctx.device().supports_sub_group_size(32);
    s.heterogeneous_batch = A.is_heterogeneous() || Bmat.is_heterogeneous();

    s.tiny_max_n = sycl_posv::posv_tiny_max_n<T>();
    s.tiny_max_nrhs = sycl_posv::kPosvTinyMaxRhs;

    // `potrf` then two routed `trsm` calls; all three are public entry points the
    // facade guarantees in every build with the device family.
    s.composed_available = true;

    // Device-queried, exactly as potrs_fused_dispatch re-checks it.
    const std::size_t local_mem = ctx.device().get_property(DeviceProperty::LOCAL_MEM_SIZE);
    const std::size_t budget = (local_mem > 4096) ? (local_mem - 4096) : 0;
    s.fused_max_rhs_elems = s.is_gpu ? static_cast<int64_t>(
                                           sycl_getrs::getrs_fused_max_rhs_elems<T>(budget))
                                     : 0;
    s.fused_max_nrhs = sycl_getrs::kGetrsFusedMaxRhs;

    return s;
}

template <Backend B, typename T>
inline dispatch::Route posv_route(const Queue& ctx,
                                  const MatrixView<T, MatrixFormat::Dense>& A,
                                  const MatrixView<T, MatrixFormat::Dense>& Bmat,
                                  Uplo uplo) {
    const auto shape = posv_op_shape<B, T>(ctx, A, Bmat, uplo);
    if (!shape) {
        return dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto};
    }
    const auto parsed = dispatch::parse_route_env(dispatch::Op::posv);
    const dispatch::Route forced =
        parsed.found ? parsed.route : dispatch::legacy_unset_default(dispatch::Op::posv);
    return dispatch::resolve_posv_route<T>(forced, *shape);
}

}  // namespace batchlas::backend
