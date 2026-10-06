// spmm (flat-kernel-selection.md §4.3, R1; docs/design/flat-kernel-selection.md#phase-5-spmm): select::run
// takes the first entry of the nearest tuned/spmm.<dtype>.<device>.txt row that can_run() admits. Direct
// is the native batched CSR driver (gather for transA == N, scale + atomic scatter otherwise); Vendor is
// cuSPARSE / rocSPARSE / netlib.
//
// Nothing on the selection path may read device memory: row_offsets(), col_indices() and the per-item nnz
// can be device-only, and spmm_buffer_size runs this same choose().

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/spmm.hh>

#include "choice.hh"
#include "../../select/select.hh"
#include "../../sycl/spmm_native.hh"
#include "../../util/template-instantiations.hh"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <variant>

namespace batchlas {
namespace ops::spmm {

using select::overloaded;

template <class T>
using Dense = MatrixView<T, MatrixFormat::Dense>;

// ConjTrans folds to T on both operands: the bodies differ from Trans only by a conjugation.
template <class T, MatrixFormat MF>
select::Key key_of(const MatrixView<T, MF>& A, const Dense<T>& C, Transpose transA, Transpose transB) {
    return {{"transA", transA == Transpose::NoTrans ? "N" : "T"},
            {"transB", transB == Transpose::NoTrans ? "N" : "T"},
            {"m", A.rows()}, {"nrhs", C.cols()}, {"batch", A.batch_size()}};
}

// Do the views describe one C = alpha op(A) op(B) + beta C? The old shape builder's checks
// (it sent anything else to the vendor, the only validation in the tree): extents agree, batch
// sizes agree, positive lds, and a CSR offset stride of at least m + 1 (the bodies read ro[i + 1]).
template <class T, MatrixFormat MF>
bool one_spmm(const MatrixView<T, MF>& A, const Dense<T>& Bm, const Dense<T>& C, Transpose transA,
              Transpose transB) {
    const std::int64_t m = A.rows(), ka = A.cols();
    if (m < 0 || ka < 0) return false;
    const bool an = transA == Transpose::NoTrans, bn = transB == Transpose::NoTrans;
    const std::int64_t opa_rows = an ? m : ka, opa_cols = an ? ka : m;
    const std::int64_t opb_rows = bn ? Bm.rows() : Bm.cols(), opb_cols = bn ? Bm.cols() : Bm.rows();
    if (opa_cols != opb_rows || C.rows() != opa_rows || C.cols() != opb_cols) return false;
    if (A.batch_size() != Bm.batch_size() || A.batch_size() != C.batch_size()) return false;
    if (Bm.ld() <= 0 || C.ld() <= 0) return false;
    if constexpr (MF == MatrixFormat::CSR) {
        if (A.offset_stride() < m + 1 || A.matrix_stride() < 0) return false;
    }
    return true;
}

// Correctness only (R3). Direct has no GPU gate on purpose: its bodies use no local memory or
// group collective, and the NETLIB (native_cpu) queue relies on it. One launch covers the batch
// with one (ld, stride) per dense operand, so neither may be heterogeneous; a CSR view varies per
// item only through nnz(b), which the bodies bound by row_offsets. m, k or nrhs 0 is a legal call
// (the driver quick-returns on the host); an empty batch is not. Two cuSPARSE terms, both off
// the old Auto path: a conjugated single-row B is an error status the vendor arm never checks (C
// unwritten, a silent wrong answer), and complex<double> N/N with one column segfaults on the host
// (known-defects #13).
// netlib's spmm throws `unsupported` per item for any transpose (netlib_lapack.cc), so Auto there
// falls to Direct; an empty batch never reaches the throw.
template <Backend B, class T, MatrixFormat MF>
bool can_run(const SpmmChoice& c, const select::Device& d, const MatrixView<T, MF>& A, const Dense<T>& Bm,
             const Dense<T>& C, Transpose transA, Transpose transB) {
    return std::visit(overloaded{
        [&](Direct) {
            if constexpr (MF != MatrixFormat::CSR) {
                return false;
            } else {
                const bool built = transA == Transpose::NoTrans ? sycl_spmm::spmm_gather_available<T>()
                                                                : sycl_spmm::spmm_scatter_available<T>();
                return built && one_spmm(A, Bm, C, transA, transB) && !Bm.is_heterogeneous() &&
                       !C.is_heterogeneous() && C.cols() >= 0 && A.batch_size() >= 1;
            }
        },
        [&](Vendor) {
            constexpr bool cx = !std::is_same_v<T, typename base_type<T>::type>;
            constexpr bool zz = std::is_same_v<T, std::complex<double>>;
            const bool nn = transA == Transpose::NoTrans && transB == Transpose::NoTrans;
            return d.has_vendor && !(B == Backend::NETLIB && !nn && A.batch_size() > 0) &&
                   !(B == Backend::CUDA && cx && transB == Transpose::ConjTrans && Bm.rows() == 1) &&
                   !(B == Backend::CUDA && zz && nn && C.cols() == 1);
        },
    }, c);
}

template <Backend B, class T, MatrixFormat MF>
Event launch(Queue& q, const SpmmChoice& c, const MatrixView<T, MF>& A, const Dense<T>& Bm, const Dense<T>& C,
             T alpha, T beta, Transpose transA, Transpose transB, Span<std::byte> ws) {
    return std::visit(overloaded{
        [&](Direct) -> Event {
            if constexpr (MF == MatrixFormat::CSR)
                return sycl_spmm::spmm_native_csr<T>(q, A, Bm, C, alpha, beta, transA, transB);
            else
                throw batchlas::internal_error("spmm: direct chosen for a non-CSR view");  // can_run refuses
        },
        [&](Vendor) -> Event {
            if constexpr (select::has_library<B>(spec.vendor))
                return backend::spmm_vendor<B, T, MF>(q, A, Bm, C, alpha, beta, transA, transB, ws);
            else select::no_vendor<B, T>(spec);
        },
    }, c);
}

// Exactly the chosen family's need (R5). Direct takes none. A Direct-routed call never asks the
// vendor sizer: it builds a plan that walks the CSR row offsets from the host.
template <Backend B, class T, MatrixFormat MF>
std::size_t workspace(Queue& q, const SpmmChoice& c, const MatrixView<T, MF>& A, const Dense<T>& Bm,
                      const Dense<T>& C, T alpha, T beta, Transpose transA, Transpose transB) {
    return std::visit(overloaded{
        [&](Direct) -> std::size_t { return 0; },
        [&](Vendor) -> std::size_t {
            if constexpr (select::has_library<B>(spec.vendor))
                return backend::spmm_vendor_buffer_size<B, T, MF>(q, A, Bm, C, alpha, beta, transA, transB);
            else select::no_vendor<B, T>(spec);
        },
    }, c);
}

}  // namespace ops::spmm

// Deliberately no validate_params: a shape the native driver cannot take resolves to the vendor,
// as before, which then reports it (or, vendor-free, there is no route).
template <Backend Back, typename T, MatrixFormat MFormat>
Event spmm(Queue& ctx, const MatrixView<T, MFormat>& A, const MatrixView<T, MatrixFormat::Dense>& B_mat,
           const MatrixView<T, MatrixFormat::Dense>& C, T alpha, T beta, Transpose transA, Transpose transB,
           Span<std::byte> workspace) {
    // The coverage row's key, as the old shape builder wrote it: m, k = A as stored, n = nrhs.
    const coverage::Shape shape{.m = A.rows(), .n = C.cols(), .k = A.cols(), .batch = A.batch_size(),
                                .transA = transA, .transB = transB};
    const select::Key key = ops::spmm::key_of<T, MFormat>(A, C, transA, transB);
    return select::run<Back, T>(
        ops::spmm::spec, ctx, key, ops::spmm::candidates<T>(),
        [&](const auto& c, const auto& d) {
            return ops::spmm::can_run<Back, T, MFormat>(c, d, A, B_mat, C, transA, transB);
        },
        shape, key, [&](const auto& c) {
            return ops::spmm::launch<Back, T, MFormat>(ctx, c, A, B_mat, C, alpha, beta, transA, transB, workspace);
        });
}

template <Backend Back, typename T, MatrixFormat MFormat>
size_t spmm_buffer_size(Queue& ctx, const MatrixView<T, MFormat>& A, const MatrixView<T, MatrixFormat::Dense>& B_mat,
                        const MatrixView<T, MatrixFormat::Dense>& C, T alpha, T beta, Transpose transA,
                        Transpose transB) {
    const auto c = select::pick<Back, T>(
        ops::spmm::spec, ctx, ops::spmm::key_of<T, MFormat>(A, C, transA, transB), ops::spmm::candidates<T>(),
        [&](const auto& k, const auto& d) {
            return ops::spmm::can_run<Back, T, MFormat>(k, d, A, B_mat, C, transA, transB);
        });
    return ops::spmm::workspace<Back, T, MFormat>(ctx, c, A, B_mat, C, alpha, beta, transA, transB);
}

#define SPMM_INSTANTIATE(B_, fp)                                    \
    BATCHLAS_INSTANTIATE_FORMAT_OP(B_, fp, MatrixFormat::CSR, spmm) \
    BATCHLAS_INSTANTIATE_FORMAT_OP(B_, fp, MatrixFormat::CSR, spmm_buffer_size)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(SPMM_INSTANTIATE)
#undef SPMM_INSTANTIATE

}  // namespace batchlas
