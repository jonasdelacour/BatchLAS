#pragma once

/// @file
/// @brief spmm: direct, vendor. evidence: docs/perf/spmm.md @ingroup api_selection_ops

#include "../../select/select.hh"

#include <array>
#include <string_view>
#include <variant>

namespace batchlas::ops::spmm {

struct Direct : select::NoFields<"direct"> {};  ///< sycl_spmm::spmm_native_csr: CSR only, no GPU gate (native_cpu too)
struct Vendor : select::NoFields<"vendor"> {};  ///< backend::spmm_vendor (cuSPARSE, rocSPARSE, netlib) less known-defect shapes

using SpmmChoice = std::variant<Direct, Vendor>;  ///< fieldless: body (by transA), column block, pair load derived

template <class T>  /// Every compiled choice, in tie-break order (§6.3).
constexpr auto candidates() { return select::all_of<SpmmChoice>(); }

inline constexpr std::array<std::string_view, 2> last_resort{"vendor", "direct"};  ///< vendor: every format
inline constexpr select::OpSpec spec{Op::spmm, select::Lib::sparse, {last_resort}};  ///< op, library, rules

/// ConjTrans folds to T; m = A.rows() as stored; no nnz key (device memory). Work ~ m nrhs batch.
inline constexpr std::array<std::string_view, 5> key_names{
    "transA:exact", "transB:exact", "m:log", "nrhs:log", "batch:log"};

/// The transcriber (tuned/README.md) spells the same grid; the old predicates read no extent.
inline constexpr std::array<int, 5> grid_m{1, 16, 256, 4096, 65536};
inline constexpr std::array<int, 5> grid_nrhs{1, 2, 4, 16, 64};
inline constexpr std::array<int, 5> grid_batch{1, 8, 128, 1024, 16384};

}  // namespace batchlas::ops::spmm
