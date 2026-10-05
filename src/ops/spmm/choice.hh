#pragma once

// spmm's selection vocabulary (docs/design/flat-select-p5/spmm.md), header-only.

#include "../../select/select.hh"

#include <array>
#include <string_view>
#include <variant>

namespace batchlas::ops::spmm {

// Fieldless: Direct's body (by transA), column block and complex pair load are derived.
struct Direct : select::NoFields<"direct"> {};  // sycl_spmm::spmm_native_csr
struct Vendor : select::NoFields<"vendor"> {};  // backend::spmm_vendor (cuSPARSE, rocSPARSE, netlib)

using SpmmChoice = std::variant<Direct, Vendor>;

template <class T>
constexpr auto candidates() {
    return std::array<SpmmChoice, 2>{Direct{}, Vendor{}};
}

inline constexpr std::array<std::string_view, 2> last_resort{"vendor", "direct"};
inline constexpr select::Rules rules{last_resort};  // vendor: every format; direct: CSR only

// ConjTrans folds to T; m = A.rows() as stored; no nnz key (device memory). Work ~ m nrhs batch.
inline constexpr std::array<std::string_view, 5> key_names{
    "transA:exact", "transB:exact", "m:log", "nrhs:log", "batch:log"};

// tools/transcribe/spmm_transcribe.cc spells the same grid; the old predicates read no extent.
inline constexpr std::array<int, 5> grid_m{1, 16, 256, 4096, 65536};
inline constexpr std::array<int, 5> grid_nrhs{1, 2, 4, 16, 64};
inline constexpr std::array<int, 5> grid_batch{1, 8, 128, 1024, 16384};

}  // namespace batchlas::ops::spmm
