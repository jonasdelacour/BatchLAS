#pragma once

// her2k's selection vocabulary (docs/design/flat-kernel-selection.md §12), header-only.

#include "../../select/select.hh"

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>

#include <array>
#include <cstdint>
#include <string_view>
#include <variant>

namespace batchlas::ops::her2k {

// Fieldless: the fold's ld is derived from n.
struct Fold : select::NoFields<"fold"> {};      // one public gemm into scratch + accumulate_hermitian<true>
struct Vendor : select::NoFields<"vendor"> {};  // backend::her2k_vendor (cublas?her2k / cblas_?her2k loop)

using Her2kChoice = std::variant<Fold, Vendor>;

template <class T>
constexpr auto candidates() {
    return std::array<Her2kChoice, 2>{Fold{}, Vendor{}};
}

inline constexpr std::array<std::string_view, 2> last_resort{"fold", "vendor"};
inline constexpr select::Rules rules{last_resort};

// n = C's order, k = op(A)'s inner extent. Work ~ n^2 k batch.
inline constexpr std::array<std::string_view, 3> key_names{"n:log:2", "k:log", "batch:log"};

// The transcriber's grid: n 127|128 and batch 1|2 straddle the old rule (n 768|769 as herk's).
inline constexpr std::array<int, 18> grid_n{1,   2,   4,   8,   16,  32,   64,   127,  128,
                                            129, 256, 512, 767, 768, 769, 1024, 2048, 4096};
inline constexpr std::array<int, 5> grid_k{1, 8, 64, 512, 4096};
inline constexpr std::array<int, 10> grid_batch{1, 2, 3, 4, 5, 8, 128, 1024, 8192, 32768};

inline constexpr std::int64_t kMaxGridBatch = 65535;  // accumulate_hermitian: batch in grid z
inline constexpr int kFoldWg = 256;                   // accumulate_hermitian's group

// Whether her2k(q, A, B, C, transA) would run `fold`: the call's own choose(), pins included (a
// bad pin throws). sytrd_blocked asks before calling her2k. Defined in her2k.cc.
template <Backend Bk, class T>
bool fold_chosen(Queue& q, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& B,
                 const MatrixView<T, MatrixFormat::Dense>& C, Transpose transA);

}  // namespace batchlas::ops::her2k
