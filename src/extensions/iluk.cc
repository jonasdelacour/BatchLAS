#include "../linalg-impl.hh"

#include <blas/functions/iluk.hh>
#include <util/mempool.hh>
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace batchlas {
namespace {

template <typename T>
using RealT = typename base_type<T>::type;

template <typename T>
inline RealT<T> abs_value(const T& v) {
    return static_cast<RealT<T>>(std::abs(v));
}

template <typename T>
inline T real_scalar(RealT<T> value) {
    if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
        return T(value, RealT<T>(0));
    } else {
        return static_cast<T>(value);
    }
}

template <typename T>
inline T stabilize_pivot_or_mark(const T& pivot,
                                 RealT<T> row_scale,
                                 const ILUKParams<T>& params,
                                 int32_t* status) {
    const auto shift_mag = abs_value(params.diagonal_shift);
    const auto pivot_mag = abs_value(pivot);
    const auto threshold = std::max(shift_mag, params.diag_pivot_threshold * std::max(row_scale, RealT<T>(1)));

    if (pivot_mag >= threshold && pivot_mag > RealT<T>(0)) {
        return pivot;
    }

    if (shift_mag > RealT<T>(0)) {
        const T shifted = pivot + params.diagonal_shift;
        if (abs_value(shifted) >= threshold && abs_value(shifted) > RealT<T>(0)) {
            return shifted;
        }
        if (shift_mag >= threshold) {
            return params.diagonal_shift;
        }
    }

    if (threshold > RealT<T>(0)) {
        return real_scalar<T>(threshold);
    }

    if (status != nullptr) {
        *status = 1;
    }
    throw std::runtime_error(
        "ILU(k): encountered a zero or effectively zero pivot without a usable diagonal shift");
}

template <typename T>
inline T stabilize_pivot_or_mark(const T& pivot,
                                 const T& diagonal_shift,
                                 int32_t* status) {
    const auto shift_mag = abs_value(diagonal_shift);
    const auto pivot_mag = abs_value(pivot);
    if (pivot_mag > shift_mag) {
        return pivot;
    }

    if (shift_mag > RealT<T>(0)) {
        const T shifted = pivot + diagonal_shift;
        if (abs_value(shifted) > shift_mag) {
            return shifted;
        }
        return diagonal_shift;
    }

    if (status != nullptr) {
        *status = 1;
    }
    return T(1);
}

template <typename T>
bool u_diagonals_are_usable(const T* vals,
                            const int* diag_positions,
                            int n,
                            int batch_size,
                            const T& diagonal_shift) {
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < n; ++i) {
            int32_t status = 0;
            (void)stabilize_pivot_or_mark(vals[diag_positions[b * n + i]], diagonal_shift, &status);
            if (status != 0) return false;
        }
    }
    return true;
}

template <typename T>
bool has_identical_batch_sparsity(const MatrixView<T, MatrixFormat::CSR>& A) {
    if (A.batch_size() <= 1) return true;
    const auto ro = A.row_offsets();
    const auto ci = A.col_indices();
    const int rows = A.rows();
    const int offset_stride = A.offset_stride();
    const int matrix_stride = A.matrix_stride();
    for (int b = 1; b < A.batch_size(); ++b) {
        const int ro_base = b * offset_stride;
        const int ci_base = b * matrix_stride;
        for (int i = 0; i < rows + 1; ++i) {
            if (ro[ro_base + i] != ro[i]) return false;
        }
        for (int i = 0; i < A.nnz(); ++i) {
            if (ci[ci_base + i] != ci[i]) return false;
        }
    }
    return true;
}

template <typename T>
RealT<T> row_scale(const std::vector<T>& values, const std::vector<uint8_t>& keep_flags) {
    RealT<T> scale = RealT<T>(0);
    for (std::size_t idx = 0; idx < values.size(); ++idx) {
        if (!keep_flags.empty() && keep_flags[idx] == 0) continue;
        scale = std::max(scale, abs_value(values[idx]));
    }
    return scale;
}

template <typename T>
void apply_drop_and_fill_control(std::vector<T>& row_values,
                                 std::vector<uint8_t>& keep_flags,
                                 int diag_index,
                                 int original_row_nnz,
                                 const ILUKParams<T>& params) {
    const auto scale = row_scale(row_values, keep_flags);
    const auto drop_threshold = params.drop_tolerance * std::max(scale, RealT<T>(1));

    std::vector<std::pair<RealT<T>, int>> candidates;
    candidates.reserve(row_values.size());
    for (int idx = 0; idx < static_cast<int>(row_values.size()); ++idx) {
        keep_flags[static_cast<std::size_t>(idx)] = 1;
        if (idx == diag_index) continue;
        if (abs_value(row_values[static_cast<std::size_t>(idx)]) <= drop_threshold) {
            if (params.modified_ilu) {
                row_values[static_cast<std::size_t>(diag_index)] += row_values[static_cast<std::size_t>(idx)];
            }
            row_values[static_cast<std::size_t>(idx)] = T(0);
            keep_flags[static_cast<std::size_t>(idx)] = 0;
            continue;
        }
        candidates.emplace_back(abs_value(row_values[static_cast<std::size_t>(idx)]), idx);
    }

    const int offdiag_quota = std::max(0, static_cast<int>(std::ceil(params.fill_factor * static_cast<RealT<T>>(original_row_nnz))) - 1);
    if (static_cast<int>(candidates.size()) > offdiag_quota) {
        std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.first > rhs.first;
        });
        for (int drop_idx = offdiag_quota; drop_idx < static_cast<int>(candidates.size()); ++drop_idx) {
            const int idx = candidates[static_cast<std::size_t>(drop_idx)].second;
            if (params.modified_ilu) {
                row_values[static_cast<std::size_t>(diag_index)] += row_values[static_cast<std::size_t>(idx)];
            }
            row_values[static_cast<std::size_t>(idx)] = T(0);
            keep_flags[static_cast<std::size_t>(idx)] = 0;
        }
    }

    keep_flags[static_cast<std::size_t>(diag_index)] = 1;
}

std::vector<std::vector<int>> symbolic_iluk_pattern_single(const Span<int>& row_offsets,
                                                           const Span<int>& col_indices,
                                                           int rows,
                                                           int offset_base,
                                                           int matrix_base,
                                                           int level_k) {
    std::vector<std::vector<std::pair<int, int>>> row_levels(rows);

    for (int i = 0; i < rows; ++i) {
        std::map<int, int> levels;
        levels[i] = 0;  // keep diagonal

        const int rs = row_offsets[offset_base + i];
        const int re = row_offsets[offset_base + i + 1];
        for (int p = rs; p < re; ++p) {
            levels[col_indices[matrix_base + p]] = 0;
        }

        for (auto it = levels.begin(); it != levels.end() && it->first < i; ++it) {
            const int j = it->first;
            const int lij_level = it->second;
            for (const auto& [col, lvl_jc] : row_levels[j]) {
                if (col <= j) continue;
                const int new_level = std::max(lij_level, lvl_jc) + 1;
                if (new_level > level_k) continue;
                auto found = levels.find(col);
                if (found == levels.end()) {
                    levels[col] = new_level;
                } else if (new_level < found->second) {
                    found->second = new_level;
                }
            }
        }

        row_levels[i].reserve(levels.size());
        for (const auto& kv : levels) row_levels[i].push_back(kv);
    }

    std::vector<std::vector<int>> rows_cols(rows);
    for (int i = 0; i < rows; ++i) {
        rows_cols[i].reserve(row_levels[i].size());
        for (const auto& [col, _lvl] : row_levels[i]) rows_cols[i].push_back(col);
    }
    return rows_cols;
}

template <Backend B, typename T>
struct ILUKApplyKernel;

// Bucket rows of a triangular factor into dependency levels. `lower` selects the
// unit-lower forward solve (row i depends on columns j < i, scanned in increasing
// row order) versus the upper backward solve (row i depends on columns j > i,
// scanned in decreasing row order). Rows sharing a level are mutually independent.
struct LevelSchedule {
    std::vector<int> rows;
    std::vector<int> level_ptr;
    int levels = 0;
};

LevelSchedule build_level_schedule(const std::vector<int>& row_offsets,
                                   const std::vector<int>& col_indices,
                                   int n,
                                   bool lower) {
    std::vector<int> level(static_cast<std::size_t>(n), 0);
    int max_level = 0;

    for (int step = 0; step < n; ++step) {
        const int i = lower ? step : (n - 1 - step);
        int lvl = 0;
        for (int p = row_offsets[static_cast<std::size_t>(i)]; p < row_offsets[static_cast<std::size_t>(i + 1)]; ++p) {
            const int j = col_indices[static_cast<std::size_t>(p)];
            const bool depends = lower ? (j < i) : (j > i);
            if (depends) {
                lvl = std::max(lvl, level[static_cast<std::size_t>(j)] + 1);
            }
        }
        level[static_cast<std::size_t>(i)] = lvl;
        max_level = std::max(max_level, lvl);
    }

    LevelSchedule out;
    out.levels = max_level + 1;
    std::vector<int> counts(static_cast<std::size_t>(out.levels) + 1, 0);
    for (int i = 0; i < n; ++i) counts[static_cast<std::size_t>(level[static_cast<std::size_t>(i)]) + 1] += 1;
    for (int l = 0; l < out.levels; ++l) counts[static_cast<std::size_t>(l) + 1] += counts[static_cast<std::size_t>(l)];

    out.level_ptr = counts;
    out.rows.assign(static_cast<std::size_t>(n), 0);
    std::vector<int> cursor(counts.begin(), counts.end() - 1);
    for (int i = 0; i < n; ++i) {
        out.rows[static_cast<std::size_t>(cursor[static_cast<std::size_t>(level[static_cast<std::size_t>(i)])]++)] = i;
    }
    return out;
}

// Everything ILU(k) computes on the host, in a storage-agnostic form. The
// sparsity pattern (row_offsets/col_indices/diag_offsets) and both level
// schedules are shared across the batch; only `values` varies per batch element.
template <typename T>
struct HostFactor {
    int n = 0;
    int batch_size = 0;
    int nnz = 0;
    std::vector<int> row_offsets;   // n + 1
    std::vector<int> col_indices;   // nnz
    std::vector<T> values;          // batch_size * nnz, batch b at offset b * nnz
    std::vector<int> diag_offsets;  // n, index into a single batch element's values
    LevelSchedule l;
    LevelSchedule u;
};

// `check_batch_sparsity` walks every batch element's pattern, which on unified
// memory the device has just written means a page migration per batch element.
// iluk_buffer_size skips it: it only sizes against batch element 0's pattern, and
// the factorization that follows validates before it uses anything.
template <typename T>
void validate_iluk_params_or_throw(const MatrixView<T, MatrixFormat::CSR>& A, const ILUKParams<T>& params,
                                   bool check_batch_sparsity = true) {
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("ILU(k): matrix must be square");
    }
    if (params.levels_of_fill < 0) {
        throw std::invalid_argument("ILU(k): levels_of_fill must be >= 0");
    }
    if (params.drop_tolerance < RealT<T>(0)) {
        throw std::invalid_argument("ILU(k): drop_tolerance must be >= 0");
    }
    if (params.fill_factor < RealT<T>(1)) {
        throw std::invalid_argument("ILU(k): fill_factor must be >= 1");
    }
    if (params.diag_pivot_threshold < RealT<T>(0)) {
        throw std::invalid_argument("ILU(k): diag_pivot_threshold must be >= 0");
    }
    if (!params.validate_batch_sparsity && A.batch_size() > 1) {
        throw std::invalid_argument(
            "ILU(k): disabling batch sparsity validation is not supported; current implementation requires identical CSR sparsity across the batch");
    }
    if (check_batch_sparsity && A.batch_size() > 1 && !has_identical_batch_sparsity(A)) {
        throw std::invalid_argument(
            "ILU(k): heterogeneous batch sparsity is not supported; batches must share an identical CSR pattern");
    }
}

template <typename T>
HostFactor<T> compute_iluk(const MatrixView<T, MatrixFormat::CSR>& A, const ILUKParams<T>& params) {
    validate_iluk_params_or_throw(A, params);

    const int n = A.rows();
    const int batch_size = A.batch_size();

    const auto ro = A.row_offsets();
    const auto ci = A.col_indices();
    auto symbolic_rows = symbolic_iluk_pattern_single(ro, ci, n, 0, 0, params.levels_of_fill);
    std::vector<int> diag_local(n, -1);
    std::vector<int> original_row_nnz(n, 0);
    for (int i = 0; i < n; ++i) {
        original_row_nnz[static_cast<std::size_t>(i)] = ro[i + 1] - ro[i];
        const int re = static_cast<int>(symbolic_rows[static_cast<std::size_t>(i)].size());
        int pos = -1;
        for (int p = 0; p < re; ++p) {
            if (symbolic_rows[static_cast<std::size_t>(i)][static_cast<std::size_t>(p)] == i) {
                pos = p;
                break;
            }
        }
        if (pos < 0) {
            throw std::runtime_error("ILU(k): symbolic phase produced a row without diagonal");
        }
        diag_local[i] = pos;
    }

    UnifiedVector<int32_t> factor_status(static_cast<std::size_t>(batch_size), 0);

    std::vector<std::vector<std::vector<T>>> batch_values(static_cast<std::size_t>(batch_size));
    std::vector<std::vector<std::vector<uint8_t>>> batch_keep(static_cast<std::size_t>(batch_size));

    const auto a_vals = A.data();

    // Scatter workspace mapping a column index to its slot in the row currently
    // being assembled. Replaces a binary search per touched entry with an O(1)
    // lookup; -1 means the column is outside this row's symbolic pattern.
    std::vector<int> col_to_slot(static_cast<std::size_t>(n), -1);

    for (int b = 0; b < batch_size; ++b) {
        auto& values_by_row = batch_values[static_cast<std::size_t>(b)];
        auto& keep_by_row = batch_keep[static_cast<std::size_t>(b)];
        values_by_row.resize(static_cast<std::size_t>(n));
        keep_by_row.resize(static_cast<std::size_t>(n));

        const int ro_base = b * A.offset_stride();
        const int val_base = b * A.matrix_stride();
        for (int i = 0; i < n; ++i) {
            const auto& row_cols = symbolic_rows[static_cast<std::size_t>(i)];
            auto& row_vals = values_by_row[static_cast<std::size_t>(i)];
            auto& row_keep = keep_by_row[static_cast<std::size_t>(i)];
            row_vals.assign(row_cols.size(), T(0));
            row_keep.assign(row_cols.size(), 1);

            const int row_nnz = static_cast<int>(row_cols.size());
            for (int p = 0; p < row_nnz; ++p) col_to_slot[static_cast<std::size_t>(row_cols[static_cast<std::size_t>(p)])] = p;

            const int ars = ro[ro_base + i];
            const int are = ro[ro_base + i + 1];
            for (int p = ars; p < are; ++p) {
                const int slot = col_to_slot[static_cast<std::size_t>(ci[val_base + p])];
                if (slot >= 0) {
                    row_vals[static_cast<std::size_t>(slot)] = a_vals[val_base + p];
                }
            }

            for (int p = 0; p < row_nnz; ++p) col_to_slot[static_cast<std::size_t>(row_cols[static_cast<std::size_t>(p)])] = -1;
        }

        for (int i = 0; i < n; ++i) {
            auto& row_vals = values_by_row[static_cast<std::size_t>(i)];
            auto& row_keep = keep_by_row[static_cast<std::size_t>(i)];
            const auto& row_cols = symbolic_rows[static_cast<std::size_t>(i)];

            const int row_nnz = static_cast<int>(row_cols.size());
            for (int p = 0; p < row_nnz; ++p) col_to_slot[static_cast<std::size_t>(row_cols[static_cast<std::size_t>(p)])] = p;

            for (int p = 0; p < row_nnz; ++p) {
                const int j = row_cols[static_cast<std::size_t>(p)];
                if (j >= i) break;

                const auto& pivot_row_vals = values_by_row[static_cast<std::size_t>(j)];
                const auto& pivot_row_keep = keep_by_row[static_cast<std::size_t>(j)];
                const int diag_j = diag_local[static_cast<std::size_t>(j)];

                // Row j < i is already finalized: its diagonal was stabilized when row j
                // was processed and the row has not changed since. Stabilization is
                // idempotent, so re-deriving the row scale here (an O(nnz_j) scan on every
                // elimination step) would recompute the value already stored.
                const T lij = row_vals[static_cast<std::size_t>(p)] / pivot_row_vals[static_cast<std::size_t>(diag_j)];
                row_vals[static_cast<std::size_t>(p)] = lij;

                // Columns are sorted, so the strict upper part of row j starts past its diagonal.
                const auto& pivot_cols = symbolic_rows[static_cast<std::size_t>(j)];
                const int pivot_nnz = static_cast<int>(pivot_cols.size());
                for (int q = diag_j + 1; q < pivot_nnz; ++q) {
                    if (pivot_row_keep[static_cast<std::size_t>(q)] == 0) continue;
                    const int slot = col_to_slot[static_cast<std::size_t>(pivot_cols[static_cast<std::size_t>(q)])];
                    if (slot >= 0) {
                        row_vals[static_cast<std::size_t>(slot)] -= lij * pivot_row_vals[static_cast<std::size_t>(q)];
                    }
                }
            }

            apply_drop_and_fill_control(row_vals, row_keep, diag_local[static_cast<std::size_t>(i)], original_row_nnz[static_cast<std::size_t>(i)], params);
            const auto final_scale = row_scale(row_vals, row_keep);
            row_vals[static_cast<std::size_t>(diag_local[static_cast<std::size_t>(i)])] = stabilize_pivot_or_mark(
                row_vals[static_cast<std::size_t>(diag_local[static_cast<std::size_t>(i)])], final_scale, params, &factor_status[static_cast<std::size_t>(b)]);

            for (int p = 0; p < row_nnz; ++p) col_to_slot[static_cast<std::size_t>(row_cols[static_cast<std::size_t>(p)])] = -1;
        }
    }

    for (int b = 0; b < batch_size; ++b) {
        if (factor_status[static_cast<std::size_t>(b)] != 0) {
            throw std::runtime_error(
                "ILU(k): encountered a zero or effectively zero pivot without a usable diagonal shift");
        }
    }

    std::vector<std::vector<uint8_t>> union_keep(static_cast<std::size_t>(n));
    std::vector<std::vector<int>> compact_rows(static_cast<std::size_t>(n));
    std::vector<std::vector<int>> compact_index(static_cast<std::size_t>(n));

    HostFactor<T> out;
    out.n = n;
    out.batch_size = batch_size;
    out.row_offsets.assign(static_cast<std::size_t>(n) + 1, 0);

    for (int i = 0; i < n; ++i) {
        const auto& row_cols = symbolic_rows[static_cast<std::size_t>(i)];
        auto& row_union = union_keep[static_cast<std::size_t>(i)];
        row_union.assign(row_cols.size(), 0);
        row_union[static_cast<std::size_t>(diag_local[static_cast<std::size_t>(i)])] = 1;
        for (int b = 0; b < batch_size; ++b) {
            const auto& row_keep = batch_keep[static_cast<std::size_t>(b)][static_cast<std::size_t>(i)];
            for (int p = 0; p < static_cast<int>(row_cols.size()); ++p) {
                row_union[static_cast<std::size_t>(p)] = static_cast<uint8_t>(row_union[static_cast<std::size_t>(p)] | row_keep[static_cast<std::size_t>(p)]);
            }
        }

        auto& compact_row = compact_rows[static_cast<std::size_t>(i)];
        auto& compact_pos = compact_index[static_cast<std::size_t>(i)];
        compact_pos.assign(row_cols.size(), -1);
        for (int p = 0; p < static_cast<int>(row_cols.size()); ++p) {
            if (row_union[static_cast<std::size_t>(p)] == 0) continue;
            compact_pos[static_cast<std::size_t>(p)] = static_cast<int>(compact_row.size());
            compact_row.push_back(row_cols[static_cast<std::size_t>(p)]);
        }
        out.row_offsets[static_cast<std::size_t>(i + 1)] = out.row_offsets[static_cast<std::size_t>(i)] + static_cast<int>(compact_row.size());
    }

    out.col_indices.reserve(static_cast<std::size_t>(out.row_offsets.back()));
    for (int i = 0; i < n; ++i) {
        const auto& compact_row = compact_rows[static_cast<std::size_t>(i)];
        out.col_indices.insert(out.col_indices.end(), compact_row.begin(), compact_row.end());
    }
    out.nnz = static_cast<int>(out.col_indices.size());

    out.diag_offsets.assign(static_cast<std::size_t>(n), 0);
    out.values.assign(static_cast<std::size_t>(out.nnz) * static_cast<std::size_t>(batch_size), T(0));
    for (int i = 0; i < n; ++i) {
        out.diag_offsets[static_cast<std::size_t>(i)] =
            out.row_offsets[static_cast<std::size_t>(i)] + compact_index[static_cast<std::size_t>(i)][static_cast<std::size_t>(diag_local[static_cast<std::size_t>(i)])];
    }
    for (int b = 0; b < batch_size; ++b) {
        const std::size_t vbase = static_cast<std::size_t>(b) * static_cast<std::size_t>(out.nnz);
        for (int i = 0; i < n; ++i) {
            const auto& row_vals = batch_values[static_cast<std::size_t>(b)][static_cast<std::size_t>(i)];
            const auto& row_keep = batch_keep[static_cast<std::size_t>(b)][static_cast<std::size_t>(i)];
            const auto& compact_pos = compact_index[static_cast<std::size_t>(i)];
            for (int p = 0; p < static_cast<int>(row_vals.size()); ++p) {
                const int new_pos = compact_pos[static_cast<std::size_t>(p)];
                if (new_pos < 0) continue;
                if (row_keep[static_cast<std::size_t>(p)] == 0 && p != diag_local[static_cast<std::size_t>(i)]) continue;
                out.values[vbase + static_cast<std::size_t>(out.row_offsets[static_cast<std::size_t>(i)] + new_pos)] =
                    row_vals[static_cast<std::size_t>(p)];
            }
        }
    }

    out.l = build_level_schedule(out.row_offsets, out.col_indices, n, /*lower=*/true);
    out.u = build_level_schedule(out.row_offsets, out.col_indices, n, /*lower=*/false);
    return out;
}

}  // namespace

template <typename T>
void iluk_build_level_schedule(ILUKPreconditioner<T>& M) {
    const int n = M.n;
    if (n <= 0) {
        throw std::invalid_argument("ILU(k): cannot build a level schedule for an empty factor");
    }
    auto lu = M.lu.view();
    const auto ro = lu.row_offsets();
    const auto ci = lu.col_indices();

    std::vector<int> row_offsets(static_cast<std::size_t>(n) + 1);
    for (int i = 0; i <= n; ++i) row_offsets[static_cast<std::size_t>(i)] = ro[i];
    std::vector<int> col_indices(static_cast<std::size_t>(row_offsets[static_cast<std::size_t>(n)]));
    for (std::size_t p = 0; p < col_indices.size(); ++p) col_indices[p] = ci[p];

    const auto l = build_level_schedule(row_offsets, col_indices, n, /*lower=*/true);
    const auto u = build_level_schedule(row_offsets, col_indices, n, /*lower=*/false);

    auto to_unified = [](const std::vector<int>& src) {
        UnifiedVector<int> dst(src.size());
        for (std::size_t i = 0; i < src.size(); ++i) dst[i] = src[i];
        return dst;
    };
    M.l_rows = to_unified(l.rows);
    M.l_level_ptr = to_unified(l.level_ptr);
    M.l_levels = l.levels;
    M.u_rows = to_unified(u.rows);
    M.u_level_ptr = to_unified(u.level_ptr);
    M.u_levels = u.levels;
    M.u_diagonals_usable = u_diagonals_are_usable(M.lu.view().data().data(), M.diag_positions.data(),
                                                  M.n, M.batch_size, M.diagonal_shift);
}

template <Backend B, typename T>
ILUKPreconditioner<T> iluk_factorize(Queue& ctx,
                                     const MatrixView<T, MatrixFormat::CSR>& A,
                                     const ILUKParams<T>& params) {
    (void)ctx;
    const auto host = compute_iluk(A, params);
    const int n = host.n;
    const int batch_size = host.batch_size;

    ILUKPreconditioner<T> result;
    result.lu = Matrix<T, MatrixFormat::CSR>(n, n, host.nnz, batch_size);
    result.diag_positions = UnifiedVector<int>(static_cast<std::size_t>(n) * static_cast<std::size_t>(batch_size));
    result.n = n;
    result.batch_size = batch_size;
    result.levels_of_fill = params.levels_of_fill;
    result.diagonal_shift = params.diagonal_shift;
    result.drop_tolerance = params.drop_tolerance;
    result.fill_factor = params.fill_factor;
    result.diag_pivot_threshold = params.diag_pivot_threshold;
    result.modified_ilu = params.modified_ilu;

    auto to_unified = [](const std::vector<int>& src) {
        UnifiedVector<int> dst(src.size());
        for (std::size_t i = 0; i < src.size(); ++i) dst[i] = src[i];
        return dst;
    };
    result.l_rows = to_unified(host.l.rows);
    result.l_level_ptr = to_unified(host.l.level_ptr);
    result.l_levels = host.l.levels;
    result.u_rows = to_unified(host.u.rows);
    result.u_level_ptr = to_unified(host.u.level_ptr);
    result.u_levels = host.u.levels;

    auto lu_view = result.lu.view();
    auto lu_ro = lu_view.row_offsets();
    auto lu_ci = lu_view.col_indices();
    auto lu_vals = lu_view.data();
    for (int b = 0; b < batch_size; ++b) {
        const int ro_base = b * lu_view.offset_stride();
        const int val_base = b * lu_view.matrix_stride();
        for (int i = 0; i < n + 1; ++i) lu_ro[ro_base + i] = host.row_offsets[static_cast<std::size_t>(i)];
        for (int p = 0; p < host.nnz; ++p) {
            lu_ci[val_base + p] = host.col_indices[static_cast<std::size_t>(p)];
            lu_vals[val_base + p] = host.values[static_cast<std::size_t>(b) * static_cast<std::size_t>(host.nnz) + static_cast<std::size_t>(p)];
        }
        for (int i = 0; i < n; ++i) {
            result.diag_positions[static_cast<std::size_t>(b * n + i)] = val_base + host.diag_offsets[static_cast<std::size_t>(i)];
        }
    }

    result.u_diagonals_usable = u_diagonals_are_usable(lu_vals.data(), result.diag_positions.data(), n,
                                                       batch_size, result.diagonal_shift);

    return result;
}

template <Backend B, typename T>
size_t iluk_buffer_size(Queue& ctx,
                        const MatrixView<T, MatrixFormat::CSR>& A,
                        const ILUKParams<T>& params) {
    validate_iluk_params_or_throw(A, params, /*check_batch_sparsity=*/false);

    const int n = A.rows();
    const int batch_size = A.batch_size();

    // Size against the symbolic pattern. The numeric phase only ever prunes
    // entries (drop tolerance, fill quota), so this is an upper bound on the
    // final nnz and the workspace factorization is guaranteed to fit.
    const auto symbolic_rows = symbolic_iluk_pattern_single(A.row_offsets(), A.col_indices(), n, 0, 0,
                                                            params.levels_of_fill);
    size_t nnz_upper = 0;
    for (const auto& row : symbolic_rows) nnz_upper += row.size();

    size_t bytes = 0;
    bytes += BumpAllocator::allocation_size<T>(ctx, nnz_upper * static_cast<size_t>(batch_size));    // values
    bytes += BumpAllocator::allocation_size<int>(ctx, nnz_upper * static_cast<size_t>(batch_size));  // col indices
    bytes += BumpAllocator::allocation_size<int>(ctx, static_cast<size_t>(n + 1) * static_cast<size_t>(batch_size));  // row offsets
    bytes += BumpAllocator::allocation_size<int>(ctx, static_cast<size_t>(n) * static_cast<size_t>(batch_size));      // diag positions
    // Level counts are bounded by n, so the pointer arrays need at most n + 1 slots.
    bytes += BumpAllocator::allocation_size<int>(ctx, static_cast<size_t>(n)) * 2;      // l_rows, u_rows
    bytes += BumpAllocator::allocation_size<int>(ctx, static_cast<size_t>(n + 1)) * 2;  // l/u level_ptr
    return bytes;
}

template <Backend B, typename T>
ILUKView<T> iluk_factorize(Queue& ctx,
                           const MatrixView<T, MatrixFormat::CSR>& A,
                           Span<std::byte> workspace,
                           const ILUKParams<T>& params,
                           size_t* bytes_used) {
    const auto host = compute_iluk(A, params);
    const int n = host.n;
    const int batch_size = host.batch_size;
    const int nnz = host.nnz;

    auto pool = BumpAllocator(workspace);
    auto values = pool.allocate<T>(ctx, static_cast<size_t>(nnz) * static_cast<size_t>(batch_size));
    auto col_indices = pool.allocate<int>(ctx, static_cast<size_t>(nnz) * static_cast<size_t>(batch_size));
    auto row_offsets = pool.allocate<int>(ctx, static_cast<size_t>(n + 1) * static_cast<size_t>(batch_size));
    auto diag_positions = pool.allocate<int>(ctx, static_cast<size_t>(n) * static_cast<size_t>(batch_size));
    auto l_rows = pool.allocate<int>(ctx, static_cast<size_t>(n));
    auto u_rows = pool.allocate<int>(ctx, static_cast<size_t>(n));
    auto l_level_ptr = pool.allocate<int>(ctx, host.l.level_ptr.size());
    auto u_level_ptr = pool.allocate<int>(ctx, host.u.level_ptr.size());

    // The pattern is identical across the batch, but the apply kernel indexes
    // values and column indices with the same matrix stride, so both are
    // replicated per batch element rather than shared.
    for (int b = 0; b < batch_size; ++b) {
        const size_t val_base = static_cast<size_t>(b) * static_cast<size_t>(nnz);
        const size_t ro_base = static_cast<size_t>(b) * static_cast<size_t>(n + 1);
        for (int i = 0; i < n + 1; ++i) row_offsets[ro_base + static_cast<size_t>(i)] = host.row_offsets[static_cast<std::size_t>(i)];
        for (int p = 0; p < nnz; ++p) {
            col_indices[val_base + static_cast<size_t>(p)] = host.col_indices[static_cast<std::size_t>(p)];
            values[val_base + static_cast<size_t>(p)] = host.values[val_base + static_cast<size_t>(p)];
        }
        for (int i = 0; i < n; ++i) {
            diag_positions[static_cast<size_t>(b * n + i)] = static_cast<int>(val_base) + host.diag_offsets[static_cast<std::size_t>(i)];
        }
    }
    for (int i = 0; i < n; ++i) {
        l_rows[static_cast<size_t>(i)] = host.l.rows[static_cast<std::size_t>(i)];
        u_rows[static_cast<size_t>(i)] = host.u.rows[static_cast<std::size_t>(i)];
    }
    for (std::size_t i = 0; i < host.l.level_ptr.size(); ++i) l_level_ptr[i] = host.l.level_ptr[i];
    for (std::size_t i = 0; i < host.u.level_ptr.size(); ++i) u_level_ptr[i] = host.u.level_ptr[i];

    ILUKView<T> view;
    view.lu = MatrixView<T, MatrixFormat::CSR>(values.data(), row_offsets.data(), col_indices.data(), nnz, n, n,
                                               nnz, n + 1, batch_size);
    view.diag_positions = diag_positions;
    view.l_rows = l_rows;
    view.l_level_ptr = l_level_ptr;
    view.l_levels = host.l.levels;
    view.u_rows = u_rows;
    view.u_level_ptr = u_level_ptr;
    view.u_levels = host.u.levels;
    view.n = n;
    view.batch_size = batch_size;
    view.diagonal_shift = params.diagonal_shift;
    view.u_diagonals_usable = u_diagonals_are_usable(values.data(), diag_positions.data(), n, batch_size,
                                                     params.diagonal_shift);
    if (bytes_used != nullptr) {
        *bytes_used = static_cast<size_t>(workspace.size() - pool.remaining().size());
    }
    return view;
}

template <Backend B, typename T>
Event iluk_apply(Queue& ctx,
                 const ILUKView<T>& M,
                 const MatrixView<T, MatrixFormat::Dense>& rhs,
                 const MatrixView<T, MatrixFormat::Dense>& out,
                 Span<std::byte>) {
    if (rhs.rows() != M.n || out.rows() != M.n) {
        throw std::invalid_argument("ILU(k) apply: rhs/out row dimension must match factor rows");
    }
    if (rhs.cols() != out.cols()) {
        throw std::invalid_argument("ILU(k) apply: rhs and out must have same column count");
    }
    if (rhs.batch_size() != M.batch_size || out.batch_size() != M.batch_size) {
        throw std::invalid_argument("ILU(k) apply: rhs/out batch size must match factor batch size");
    }
    if (M.l_levels <= 0 || M.u_levels <= 0 ||
        M.l_rows.size() != static_cast<std::size_t>(M.n) ||
        M.u_rows.size() != static_cast<std::size_t>(M.n)) {
        throw std::invalid_argument(
            "ILU(k) apply: preconditioner has no triangular-solve level schedule; "
            "call iluk_build_level_schedule() when constructing one by hand");
    }
    if (!M.u_diagonals_usable) {
        throw std::runtime_error(
            "ILU(k) apply: encountered a zero or effectively zero U diagonal without a usable diagonal shift");
    }

    const int n = M.n;
    const int nrhs = rhs.cols();
    const int batch = M.batch_size;

    auto lu_kv = M.lu.kernel_view();
    auto rhs_kv = rhs.kernel_view();
    auto out_kv = out.kernel_view();

    auto lu_vals = lu_kv.data_;
    auto lu_ro = lu_kv.row_offsets_;
    auto lu_ci = lu_kv.col_indices_;

    auto rhs_data = rhs_kv.data_;
    auto out_data = out_kv.data_;
    auto diag_pos = M.diag_positions.data();
    const T diag_shift = M.diagonal_shift;

    const size_t total_systems = static_cast<size_t>(batch * nrhs);

    // One work-group per (batch, rhs column) system. Within a group the rows of a
    // dependency level are split across work-items and a group barrier separates
    // levels, so the serial chain is the level count rather than n. Each group owns
    // its system exclusively, so no cross-group synchronization is needed.
    const int l_levels = M.l_levels;
    const int u_levels = M.u_levels;
    auto l_rows = M.l_rows.data();
    auto l_level_ptr = M.l_level_ptr.data();
    auto u_rows = M.u_rows.data();
    auto u_level_ptr = M.u_level_ptr.data();

    constexpr int wg_size = 128;
    ctx->parallel_for<ILUKApplyKernel<B, T>>(
        sycl::nd_range<1>(total_systems * wg_size, wg_size), [=](sycl::nd_item<1> item) {
        const int linear = static_cast<int>(item.get_group(0));
        const int b = linear / nrhs;
        const int col = linear % nrhs;
        const int lid = static_cast<int>(item.get_local_id(0));

        const int ro_base = b * lu_kv.offset_stride_;
        const int lu_base = b * lu_kv.matrix_stride_;
        const int rhs_base = b * rhs_kv.stride_ + col * rhs_kv.ld_;
        const int out_base = b * out_kv.stride_ + col * out_kv.ld_;

        // Forward solve into out (as temporary y).
        for (int lv = 0; lv < l_levels; ++lv) {
            const int begin = l_level_ptr[lv];
            const int end = l_level_ptr[lv + 1];
            for (int t = begin + lid; t < end; t += wg_size) {
                const int i = l_rows[t];
                T sum = rhs_data[rhs_base + i];
                const int rs = lu_ro[ro_base + i];
                const int re = lu_ro[ro_base + i + 1];
                for (int p = rs; p < re; ++p) {
                    const int j = lu_ci[lu_base + p];
                    if (j >= i) break;
                    sum -= lu_vals[lu_base + p] * out_data[out_base + j];
                }
                out_data[out_base + i] = sum;
            }
            sycl::group_barrier(item.get_group());
        }

        for (int lv = 0; lv < u_levels; ++lv) {
            const int begin = u_level_ptr[lv];
            const int end = u_level_ptr[lv + 1];
            for (int t = begin + lid; t < end; t += wg_size) {
                const int i = u_rows[t];
                T sum = out_data[out_base + i];
                const int diag_abs = diag_pos[b * n + i];
                const int re = lu_ro[ro_base + i + 1];
                // Usability of every U diagonal was established when the factor was
                // built; this only substitutes the shifted value.
                T diag = stabilize_pivot_or_mark(lu_vals[diag_abs], diag_shift, nullptr);
                // Columns are sorted ascending, so the strict upper part of row i
                // starts one past the diagonal.
                for (int p = diag_abs - lu_base + 1; p < re; ++p) {
                    sum -= lu_vals[lu_base + p] * out_data[out_base + lu_ci[lu_base + p]];
                }
                out_data[out_base + i] = sum / diag;
            }
            sycl::group_barrier(item.get_group());
        }
    });

    // No host sync here: the caller owns synchronization, so repeated applications
    // inside an iterative solver stay pipelined.
    return ctx.get_event();
}

template <Backend B, typename T>
size_t iluk_apply_buffer_size(Queue&, const ILUKView<T>&, const MatrixView<T, MatrixFormat::Dense>&, const MatrixView<T, MatrixFormat::Dense>&) {
    return 0;
}

#define ILUK_INSTANTIATE_COMMON(FP) \
    template void iluk_build_level_schedule<FP>(ILUKPreconditioner<FP>&);

ILUK_INSTANTIATE_COMMON(float)
ILUK_INSTANTIATE_COMMON(double)
ILUK_INSTANTIATE_COMMON(std::complex<float>)
ILUK_INSTANTIATE_COMMON(std::complex<double>)
#undef ILUK_INSTANTIATE_COMMON

#define ILUK_INSTANTIATE(BACK, FP) \
    template ILUKPreconditioner<FP> iluk_factorize<BACK, FP>(Queue&, const MatrixView<FP, MatrixFormat::CSR>&, const ILUKParams<FP>&); \
    template ILUKView<FP> iluk_factorize<BACK, FP>(Queue&, const MatrixView<FP, MatrixFormat::CSR>&, Span<std::byte>, const ILUKParams<FP>&, size_t*); \
    template size_t iluk_buffer_size<BACK, FP>(Queue&, const MatrixView<FP, MatrixFormat::CSR>&, const ILUKParams<FP>&); \
    template Event iluk_apply<BACK, FP>(Queue&, const ILUKView<FP>&, const MatrixView<FP, MatrixFormat::Dense>&, const MatrixView<FP, MatrixFormat::Dense>&, Span<std::byte>); \
    template size_t iluk_apply_buffer_size<BACK, FP>(Queue&, const ILUKView<FP>&, const MatrixView<FP, MatrixFormat::Dense>&, const MatrixView<FP, MatrixFormat::Dense>&);

#if BATCHLAS_HAS_CUDA_BACKEND
ILUK_INSTANTIATE(Backend::CUDA, float)
ILUK_INSTANTIATE(Backend::CUDA, double)
ILUK_INSTANTIATE(Backend::CUDA, std::complex<float>)
ILUK_INSTANTIATE(Backend::CUDA, std::complex<double>)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
ILUK_INSTANTIATE(Backend::ROCM, float)
ILUK_INSTANTIATE(Backend::ROCM, double)
ILUK_INSTANTIATE(Backend::ROCM, std::complex<float>)
ILUK_INSTANTIATE(Backend::ROCM, std::complex<double>)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
ILUK_INSTANTIATE(Backend::NETLIB, float)
ILUK_INSTANTIATE(Backend::NETLIB, double)
ILUK_INSTANTIATE(Backend::NETLIB, std::complex<float>)
ILUK_INSTANTIATE(Backend::NETLIB, std::complex<double>)
#endif

#undef ILUK_INSTANTIATE

}  // namespace batchlas
