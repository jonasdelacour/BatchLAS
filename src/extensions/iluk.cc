#include "../linalg-impl.hh"

#include <blas/functions/iluk.hh>
#include <util/mempool.hh>
#include <optional>
#include <cstdlib>
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
RealT<T> row_scale(const T* values, const uint8_t* keep_flags, int len) {
    RealT<T> scale = RealT<T>(0);
    for (int idx = 0; idx < len; ++idx) {
        if (keep_flags != nullptr && keep_flags[idx] == 0) continue;
        scale = std::max(scale, abs_value(values[idx]));
    }
    return scale;
}

// `candidates` is caller-owned scratch reused across rows. Allocating it per row
// put a heap allocation in the innermost loop, which became the limiting factor
// once the batch loop was parallelised.
template <typename T>
void apply_drop_and_fill_control(T* row_values,
                                 uint8_t* keep_flags,
                                 int len,
                                 int diag_index,
                                 int original_row_nnz,
                                 const ILUKParams<T>& params,
                                 std::vector<std::pair<RealT<T>, int>>& candidates) {
    const auto scale = row_scale(row_values, keep_flags, len);
    const auto drop_threshold = params.drop_tolerance * std::max(scale, RealT<T>(1));

    candidates.clear();
    for (int idx = 0; idx < len; ++idx) {
        keep_flags[idx] = 1;
        if (idx == diag_index) continue;
        if (abs_value(row_values[idx]) <= drop_threshold) {
            if (params.modified_ilu) {
                row_values[diag_index] += row_values[idx];
            }
            row_values[idx] = T(0);
            keep_flags[idx] = 0;
            continue;
        }
        candidates.emplace_back(abs_value(row_values[idx]), idx);
    }

    const int offdiag_quota = std::max(0, static_cast<int>(std::ceil(params.fill_factor * static_cast<RealT<T>>(original_row_nnz))) - 1);
    if (static_cast<int>(candidates.size()) > offdiag_quota) {
        // Ties are broken by column position so the surviving pattern is a function of
        // the input alone. Leaving it to std::sort's unspecified order would make the
        // factor depend on the sort implementation, and would stop the host and device
        // paths from producing the same result.
        std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
            if (lhs.first != rhs.first) return lhs.first > rhs.first;
            return lhs.second < rhs.second;
        });
        for (int drop_idx = offdiag_quota; drop_idx < static_cast<int>(candidates.size()); ++drop_idx) {
            const int idx = candidates[static_cast<std::size_t>(drop_idx)].second;
            if (params.modified_ilu) {
                row_values[diag_index] += row_values[idx];
            }
            row_values[idx] = T(0);
            keep_flags[idx] = 0;
        }
    }

    keep_flags[diag_index] = 1;
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

// Which numeric path to take. The device path has a fixed cost per call -- one
// kernel launch per dependency level, plus a few host round trips -- that does not
// shrink for small problems, while the host path costs time proportional to the
// batch. Measured on a 2D Laplacian the two cross at roughly 32 elements.
// BATCHLAS_ILUK_DEVICE forces either side, which is also how the two are checked
// against each other.
bool iluk_prefer_device(int batch_size) {
    if (const char* v = std::getenv("BATCHLAS_ILUK_DEVICE")) {
        if (v[0] == '0') return false;
        if (v[0] == '1') return true;
    }
    return batch_size >= 32;
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
    const auto symbolic_rows = symbolic_iluk_pattern_single(ro, ci, n, 0, 0, params.levels_of_fill);

    // Flatten the symbolic pattern into CSR-style arrays shared by the whole batch.
    // Keeping it as vector<vector<int>>, and the per-batch values as
    // vector<vector<T>>, meant one heap allocation per row per batch element --
    // n * batch_size of them. That is invisible at batch 1 and is the dominant cost
    // once the batch loop below runs on several threads, since they all contend on
    // the same allocator.
    std::vector<int> sym_ro(static_cast<std::size_t>(n) + 1, 0);
    for (int i = 0; i < n; ++i) {
        sym_ro[static_cast<std::size_t>(i + 1)] =
            sym_ro[static_cast<std::size_t>(i)] + static_cast<int>(symbolic_rows[static_cast<std::size_t>(i)].size());
    }
    const int sym_nnz = sym_ro[static_cast<std::size_t>(n)];
    std::vector<int> sym_ci(static_cast<std::size_t>(sym_nnz));
    for (int i = 0; i < n; ++i) {
        std::copy(symbolic_rows[static_cast<std::size_t>(i)].begin(), symbolic_rows[static_cast<std::size_t>(i)].end(),
                  sym_ci.begin() + sym_ro[static_cast<std::size_t>(i)]);
    }

    // diag_local[i] is the diagonal's offset within row i of the symbolic pattern.
    std::vector<int> diag_local(static_cast<std::size_t>(n), -1);
    std::vector<int> original_row_nnz(static_cast<std::size_t>(n), 0);
    for (int i = 0; i < n; ++i) {
        original_row_nnz[static_cast<std::size_t>(i)] = ro[i + 1] - ro[i];
        const int rs = sym_ro[static_cast<std::size_t>(i)];
        const int re = sym_ro[static_cast<std::size_t>(i + 1)];
        int pos = -1;
        for (int p = rs; p < re; ++p) {
            if (sym_ci[static_cast<std::size_t>(p)] == i) { pos = p - rs; break; }
        }
        if (pos < 0) {
            throw std::runtime_error("ILU(k): symbolic phase produced a row without diagonal");
        }
        diag_local[static_cast<std::size_t>(i)] = pos;
    }

    UnifiedVector<int32_t> factor_status(static_cast<std::size_t>(batch_size), 0);

    const std::size_t stride = static_cast<std::size_t>(sym_nnz);
    std::vector<T> sym_values(stride * static_cast<std::size_t>(batch_size), T(0));
    std::vector<uint8_t> sym_keep(stride * static_cast<std::size_t>(batch_size), 1);

    const auto a_vals = A.data();

    // Batch elements share a sparsity pattern but nothing else: each one's numeric
    // phase reads only its own slice of A and writes only its own slice of
    // sym_values, so the batch loop is embarrassingly parallel. It is also the part
    // that scales with batch size -- at batch 1024 it dominated everything else the
    // solver did.
    //
    // `col_to_slot` and `candidates` are per-thread scratch. col_to_slot maps a
    // column index to its slot in the row being assembled: an O(1) lookup in place
    // of a binary search per touched entry, with -1 meaning the column is outside
    // this row's pattern.
    auto factor_batch_element = [&](int b, std::vector<int>& col_to_slot,
                                    std::vector<std::pair<RealT<T>, int>>& candidates) {
        T* values = sym_values.data() + static_cast<std::size_t>(b) * stride;
        uint8_t* keep = sym_keep.data() + static_cast<std::size_t>(b) * stride;

        const int ro_base = b * A.offset_stride();
        const int val_base = b * A.matrix_stride();
        for (int i = 0; i < n; ++i) {
            const int rs = sym_ro[static_cast<std::size_t>(i)];
            const int row_nnz = sym_ro[static_cast<std::size_t>(i + 1)] - rs;
            for (int p = 0; p < row_nnz; ++p) {
                values[rs + p] = T(0);
                keep[rs + p] = 1;
                col_to_slot[static_cast<std::size_t>(sym_ci[static_cast<std::size_t>(rs + p)])] = p;
            }

            const int ars = ro[ro_base + i];
            const int are = ro[ro_base + i + 1];
            for (int p = ars; p < are; ++p) {
                const int slot = col_to_slot[static_cast<std::size_t>(ci[val_base + p])];
                if (slot >= 0) values[rs + slot] = a_vals[val_base + p];
            }

            for (int p = 0; p < row_nnz; ++p) col_to_slot[static_cast<std::size_t>(sym_ci[static_cast<std::size_t>(rs + p)])] = -1;
        }

        for (int i = 0; i < n; ++i) {
            const int rs = sym_ro[static_cast<std::size_t>(i)];
            const int row_nnz = sym_ro[static_cast<std::size_t>(i + 1)] - rs;
            T* row_vals = values + rs;

            for (int p = 0; p < row_nnz; ++p) col_to_slot[static_cast<std::size_t>(sym_ci[static_cast<std::size_t>(rs + p)])] = p;

            for (int p = 0; p < row_nnz; ++p) {
                const int j = sym_ci[static_cast<std::size_t>(rs + p)];
                if (j >= i) break;

                const int prs = sym_ro[static_cast<std::size_t>(j)];
                const int pivot_nnz = sym_ro[static_cast<std::size_t>(j + 1)] - prs;
                const T* pivot_vals = values + prs;
                const uint8_t* pivot_keep = keep + prs;
                const int diag_j = diag_local[static_cast<std::size_t>(j)];

                // Row j < i is already finalized: its diagonal was stabilized when row j
                // was processed and the row has not changed since. Stabilization is
                // idempotent, so re-deriving the row scale here (an O(nnz_j) scan on every
                // elimination step) would recompute the value already stored.
                const T lij = row_vals[p] / pivot_vals[diag_j];
                row_vals[p] = lij;

                // Columns are sorted, so the strict upper part of row j starts past its diagonal.
                for (int q = diag_j + 1; q < pivot_nnz; ++q) {
                    if (pivot_keep[q] == 0) continue;
                    const int slot = col_to_slot[static_cast<std::size_t>(sym_ci[static_cast<std::size_t>(prs + q)])];
                    if (slot >= 0) row_vals[slot] -= lij * pivot_vals[q];
                }
            }

            const int diag_i = diag_local[static_cast<std::size_t>(i)];
            apply_drop_and_fill_control(row_vals, keep + rs, row_nnz, diag_i,
                                        original_row_nnz[static_cast<std::size_t>(i)], params, candidates);
            const auto final_scale = row_scale(row_vals, keep + rs, row_nnz);
            row_vals[diag_i] = stabilize_pivot_or_mark(row_vals[diag_i], final_scale, params,
                                                       &factor_status[static_cast<std::size_t>(b)]);

            for (int p = 0; p < row_nnz; ++p) col_to_slot[static_cast<std::size_t>(sym_ci[static_cast<std::size_t>(rs + p)])] = -1;
        }
    };

    {
        std::vector<int> col_to_slot(static_cast<std::size_t>(n), -1);
        std::vector<std::pair<RealT<T>, int>> candidates;
        for (int b = 0; b < batch_size; ++b) factor_batch_element(b, col_to_slot, candidates);
    }

    for (int b = 0; b < batch_size; ++b) {
        if (factor_status[static_cast<std::size_t>(b)] != 0) {
            throw std::runtime_error(
                "ILU(k): encountered a zero or effectively zero pivot without a usable diagonal shift");
        }
    }

    // Compaction: an entry survives if any batch element kept it, so the batch keeps
    // sharing one pattern. compact_pos maps a symbolic slot to its compacted slot.
    HostFactor<T> out;
    out.n = n;
    out.batch_size = batch_size;
    out.row_offsets.assign(static_cast<std::size_t>(n) + 1, 0);
    std::vector<int> compact_pos(static_cast<std::size_t>(sym_nnz), -1);

    for (int i = 0; i < n; ++i) {
        const int rs = sym_ro[static_cast<std::size_t>(i)];
        const int row_nnz = sym_ro[static_cast<std::size_t>(i + 1)] - rs;
        int kept = 0;
        for (int p = 0; p < row_nnz; ++p) {
            bool keep_entry = (p == diag_local[static_cast<std::size_t>(i)]);
            for (int b = 0; b < batch_size && !keep_entry; ++b) {
                if (sym_keep[static_cast<std::size_t>(b) * stride + static_cast<std::size_t>(rs + p)] != 0) keep_entry = true;
            }
            if (keep_entry) {
                compact_pos[static_cast<std::size_t>(rs + p)] = kept++;
                out.col_indices.push_back(sym_ci[static_cast<std::size_t>(rs + p)]);
            }
        }
        out.row_offsets[static_cast<std::size_t>(i + 1)] = out.row_offsets[static_cast<std::size_t>(i)] + kept;
    }
    out.nnz = static_cast<int>(out.col_indices.size());

    out.diag_offsets.assign(static_cast<std::size_t>(n), 0);
    for (int i = 0; i < n; ++i) {
        out.diag_offsets[static_cast<std::size_t>(i)] =
            out.row_offsets[static_cast<std::size_t>(i)] +
            compact_pos[static_cast<std::size_t>(sym_ro[static_cast<std::size_t>(i)] + diag_local[static_cast<std::size_t>(i)])];
    }

    out.values.assign(static_cast<std::size_t>(out.nnz) * static_cast<std::size_t>(batch_size), T(0));
    for (int b = 0; b < batch_size; ++b) {
        const std::size_t src_base = static_cast<std::size_t>(b) * stride;
        const std::size_t dst_base = static_cast<std::size_t>(b) * static_cast<std::size_t>(out.nnz);
        for (int i = 0; i < n; ++i) {
            const int rs = sym_ro[static_cast<std::size_t>(i)];
            const int row_nnz = sym_ro[static_cast<std::size_t>(i + 1)] - rs;
            for (int p = 0; p < row_nnz; ++p) {
                const int new_pos = compact_pos[static_cast<std::size_t>(rs + p)];
                if (new_pos < 0) continue;
                if (sym_keep[src_base + static_cast<std::size_t>(rs + p)] == 0 && p != diag_local[static_cast<std::size_t>(i)]) continue;
                out.values[dst_base + static_cast<std::size_t>(out.row_offsets[static_cast<std::size_t>(i)] + new_pos)] =
                    sym_values[src_base + static_cast<std::size_t>(rs + p)];
            }
        }
    }

    out.l = build_level_schedule(out.row_offsets, out.col_indices, n, /*lower=*/true);
    out.u = build_level_schedule(out.row_offsets, out.col_indices, n, /*lower=*/false);
    return out;
}

// Copy a host-computed factor into caller storage. Only used on the small-batch
// path, where the copy is proportional to a batch the host already walked.
template <typename T>
void write_host_factor(const HostFactor<T>& host,
                       int n,
                       int batch,
                       Span<T> values,
                       Span<int> col_indices,
                       Span<int> row_offsets,
                       Span<int> diag_positions,
                       int matrix_stride,
                       int offset_stride) {
    const int nnz = host.nnz;
    for (int b = 0; b < batch; ++b) {
        const std::size_t src_base = static_cast<std::size_t>(b) * static_cast<std::size_t>(nnz);
        const std::size_t val_base = static_cast<std::size_t>(b) * static_cast<std::size_t>(matrix_stride);
        const std::size_t ro_base = static_cast<std::size_t>(b) * static_cast<std::size_t>(offset_stride);
        for (int i = 0; i < n + 1; ++i) row_offsets[ro_base + static_cast<std::size_t>(i)] = host.row_offsets[static_cast<std::size_t>(i)];
        for (int p = 0; p < nnz; ++p) {
            col_indices[val_base + static_cast<std::size_t>(p)] = host.col_indices[static_cast<std::size_t>(p)];
            values[val_base + static_cast<std::size_t>(p)] = host.values[src_base + static_cast<std::size_t>(p)];
        }
        for (int i = 0; i < n; ++i) {
            diag_positions[static_cast<std::size_t>(b * n + i)] =
                static_cast<int>(val_base) + host.diag_offsets[static_cast<std::size_t>(i)];
        }
    }
}

// ---------------------------------------------------------------------------
// Device factorization
//
// The batch shares one sparsity pattern, so every *index* the numeric phase
// needs -- which slot each elimination reads, which slot it updates, which rows
// may run concurrently -- depends only on that pattern and can be computed once
// on the host regardless of batch size. What is left for the device is pure
// arithmetic over each batch element's values, which is what actually scales.
// ---------------------------------------------------------------------------

template <Backend B, typename T> struct ILUKSparsityCheckKernel;
template <Backend B, typename T> struct ILUKZeroKernel;
template <Backend B, typename T> struct ILUKScatterKernel;
template <Backend B, typename T> struct ILUKRowOffsetKernel;
template <Backend B, typename T> struct ILUKDiagPosKernel;
template <Backend B, typename T> struct ILUKEliminateKernel;
template <Backend B, typename T> struct ILUKUnionKernel;
template <Backend B, typename T> struct ILUKGatherKernel;
template <Backend B, typename T> struct ILUKDiagCheckKernel;

// Kernel-safe counterpart of stabilize_pivot_or_mark: reports an unusable pivot
// through `status` instead of throwing, since the host cannot catch an exception
// raised inside a kernel. The caller rechecks status once, after the queue drains.
template <typename T>
inline T stabilize_pivot_no_throw(const T& pivot,
                                  RealT<T> scale,
                                  const T& diagonal_shift,
                                  RealT<T> diag_pivot_threshold,
                                  int32_t* status) {
    const auto shift_mag = abs_value(diagonal_shift);
    const auto pivot_mag = abs_value(pivot);
    const auto one = RealT<T>(1);
    const auto scaled = diag_pivot_threshold * (scale > one ? scale : one);
    const auto threshold = shift_mag > scaled ? shift_mag : scaled;

    if (pivot_mag >= threshold && pivot_mag > RealT<T>(0)) return pivot;
    if (shift_mag > RealT<T>(0)) {
        const T shifted = pivot + diagonal_shift;
        if (abs_value(shifted) >= threshold && abs_value(shifted) > RealT<T>(0)) return shifted;
        if (shift_mag >= threshold) return diagonal_shift;
    }
    if (threshold > RealT<T>(0)) return real_scalar<T>(threshold);
    if (status != nullptr) *status = 1;
    return T(1);
}

// Everything about an ILU(k) factorization that is fixed by A's sparsity pattern.
// Building this costs the same as one batch element's structural walk no matter
// how large the batch is.
struct ILUKSymbolic {
    int n = 0;
    int sym_nnz = 0;
    int a_nnz = 0;
    std::vector<int> sym_ro;          // n + 1
    std::vector<int> sym_ci;          // sym_nnz
    std::vector<int> diag_abs;        // n, absolute slot of each row's diagonal
    std::vector<int> original_row_nnz;// n, row counts of A (drives the fill quota)
    std::vector<int> a_to_sym;        // a_nnz, where each entry of A lands
    std::vector<int> is_diag;         // sym_nnz

    // Elimination schedule. Row i performs steps [step_ptr[i], step_ptr[i+1]);
    // step e divides slot step_target[e] by the pivot row's diagonal at
    // step_pivot_diag[e], then applies updates [step_upd_ptr[e], step_upd_ptr[e+1]).
    std::vector<int> step_ptr;        // n + 1
    std::vector<int> step_target;
    std::vector<int> step_pivot_diag;
    std::vector<int> step_upd_ptr;    // steps + 1
    std::vector<int> upd_src;
    std::vector<int> upd_dst;

    // Rows within one level depend only on earlier levels, so a level's rows can
    // be eliminated concurrently across the whole batch.
    LevelSchedule levels;
};

ILUKSymbolic build_iluk_symbolic(const Span<int>& ro, const Span<int>& ci, int n, int a_nnz, int level_k) {
    ILUKSymbolic s;
    s.n = n;
    s.a_nnz = a_nnz;
    const auto rows = symbolic_iluk_pattern_single(ro, ci, n, 0, 0, level_k);

    s.sym_ro.assign(static_cast<std::size_t>(n) + 1, 0);
    for (int i = 0; i < n; ++i) {
        s.sym_ro[static_cast<std::size_t>(i + 1)] =
            s.sym_ro[static_cast<std::size_t>(i)] + static_cast<int>(rows[static_cast<std::size_t>(i)].size());
    }
    s.sym_nnz = s.sym_ro[static_cast<std::size_t>(n)];
    s.sym_ci.resize(static_cast<std::size_t>(s.sym_nnz));
    for (int i = 0; i < n; ++i) {
        std::copy(rows[static_cast<std::size_t>(i)].begin(), rows[static_cast<std::size_t>(i)].end(),
                  s.sym_ci.begin() + s.sym_ro[static_cast<std::size_t>(i)]);
    }

    s.diag_abs.assign(static_cast<std::size_t>(n), -1);
    s.original_row_nnz.assign(static_cast<std::size_t>(n), 0);
    s.is_diag.assign(static_cast<std::size_t>(s.sym_nnz), 0);
    std::vector<int> diag_local(static_cast<std::size_t>(n), -1);
    for (int i = 0; i < n; ++i) {
        s.original_row_nnz[static_cast<std::size_t>(i)] = ro[i + 1] - ro[i];
        const int rs = s.sym_ro[static_cast<std::size_t>(i)];
        const int re = s.sym_ro[static_cast<std::size_t>(i + 1)];
        for (int p = rs; p < re; ++p) {
            if (s.sym_ci[static_cast<std::size_t>(p)] == i) {
                s.diag_abs[static_cast<std::size_t>(i)] = p;
                diag_local[static_cast<std::size_t>(i)] = p - rs;
                s.is_diag[static_cast<std::size_t>(p)] = 1;
                break;
            }
        }
        if (s.diag_abs[static_cast<std::size_t>(i)] < 0) {
            throw std::runtime_error("ILU(k): symbolic phase produced a row without diagonal");
        }
    }

    // Where each entry of A lands in the symbolic pattern.
    s.a_to_sym.assign(static_cast<std::size_t>(a_nnz), -1);
    std::vector<int> col_to_slot(static_cast<std::size_t>(n), -1);
    for (int i = 0; i < n; ++i) {
        const int rs = s.sym_ro[static_cast<std::size_t>(i)];
        const int re = s.sym_ro[static_cast<std::size_t>(i + 1)];
        for (int p = rs; p < re; ++p) col_to_slot[static_cast<std::size_t>(s.sym_ci[static_cast<std::size_t>(p)])] = p;
        for (int p = ro[i]; p < ro[i + 1]; ++p) {
            s.a_to_sym[static_cast<std::size_t>(p)] = col_to_slot[static_cast<std::size_t>(ci[p])];
        }
        for (int p = rs; p < re; ++p) col_to_slot[static_cast<std::size_t>(s.sym_ci[static_cast<std::size_t>(p)])] = -1;
    }

    // Elimination schedule. The host path skips updates whose source was dropped;
    // dropped entries are set to zero, so applying them anyway is a no-op and the
    // schedule stays independent of the values.
    s.step_ptr.assign(static_cast<std::size_t>(n) + 1, 0);
    s.step_upd_ptr.push_back(0);
    for (int i = 0; i < n; ++i) {
        const int rs = s.sym_ro[static_cast<std::size_t>(i)];
        const int re = s.sym_ro[static_cast<std::size_t>(i + 1)];
        for (int p = rs; p < re; ++p) col_to_slot[static_cast<std::size_t>(s.sym_ci[static_cast<std::size_t>(p)])] = p;

        for (int p = rs; p < re; ++p) {
            const int j = s.sym_ci[static_cast<std::size_t>(p)];
            if (j >= i) break;
            s.step_target.push_back(p);
            s.step_pivot_diag.push_back(s.diag_abs[static_cast<std::size_t>(j)]);

            const int prs = s.sym_ro[static_cast<std::size_t>(j)];
            const int pre = s.sym_ro[static_cast<std::size_t>(j + 1)];
            for (int q = s.diag_abs[static_cast<std::size_t>(j)] + 1; q < pre; ++q) {
                const int dst = col_to_slot[static_cast<std::size_t>(s.sym_ci[static_cast<std::size_t>(q)])];
                if (dst >= 0) {
                    s.upd_src.push_back(q);
                    s.upd_dst.push_back(dst);
                }
            }
            (void)prs;
            s.step_upd_ptr.push_back(static_cast<int>(s.upd_src.size()));
        }
        s.step_ptr[static_cast<std::size_t>(i + 1)] = static_cast<int>(s.step_target.size());

        for (int p = rs; p < re; ++p) col_to_slot[static_cast<std::size_t>(s.sym_ci[static_cast<std::size_t>(p)])] = -1;
    }

    s.levels = build_level_schedule(s.sym_ro, s.sym_ci, n, /*lower=*/true);
    return s;
}

template <typename U>
UnifiedVector<U> to_device(const std::vector<U>& src) {
    UnifiedVector<U> dst(src.size());
    if (!src.empty()) std::copy(src.begin(), src.end(), dst.data());
    return dst;
}

// The symbolic index arrays are small but numerous, and one managed allocation
// each turned out to dominate the cost of factorizing a single small system.
// Packing them into one buffer keeps that fixed overhead to a single allocation.
struct PackedInts {
    UnifiedVector<int> buffer;
    std::vector<int> offsets;
    int tail_offset = 0;

    const int* at(std::size_t which) const { return buffer.data() + offsets[which]; }
    int* tail() { return buffer.data() + tail_offset; }
};

// `tail` reserves uninitialised trailing room in the same allocation, used for the
// compaction scratch and status flags so the whole factorization needs one integer
// buffer rather than one per array.
PackedInts pack_ints(std::initializer_list<const std::vector<int>*> arrays, std::size_t tail) {
    PackedInts packed;
    std::size_t total = 0;
    packed.offsets.reserve(arrays.size());
    for (const auto* a : arrays) {
        packed.offsets.push_back(static_cast<int>(total));
        total += a->size();
    }
    packed.tail_offset = static_cast<int>(total);
    packed.buffer = UnifiedVector<int>(total + tail == 0 ? 1 : total + tail);
    std::size_t cursor = 0;
    for (const auto* a : arrays) {
        if (!a->empty()) std::copy(a->begin(), a->end(), packed.buffer.data() + cursor);
        cursor += a->size();
    }
    return packed;
}

// Batch elements must share a pattern. Checking that on the host means reading
// every element's indices back across the bus, which is exactly the kind of
// batch-proportional host work this path exists to avoid.
template <Backend B, typename T>
void check_batch_sparsity_on_device(Queue& ctx, const MatrixView<T, MatrixFormat::CSR>& A) {
    const int batch = A.batch_size();
    if (batch <= 1) return;
    const int rows = A.rows();
    const int nnz = A.nnz();
    UnifiedVector<int32_t> mismatch(1, 0);
    auto kv = A.kernel_view();
    auto* flag = mismatch.data();
    const int per_element = rows + 1 + nnz;
    ctx->parallel_for<ILUKSparsityCheckKernel<B, T>>(
        sycl::range<1>(static_cast<size_t>(batch - 1) * per_element), [=](sycl::id<1> id) {
        const int linear = static_cast<int>(id[0]);
        const int b = linear / per_element + 1;
        const int k = linear % per_element;
        if (k <= rows) {
            if (kv.row_offsets_[b * kv.offset_stride_ + k] != kv.row_offsets_[k]) *flag = 1;
        } else {
            const int p = k - rows - 1;
            if (kv.col_indices_[b * kv.matrix_stride_ + p] != kv.col_indices_[p]) *flag = 1;
        }
    });
    ctx.wait_and_throw();
    if (mismatch[0] != 0) {
        throw std::invalid_argument(
            "ILU(k): heterogeneous batch sparsity is not supported; batches must share an identical CSR pattern");
    }
}

template <typename T>
struct DeviceFactorOut {
    int nnz = 0;
    std::vector<int> row_offsets;    // n + 1, shared by the batch
    std::vector<int> col_indices;    // nnz, shared by the batch
    std::vector<int> diag_offsets;   // n, offset of each diagonal within one element
    LevelSchedule l;
    LevelSchedule u;
    UnifiedVector<T> work;           // sym_nnz * batch, laid out slot-major
    // The single integer buffer: symbolic index arrays followed by the compaction
    // scratch (keep-union flags, compact_to_sym, compacted column indices, row
    // offsets, diagonal offsets) and the status flags.
    UnifiedVector<int> ints;
    int compact_offset = 0, col_offset = 0, row_offset = 0, diag_offset = 0;
    bool u_diagonals_usable = false;
};

// The working values are stored slot-major (`work[slot * batch + b]`) rather than
// batch-major. One work-item handles one (row, batch element) pair, so adjacent
// work-items differ in b -- slot-major makes those accesses contiguous. The final
// gather transposes into the CSR batch-major layout the apply kernel expects.
template <Backend B, typename T>
DeviceFactorOut<T> run_device_numeric(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::CSR>& A,
                                      const ILUKSymbolic& sym,
                                      const ILUKParams<T>& params) {
    const int n = sym.n;
    const int batch = A.batch_size();
    const int sym_nnz = sym.sym_nnz;

    // Layout of the reserved tail: keep-union flags, compact_to_sym, compacted column
    // indices (each at most sym_nnz), row offsets, diagonal offsets, then two status ints.
    const std::size_t ku_off = 0;
    const std::size_t c2s_off = ku_off + static_cast<std::size_t>(sym_nnz);
    const std::size_t ci_off = c2s_off + static_cast<std::size_t>(sym_nnz);
    const std::size_t ro_off = ci_off + static_cast<std::size_t>(sym_nnz);
    const std::size_t do_off = ro_off + static_cast<std::size_t>(n) + 1;
    const std::size_t status_off = do_off + static_cast<std::size_t>(n);
    auto packed = pack_ints({&sym.sym_ro, &sym.diag_abs, &sym.original_row_nnz, &sym.a_to_sym,
                                   &sym.step_ptr, &sym.step_target, &sym.step_pivot_diag,
                                   &sym.step_upd_ptr, &sym.upd_src, &sym.upd_dst,
                                   &sym.levels.rows, &sym.levels.level_ptr, &sym.is_diag},
                            status_off + 2);
    // Index map for `packed`: 0 sym_ro, 1 diag_abs, 2 original_row_nnz, 3 a_to_sym,
    // 4 step_ptr, 5 step_target, 6 step_pivot_diag, 7 step_upd_ptr, 8 upd_src,
    // 9 upd_dst, 10 levels.rows, 11 levels.level_ptr, 12 is_diag.

    DeviceFactorOut<T> out;
    out.work = UnifiedVector<T>(static_cast<std::size_t>(sym_nnz) * static_cast<std::size_t>(batch));
    // status[0] flags an unusable pivot during elimination, status[1] an unusable U
    // diagonal afterwards.
    int* status = packed.tail() + status_off;
    status[0] = 0;
    status[1] = 0;

    auto* work = out.work.data();
    auto* st = status;
    const auto akv = A.kernel_view();
    const auto* a_to_sym = packed.at(3);

    // Scatter A into the symbolic pattern; slots with no counterpart stay zero.
    const int a_nnz = sym.a_nnz;
    ctx->parallel_for<ILUKZeroKernel<B, T>>(
        sycl::range<1>(static_cast<size_t>(sym_nnz) * batch), [=](sycl::id<1> id) {
        work[id[0]] = T(0);
    });
    ctx.wait_and_throw();  // the zero fill must land before A is scattered over it
    ctx->parallel_for<ILUKScatterKernel<B, T>>(
        sycl::range<1>(static_cast<size_t>(a_nnz) * batch), [=](sycl::id<1> id) {
        const int linear = static_cast<int>(id[0]);
        const int p = linear / batch;
        const int b = linear % batch;
        const int slot = a_to_sym[p];
        if (slot >= 0) work[static_cast<size_t>(slot) * batch + b] = akv.data_[b * akv.matrix_stride_ + p];
    });

    const auto* sym_ro = packed.at(0);
    const auto* diag_abs = packed.at(1);
    const auto* orig_nnz = packed.at(2);
    const auto* step_ptr = packed.at(4);
    const auto* step_target = packed.at(5);
    const auto* step_pivot = packed.at(6);
    const auto* step_upd_ptr = packed.at(7);
    const auto* upd_src = packed.at(8);
    const auto* upd_dst = packed.at(9);
    const auto* level_rows = packed.at(10);

    const auto drop_tolerance = params.drop_tolerance;
    const auto fill_factor = params.fill_factor;
    const auto diag_pivot_threshold = params.diag_pivot_threshold;
    const T diagonal_shift = params.diagonal_shift;
    const bool modified_ilu = params.modified_ilu;

    // One kernel launch per dependency level. The launch count follows the
    // factor's depth, which is a property of the pattern, so it does not grow
    // with the batch -- the batch only makes each launch wider.
    // Level kernels must run in order, but that ordering is the queue's job. Waiting
    // on the host between levels cost one round trip per level -- 53 of them for a
    // 4096-row ILU(1) factor -- which is why an in-order queue is used here instead.
    // std::optional for the same reason as the syev/gesvd/ormqr dispatchers: the
    // Queue constructor builds a real sycl::queue, so declaring it by value pays
    // that construction on the common in-order path, which never uses it.
    std::optional<Queue> ordered;
    if (!ctx.in_order()) {
        ordered.emplace(ctx, /*in_order=*/true);
        Event dep = ctx.get_event();
        ordered->enqueue(dep);
    }
    Queue& q = ctx.in_order() ? ctx : *ordered;
    for (int lv = 0; lv < sym.levels.levels; ++lv) {
        const int begin = sym.levels.level_ptr[static_cast<std::size_t>(lv)];
        const int end = sym.levels.level_ptr[static_cast<std::size_t>(lv) + 1];
        const int count = end - begin;
        if (count <= 0) continue;
        q->parallel_for<ILUKEliminateKernel<B, T>>(
            sycl::range<1>(static_cast<size_t>(count) * batch), [=](sycl::id<1> id) {
            const int linear = static_cast<int>(id[0]);
            const int t = linear / batch;
            const int b = linear % batch;
            const int i = level_rows[begin + t];

            auto at = [&](int slot) -> T& { return work[static_cast<size_t>(slot) * batch + b]; };

            for (int e = step_ptr[i]; e < step_ptr[i + 1]; ++e) {
                const T lij = at(step_target[e]) / at(step_pivot[e]);
                at(step_target[e]) = lij;
                for (int u = step_upd_ptr[e]; u < step_upd_ptr[e + 1]; ++u) {
                    at(upd_dst[u]) -= lij * at(upd_src[u]);
                }
            }

            const int rs = sym_ro[i];
            const int re = sym_ro[i + 1];
            const int dg = diag_abs[i];

            RealT<T> scale = RealT<T>(0);
            for (int p = rs; p < re; ++p) {
                const auto m = abs_value(at(p));
                if (m > scale) scale = m;
            }
            const auto one = RealT<T>(1);
            const auto threshold = drop_tolerance * (scale > one ? scale : one);

            int kept = 0;
            for (int p = rs; p < re; ++p) {
                if (p == dg) continue;
                if (abs_value(at(p)) <= threshold) {
                    if (modified_ilu) at(dg) += at(p);
                    at(p) = T(0);
                } else {
                    ++kept;
                }
            }

            // Fill quota: keep the largest `quota` off-diagonal entries. Repeatedly
            // dropping the smallest survivor is equivalent to sorting and truncating,
            // and avoids needing per-row scratch in the kernel. With the default
            // fill_factor this loop does not run at all.
            const int quota_raw = static_cast<int>(sycl::ceil(fill_factor * static_cast<RealT<T>>(orig_nnz[i]))) - 1;
            const int quota = quota_raw > 0 ? quota_raw : 0;
            while (kept > quota) {
                int victim = -1;
                RealT<T> smallest = RealT<T>(0);
                for (int p = rs; p < re; ++p) {
                    if (p == dg) continue;
                    const auto m = abs_value(at(p));
                    if (m == RealT<T>(0)) continue;
                    // Smallest magnitude wins; on a tie the later column is dropped, which
                    // is the same survivor set the host path's tie-break produces.
                    if (victim < 0 || m < smallest || (m == smallest && p > victim)) {
                        victim = p;
                        smallest = m;
                    }
                }
                if (victim < 0) break;
                if (modified_ilu) at(dg) += at(victim);
                at(victim) = T(0);
                --kept;
            }

            RealT<T> final_scale = RealT<T>(0);
            for (int p = rs; p < re; ++p) {
                const auto m = abs_value(at(p));
                if (m > final_scale) final_scale = m;
            }
            at(dg) = stabilize_pivot_no_throw(at(dg), final_scale, diagonal_shift, diag_pivot_threshold, st);
        });
    }
    q.wait_and_throw();

    if (status[0] != 0) {
        throw std::runtime_error(
            "ILU(k): encountered a zero or effectively zero pivot without a usable diagonal shift");
    }

    // An entry survives compaction if any batch element kept it. Reducing over the
    // batch on the device leaves the host with sym_nnz flags to scan -- a figure
    // that does not depend on the batch size.
    // One scratch buffer covering keep_union, compact_to_sym, the compacted row
    // offsets, column indices and diagonal offsets. compact_to_sym cannot exceed
    // sym_nnz, so the whole layout is known before compaction runs.
    auto* ku = packed.tail() + ku_off;
    const auto* is_diag = packed.at(12);
    ctx->parallel_for<ILUKUnionKernel<B, T>>(sycl::range<1>(static_cast<size_t>(sym_nnz)), [=](sycl::id<1> id) {
        const size_t slot = id[0];
        int keep = is_diag[slot];
        for (int b = 0; b < batch && keep == 0; ++b) {
            // Dropped entries were zeroed, and any entry at or below the drop
            // threshold was dropped, so a non-zero value is exactly a kept one.
            if (abs_value(work[slot * batch + b]) != RealT<T>(0)) keep = 1;
        }
        ku[slot] = keep;
    });
    ctx.wait_and_throw();

    // Compaction is a scan over sym_nnz flags -- a figure fixed by the pattern, so
    // this host pass does not grow with the batch.
    int* c2s_w = packed.tail() + c2s_off;
    int* ci_w = packed.tail() + ci_off;
    int* ro_w = packed.tail() + ro_off;
    int* do_w = packed.tail() + do_off;
    out.row_offsets.assign(static_cast<std::size_t>(n) + 1, 0);
    out.diag_offsets.assign(static_cast<std::size_t>(n), 0);
    out.col_indices.reserve(static_cast<std::size_t>(sym_nnz));
    int cursor = 0;
    ro_w[0] = 0;
    for (int i = 0; i < n; ++i) {
        const int rs = sym.sym_ro[static_cast<std::size_t>(i)];
        const int re = sym.sym_ro[static_cast<std::size_t>(i + 1)];
        int kept = 0;
        for (int p = rs; p < re; ++p) {
            if (ku[p] == 0) continue;
            if (p == sym.diag_abs[static_cast<std::size_t>(i)]) {
                out.diag_offsets[static_cast<std::size_t>(i)] = out.row_offsets[static_cast<std::size_t>(i)] + kept;
            }
            c2s_w[cursor] = p;
            ci_w[cursor] = sym.sym_ci[static_cast<std::size_t>(p)];
            out.col_indices.push_back(sym.sym_ci[static_cast<std::size_t>(p)]);
            ++cursor;
            ++kept;
        }
        out.row_offsets[static_cast<std::size_t>(i + 1)] = out.row_offsets[static_cast<std::size_t>(i)] + kept;
        ro_w[i + 1] = out.row_offsets[static_cast<std::size_t>(i + 1)];
    }
    out.nnz = cursor;
    for (int i = 0; i < n; ++i) do_w[i] = out.diag_offsets[static_cast<std::size_t>(i)];

    out.l = build_level_schedule(out.row_offsets, out.col_indices, n, /*lower=*/true);
    out.u = build_level_schedule(out.row_offsets, out.col_indices, n, /*lower=*/false);

    // Whether every U diagonal can be shifted into usability, reduced on the device
    // so the host reads one flag rather than n * batch values.
    auto* df = status + 1;
    const auto* c2s = packed.tail() + c2s_off;
    const auto* diag_off = packed.tail() + do_off;
    ctx->parallel_for<ILUKDiagCheckKernel<B, T>>(
        sycl::range<1>(static_cast<size_t>(n) * batch), [=](sycl::id<1> id) {
        const int linear = static_cast<int>(id[0]);
        const int i = linear / batch;
        const int b = linear % batch;
        const int slot = c2s[diag_off[i]];
        int32_t local = 0;
        (void)stabilize_pivot_no_throw(work[static_cast<size_t>(slot) * batch + b], RealT<T>(0),
                                       diagonal_shift, RealT<T>(0), &local);
        if (local != 0) *df = 1;
    });
    ctx.wait_and_throw();
    out.u_diagonals_usable = (status[1] == 0);
    out.ints = std::move(packed.buffer);
    out.compact_offset = static_cast<int>(packed.tail_offset + c2s_off);
    out.col_offset = static_cast<int>(packed.tail_offset + ci_off);
    out.row_offset = static_cast<int>(packed.tail_offset + ro_off);
    out.diag_offset = static_cast<int>(packed.tail_offset + do_off);
    return out;
}

// Transpose the slot-major working values into the batch-major CSR the apply
// kernel reads, and fill in the replicated pattern and diagonal positions.
template <Backend B, typename T>
void write_factor_to_storage(Queue& ctx,
                             const DeviceFactorOut<T>& dev,
                             int n,
                             int batch,
                             Span<T> values,
                             Span<int> col_indices,
                             Span<int> row_offsets,
                             Span<int> diag_positions,
                             int matrix_stride,
                             int offset_stride) {
    const int nnz = dev.nnz;
    auto* vals = values.data();
    auto* ci = col_indices.data();
    auto* ro = row_offsets.data();
    auto* dp = diag_positions.data();
    const auto* c2s = dev.ints.data() + dev.compact_offset;
    const auto* src = dev.work.data();
    const auto* host_ro = dev.ints.data() + dev.row_offset;
    const auto* host_ci = dev.ints.data() + dev.col_offset;
    const auto* diag_off = dev.ints.data() + dev.diag_offset;

    ctx->parallel_for<ILUKGatherKernel<B, T>>(
        sycl::range<1>(static_cast<size_t>(nnz) * batch), [=](sycl::id<1> id) {
        const int linear = static_cast<int>(id[0]);
        const int q = linear / batch;
        const int b = linear % batch;
        vals[static_cast<size_t>(b) * matrix_stride + q] = src[static_cast<size_t>(c2s[q]) * batch + b];
        ci[static_cast<size_t>(b) * matrix_stride + q] = host_ci[q];
    });
    ctx->parallel_for<ILUKRowOffsetKernel<B, T>>(
        sycl::range<1>(static_cast<size_t>(n + 1) * batch), [=](sycl::id<1> id) {
        const int linear = static_cast<int>(id[0]);
        const int i = linear / batch;
        const int b = linear % batch;
        ro[static_cast<size_t>(b) * offset_stride + i] = host_ro[i];
    });
    ctx->parallel_for<ILUKDiagPosKernel<B, T>>(
        sycl::range<1>(static_cast<size_t>(n) * batch), [=](sycl::id<1> id) {
        const int linear = static_cast<int>(id[0]);
        const int i = linear / batch;
        const int b = linear % batch;
        dp[static_cast<size_t>(b) * n + i] = b * matrix_stride + diag_off[i];
    });
    ctx.wait_and_throw();
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
    validate_iluk_params_or_throw(A, params, /*check_batch_sparsity=*/false);
    check_batch_sparsity_on_device<B, T>(ctx, A);

    const int n = A.rows();
    const int batch_size = A.batch_size();

    if (!iluk_prefer_device(batch_size)) {
        const auto host = compute_iluk(A, params);
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
        result.l_rows = to_device(host.l.rows);
        result.l_level_ptr = to_device(host.l.level_ptr);
        result.l_levels = host.l.levels;
        result.u_rows = to_device(host.u.rows);
        result.u_level_ptr = to_device(host.u.level_ptr);
        result.u_levels = host.u.levels;
        auto host_view = result.lu.view();
        write_host_factor(host, n, batch_size, host_view.data(), host_view.col_indices(),
                          host_view.row_offsets(), result.diag_positions,
                          host_view.matrix_stride(), host_view.offset_stride());
        result.u_diagonals_usable = u_diagonals_are_usable(host_view.data().data(),
                                                           result.diag_positions.data(), n, batch_size,
                                                           result.diagonal_shift);
        return result;
    }

    const auto sym = build_iluk_symbolic(A.row_offsets(), A.col_indices(), n, A.nnz(), params.levels_of_fill);
    auto dev = run_device_numeric<B, T>(ctx, A, sym, params);

    ILUKPreconditioner<T> result;
    result.lu = Matrix<T, MatrixFormat::CSR>(n, n, dev.nnz, batch_size);
    result.diag_positions = UnifiedVector<int>(static_cast<std::size_t>(n) * static_cast<std::size_t>(batch_size));
    result.n = n;
    result.batch_size = batch_size;
    result.levels_of_fill = params.levels_of_fill;
    result.diagonal_shift = params.diagonal_shift;
    result.drop_tolerance = params.drop_tolerance;
    result.fill_factor = params.fill_factor;
    result.diag_pivot_threshold = params.diag_pivot_threshold;
    result.modified_ilu = params.modified_ilu;

    result.l_rows = to_device(dev.l.rows);
    result.l_level_ptr = to_device(dev.l.level_ptr);
    result.l_levels = dev.l.levels;
    result.u_rows = to_device(dev.u.rows);
    result.u_level_ptr = to_device(dev.u.level_ptr);
    result.u_levels = dev.u.levels;

    auto lu_view = result.lu.view();
    write_factor_to_storage<B, T>(ctx, dev, n, batch_size, lu_view.data(), lu_view.col_indices(),
                                  lu_view.row_offsets(), result.diag_positions,
                                  lu_view.matrix_stride(), lu_view.offset_stride());
    result.u_diagonals_usable = dev.u_diagonals_usable;
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
    validate_iluk_params_or_throw(A, params, /*check_batch_sparsity=*/false);
    check_batch_sparsity_on_device<B, T>(ctx, A);

    const int n = A.rows();
    const int batch_size = A.batch_size();
    const bool on_device = iluk_prefer_device(batch_size);

    std::optional<HostFactor<T>> host;
    std::optional<DeviceFactorOut<T>> dev_opt;
    if (on_device) {
        const auto sym = build_iluk_symbolic(A.row_offsets(), A.col_indices(), n, A.nnz(), params.levels_of_fill);
        dev_opt = run_device_numeric<B, T>(ctx, A, sym, params);
    } else {
        host = compute_iluk(A, params);
    }
    const int nnz = on_device ? dev_opt->nnz : host->nnz;
    const LevelSchedule& sched_l = on_device ? dev_opt->l : host->l;
    const LevelSchedule& sched_u = on_device ? dev_opt->u : host->u;

    auto pool = BumpAllocator(workspace);
    auto values = pool.allocate<T>(ctx, static_cast<size_t>(nnz) * static_cast<size_t>(batch_size));
    auto col_indices = pool.allocate<int>(ctx, static_cast<size_t>(nnz) * static_cast<size_t>(batch_size));
    auto row_offsets = pool.allocate<int>(ctx, static_cast<size_t>(n + 1) * static_cast<size_t>(batch_size));
    auto diag_positions = pool.allocate<int>(ctx, static_cast<size_t>(n) * static_cast<size_t>(batch_size));
    auto l_rows = pool.allocate<int>(ctx, static_cast<size_t>(n));
    auto u_rows = pool.allocate<int>(ctx, static_cast<size_t>(n));
    auto l_level_ptr = pool.allocate<int>(ctx, sched_l.level_ptr.size());
    auto u_level_ptr = pool.allocate<int>(ctx, sched_u.level_ptr.size());

    // The pattern is identical across the batch, but the apply kernel indexes
    // values and column indices with the same matrix stride, so both are
    // replicated per batch element rather than shared.
    if (on_device) {
        write_factor_to_storage<B, T>(ctx, *dev_opt, n, batch_size, values, col_indices, row_offsets,
                                      diag_positions, nnz, n + 1);
    } else {
        write_host_factor(*host, n, batch_size, values, col_indices, row_offsets, diag_positions, nnz, n + 1);
    }

    // The level schedules are a few n-sized arrays derived from the compacted
    // pattern, so copying them on the host costs nothing that grows with batch.
    for (int i = 0; i < n; ++i) {
        l_rows[static_cast<size_t>(i)] = sched_l.rows[static_cast<std::size_t>(i)];
        u_rows[static_cast<size_t>(i)] = sched_u.rows[static_cast<std::size_t>(i)];
    }
    for (std::size_t i = 0; i < sched_l.level_ptr.size(); ++i) l_level_ptr[i] = sched_l.level_ptr[i];
    for (std::size_t i = 0; i < sched_u.level_ptr.size(); ++i) u_level_ptr[i] = sched_u.level_ptr[i];

    ILUKView<T> view;
    view.lu = MatrixView<T, MatrixFormat::CSR>(values.data(), row_offsets.data(), col_indices.data(), nnz, n, n,
                                               nnz, n + 1, batch_size);
    view.diag_positions = diag_positions;
    view.l_rows = l_rows;
    view.l_level_ptr = l_level_ptr;
    view.l_levels = sched_l.levels;
    view.u_rows = u_rows;
    view.u_level_ptr = u_level_ptr;
    view.u_levels = sched_u.levels;
    view.n = n;
    view.batch_size = batch_size;
    view.diagonal_shift = params.diagonal_shift;
    view.u_diagonals_usable = on_device
        ? dev_opt->u_diagonals_usable
        : u_diagonals_are_usable(values.data(), diag_positions.data(), n, batch_size, params.diagonal_shift);
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
