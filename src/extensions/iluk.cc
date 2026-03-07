#include "../linalg-impl.hh"

#include <blas/functions/iluk.hh>
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

}  // namespace

template <Backend B, typename T>
ILUKPreconditioner<T> iluk_factorize(Queue& ctx,
                                     const MatrixView<T, MatrixFormat::CSR>& A,
                                     const ILUKParams<T>& params) {
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

    const int n = A.rows();
    const int batch_size = A.batch_size();

    if (!params.validate_batch_sparsity && batch_size > 1) {
        throw std::invalid_argument(
            "ILU(k): disabling batch sparsity validation is not supported; current implementation requires identical CSR sparsity across the batch");
    }
    if (batch_size > 1 && !has_identical_batch_sparsity(A)) {
        throw std::invalid_argument(
            "ILU(k): heterogeneous batch sparsity is not supported; batches must share an identical CSR pattern");
    }

    const auto ro = A.row_offsets();
    const auto ci = A.col_indices();
    auto symbolic_rows = symbolic_iluk_pattern_single(ro, ci, n, 0, 0, params.levels_of_fill);
    std::vector<int> diag_local(n, -1);
    std::vector<int> original_row_nnz(n, 0);
    for (int i = 0; i < n; ++i) {
        original_row_nnz[static_cast<std::size_t>(i)] = ro[i + 1] - ro[i];
        const int rs = 0;
        const int re = static_cast<int>(symbolic_rows[static_cast<std::size_t>(i)].size());
        int pos = -1;
        for (int p = rs; p < re; ++p) {
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

            const int ars = ro[ro_base + i];
            const int are = ro[ro_base + i + 1];
            for (int p = ars; p < are; ++p) {
                const int col = ci[val_base + p];
                const auto it = std::lower_bound(row_cols.begin(), row_cols.end(), col);
                if (it != row_cols.end() && *it == col) {
                    row_vals[static_cast<std::size_t>(it - row_cols.begin())] = a_vals[val_base + p];
                }
            }
        }

        for (int i = 0; i < n; ++i) {
            auto& row_vals = values_by_row[static_cast<std::size_t>(i)];
            auto& row_keep = keep_by_row[static_cast<std::size_t>(i)];
            const auto& row_cols = symbolic_rows[static_cast<std::size_t>(i)];

            for (int p = 0; p < static_cast<int>(row_cols.size()); ++p) {
                const int j = row_cols[static_cast<std::size_t>(p)];
                if (j >= i) break;

                auto& pivot_row_vals = values_by_row[static_cast<std::size_t>(j)];
                auto& pivot_row_keep = keep_by_row[static_cast<std::size_t>(j)];
                const auto pivot_scale = row_scale(pivot_row_vals, pivot_row_keep);
                const int diag_j = diag_local[static_cast<std::size_t>(j)];
                pivot_row_vals[static_cast<std::size_t>(diag_j)] = stabilize_pivot_or_mark(
                    pivot_row_vals[static_cast<std::size_t>(diag_j)], pivot_scale, params, &factor_status[static_cast<std::size_t>(b)]);

                const T lij = row_vals[static_cast<std::size_t>(p)] / pivot_row_vals[static_cast<std::size_t>(diag_j)];
                row_vals[static_cast<std::size_t>(p)] = lij;

                const auto& pivot_cols = symbolic_rows[static_cast<std::size_t>(j)];
                for (int q = 0; q < static_cast<int>(pivot_cols.size()); ++q) {
                    if (pivot_row_keep[static_cast<std::size_t>(q)] == 0) continue;
                    const int col = pivot_cols[static_cast<std::size_t>(q)];
                    if (col <= j) continue;

                    const auto it = std::lower_bound(row_cols.begin(), row_cols.end(), col);
                    if (it != row_cols.end() && *it == col) {
                        const auto target = static_cast<std::size_t>(it - row_cols.begin());
                        row_vals[target] -= lij * pivot_row_vals[static_cast<std::size_t>(q)];
                    }
                }
            }

            apply_drop_and_fill_control(row_vals, row_keep, diag_local[static_cast<std::size_t>(i)], original_row_nnz[static_cast<std::size_t>(i)], params);
            const auto final_scale = row_scale(row_vals, row_keep);
            row_vals[static_cast<std::size_t>(diag_local[static_cast<std::size_t>(i)])] = stabilize_pivot_or_mark(
                row_vals[static_cast<std::size_t>(diag_local[static_cast<std::size_t>(i)])], final_scale, params, &factor_status[static_cast<std::size_t>(b)]);
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
    std::vector<int> row_offsets(n + 1, 0);

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
        row_offsets[static_cast<std::size_t>(i + 1)] = row_offsets[static_cast<std::size_t>(i)] + static_cast<int>(compact_row.size());
    }

    std::vector<int> col_indices;
    col_indices.reserve(static_cast<std::size_t>(row_offsets.back()));
    for (int i = 0; i < n; ++i) {
        const auto& compact_row = compact_rows[static_cast<std::size_t>(i)];
        col_indices.insert(col_indices.end(), compact_row.begin(), compact_row.end());
    }

    const int nnz = static_cast<int>(col_indices.size());
    ILUKPreconditioner<T> result;
    result.lu = Matrix<T, MatrixFormat::CSR>(n, n, nnz, batch_size);
    result.diag_positions = UnifiedVector<int>(static_cast<std::size_t>(n * batch_size));
    result.n = n;
    result.batch_size = batch_size;
    result.levels_of_fill = params.levels_of_fill;
    result.diagonal_shift = params.diagonal_shift;
    result.drop_tolerance = params.drop_tolerance;
    result.fill_factor = params.fill_factor;
    result.diag_pivot_threshold = params.diag_pivot_threshold;
    result.modified_ilu = params.modified_ilu;

    auto lu_view = result.lu.view();
    auto lu_ro = lu_view.row_offsets();
    auto lu_ci = lu_view.col_indices();
    auto lu_vals = lu_view.data();
    for (int b = 0; b < batch_size; ++b) {
        const int ro_base = b * lu_view.offset_stride();
        const int val_base = b * lu_view.matrix_stride();
        for (int i = 0; i < n + 1; ++i) lu_ro[ro_base + i] = row_offsets[static_cast<std::size_t>(i)];
        for (int p = 0; p < nnz; ++p) {
            lu_ci[val_base + p] = col_indices[static_cast<std::size_t>(p)];
            lu_vals[val_base + p] = T(0);
        }

        for (int i = 0; i < n; ++i) {
            const auto& row_vals = batch_values[static_cast<std::size_t>(b)][static_cast<std::size_t>(i)];
            const auto& row_keep = batch_keep[static_cast<std::size_t>(b)][static_cast<std::size_t>(i)];
            const auto& compact_pos = compact_index[static_cast<std::size_t>(i)];
            for (int p = 0; p < static_cast<int>(row_vals.size()); ++p) {
                const int new_pos = compact_pos[static_cast<std::size_t>(p)];
                if (new_pos < 0) continue;
                if (row_keep[static_cast<std::size_t>(p)] == 0 && p != diag_local[static_cast<std::size_t>(i)]) continue;
                lu_vals[val_base + row_offsets[static_cast<std::size_t>(i)] + new_pos] = row_vals[static_cast<std::size_t>(p)];
            }
            result.diag_positions[static_cast<std::size_t>(b * n + i)] = val_base + row_offsets[static_cast<std::size_t>(i)] + compact_pos[static_cast<std::size_t>(diag_local[static_cast<std::size_t>(i)])];
        }
    }

    return result;
}

template <Backend B, typename T>
Event iluk_apply(Queue& ctx,
                 const ILUKPreconditioner<T>& M,
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
    UnifiedVector<int32_t> apply_status(total_systems, 0);
    auto apply_status_ptr = apply_status.data();
    ctx->parallel_for<ILUKApplyKernel<B, T>>(sycl::range<1>(total_systems), [=](sycl::id<1> id) {
        const int linear = static_cast<int>(id[0]);
        const int b = linear / nrhs;
        const int col = linear % nrhs;

        const int ro_base = b * lu_kv.offset_stride_;
        const int lu_base = b * lu_kv.matrix_stride_;
        const int rhs_base = b * rhs_kv.stride_ + col * rhs_kv.ld_;
        const int out_base = b * out_kv.stride_ + col * out_kv.ld_;
        // Forward solve into out (as temporary y).
        for (int i = 0; i < n; ++i) {
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

        for (int i = n - 1; i >= 0; --i) {
            T sum = out_data[out_base + i];
            const int rs = lu_ro[ro_base + i];
            const int re = lu_ro[ro_base + i + 1];
            T diag = stabilize_pivot_or_mark(lu_vals[diag_pos[b * n + i]],
                                             diag_shift,
                                             &apply_status_ptr[linear]);
            for (int p = rs; p < re; ++p) {
                const int j = lu_ci[lu_base + p];
                if (j > i) {
                    sum -= lu_vals[lu_base + p] * out_data[out_base + j];
                }
            }
            out_data[out_base + i] = sum / diag;
        }
    });

    ctx.wait_and_throw();

    for (size_t idx = 0; idx < total_systems; ++idx) {
        if (apply_status[idx] != 0) {
            throw std::runtime_error(
                "ILU(k) apply: encountered a zero or effectively zero U diagonal without a usable diagonal shift");
        }
    }

    return ctx.get_event();
}

template <Backend B, typename T>
size_t iluk_apply_buffer_size(Queue&, const ILUKPreconditioner<T>&, const MatrixView<T, MatrixFormat::Dense>&, const MatrixView<T, MatrixFormat::Dense>&) {
    return 0;
}

#define ILUK_INSTANTIATE(BACK, FP) \
    template ILUKPreconditioner<FP> iluk_factorize<BACK, FP>(Queue&, const MatrixView<FP, MatrixFormat::CSR>&, const ILUKParams<FP>&); \
    template Event iluk_apply<BACK, FP>(Queue&, const ILUKPreconditioner<FP>&, const MatrixView<FP, MatrixFormat::Dense>&, const MatrixView<FP, MatrixFormat::Dense>&, Span<std::byte>); \
    template size_t iluk_apply_buffer_size<BACK, FP>(Queue&, const ILUKPreconditioner<FP>&, const MatrixView<FP, MatrixFormat::Dense>&, const MatrixView<FP, MatrixFormat::Dense>&);

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
