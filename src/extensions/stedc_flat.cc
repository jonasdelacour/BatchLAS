#include <algorithm>
#include <limits>
#include <stdexcept>

#include <blas/matrix.hh>
#include <blas/functions.hh>
#include <blas/extensions.hh>
#include <util/mempool.hh>
#include <util/sycl-local-accessor-helpers.hh>
#include <internal/sort.hh>
#include <batchlas/backend_config.h>
#include <batchlas/tuning_params.hh>

#include "../math-helpers.hh"
#include "../util/kernel-trace.hh"
#include "../util/template-instantiations.hh"
#include "stedc_flattened.hh"
#include "stedc_merge_kernels.hh"
#include "stedc_secular.hh"

namespace batchlas {

namespace {

template <Backend B, typename T>
inline StedcParams<T> resolve_flat_tuning(int64_t n, StedcParams<T> params) {
    const int32_t nn = static_cast<int32_t>(n);

    if constexpr (B != Backend::NETLIB) {
        if (params.recursion_threshold <= 0) {
            params.recursion_threshold = tuning::stedc_recursion_threshold_for_n(nn);
        }
        if (params.merge_variant == StedcMergeVariant::Auto) {
            params.merge_variant = static_cast<StedcMergeVariant>(tuning::stedc_merge_variant_for_n(nn));
        }
        if (params.secular_threads_per_root <= 0) {
            params.secular_threads_per_root = tuning::stedc_threads_per_root_for_n(nn);
        }
        if (params.secular_cta_wg_size_multiplier <= 0) {
            params.secular_cta_wg_size_multiplier = tuning::stedc_wg_multiplier_for_n(nn);
        }
    } else {
        params.secular_solver = StedcSecularSolver::Legacy;
        params.merge_variant = StedcMergeVariant::Baseline;
    }

    const int64_t safe_n = std::max<int64_t>(1, n);
    params.recursion_threshold = std::max<int64_t>(1, std::min<int64_t>(params.recursion_threshold, safe_n));
    return params;
}

template <typename T>
T tail_sentinel_from_sorted(const VectorView<T>& values, int32_t logical_n, int32_t bid) {
    if (logical_n <= 0) {
        return T(1);
    }
    return values(logical_n - 1, bid) + T(1);
}

template <typename T>
MatrixView<T> make_level_matrix_view(Span<T> storage, const FlatLevel& level, std::size_t batch_size) {
    const int32_t cap = level.capacity_n;
    const int32_t super_batch = static_cast<int32_t>(flat_level_super_batch_size(level, batch_size));
    return MatrixView<T>(storage.data(), cap, cap, cap, cap * cap, super_batch);
}

template <typename T>
VectorView<T> make_level_vector_view(Span<T> storage, const FlatLevel& level, std::size_t batch_size) {
    const int32_t cap = level.capacity_n;
    const int32_t super_batch = static_cast<int32_t>(flat_level_super_batch_size(level, batch_size));
    return VectorView<T>(storage.data(), cap, super_batch, 1, cap);
}

template <typename T>
VectorView<T> make_level_offdiag_view(Span<T> storage, const FlatLevel& level, std::size_t batch_size) {
    const int32_t cap = std::max<int32_t>(1, level.capacity_n - 1);
    const int32_t super_batch = static_cast<int32_t>(flat_level_super_batch_size(level, batch_size));
    return VectorView<T>(storage.data(), cap, super_batch, 1, cap);
}

void fill_level_metadata(const FlatLevel& level,
                         std::size_t batch_size,
                         Span<int32_t> logical_n,
                         Span<int32_t> left_n,
                         Span<int32_t> offsets,
                         Span<int32_t> boundaries) {
    const std::size_t super_batch = flat_level_super_batch_size(level, batch_size);
    for (std::size_t slot = 0; slot < super_batch; ++slot) {
        const std::size_t node_ix = slot / batch_size;
        const FlatNode& node = level.nodes[node_ix];
        logical_n[slot] = node.logical_n;
        left_n[slot] = node.left_n;
        offsets[slot] = node.offset;
        boundaries[slot] = node.left_n > 0 ? (node.offset + node.left_n - 1) : -1;
    }
}

template <typename T>
class StedcFlatLeafPack;
template <typename T>
class StedcFlatParentPack;
template <typename T>
class StedcFlatBuildV;
template <typename T>
class StedcFlatDeflation;
template <typename T>
class StedcFlatBaselineSecularSolve;
template <typename T>
class StedcFlatBaselineRescaleV;
template <typename T>
class StedcFlatBaselineMatrixUpdate;
template <typename T>
class StedcFlatAssignEigenvalues;

template <typename T>
class StedcFlatInitPermMap;

template <typename T>
class StedcFlatInvertPermMap;

template <typename T>
void launch_init_perm_map(Queue& ctx,
                          const VectorView<int32_t>& perm_map,
                          Span<int32_t> logical_n) {
    ctx->submit([&](sycl::handler& h) {
        h.parallel_for<StedcFlatInitPermMap<T>>(sycl::nd_range<1>(perm_map.batch_size() * 128, 128), [=](sycl::nd_item<1> item) {
            const int32_t bid = static_cast<int32_t>(item.get_group_linear_id());
            const int32_t tid = static_cast<int32_t>(item.get_local_linear_id());
            const int32_t bdim = static_cast<int32_t>(item.get_local_range(0));
            const int32_t logical = logical_n[bid];
            for (int32_t i = tid; i < logical; i += bdim) {
                perm_map(i, bid) = i;
            }
            for (int32_t i = logical + tid; i < perm_map.size(); i += bdim) {
                perm_map(i, bid) = i;
            }
        });
    });
}

template <typename T>
void launch_invert_perm_map(Queue& ctx,
                            const VectorView<int32_t>& perm_map,
                            const VectorView<int32_t>& inverse_perm_map,
                            Span<int32_t> logical_n) {
    ctx->submit([&](sycl::handler& h) {
        h.parallel_for<StedcFlatInvertPermMap<T>>(sycl::nd_range<1>(perm_map.batch_size() * 128, 128), [=](sycl::nd_item<1> item) {
            const int32_t bid = static_cast<int32_t>(item.get_group_linear_id());
            const int32_t tid = static_cast<int32_t>(item.get_local_linear_id());
            const int32_t bdim = static_cast<int32_t>(item.get_local_range(0));
            const int32_t logical = logical_n[bid];
            for (int32_t i = tid; i < logical; i += bdim) {
                inverse_perm_map(perm_map(i, bid), bid) = i;
            }
            for (int32_t i = logical + tid; i < inverse_perm_map.size(); i += bdim) {
                inverse_perm_map(i, bid) = i;
            }
        });
    });
}

template <typename T>
void launch_leaf_pack(Queue& ctx,
                      const VectorView<T>& d,
                      const VectorView<T>& e,
                      const VectorView<T>& leaf_diag,
                      const VectorView<T>& leaf_offdiag,
                      Span<int32_t> logical_n,
                      Span<int32_t> offsets,
                      Span<int32_t> split_flags,
                      int32_t root_batch_size) {
    BATCHLAS_KERNEL_TRACE_SCOPE("stedc_flat:leaf_pack");
    ctx->submit([&](sycl::handler& h) {
        h.parallel_for<StedcFlatLeafPack<T>>(sycl::nd_range<1>(leaf_diag.batch_size() * 128, 128), [=](sycl::nd_item<1> item) {
            const int32_t bid = static_cast<int32_t>(item.get_group_linear_id());
            const int32_t tid = static_cast<int32_t>(item.get_local_linear_id());
            if (tid != 0) {
                return;
            }

            const int32_t logical = logical_n[bid];
            const int32_t offset = offsets[bid];
            const int32_t root_batch = bid % root_batch_size;
            T tail_base = T(1);

            for (int32_t i = 0; i < logical; ++i) {
                const int32_t gi = offset + i;
                T diag = d(gi, root_batch);
                T row_sum = T(0);
                if (gi > 0) {
                    row_sum += sycl::fabs(e(gi - 1, root_batch));
                    if (split_flags[gi - 1] != 0) {
                        diag -= sycl::fabs(e(gi - 1, root_batch));
                    }
                }
                if (gi < d.size() - 1) {
                    row_sum += sycl::fabs(e(gi, root_batch));
                    if (split_flags[gi] != 0) {
                        diag -= sycl::fabs(e(gi, root_batch));
                    }
                }
                leaf_diag(i, bid) = diag;
                tail_base = sycl::fmax(tail_base, diag + row_sum + T(1));
            }

            for (int32_t i = 0; i < logical - 1; ++i) {
                leaf_offdiag(i, bid) = e(offset + i, root_batch);
            }
            for (int32_t i = std::max<int32_t>(0, logical - 1); i < leaf_offdiag.size(); ++i) {
                leaf_offdiag(i, bid) = T(0);
            }

            for (int32_t i = logical; i < leaf_diag.size(); ++i) {
                leaf_diag(i, bid) = tail_base + T(i - logical);
            }
        });
    });
}

template <typename T>
void launch_parent_pack(Queue& ctx,
                        const MatrixView<T>& child_q,
                        const VectorView<T>& child_lambda,
                        const MatrixView<T>& parent_q,
                        const VectorView<T>& parent_lambda,
                        const VectorView<T>& e,
                        Span<int32_t> logical_n,
                        Span<int32_t> left_n,
                        Span<int32_t> boundaries,
                        Span<T> rho,
                        int32_t root_batch_size) {
    BATCHLAS_KERNEL_TRACE_SCOPE("stedc_flat:parent_pack");
    ctx->submit([&](sycl::handler& h) {
        auto child_q_view = child_q.kernel_view();
        auto parent_q_view = parent_q.kernel_view();
        const int32_t parent_rows = parent_q.rows();
        const int32_t parent_cols = parent_q.cols();
        h.parallel_for<StedcFlatParentPack<T>>(sycl::nd_range<1>(parent_lambda.batch_size() * 128, 128), [=](sycl::nd_item<1> item) {
            const int32_t bid = static_cast<int32_t>(item.get_group_linear_id());
            const int32_t tid = static_cast<int32_t>(item.get_local_linear_id());
            if (tid != 0) {
                return;
            }

            const int32_t logical = logical_n[bid];
            const int32_t left = left_n[bid];
            const int32_t right = logical - left;
            const int32_t node_ix = bid / root_batch_size;
            const int32_t root_batch = bid % root_batch_size;
            const int32_t left_child_bid = (node_ix * 2) * root_batch_size + root_batch;
            const int32_t right_child_bid = (node_ix * 2 + 1) * root_batch_size + root_batch;
            const int32_t boundary = boundaries[bid];

            auto left_q = child_q_view.batch_item(left_child_bid);
            auto right_q = child_q_view.batch_item(right_child_bid);
            auto dst_q = parent_q_view.batch_item(bid);

            rho[bid] = boundary >= 0 ? e(boundary, root_batch) : T(0);

            T tail_base = tail_sentinel_from_sorted(child_lambda, left, left_child_bid);
            tail_base = sycl::fmax(tail_base, tail_sentinel_from_sorted(child_lambda, right, right_child_bid));
            tail_base += T(1);

            for (int32_t col = 0; col < parent_cols; ++col) {
                for (int32_t row = 0; row < parent_rows; ++row) {
                    dst_q(row, col) = (row == col) ? T(1) : T(0);
                }
            }

            for (int32_t i = 0; i < left; ++i) {
                parent_lambda(i, bid) = child_lambda(i, left_child_bid);
            }
            for (int32_t i = 0; i < right; ++i) {
                parent_lambda(left + i, bid) = child_lambda(i, right_child_bid);
            }
            for (int32_t i = logical; i < parent_lambda.size(); ++i) {
                parent_lambda(i, bid) = tail_base + T(i - logical);
            }

            for (int32_t col = 0; col < left; ++col) {
                for (int32_t row = 0; row < left; ++row) {
                    dst_q(row, col) = left_q(row, col);
                }
            }
            for (int32_t col = 0; col < right; ++col) {
                for (int32_t row = 0; row < right; ++row) {
                    dst_q(left + row, left + col) = right_q(row, col);
                }
            }
        });
    });
}

template <typename T>
void launch_build_v(Queue& ctx,
                    const MatrixView<T>& parent_q,
                    const VectorView<T>& v,
                    Span<int32_t> logical_n,
                    Span<int32_t> left_n) {
    BATCHLAS_KERNEL_TRACE_SCOPE("stedc_flat:build_v");
    ctx->submit([&](sycl::handler& h) {
        auto q_view = parent_q.kernel_view();
        h.parallel_for<StedcFlatBuildV<T>>(sycl::nd_range<1>(v.batch_size() * 128, 128), [=](sycl::nd_item<1> item) {
            const int32_t bid = static_cast<int32_t>(item.get_group_linear_id());
            const int32_t tid = static_cast<int32_t>(item.get_local_linear_id());
            const int32_t bdim = static_cast<int32_t>(item.get_local_range(0));
            const int32_t logical = logical_n[bid];
            const int32_t left = left_n[bid];
            const int32_t right = logical - left;
            auto q_bid = q_view.batch_item(bid);

            for (int32_t i = tid; i < left; i += bdim) {
                v(i, bid) = q_bid(left - 1, i) / sycl::sqrt(T(2));
            }
            for (int32_t i = tid; i < right; i += bdim) {
                v(left + i, bid) = q_bid(left, left + i) / sycl::sqrt(T(2));
            }
            for (int32_t i = logical + tid; i < v.size(); i += bdim) {
                v(i, bid) = T(0);
            }
        });
    });
}

template <typename T>
void launch_deflation(Queue& ctx,
                      const VectorView<T>& eigenvalues,
                      const VectorView<T>& v,
                      const MatrixView<T>& q,
                      const VectorView<int32_t>& perm_map,
                      const VectorView<int32_t>& keep_indices,
                      Span<int32_t> logical_n,
                      Span<T> rho,
                      Span<int32_t> n_reduced) {
    BATCHLAS_KERNEL_TRACE_SCOPE("stedc_flat:deflation");
    const T reltol = T(64.0) * std::numeric_limits<T>::epsilon();
    ctx->submit([&](sycl::handler& h) {
        auto q_view = q.kernel_view();
        auto perm_local = sycl::local_accessor<int32_t, 1>(sycl::range<1>(eigenvalues.size()), h);
        auto scan_mem_include = sycl::local_accessor<int32_t, 1>(sycl::range<1>(eigenvalues.size()), h);
        auto scan_mem_exclude = sycl::local_accessor<int32_t, 1>(sycl::range<1>(eigenvalues.size()), h);
        auto norm_mem = sycl::local_accessor<T, 1>(sycl::range<1>(eigenvalues.size()), h);
        h.parallel_for<StedcFlatDeflation<T>>(sycl::nd_range<1>(eigenvalues.batch_size() * 128, 128), [=](sycl::nd_item<1> item) {
            const int32_t bid = static_cast<int32_t>(item.get_group_linear_id());
            const int32_t tid = static_cast<int32_t>(item.get_local_linear_id());
            const int32_t bdim = static_cast<int32_t>(item.get_local_range(0));
            const auto cta = item.get_group();

            const int32_t logical = logical_n[bid];
            auto q_bid = q_view.batch_item(bid);

            for (int32_t i = tid; i < keep_indices.size(); i += bdim) {
                keep_indices(i, bid) = 0;
                if (i < logical) {
                    perm_local[i] = perm_map(i, bid);
                    scan_mem_exclude[i] = 0;
                }
            }
            sycl::group_barrier(cta);

            for (int32_t j = 0; j < logical - 1; ++j) {
                const int32_t sorted_j = perm_local[j];
                const int32_t sorted_j1 = perm_local[j + 1];
                const T a = eigenvalues(sorted_j, bid);
                const T b = eigenvalues(sorted_j1, bid);
                const T gap = sycl::fabs(b - a);
                const T scale = reltol * sycl::fmax(T(1), sycl::fmax(sycl::fabs(a), sycl::fabs(b)));
                if (gap <= scale) {
                    auto [c, s, r] = internal::lartg(v(sorted_j1, bid), v(sorted_j, bid));
                    sycl::group_barrier(cta);
                    if (tid == 0) {
                        v(sorted_j, bid) = T(0);
                        v(sorted_j1, bid) = r;
                    }
                    const int32_t pj = sorted_j;
                    const int32_t pj1 = sorted_j1;
                    if (pj < 0 || pj >= logical || pj1 < 0 || pj1 >= logical) {
                        continue;
                    }
                    for (int32_t row = tid; row < logical; row += bdim) {
                        const T qi = q_bid(row, pj);
                        const T qj = q_bid(row, pj1);
                        q_bid(row, pj) = qi * c - qj * s;
                        q_bid(row, pj1) = qj * c + qi * s;
                    }
                }
            }
            sycl::group_barrier(cta);

            for (int32_t i = tid; i < logical; i += bdim) {
                norm_mem[i] = sycl::fabs(eigenvalues(perm_local[i], bid));
            }
            auto eig_max = sycl::joint_reduce(cta,
                                              util::get_raw_ptr(norm_mem),
                                              util::get_raw_ptr(norm_mem) + logical,
                                              sycl::maximum<T>());

            auto v_norm = internal::nrm2<T>(cta, v);
            for (int32_t i = tid; i < logical; i += bdim) {
                norm_mem[i] = sycl::fabs(v(perm_local[i], bid) / v_norm);
            }
            auto v_max = sycl::joint_reduce(cta,
                                            util::get_raw_ptr(norm_mem),
                                            util::get_raw_ptr(norm_mem) + logical,
                                            sycl::maximum<T>());
            auto tol = T(8.0) * std::numeric_limits<T>::epsilon() * sycl::fmax(eig_max, v_max);

            for (int32_t i = tid; i < logical; i += bdim) {
                if (sycl::fabs(rho[bid] * norm_mem[i]) > tol) {
                    keep_indices(i, bid) = 1;
                } else {
                    scan_mem_exclude[i] = 1;
                }
            }

            sycl::group_barrier(cta);

            sycl::joint_exclusive_scan(cta,
                                       keep_indices.batch_item(bid).data_ptr(),
                                       keep_indices.batch_item(bid).data_ptr() + logical,
                                       util::get_raw_ptr(scan_mem_include),
                                       0,
                                       sycl::plus<int32_t>());
            sycl::joint_exclusive_scan(cta,
                                       util::get_raw_ptr(scan_mem_exclude),
                                       util::get_raw_ptr(scan_mem_exclude) + logical,
                                       util::get_raw_ptr(scan_mem_exclude),
                                       0,
                                       sycl::plus<int32_t>());

            for (int32_t i = tid; i < logical; i += bdim) {
                if (keep_indices(i, bid) == 1) {
                    perm_map(scan_mem_include[i], bid) = perm_local[i];
                } else {
                    perm_map(logical - 1 - scan_mem_exclude[i], bid) = perm_local[i];
                }
            }

            if (tid == 0) {
                n_reduced[bid] = scan_mem_include[logical - 1] + keep_indices(logical - 1, bid);
            }
        });
    });
}

template <typename T>
void launch_baseline_merge(Queue& ctx,
                           const VectorView<T>& eigenvalues,
                           const VectorView<T>& v,
                           Span<T> rho,
                           Span<int32_t> n_reduced,
                           const MatrixView<T>& qprime,
                           const VectorView<T>& temp_lambdas,
                           const StedcParams<T>& params) {
    {
        BATCHLAS_KERNEL_TRACE_SCOPE("stedc_flat:merge_baseline.solve");
        ctx->submit([&](sycl::handler& h) {
            auto q_view = qprime.kernel_view();
            h.parallel_for<StedcFlatBaselineSecularSolve<T>>(sycl::nd_range<1>(eigenvalues.batch_size() * 128, 128), [=](sycl::nd_item<1> item) {
                const int32_t bid = static_cast<int32_t>(item.get_group_linear_id());
                const int32_t tid = static_cast<int32_t>(item.get_local_linear_id());
                const int32_t bdim = static_cast<int32_t>(item.get_local_range(0));
                const int32_t dd = n_reduced[bid];
                if (dd <= 0) {
                    return;
                }
                auto q_bid = q_view.batch_item(bid);
                const T sign = (rho[bid] >= T(0)) ? T(1) : T(-1);

                for (int32_t k = tid; k < dd * dd; k += bdim) {
                    const int32_t i = k % dd;
                    const int32_t j = k / dd;
                    q_bid(i, j) = eigenvalues(i, bid);
                }
                sycl::group_barrier(item.get_group());

                for (int32_t k = tid; k < dd; k += bdim) {
                    auto dview = VectorView<T>(q_bid.data() + k * q_bid.ld(), dd);
                    if (k == dd - 1) {
                        temp_lambdas(k, bid) = sec_solve_ext_roc(dd, dview, v.batch_item(bid), sycl::fabs(T(2) * rho[bid])) * sign;
                    } else {
                        temp_lambdas(k, bid) = sec_solve_roc(dd, dview, v.batch_item(bid), sycl::fabs(T(2) * rho[bid]), k) * sign;
                    }
                }
            });
        });
    }

    if (!params.enable_rescale) {
        return;
    }

    {
        BATCHLAS_KERNEL_TRACE_SCOPE("stedc_flat:merge_baseline.rescale");
        ctx->submit([&](sycl::handler& h) {
            auto q_view = qprime.kernel_view();
            h.parallel_for<StedcFlatBaselineRescaleV<T>>(sycl::nd_range<1>(eigenvalues.batch_size() * 128, 128), [=](sycl::nd_item<1> item) {
                const int32_t bid = static_cast<int32_t>(item.get_group_linear_id());
                const int32_t tid = static_cast<int32_t>(item.get_local_linear_id());
                if (tid != 0) {
                    return;
                }
                const int32_t dd = n_reduced[bid];
                if (dd <= 0) {
                    return;
                }
                auto q_bid = q_view.batch_item(bid);
                for (int32_t eig = 0; eig < dd; ++eig) {
                    const T Di = eigenvalues(eig, bid);
                    T valf = T(1);
                    for (int32_t j = 0; j < dd; ++j) {
                        valf *= (j == eig) ? q_bid(eig, j) : q_bid(eig, j) / (Di - eigenvalues(j, bid));
                    }
                    const T mag = sycl::sqrt(sycl::fabs(valf));
                    const T sgn = (v(eig, bid) >= T(0)) ? T(1) : T(-1);
                    v(eig, bid) = sgn * mag;
                }
            });
        });
    }
}

template <typename T>
void launch_baseline_matrix_update(Queue& ctx,
                                   const VectorView<T>& v,
                                   Span<int32_t> n_reduced,
                                   const MatrixView<T>& qprime) {
    BATCHLAS_KERNEL_TRACE_SCOPE("stedc_flat:merge_baseline.update");
    ctx->submit([&](sycl::handler& h) {
        auto q_view = qprime.kernel_view();
        h.parallel_for<StedcFlatBaselineMatrixUpdate<T>>(sycl::nd_range<1>(qprime.batch_size() * 128, 128), [=](sycl::nd_item<1> item) {
            const int32_t bid = static_cast<int32_t>(item.get_group_linear_id());
            const int32_t tid = static_cast<int32_t>(item.get_local_linear_id());
            if (tid != 0) {
                return;
            }
            const int32_t dd = n_reduced[bid];
            if (dd <= 0) {
                return;
            }
            auto q_bid = q_view.batch_item(bid);
            for (int32_t eig = 0; eig < dd; ++eig) {
                T norm_sq = T(0);
                for (int32_t i = 0; i < dd; ++i) {
                    q_bid(i, eig) = v(i, bid) / q_bid(i, eig);
                    norm_sq += q_bid(i, eig) * q_bid(i, eig);
                }
                const T norm = sycl::sqrt(sycl::fmax(norm_sq, std::numeric_limits<T>::epsilon()));
                for (int32_t i = 0; i < dd; ++i) {
                    q_bid(i, eig) /= norm;
                }
            }
        });
    });
}

template <typename T>
void launch_assign_temp_lambdas(Queue& ctx,
                                const VectorView<T>& eigenvalues,
                                const VectorView<T>& temp_lambdas,
                                Span<int32_t> n_reduced) {
    BATCHLAS_KERNEL_TRACE_SCOPE("stedc_flat:assign_lambdas");
    ctx->submit([&](sycl::handler& h) {
        h.parallel_for<StedcFlatAssignEigenvalues<T>>(sycl::nd_range<1>(eigenvalues.batch_size() * 128, 128), [=](sycl::nd_item<1> item) {
            const int32_t bid = static_cast<int32_t>(item.get_group_linear_id());
            const int32_t tid = static_cast<int32_t>(item.get_local_linear_id());
            if (tid != 0) {
                return;
            }
            const int32_t dd = n_reduced[bid];
            for (int32_t i = 0; i < dd; ++i) {
                eigenvalues(i, bid) = temp_lambdas(i, bid);
            }
        });
    });
}

template <Backend B, typename T>
bool flat_merge_depth(Queue& ctx,
                       const FlatLevel& parent_level,
                       const FlatLevel& child_level,
                       std::size_t batch_size,
                       const VectorView<T>& original_e,
                       const MatrixView<T>& child_q,
                       const MatrixView<T>& q_output,
                       const VectorView<T>& child_lambda,
                       const MatrixView<T>& parent_q_input,
                       const VectorView<T>& parent_lambda,
                       const MatrixView<T>& qprime,
                       const VectorView<T>& temp_lambdas,
                       const VectorView<T>& v,
                       const VectorView<int32_t>& perm_map,
                       const VectorView<int32_t>& keep_indices,
                       const VectorView<int32_t>& permutation,
                       Span<int32_t> logical_n_mem,
                       Span<int32_t> left_n_mem,
                       Span<int32_t> offset_mem,
                       Span<int32_t> boundary_mem,
                       Span<int32_t> n_reduced,
                       Span<T> rho,
                       const StedcParams<T>& params) {
    const std::size_t super_batch = flat_level_super_batch_size(parent_level, batch_size);
    auto logical_n = logical_n_mem.subspan(0, super_batch);
    auto left_n = left_n_mem.subspan(0, super_batch);
    auto offsets = offset_mem.subspan(0, super_batch);
    auto boundaries = boundary_mem.subspan(0, super_batch);
    auto n_reduced_current = n_reduced.subspan(0, super_batch);
    auto rho_current = rho.subspan(0, super_batch);
    auto merge_params = params;

    fill_level_metadata(parent_level, batch_size, logical_n, left_n, offsets, boundaries);

    launch_parent_pack(ctx,
                       child_q,
                       child_lambda,
                       parent_q_input,
                       parent_lambda,
                       original_e,
                       logical_n,
                       left_n,
                       boundaries,
                       rho_current,
                       static_cast<int32_t>(batch_size));
    launch_build_v(ctx, parent_q_input, v, logical_n, left_n);
    launch_init_perm_map<T>(ctx, perm_map, logical_n);

    {
        BATCHLAS_KERNEL_TRACE_SCOPE("stedc_flat:argsort_initial");
        argsort_active(ctx, parent_lambda, perm_map, logical_n, SortOrder::Ascending, true);
    }

    launch_deflation(ctx, parent_lambda, v, parent_q_input, perm_map, keep_indices, logical_n, rho_current, n_reduced_current);
    {
        BATCHLAS_KERNEL_TRACE_SCOPE("stedc_flat:materialize_lambda_deflate");
        permute_active(ctx, parent_lambda, perm_map, logical_n);
    }
    {
        BATCHLAS_KERNEL_TRACE_SCOPE("stedc_flat:materialize_v_deflate");
        permute_active(ctx, v, perm_map, logical_n);
    }

    qprime.fill_identity(ctx);
    if (merge_params.secular_solver == StedcSecularSolver::Legacy) {
        BATCHLAS_KERNEL_TRACE_SCOPE("stedc_flat:merge_legacy");
        secular_solver(ctx, parent_lambda, v, qprime, temp_lambdas, n_reduced_current, rho_current, T(10.0));
    } else if (merge_params.merge_variant == StedcMergeVariant::Baseline) {
        launch_baseline_merge(ctx, parent_lambda, v, rho_current, n_reduced_current, qprime, temp_lambdas, merge_params);
        launch_baseline_matrix_update(ctx, v, n_reduced_current, qprime);
    } else {
        BATCHLAS_KERNEL_TRACE_SCOPE(params.merge_variant == StedcMergeVariant::Fused ? "stedc_flat:merge_fused" : "stedc_flat:merge_fused_cta");
        stedc_merge_dispatch<B, T>(ctx, parent_lambda, v, rho_current, n_reduced_current, qprime, temp_lambdas, merge_params);
    }

    launch_assign_temp_lambdas(ctx, parent_lambda, temp_lambdas, n_reduced_current);
    {
        BATCHLAS_KERNEL_TRACE_SCOPE("stedc_flat:argsort_final");
        argsort_active(ctx, parent_lambda, permutation, logical_n, SortOrder::Ascending, true);
    }
    {
        BATCHLAS_KERNEL_TRACE_SCOPE("stedc_flat:permute_lambda_final");
        permute_active(ctx, parent_lambda, permutation, logical_n);
    }
    {
        BATCHLAS_KERNEL_TRACE_SCOPE("stedc_flat:update_q");
        MatrixView<T>::copy(ctx, q_output, qprime);
        launch_invert_perm_map<T>(ctx, perm_map, keep_indices, logical_n);
        permuted_copy_active_2d(ctx, q_output, qprime, keep_indices, permutation, logical_n);
        auto q_input_active = parent_q_input.with_active_dims(logical_n, logical_n);
        auto qprime_active = qprime.with_active_dims(logical_n, logical_n);
        auto q_output_active = q_output.with_active_dims(logical_n, logical_n);
        gemm<B>(ctx, q_input_active, qprime_active, q_output_active, T(1), T(0), Transpose::NoTrans, Transpose::NoTrans);
    }

    return true;
}

template <Backend B, typename T>
size_t flat_workspace_size_impl(Queue& ctx, std::size_t n, std::size_t batch_size, JobType jobz, StedcParams<T> params) {
    (void)jobz;
    if (n == 0 || batch_size == 0) {
        return 0;
    }

    params = resolve_flat_tuning<B, T>(static_cast<int64_t>(n), params);
    const auto schedule = build_flat_schedule(static_cast<int64_t>(n), params.recursion_threshold);

    std::size_t max_matrix_elems = 0;
    std::size_t max_vector_elems = 0;
    std::size_t max_super_batch = 0;
    for (const auto& level : schedule.levels) {
        max_matrix_elems = std::max(max_matrix_elems, flat_level_matrix_elements(level, batch_size));
        max_vector_elems = std::max(max_vector_elems, flat_level_vector_elements(level, batch_size));
        max_super_batch = std::max(max_super_batch, flat_level_super_batch_size(level, batch_size));
    }

    const auto& leaf_level = flat_leaf_level(schedule);
    const auto leaf_diag = VectorView<T>(nullptr, leaf_level.capacity_n, static_cast<int32_t>(flat_level_super_batch_size(leaf_level, batch_size)), 1, leaf_level.capacity_n);
    const auto leaf_offdiag = VectorView<T>(nullptr, std::max<int32_t>(1, leaf_level.capacity_n - 1), static_cast<int32_t>(flat_level_super_batch_size(leaf_level, batch_size)), 1, std::max<int32_t>(1, leaf_level.capacity_n - 1));
    const auto leaf_evals = VectorView<T>(nullptr, leaf_level.capacity_n, static_cast<int32_t>(flat_level_super_batch_size(leaf_level, batch_size)), 1, leaf_level.capacity_n);
    auto steqr_params = params.leaf_steqr_params;
    steqr_params.sort = true;
    steqr_params.sort_order = SortOrder::Ascending;
    const std::size_t leaf_ws = (leaf_level.capacity_n <= 32)
        ? steqr_cta_buffer_size<T>(ctx, leaf_diag, leaf_offdiag, leaf_evals, JobType::EigenVectors, steqr_params)
        : steqr_buffer_size<T>(ctx, leaf_diag, leaf_offdiag, leaf_evals, JobType::EigenVectors, steqr_params);

    std::size_t bytes = 0;
    bytes += BumpAllocator::allocation_size<T>(ctx, max_matrix_elems); // q_current
    bytes += BumpAllocator::allocation_size<T>(ctx, max_matrix_elems); // q_scratch
    bytes += BumpAllocator::allocation_size<T>(ctx, max_matrix_elems); // qprime
    bytes += BumpAllocator::allocation_size<T>(ctx, max_vector_elems); // lambda_current
    bytes += BumpAllocator::allocation_size<T>(ctx, max_vector_elems); // lambda_scratch
    bytes += BumpAllocator::allocation_size<T>(ctx, max_vector_elems); // leaf/merge offdiag
    bytes += BumpAllocator::allocation_size<T>(ctx, max_vector_elems); // v
    bytes += BumpAllocator::allocation_size<T>(ctx, max_vector_elems); // temp_lambdas
    bytes += BumpAllocator::allocation_size<int32_t>(ctx, max_vector_elems); // perm_map
    bytes += BumpAllocator::allocation_size<int32_t>(ctx, max_vector_elems); // keep_indices
    bytes += BumpAllocator::allocation_size<int32_t>(ctx, max_vector_elems); // permutation
    bytes += BumpAllocator::allocation_size<int32_t>(ctx, max_super_batch);  // logical_n
    bytes += BumpAllocator::allocation_size<int32_t>(ctx, max_super_batch);  // left_n
    bytes += BumpAllocator::allocation_size<int32_t>(ctx, max_super_batch);  // offsets
    bytes += BumpAllocator::allocation_size<int32_t>(ctx, max_super_batch);  // boundaries
    bytes += BumpAllocator::allocation_size<int32_t>(ctx, max_super_batch);  // n_reduced
    bytes += BumpAllocator::allocation_size<T>(ctx, max_super_batch);        // rho
    bytes += BumpAllocator::allocation_size<int32_t>(ctx, schedule.split_flags.size());
    bytes += BumpAllocator::allocation_size<std::byte>(ctx, leaf_ws);
    return bytes;
}

} // namespace

template <Backend B, typename T>
Event stedc_flat(Queue& ctx,
                 const VectorView<T>& d,
                 const VectorView<T>& e,
                 const VectorView<T>& eigenvalues,
                 const Span<std::byte>& ws,
                 JobType jobz,
                 StedcParams<T> params,
                 const MatrixView<T, MatrixFormat::Dense>& eigvects) {
    if constexpr (B == Backend::NETLIB) {
        return stedc<B, T>(ctx, d, e, eigenvalues, ws, jobz, params, eigvects);
    } else {
    if (d.size() != e.size() + 1) {
        throw std::runtime_error("The size of e must be one less than the size of d.");
    }
    if (d.size() != eigenvalues.size()) {
        throw std::runtime_error("The size of eigenvalues must match the size of d.");
    }
    if (d.batch_size() != e.batch_size() || d.batch_size() != eigenvalues.batch_size()) {
        throw std::runtime_error("The batch sizes of d, e, and eigenvalues must match.");
    }
    if (jobz == JobType::EigenVectors) {
        if (eigvects.rows() != d.size() || eigvects.cols() != d.size() || eigvects.batch_size() != d.batch_size()) {
            throw std::runtime_error("The dimensions of eigvects must match the size of d and its batch size.");
        }
    }

    if (jobz == JobType::NoEigenVectors) {
        return stedc<B, T>(ctx, d, e, eigenvalues, ws, jobz, params, eigvects);
    }

    auto effective_params = resolve_flat_tuning<B, T>(d.size(), params);
    const auto required_ws = stedc_flat_workspace_size<B, T>(ctx, static_cast<std::size_t>(d.size()), static_cast<std::size_t>(d.batch_size()), jobz, effective_params);
    if (ws.size() < required_ws) {
        throw std::runtime_error("stedc_flat: workspace is smaller than stedc_flat_workspace_size.");
    }

    const auto schedule = build_flat_schedule(d.size(), effective_params.recursion_threshold);
    if (schedule.levels.size() == 1) {
        auto steqr_params = effective_params.leaf_steqr_params;
        steqr_params.sort = true;
        steqr_params.sort_order = SortOrder::Ascending;
        return steqr<B, T>(ctx, d, e, eigenvalues, ws, jobz, steqr_params, eigvects);
    }

    auto pool = BumpAllocator(ws);

    std::size_t max_matrix_elems = 0;
    std::size_t max_vector_elems = 0;
    std::size_t max_super_batch = 0;
    for (const auto& level : schedule.levels) {
        max_matrix_elems = std::max(max_matrix_elems, flat_level_matrix_elements(level, d.batch_size()));
        max_vector_elems = std::max(max_vector_elems, flat_level_vector_elements(level, d.batch_size()));
        max_super_batch = std::max(max_super_batch, flat_level_super_batch_size(level, d.batch_size()));
    }

    auto q_current_mem = pool.allocate<T>(ctx, max_matrix_elems);
    auto q_scratch_mem = pool.allocate<T>(ctx, max_matrix_elems);
    auto qprime_mem = pool.allocate<T>(ctx, max_matrix_elems);
    auto lambda_current_mem = pool.allocate<T>(ctx, max_vector_elems);
    auto lambda_scratch_mem = pool.allocate<T>(ctx, max_vector_elems);
    auto offdiag_mem = pool.allocate<T>(ctx, max_vector_elems);
    auto v_mem = pool.allocate<T>(ctx, max_vector_elems);
    auto temp_lambda_mem = pool.allocate<T>(ctx, max_vector_elems);
    auto perm_map_mem = pool.allocate<int32_t>(ctx, max_vector_elems);
    auto keep_mem = pool.allocate<int32_t>(ctx, max_vector_elems);
    auto permutation_mem = pool.allocate<int32_t>(ctx, max_vector_elems);
    auto logical_n_mem = pool.allocate<int32_t>(ctx, max_super_batch);
    auto left_n_mem = pool.allocate<int32_t>(ctx, max_super_batch);
    auto offset_mem = pool.allocate<int32_t>(ctx, max_super_batch);
    auto boundary_mem = pool.allocate<int32_t>(ctx, max_super_batch);
    auto n_reduced_mem = pool.allocate<int32_t>(ctx, max_super_batch);
    auto rho_mem = pool.allocate<T>(ctx, max_super_batch);
    auto split_flag_mem = pool.allocate<int32_t>(ctx, schedule.split_flags.size());

    for (std::size_t i = 0; i < schedule.split_flags.size(); ++i) {
        split_flag_mem[i] = schedule.split_flags[i];
    }

    const auto& leaf_level = flat_leaf_level(schedule);
    fill_level_metadata(leaf_level, d.batch_size(), logical_n_mem, left_n_mem, offset_mem, boundary_mem);

    auto leaf_diag = make_level_vector_view(lambda_current_mem, leaf_level, d.batch_size());
    auto leaf_offdiag = make_level_offdiag_view(offdiag_mem, leaf_level, d.batch_size());
    auto leaf_q = make_level_matrix_view(q_current_mem, leaf_level, d.batch_size());

    launch_leaf_pack(ctx, d, e, leaf_diag, leaf_offdiag, logical_n_mem, offset_mem, split_flag_mem, d.batch_size());

    auto steqr_params = effective_params.leaf_steqr_params;
    steqr_params.sort = true;
    steqr_params.sort_order = SortOrder::Ascending;
    const std::size_t leaf_ws_bytes = (leaf_level.capacity_n <= 32)
        ? steqr_cta_buffer_size<T>(ctx, leaf_diag, leaf_offdiag, leaf_diag, JobType::EigenVectors, steqr_params)
        : steqr_buffer_size<T>(ctx, leaf_diag, leaf_offdiag, leaf_diag, JobType::EigenVectors, steqr_params);
    auto leaf_ws = pool.allocate<std::byte>(ctx, leaf_ws_bytes);

    {
        BATCHLAS_KERNEL_TRACE_SCOPE("stedc_flat:leaf_steqr");
        ctx->single_task([]() {});
    }
    if (leaf_level.capacity_n <= 32) {
        steqr_cta<B, T>(ctx, leaf_diag, leaf_offdiag, leaf_diag, leaf_ws, JobType::EigenVectors, steqr_params, leaf_q);
    } else {
        steqr<B, T>(ctx, leaf_diag, leaf_offdiag, leaf_diag, leaf_ws, JobType::EigenVectors, steqr_params, leaf_q);
    }

    // The first merge depth immediately consumes the leaf eigenpairs/eigenvectors.
    // Synchronize once here because the leaf solvers may complete outside simple queue-order assumptions.
    ctx.wait();

    for (std::size_t level_ix = schedule.levels.size() - 2; level_ix < schedule.levels.size(); --level_ix) {
        const auto& parent_level = schedule.levels[level_ix];
        const auto& child_level = schedule.levels[level_ix + 1];

        auto parent_q_input = make_level_matrix_view(q_scratch_mem, parent_level, d.batch_size());
        auto parent_lambda = make_level_vector_view(lambda_scratch_mem, parent_level, d.batch_size());
        auto qprime = make_level_matrix_view(qprime_mem, parent_level, d.batch_size());
        auto temp_lambdas = make_level_vector_view(temp_lambda_mem, parent_level, d.batch_size());
        auto v = make_level_vector_view(v_mem, parent_level, d.batch_size());
        auto perm_map = VectorView<int32_t>(perm_map_mem.data(), parent_level.capacity_n, static_cast<int32_t>(flat_level_super_batch_size(parent_level, d.batch_size())), 1, parent_level.capacity_n);
        auto keep_indices = VectorView<int32_t>(keep_mem.data(), parent_level.capacity_n, static_cast<int32_t>(flat_level_super_batch_size(parent_level, d.batch_size())), 1, parent_level.capacity_n);
        auto permutation = VectorView<int32_t>(permutation_mem.data(), parent_level.capacity_n, static_cast<int32_t>(flat_level_super_batch_size(parent_level, d.batch_size())), 1, parent_level.capacity_n);
        auto child_q = make_level_matrix_view(q_current_mem, child_level, d.batch_size());
        auto child_lambda = make_level_vector_view(lambda_current_mem, child_level, d.batch_size());
        auto q_output = make_level_matrix_view(q_current_mem, parent_level, d.batch_size());

        const bool result_stored_in_q_output = flat_merge_depth<B, T>(ctx,
                                          parent_level,
                                          child_level,
                                          d.batch_size(),
                                          e,
                                          child_q,
                                          q_output,
                                          child_lambda,
                                          parent_q_input,
                                          parent_lambda,
                                          qprime,
                                          temp_lambdas,
                                          v,
                                          perm_map,
                                          keep_indices,
                                          permutation,
                                          logical_n_mem,
                                          left_n_mem,
                                          offset_mem,
                                          boundary_mem,
                                          n_reduced_mem,
                                          rho_mem,
                                          effective_params);

        // The next iteration overwrites shared host-visible metadata spans reused by all levels.
        // Ensure the current level has finished consuming them before repacking the next level.
        ctx.wait();

        if (!result_stored_in_q_output) {
            // The generic path writes the merged Q into q_scratch_mem while q_current_mem is reused as
            // temporary storage for permutation steps. Swap the roles instead of copying the full matrix back.
            std::swap(q_current_mem, q_scratch_mem);
        }

        std::swap(lambda_current_mem, lambda_scratch_mem);

        if (level_ix == 0) {
            auto root_lambda = make_level_vector_view(lambda_current_mem, parent_level, d.batch_size());
            auto root_q = make_level_matrix_view(q_current_mem, parent_level, d.batch_size());
            VectorView<T>::copy(ctx, eigenvalues, root_lambda);
            if (jobz == JobType::EigenVectors) {
                MatrixView<T>::copy(ctx, eigvects, root_q);
            }
            break;
        }
    }

        return ctx.get_event();
    }
}

template <Backend B, typename T>
size_t stedc_flat_workspace_size(Queue& ctx, size_t n, size_t batch_size, JobType jobz, StedcParams<T> params) {
    if constexpr (B == Backend::NETLIB) {
        return stedc_workspace_size<B, T>(ctx, n, batch_size, jobz, params);
    } else {
        return flat_workspace_size_impl<B, T>(ctx, n, batch_size, jobz, params);
    }
}

#define STEDC_FLAT_INSTANTIATE(back, fp) \
template Event stedc_flat<back, BATCHLAS_UNPAREN fp>(Queue&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, const Span<std::byte>&, JobType, StedcParams<BATCHLAS_UNPAREN fp>, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&); \
template size_t stedc_flat_workspace_size<back, BATCHLAS_UNPAREN fp>(Queue&, size_t, size_t, JobType, StedcParams<BATCHLAS_UNPAREN fp>);

#define STEDC_FLAT_INSTANTIATE_FOR_BACKEND(back) \
    BATCHLAS_FOR_EACH_REAL_TYPE_1(STEDC_FLAT_INSTANTIATE, back)

#if BATCHLAS_HAS_HOST_BACKEND
STEDC_FLAT_INSTANTIATE_FOR_BACKEND(Backend::NETLIB)
#endif

#if BATCHLAS_HAS_CUDA_BACKEND
STEDC_FLAT_INSTANTIATE_FOR_BACKEND(Backend::CUDA)
#endif

#undef STEDC_FLAT_INSTANTIATE_FOR_BACKEND
#undef STEDC_FLAT_INSTANTIATE

} // namespace batchlas
