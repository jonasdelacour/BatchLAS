#include <vector>
#include <stdexcept>
#include <limits>
#include <blas/matrix.hh>
#include <blas/functions.hh>
#include <blas/extensions.hh>
#include <util/mempool.hh>
#include <internal/sort.hh>
#include <batchlas/backend_config.h>
#include "../math-helpers.hh"
#include "../util/kernel-trace.hh"

namespace batchlas {

// Forward declarations for secular solvers defined in stedc.cc
template <typename T>
SYCL_EXTERNAL T sec_solve_ext_roc(const int32_t dd, const VectorView<T>& D, const VectorView<T>& z, const T p);

template <typename T>
SYCL_EXTERNAL T sec_solve_roc(int32_t dd, const VectorView<T>& d, const VectorView<T>& z, const T& rho, const int32_t k);

// Simple block descriptor for the flattened STEDC schedule.
struct StedcBlock {
    size_t offset; // start index in the tridiagonal
    size_t size;   // block size
};

// Kernel configuration constants
namespace kernel_config {
    constexpr size_t BLOCK_SIZE_128 = 128;
    constexpr size_t BLOCK_SIZE_32 = 32;
    constexpr size_t GRID_MULT_128 = 128;  // multiplier for nd_range with 128 threads
    constexpr size_t GRID_MULT_32 = 32;    // multiplier for nd_range with 32 threads
}

// Helper for converting size_t to int64_t (used in Slice operations)
inline int64_t to_idx(size_t v) { return static_cast<int64_t>(v); }

// Helper to create slices for a range [offset, offset + n)
inline Slice make_slice(size_t offset, size_t size) {
    return Slice{to_idx(offset), to_idx(offset + size)};
}

// Merge two adjacent blocks [offset, offset+n) where n = n1+n2.
// This is adapted from the recursive stedc_impl merge step, without any recursion.
template <Backend B, typename T>
static Event stedc_merge_flat(
    Queue& ctx,
    size_t offset,
    size_t n,
    size_t n_left,
    const VectorView<T>& d,
    const VectorView<T>& e,
    const VectorView<T>& eigenvalues,
    const Span<std::byte>& ws,
    JobType jobz,
    StedcParams<T> params,
    const MatrixView<T, MatrixFormat::Dense>& eigvects,
    const MatrixView<T, MatrixFormat::Dense>& temp_Q)
{
    // Slices for the two halves (only for eigenvector blocks; not needed for d/e slices here)
    auto slice_left = make_slice(offset, n_left);
    auto slice_right = make_slice(offset + n_left, n - n_left);
    auto slice_all = make_slice(offset, n);
    
    auto E1 = eigvects(slice_left, slice_left);
    auto E2 = eigvects(slice_right, slice_right);
    auto Q1 = temp_Q(slice_left, slice_left);
    auto Q2 = temp_Q(slice_right, slice_right);

    auto lambda1 = eigenvalues(slice_left);
    auto lambda2 = eigenvalues(slice_right);

    auto pool = BumpAllocator(ws);
    auto batch_size = d.batch_size();
    auto rho = pool.allocate<T>(ctx, batch_size);

    {
        BATCHLAS_KERNEL_TRACE_SCOPE("stedc_flat:rho_copy");
        ctx->parallel_for(sycl::range(batch_size), [=](sycl::id<1> idx) {
            rho[idx[0]] = e(offset + n_left - 1, idx[0]);
        });
    }

    // Workspace for the recursive children is unused here; reuse the whole ws for this merge.
    auto permutation = VectorView<int32_t>(pool.allocate<int32_t>(ctx, n * batch_size), n, batch_size);
    auto v = VectorView<T>(pool.allocate<T>(ctx, n * batch_size), n, batch_size);

    {
        BATCHLAS_KERNEL_TRACE_SCOPE("stedc_flat:build_v");
        ctx->submit([&](sycl::handler& h) {
            auto E1view = E1.kernel_view();
            auto E2view = E2.kernel_view();
            h.parallel_for(sycl::nd_range<1>(batch_size * kernel_config::BLOCK_SIZE_128, kernel_config::BLOCK_SIZE_128), 
                          [=](sycl::nd_item<1> item) {
                auto bid = item.get_group_linear_id();
                auto bdim = item.get_local_range(0);
                auto tid = item.get_local_linear_id();
                for (int i = tid; i < static_cast<int>(n_left); i += static_cast<int>(bdim)) {
                    v(i, bid) = E1view(static_cast<int>(n_left) - 1, i, bid) / sycl::sqrt(T(2));
                }
                for (int i = tid; i < static_cast<int>(n - n_left); i += static_cast<int>(bdim)) {
                    v(i + static_cast<int>(n_left), bid) = E2view(0, i, bid) / sycl::sqrt(T(2));
                }
            });
        });
    }

    {
        BATCHLAS_KERNEL_TRACE_SCOPE("stedc_flat:argsort_eigs");
        argsort(ctx, eigenvalues(slice_all), permutation, SortOrder::Ascending, true);
    }
    {
        BATCHLAS_KERNEL_TRACE_SCOPE("stedc_flat:permute_eigs");
        permute(ctx, eigenvalues(slice_all), permutation);
    }
    {
        BATCHLAS_KERNEL_TRACE_SCOPE("stedc_flat:permute_v");
        permute(ctx, v, permutation);
    }
    {
        BATCHLAS_KERNEL_TRACE_SCOPE("stedc_flat:permuted_copy_eigvects");
        permuted_copy(ctx,
                      eigvects(slice_all, slice_all),
                      temp_Q(slice_all, slice_all),
                      permutation);
    }

    auto keep_indices = VectorView<int32_t>(pool.allocate<int32_t>(ctx, n * batch_size), n, batch_size);
    auto n_reduced = pool.allocate<int32_t>(ctx, batch_size);

    T reltol = T(64.0) * std::numeric_limits<T>::epsilon();
    {
        BATCHLAS_KERNEL_TRACE_SCOPE("stedc_flat:reduce_deflate");
        ctx->submit([&](sycl::handler& h) {
            auto Q = temp_Q(slice_all, slice_all).kernel_view();
            auto scan_mem_include = sycl::local_accessor<int32_t, 1>(sycl::range<1>(n), h);
            auto scan_mem_exclude = sycl::local_accessor<int32_t, 1>(sycl::range<1>(n), h);
            auto norm_mem = sycl::local_accessor<T, 1>(sycl::range<1>(n), h);
            h.parallel_for(sycl::nd_range<1>(batch_size * kernel_config::BLOCK_SIZE_128, kernel_config::BLOCK_SIZE_128), 
                          [=](sycl::nd_item<1> item) {
                auto bid = item.get_group_linear_id();
                auto bdim = item.get_local_range(0);
                auto tid = item.get_local_linear_id();
                auto cta = item.get_group();

                // Initialize indices and flags
                for (int k = tid; k < static_cast<int>(n); k += static_cast<int>(bdim)) {
                    keep_indices(k, bid) = 0;
                    scan_mem_exclude[k] = 0;
                    permutation(k, bid) = -1;
                }
                sycl::group_barrier(cta);

                // Deflate nearly-equal eigenvalues
                for (int j = 0; j < static_cast<int>(n - 1); j++) {
                    if (sycl::fabs(eigenvalues(offset + j + 1, bid) - eigenvalues(offset + j, bid)) <=
                        reltol * sycl::max(T(1), sycl::max(sycl::fabs(eigenvalues(offset + j + 1, bid)), sycl::fabs(eigenvalues(offset + j, bid))))) {
                        auto f = v(j + 1, bid);
                        auto g = v(j, bid);
                        auto [c, s, r] = internal::lartg(f, g);
                        sycl::group_barrier(cta);
                        if (tid == 0) {
                            v(j, bid) = T(0.0);
                            v(j + 1, bid) = r;
                        }
                        for (int k = tid; k < static_cast<int>(n); k += static_cast<int>(bdim)) {
                            auto Qi = Q(k, j, bid), Qj = Q(k, j + 1, bid);
                            Q(k, j, bid) = Qi * c - Qj * s;
                            Q(k, j + 1, bid) = Qj * c + Qi * s;
                        }
                    }
                }

                sycl::group_barrier(cta);

                // Compute eigenvalue max and v-vector max for tolerance
                for (int k = tid; k < static_cast<int>(n); k += static_cast<int>(bdim)) { 
                    norm_mem[k] = sycl::fabs(eigenvalues(offset + k, bid)); 
                }
                auto eig_max = sycl::joint_reduce(cta,
                                                 norm_mem.template get_multi_ptr<sycl::access::decorated::no>().get(),
                                                 norm_mem.template get_multi_ptr<sycl::access::decorated::no>().get() + static_cast<int>(n),
                                                 sycl::maximum<T>());

                auto v_norm = internal::nrm2<T>(cta, v);
                for (int k = tid; k < static_cast<int>(n); k += static_cast<int>(bdim)) { 
                    norm_mem[k] = sycl::fabs(v(k, bid) / v_norm); 
                }
                auto v_max = sycl::joint_reduce(cta,
                                               norm_mem.template get_multi_ptr<sycl::access::decorated::no>().get(),
                                               norm_mem.template get_multi_ptr<sycl::access::decorated::no>().get() + static_cast<int>(n),
                                               sycl::maximum<T>());
                auto tol = T(8.0) * std::numeric_limits<T>::epsilon() * sycl::max(eig_max, v_max);

                // Deflation check
                for (int k = tid; k < static_cast<int>(n); k += static_cast<int>(bdim)) {
                    if (sycl::fabs(rho[bid] * norm_mem[k]) > tol) {
                        keep_indices(k, bid) = 1;
                    } else {
                        scan_mem_exclude[k] = 1;
                    }
                }

                sycl::group_barrier(cta);

                // Scan for reordering
                sycl::joint_exclusive_scan(cta, keep_indices.batch_item(bid).data_ptr(), 
                                          keep_indices.batch_item(bid).data_ptr() + n, 
                                          scan_mem_include.get_multi_ptr<sycl::access::decorated::no>().get(), 
                                          0, sycl::plus<int32_t>());
                sycl::joint_exclusive_scan(cta, scan_mem_exclude.get_multi_ptr<sycl::access::decorated::no>().get(), 
                                          scan_mem_exclude.get_multi_ptr<sycl::access::decorated::no>().get() + n, 
                                          scan_mem_exclude.get_multi_ptr<sycl::access::decorated::no>().get(), 
                                          0, sycl::plus<int32_t>());

                // Build permutation from scan results
                for (int k = tid; k < static_cast<int>(n); k += static_cast<int>(bdim)) {
                    if (keep_indices(k, bid) == 1) {
                        permutation(scan_mem_include[k], bid) = k;
                    } else {
                        permutation(static_cast<int>(n) - 1 - scan_mem_exclude[k], bid) = k;
                    }
                }

                if (tid == 0) {
                    n_reduced[bid] = scan_mem_include[n - 1] + keep_indices(static_cast<int>(n - 1), bid);
                }
            });
        });
    }

    {
        BATCHLAS_KERNEL_TRACE_SCOPE("stedc_flat:permute_Q");
        permute(ctx, temp_Q(slice_all, slice_all), eigvects(slice_all, slice_all), permutation);
    }
    {
        BATCHLAS_KERNEL_TRACE_SCOPE("stedc_flat:permute_eigs2");
        permute(ctx, eigenvalues(slice_all), permutation);
    }
    {
        BATCHLAS_KERNEL_TRACE_SCOPE("stedc_flat:permute_v2");
        permute(ctx, v, permutation);
    }

    auto temp_lambdas = VectorView<T>(pool.allocate<T>(ctx, n * batch_size), n, batch_size);
    MatrixView<T> Qprime = MatrixView<T>(pool.allocate<T>(ctx, n * n * batch_size).data(), static_cast<int64_t>(n), static_cast<int64_t>(n), static_cast<int64_t>(n), static_cast<int64_t>(n * n), batch_size);

    {
        BATCHLAS_KERNEL_TRACE_SCOPE("stedc_flat:Qprime_identity");
        Qprime.fill_identity(ctx);
    }

    {
        BATCHLAS_KERNEL_TRACE_SCOPE("stedc_flat:secular_eqs");
        ctx->submit([&](sycl::handler& h) {
            auto Qview = Qprime.kernel_view();
            h.parallel_for(sycl::nd_range<1>(batch_size * kernel_config::BLOCK_SIZE_128, kernel_config::BLOCK_SIZE_128), 
                          [=](sycl::nd_item<1> item) {
                auto bid = item.get_group_linear_id();
                auto bdim = item.get_local_range(0);
                auto tid = item.get_local_linear_id();
                auto cta = item.get_group();
                auto Q_bid = Qview.batch_item(bid);
                auto sign = (e(offset + n_left - 1, bid) >= 0) ? 1 : -1;
                auto nloc = n_reduced[bid];
                
                // Load diagonal as initial estimate
                for (int k = tid; k < static_cast<int>(nloc * nloc); k += static_cast<int>(bdim)) {
                    auto i = k % static_cast<int>(nloc);
                    auto j = k / static_cast<int>(nloc);
                    Q_bid(i, j) = eigenvalues(offset + i, bid);
                }
                sycl::group_barrier(cta);
                
                // Solve secular equations for each eigenvalue
                for (int k = tid; k < static_cast<int>(nloc); k += static_cast<int>(bdim)) {
                    auto dview = Q_bid(Slice{}, k);
                    temp_lambdas(k, bid) = (k == static_cast<int>(nloc) - 1) 
                        ? sec_solve_ext_roc(static_cast<int32_t>(nloc), dview, v.batch_item(bid), sycl::fabs(T(2) * rho[bid])) * sign
                        : sec_solve_roc(static_cast<int32_t>(nloc), dview, v.batch_item(bid), sycl::fabs(T(2) * rho[bid]), k) * sign;
                }
                sycl::group_barrier(cta);
            });
        });
    }

    {
        BATCHLAS_KERNEL_TRACE_SCOPE("stedc_flat:normalize_Q");
        ctx->submit([&](sycl::handler& h) {
            auto Qview = Qprime.kernel_view();
            h.parallel_for(sycl::nd_range<1>(batch_size * kernel_config::BLOCK_SIZE_128, kernel_config::BLOCK_SIZE_128), 
                          [=](sycl::nd_item<1> item) {
                auto bid = item.get_group_linear_id();
                auto g = item.get_group();
                auto tid = item.get_local_linear_id();
                auto bdim = item.get_local_range(0);
                auto Qbid = Qview.batch_item(bid);
                auto dd = n_reduced[bid];
                
                for (int eid = 0; eid < dd; ++eid) {
                    auto Di = eigenvalues(offset + eid, bid);
                    T partial = T(1);
                    for (int j = tid; j < dd; j += static_cast<int>(bdim)) {
                        partial *= (j == eid) ? Qbid(eid, j) : Qbid(eid, j) / (Di - eigenvalues(offset + j, bid));
                    }
                    T valf = sycl::reduce_over_group(g, partial, sycl::multiplies<T>());
                    if (tid == 0) {
                        T mag = sycl::sqrt(sycl::fabs(valf));
                        T sgn = v(eid, bid) >= T(0) ? T(1) : T(-1);
                        v(eid, bid) = sgn * mag;
                    }
                }
            });
        });
    }

    {
        BATCHLAS_KERNEL_TRACE_SCOPE("stedc_flat:ortho_Q");
        ctx->submit([&](sycl::handler& h) {
            auto Qview = Qprime.kernel_view();
            h.parallel_for(sycl::nd_range<1>(batch_size * kernel_config::BLOCK_SIZE_128, kernel_config::BLOCK_SIZE_128), 
                          [=](sycl::nd_item<1> item) {
                auto bid = item.get_group_linear_id();
                auto cta = item.get_group();
                auto tid = item.get_local_linear_id();
                auto bdim = item.get_local_range(0);
                auto dd = n_reduced[bid];
                auto Qbid = Qview.batch_item(bid);
                
                for (int eig = 0; eig < dd; ++eig) {
                    for (int i = tid; i < dd; i += static_cast<int>(bdim)) {
                        Qbid(i, eig) = v(i, bid) / Qbid(i, eig);
                    }
                    auto nrm2 = internal::nrm2(cta, Qview(Slice{0, to_idx(dd)}, eig));
                    for (int i = tid; i < dd; i += static_cast<int>(bdim)) {
                        Qbid(i, eig) /= nrm2;
                    }
                }
            });
        });
    }

    {
        BATCHLAS_KERNEL_TRACE_SCOPE("stedc_flat:update_eigvects");
        gemm<B>(ctx,
                temp_Q(slice_all, slice_all),
                Qprime,
                eigvects(slice_all, slice_all),
                T(1.0),
                T(0.0),
                Transpose::NoTrans,
                Transpose::NoTrans);
    }

    {
        BATCHLAS_KERNEL_TRACE_SCOPE("stedc_flat:final_eigenvalues");
        ctx->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(batch_size * kernel_config::BLOCK_SIZE_32, kernel_config::BLOCK_SIZE_32), 
                          [=](sycl::nd_item<1> item) {
                auto bid = item.get_group_linear_id();
                auto bdim = item.get_local_range(0);
                auto tid = item.get_local_linear_id();
                for (int k = tid; k < n_reduced[bid]; k += static_cast<int>(bdim)) {
                    eigenvalues(offset + k, bid) = temp_lambdas(k, bid);
                }
            });
        });
    }

    argsort(ctx, eigenvalues(slice_all), permutation, SortOrder::Ascending, true);
    permute(ctx, eigenvalues(slice_all), permutation);
    permute(ctx, eigvects(slice_all, slice_all), temp_Q(slice_all, slice_all), permutation);

    return ctx.get_event();
}

// Flattened STEDC entry point (keeps original recursive version untouched).
template <Backend B, typename T>
Event stedc_flat(Queue& ctx,
                 const VectorView<T>& d,
                 const VectorView<T>& e,
                 const VectorView<T>& eigenvalues,
                 const Span<std::byte>& ws,
                 JobType jobz,
                 StedcParams<T> params,
                 const MatrixView<T, MatrixFormat::Dense>& eigvects)
{
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

    if (jobz == JobType::EigenVectors) {
        eigvects.fill_identity(ctx);
    } else {
        eigvects.fill_zeros(ctx);
    }

    auto pool = BumpAllocator(ws);
    auto n_total = d.size();
    auto alloc_size = BumpAllocator::allocation_size<T>(ctx, n_total * n_total * d.batch_size());
    auto temp_Q = MatrixView<T>(pool.allocate<T>(ctx, n_total * n_total * d.batch_size()).data(), n_total, n_total, n_total, n_total * n_total, d.batch_size());
    auto ws_remaining = ws.subspan(alloc_size);
    
    const size_t leaf = static_cast<size_t>(params.recursion_threshold);
    const size_t n_blocks = (n_total + leaf - 1) / leaf;
    const size_t n_boundaries = (n_blocks > 0) ? (n_blocks - 1) : 0;

    // Decouple subproblems: adjust diagonal entries at block boundaries
    if (n_boundaries > 0) {
        const size_t padded_batch = ((d.batch_size() + 31) / 32) * 32;
        ctx->submit([&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<2>(sycl::range<2>(n_boundaries, padded_batch), sycl::range<2>(1, 32)), 
                          [=](sycl::nd_item<2> item) {
                auto b = item.get_global_id(0);
                auto bid = item.get_global_id(1);
                if (bid >= d.batch_size()) return;

                size_t boundary = (b + 1) * leaf;
                if (boundary == 0 || boundary >= n_total) return;
                
                auto rho = e(boundary - 1, bid);
                d(boundary - 1, bid) -= sycl::fabs(rho);
                d(boundary, bid) -= sycl::fabs(rho);
            });
        });
    }

    // Build block schedule
    std::vector<StedcBlock> blocks;
    for (size_t off = 0; off < n_total; ) {
        auto sz = std::min(leaf, n_total - off);
        blocks.push_back({off, sz});
        off += sz;
    }

    // Leaf solves
    for (auto b : blocks) {
        auto dblk = d(Slice(b.offset, b.offset + b.size));
        auto eblk = e(Slice(b.offset, b.offset + b.size - 1));
        auto evalblk = eigenvalues(Slice(b.offset, b.offset + b.size));
        auto Eblk = eigvects(make_slice(b.offset, b.size), make_slice(b.offset, b.size));
        steqr<B, T>(ctx, dblk, eblk, evalblk, ws_remaining, jobz, 
                    SteqrParams<T>{32, 10, std::numeric_limits<T>::epsilon(), false, false, false}, Eblk);
    }

    // Iterative merges with doubling block size
    std::vector<StedcBlock> next;
    while (blocks.size() > 1) {
        next.clear();
        for (size_t i = 0; i < blocks.size(); ) {
            if (i + 1 >= blocks.size()) {
                next.push_back(blocks[i]);
                break;
            }
            auto left = blocks[i];
            auto right = blocks[i + 1];
            if (left.offset + left.size != right.offset) {
                throw std::runtime_error("Blocks are not contiguous; cannot merge.");
            }
            auto merged_size = left.size + right.size;
            stedc_merge_flat<B, T>(ctx,
                                   left.offset,
                                   merged_size,
                                   left.size,
                                   d,
                                   e,
                                   eigenvalues,
                                   ws_remaining,
                                   jobz,
                                   params,
                                   eigvects,
                                   temp_Q);
            next.push_back({left.offset, merged_size});
            i += 2;
        }
        blocks.swap(next);
    }

    return ctx.get_event();
}

// Explicit instantiations for the flattened path.
#if BATCHLAS_HAS_HOST_BACKEND
template Event stedc_flat<Backend::NETLIB, float>(Queue&, const VectorView<float>&, const VectorView<float>&, const VectorView<float>&, const Span<std::byte>&, JobType, StedcParams<float>, const MatrixView<float, MatrixFormat::Dense>&);
template Event stedc_flat<Backend::NETLIB, double>(Queue&, const VectorView<double>&, const VectorView<double>&, const VectorView<double>&, const Span<std::byte>&, JobType, StedcParams<double>, const MatrixView<double, MatrixFormat::Dense>&);
#endif

#if BATCHLAS_HAS_CUDA_BACKEND
template Event stedc_flat<Backend::CUDA, float>(Queue&, const VectorView<float>&, const VectorView<float>&, const VectorView<float>&, const Span<std::byte>&, JobType, StedcParams<float>, const MatrixView<float, MatrixFormat::Dense>&);
template Event stedc_flat<Backend::CUDA, double>(Queue&, const VectorView<double>&, const VectorView<double>&, const VectorView<double>&, const Span<std::byte>&, JobType, StedcParams<double>, const MatrixView<double, MatrixFormat::Dense>&);
#endif

} // namespace batchlas
