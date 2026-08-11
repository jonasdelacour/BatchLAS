#include <batchlas/blas/matrix.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/extra.hh>
#include <batchlas/util/kernel-heuristics.hh>
#include <batchlas/util/mempool.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>
#include <batchlas/backend_config.h>
#include "../math-helpers.hh"
#include "../queue.hh"
#include "../util/template-instantiations.hh"
#include "../sort.hh"

namespace batchlas {

template <typename T> struct FrancisKernel {};
template <typename T> struct RotationKernel {};

template <typename T>
auto wilkinson_shift(const T& a, const T& b, const T& c) {
    // Compute the Wilkinson shift assuming that a, b, c represents the
    // bottom-right 2x2 block of a matrix
    // |     :  :|
    // |     :  :|
    // |.... a, b|
    // |.... b, c|
    // Returns the eigenvalue closest to c
    const auto [lambda1, lambda2] = internal::eigenvalues_2x2(a, b, c);
    return std::abs(lambda1 - c) < std::abs(lambda2 - c) ? lambda1 : lambda2;
}

template <typename T>
auto givens_rotation(const T& a, const T& b) {
    auto [c_l, s_l, r] = internal::lartg(a, b);
    return std::array<T, 2>{c_l, -s_l};
}

template <typename T>
T apply_givens_rotation(const VectorView<T>& d,
                        const VectorView<T>& e,
                        const T& prev_bulge,
                        size_t i,
                        size_t j,
                        const std::array<T, 2>& givens,
                        bool QR) {
    // Apply similarity transform to rows/cols i and j of a tridiagonal matrix T
    // in a virtual indexing:
    //   - if QR == true:  virtual index k == physical index k  (top-down QR)
    //   - if QR == false: virtual index k == physical index (n-1-k) (bottom-up QL)
    // This way, the same bulge-chasing logic implements both QR and QL iterations.
    const T c = givens[0]; // Gamma
    const T s = givens[1]; // Sigma

    const size_t n  = d.size();      // number of diagonal entries
    const size_t ne = e.size();      // number of off-diagonal entries (n-1)

    // Virtual -> physical index mapping helpers
    auto d_get = [&](size_t k) -> T {
        return QR ? d(k) : d(n - 1 - k);
    };
    auto d_set = [&](size_t k, T val) {
        if (QR) {
            d(k) = val;
        } else {
            d(n - 1 - k) = val;
        }
    };
    auto e_get = [&](size_t k) -> T {
        // e(k) couples d(k) and d(k+1) in virtual indexing
        return QR ? e(k) : e(ne - 1 - k);
    };
    auto e_set = [&](size_t k, T val) {
        if (QR) {
            e(k) = val;
        } else {
            e(ne - 1 - k) = val;
        }
    };

    // Read current 2x2/3x3 “front” of the bulge in virtual indexing
    T di = d_get(i);
    T dj = d_get(j);
    T ei = e_get(i);
    T ej = (j < ne) ? e_get(j) : T(0);

    // Update diagonal entries
    T di_new = c * (c * di - ei * s) - s * (ei * c - s * dj);
    T dj_new = c * (c * dj + ei * s) + s * (ei * c + s * di);
    d_set(i, di_new);
    d_set(j, dj_new);

    // Update off-diagonals adjacent to rows/cols i,j in virtual indexing
    if (i > 0) {
        T e_im1 = e_get(i - 1);
        e_set(i - 1, e_im1 * c - prev_bulge * s);
    }

    T ei_new = c * (c * ei + s * di) - s * (c * dj + s * ei);
    e_set(i, ei_new);

    if (j < ne) {
        e_set(j, c * ej);
    }

    // Return the new bulge element in virtual indexing
    return -ej * s;
}

template <typename T>
struct SteqrImplScratch {
    Span<int32_t> scan;
    VectorView<std::array<int32_t, 2>> temp_deflation_indices;
};

// Single description of the per-pass scratch steqr_impl carves off the pool it
// is handed; see workspace_bytes() in util/mempool.hh.
template <typename T>
SteqrImplScratch<T> steqr_impl_layout(Queue& ctx, BumpAllocator& pool, int64_t n, int64_t batch_size) {
    auto scan = pool.allocate<int32_t>(ctx, batch_size);
    auto tdi = pool.allocate<std::array<int32_t, 2>>(
        ctx, VectorView<std::array<int32_t, 2>>::required_span_length(n / 2, 1, n / 2, batch_size));
    return {scan, VectorView<std::array<int32_t, 2>>(tdi, n / 2, batch_size, 1, n / 2)};
}

template <typename T>
Event steqr_impl(Queue& ctx,
                    const VectorView<T>& d, //Diagonal elements
                    const VectorView<T>& e, //Off-diagonal elements
                    JobType jobz, //Eigenvector computation flag
                    const MatrixView<T, MatrixFormat::Dense>& Q, //Eigenvector storage
                    const MatrixView<std::array<T,2>, MatrixFormat::Dense>& givens_rotations, //Storage for Givens rotations
                    const Span<std::array<int32_t,3>>& deflation_indices, //#sub_problems deflation indices i.e. where e is zero
                    const Span<ApplyOrder>& order_view, //Order of application of rotations, has #sub_problems number of entries
                    const Span<int32_t>& sweep_counts, //Number of sweeps actually performed per sub-problem
                    BumpAllocator allocator,
                    size_t max_sweeps, //Maximum number of sweeps to perform
                    T zero_threshold) {
    BATCHLAS_KERNEL_TRACE_SCOPE("steqr");
    // Perform the Francis sweep for the i-th step
    // This function will apply a francis sweep of Givens rotations
    auto n = d.size();
    auto batch_size = d.batch_size();
    bool store_givens = jobz == JobType::EigenVectors;
    auto ncus = ctx.device().get_property(DeviceProperty::MAX_COMPUTE_UNITS);
    auto scratch = steqr_impl_layout<T>(ctx, allocator, n, batch_size);
    auto scan_view = scratch.scan;
    auto temp_deflation_indices = scratch.temp_deflation_indices;
    
    
    {
        BATCHLAS_KERNEL_TRACE_SCOPE("steqr:deflation_ranges");
        ctx -> submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range(batch_size), [=](sycl::id<1> id) {
            auto i = id[0];
            auto ebid = e.batch_item(i);
            auto dbid = d.batch_item(i);
            auto ix = 0;
            auto sub_problem_ix = 0;
            while (ix < n - 1) {
                auto start_ix = ix;
                if (ebid(ix) != T(0)) {
                    for (ix = ix + 1; ix < n - 1; ++ix) {
                        if (ebid(ix) == T(0)) break;
                    }
                } else {
                    ix++;
                    continue;
                }
                auto end_ix = ix + 1;
                temp_deflation_indices(sub_problem_ix, i) = {start_ix, end_ix};
                sub_problem_ix++;
            }
            scan_view[i] = sub_problem_ix;
        });
        });
    }

    {
        BATCHLAS_KERNEL_TRACE_SCOPE("steqr:deflation_scan");
        internal::scan_inclusive_inplace<int32_t>(ctx, scan_view);
    }
    
    {
        BATCHLAS_KERNEL_TRACE_SCOPE("steqr:deflation_writeout");
        ctx -> submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range(batch_size), [=](sycl::id<1> id) {
            auto i = id[0];
            auto num_sub_problems = scan_view[i] - (i == 0 ? 0 : scan_view[i - 1]);
            auto offset = (i == 0 ? 0 : scan_view[i - 1]);
            for (size_t j = 0; j < num_sub_problems; ++j) {
                deflation_indices[offset + j] = {static_cast<int32_t>(i), temp_deflation_indices(j, i)[0], temp_deflation_indices(j, i)[1]};
            }
        });
        });
    }

    {
        BATCHLAS_KERNEL_TRACE_SCOPE("steqr:francis_sweep");
        ctx->submit([&](sycl::handler& cgh) {
        auto rotations_view = givens_rotations.kernel_view();
        //auto bsize = n_problems < ncus ? 1 : internal::ceil_div(size_t(n_problems), ncus);
        cgh.parallel_for<FrancisKernel<T>>(sycl::nd_range(sycl::range(ncus*128), sycl::range(64)), [=](sycl::nd_item<1> item) {
            auto i = item.get_global_id(0);
            auto n_problems = scan_view[batch_size - 1];

            for (int gid = i; gid < n_problems; gid += item.get_global_range(0)) {
            auto [batch_ix, start_ix, end_ix] = deflation_indices[gid];

            auto dbid = d.batch_item(batch_ix);
            auto ebid = e.batch_item(batch_ix);
            auto d_ = dbid(Slice(start_ix, end_ix));
            auto e_ = ebid(Slice(start_ix, end_ix - 1));
            auto n = end_ix - start_ix;

            if (store_givens) sweep_counts[gid] = 0;
            if (n == 1) continue; //Nothing to do for 1x1 blocks
            if (n == 2) { //Analytically compute eigenvalues for 2x2 blocks
                if (store_givens) {
                    auto [rt1, rt2, c, s] = internal::laev2(d_(0), e_(0), d_(1));
                    d_(0) = rt1;
                    d_(1) = rt2;
                    e_(0) = T(0);
                    rotations_view(0, 0, gid) = {c, -s};
                    order_view[gid] = ApplyOrder::Forward;
                    sweep_counts[gid] = 1;
                } else {
                    auto [rt1, rt2] = internal::eigenvalues_2x2(d_(0), e_(0), d_(1));
                    d_(0) = rt1;
                    d_(1) = rt2;
                    e_(0) = T(0);
                }
                continue;
            }
            // QR / QL switch: QR sweeps from top, QL sweeps from bottom.
            // We implement QL by viewing (d_, e_) in reversed order via virtual indices.
            bool QR = std::abs(d_(0)) <= std::abs(d_(n - 1));
            order_view[gid] = QR ? ApplyOrder::Forward : ApplyOrder::Backward;
            for (size_t k = 0; k < max_sweeps; ++k) {
                auto anorm = std::abs(d_(n - 1));
                for (size_t idx = 0; idx < n - 1; ++idx)
                    anorm = std::fmax(anorm, std::fmax(std::abs(d_(idx)), std::abs(e_(idx))));

                if (anorm > internal::ssfmax<T>()) {
                    auto alpha = internal::ssfmax<T>() / anorm;
                    // Scale down to avoid overflow
                    for (size_t idx = 0; idx < n; ++idx) d_(idx) *= alpha;
                    for (size_t idx = 0; idx < n - 1; ++idx) e_(idx) *= alpha;
                } else if (anorm < internal::ssfmin<T>() && anorm != T(0)) {
                    // Scale up to avoid underflow
                    auto alpha = internal::ssfmin<T>() / anorm;
                    for (size_t idx = 0; idx < n; ++idx) d_(idx) *= alpha;
                    for (size_t idx = 0; idx < n - 1; ++idx) e_(idx) *= alpha;
                }

                // Virtual accessors (see apply_givens_rotation for the same mapping)
                auto d_get = [&](size_t idx) -> T {
                    return QR ? d_(idx) : d_(n - 1 - idx);
                };
                auto e_get = [&](size_t idx) -> T {
                    // idx is in [0, n-2] in virtual indexing
                    return QR ? e_(idx) : e_(n - 2 - idx);
                };

                // Form Wilkinson shift in virtual indexing: trailing 2x2 of virtual block
                const auto shift = wilkinson_shift(d_get(n - 2),
                                                   e_get(n - 2),
                                                   d_get(n - 1));

                // First Givens rotation eliminates the first subdiagonal in virtual indexing
                auto [c0, s0] = givens_rotation(d_get(0) - shift, e_get(0));
                if (store_givens) {
                    rotations_view(0, k, gid) = {c0, s0};
                }
                auto bulge = apply_givens_rotation(d_, e_, T(0), 0, 1, {c0, s0}, QR);

                // Chase the bulge across the block in virtual indexing
                for (size_t j = 1; j < n - 1; ++j) {
                    auto [cj, sj] = givens_rotation(e_get(j - 1), bulge);
                    if (store_givens) {
                        rotations_view(j, k, gid) = {cj, sj};
                    }
                    bulge = apply_givens_rotation(d_, e_, bulge, j, j + 1, {cj, sj}, QR);
                }

                bool deflatable = false;
                for (size_t j = 0; j < n - 1; ++j) {
                    // Check for deflation
                    if (std::abs(e_(j)) * std::abs(e_(j)) <= internal::eps2<T>() * std::abs(d_(j))*std::abs(d_(j + 1)) +
                        internal::safmin<T>()) {
                        e_(j) = T(0);
                        deflatable = true;
                    }
                }

                if (anorm > internal::ssfmax<T>()) {
                    auto alpha = anorm / internal::ssfmax<T>();
                    // Scale back up
                    for (size_t idx = 0; idx < n; ++idx) d_(idx) *= alpha;
                    for (size_t idx = 0; idx < n - 1; ++idx) e_(idx) *= alpha;
                } else if (anorm < internal::ssfmin<T>() && anorm != T(0)) {
                    auto alpha = anorm / internal::ssfmin<T>();
                    // Scale back down
                    for (size_t idx = 0; idx < n; ++idx) d_(idx) *= alpha;
                    for (size_t idx = 0; idx < n - 1; ++idx) e_(idx) *= alpha;
                }

                if (store_givens) sweep_counts[gid] = static_cast<int32_t>(k + 1);
                if (deflatable) break;
            }
        }
        });
    });

    ctx -> submit([&](sycl::handler& cgh) {
        auto Qview = Q.kernel_view();
        auto rotations_view = givens_rotations.kernel_view();
        cgh.parallel_for<RotationKernel<T>>(sycl::nd_range(sycl::range(Q.rows()*Q.batch_size()*2), sycl::range(Q.rows())), [=](sycl::nd_item<1> item) {
            auto bid = item.get_group(0);
            auto k = item.get_local_linear_id();
            auto n_problems = scan_view[batch_size - 1];
            for (int gid = bid; gid < n_problems; gid += item.get_group_range(0)) {
            auto [batch_ix, start_ix, end_ix] = deflation_indices[gid];
            auto Q_ = Qview.batch_item(batch_ix)(Slice(), Slice(start_ix, end_ix));
            const int ncols = static_cast<int>(Q_.cols());
            bool forward = order_view[gid] == ApplyOrder::Forward;
            auto col_index = [&](int v) -> int {
                return (forward)
                        ? v
                        : (ncols - 1 - v);
            };

            // Only replay the sweeps that were actually performed for this sub-problem.
            // Slots beyond that are stale/uninitialised and must not be applied.
            // sweep_counts is empty when eigenvectors were not requested; nothing to replay then.
            const int n_sweeps = sweep_counts.size() == 0 ? 0 : static_cast<int>(sweep_counts[gid]);
            for (int i = 0; i < n_sweeps; ++i) {
                for (int j = 0; j < ncols - 1; ++j) {
                    auto [c, s] = rotations_view(j, i, gid);
                    if (c == T(1) && s == T(0)) continue; // Skip identity rotations

                    // Map virtual indices (j, j+1) to physical column indices.
                    int ix1 = col_index(j);
                    int ix2 = col_index(j + 1);

                    const T x = Q_(k, ix1);
                    const T y = Q_(k, ix2);
                    Q_(k, ix1) = c * x - s * y;
                    Q_(k, ix2) = s * x + c * y;
                }
            }
            }
        });
        });
    }

    return ctx.get_event();
}

template <Backend B, typename T>
Event block_rot_impl(Queue& ctx, const MatrixView<std::array<T,2>, MatrixFormat::Dense>& givens_rotations, 
              const MatrixView<T, MatrixFormat::Dense>& eigvects, const Span<std::byte>& ws, size_t Nb) {
    //Nb is the block size of the intermediate givens matrices we form and apply
    BumpAllocator pool(ws);
    auto N = 2 * Nb;
    auto batch_size = givens_rotations.batch_size();
    auto givens_block = MatrixView<T>(pool.allocate<T>(ctx, batch_size * N * N).data(), N, N, N, N * N, batch_size);
    auto eigvects_dual = MatrixView<T>(pool.allocate<T>(ctx, batch_size * eigvects.rows() * N).data(), eigvects.rows(), N, eigvects.rows(), eigvects.rows() * N, batch_size);

    //ctx->memcpy(static_cast<void*>(eigvects_dual.data().data()), static_cast<void*>(eigvects.data_ptr()), eigvects.data().size_bytes());
    //givens_block.fill_identity(ctx);
    //Apply Prologue rotations: (Example here if Nb = 4)
    // G[0,0] , G[0,1] , G[0,2] , G[0,3]
    // G[1,0] , G[1,1] , G[1,2] 
    // G[2,0] , G[2,1]
    // G[3,0]
    
    ctx -> submit([&](sycl::handler& cgh) {
        auto rotations = givens_rotations.kernel_view();
        auto smem_possible = ctx.device().get_property(DeviceProperty::LOCAL_MEM_SIZE) >= sizeof(T) * N * N;
        sycl::local_accessor<T, 1> local_mem(sycl::range<1>(smem_possible ? N * N : 0), cgh);
        auto givens_view = givens_block.kernel_view();
        cgh.parallel_for(sycl::nd_range(sycl::range(batch_size*(Nb + 1)), sycl::range(Nb + 1)), [=](sycl::nd_item<1> item) {
            auto bid = item.get_group(0);
            auto k = item.get_local_linear_id();
            auto G = smem_possible ? KernelMatrixView<T, MatrixFormat::Dense>(util::get_raw_ptr(local_mem), N, N) :  givens_view.batch_item(bid);
            auto Gglobal = givens_view.batch_item(bid);
            for (size_t j = 0; j < N; ++j){
                G(k, j) = k == j ? T(1) : T(0); //Identity fill-in
            }
            for (size_t i = 0; i < Nb; ++i) {
                size_t j_start = (k > i + 1) ? (k - i - 1) : 0;
                for (size_t j = 0; j < Nb - i; ++j) {
                    auto [c, s] = rotations(j, i, bid);
                    if (c == T(1) && s == T(0)) continue; //Skip identity rotations
                    if (k < (2 + j + i)) {
                        T temp = c * G(k, j) - s * G(k, j + 1);
                        G(k, j + 1) = s * G(k, j) + c * G(k, j + 1);
                        G(k, j) = temp;
                    }
                }
            }
            if (smem_possible) {
                //Copy back to global memory
                for (size_t j = 0; j < N; ++j) {
                    Gglobal(k, j) = G(k, j);
                }
            }
        });
    });
    
    auto p_slice = Slice(0, int64_t(Nb + 1));
    gemm<B>(ctx,
            eigvects(Slice(),p_slice),
            givens_block(p_slice,p_slice),
            eigvects_dual(Slice(),p_slice),
            GemmOptions<T>{});
    MatrixView<T>::copy(ctx, eigvects(Slice(),p_slice), eigvects_dual(Slice(),p_slice));

    //Apply regular rotation sets
    auto row_remainder = givens_rotations.rows() % Nb;
    for (int64_t i = Nb; i < givens_rotations.rows() - row_remainder; i += Nb) {
        //givens_block.fill_identity(ctx).wait();
        //ctx->wait();
        //std::cout << "Post identityfill regular sweep givens_block:\n" << givens_block << std::endl;

        ctx->submit([&](sycl::handler& cgh) {
            auto rotations = givens_rotations.kernel_view();
            auto smem_possible = ctx.device().get_property(DeviceProperty::LOCAL_MEM_SIZE) >= sizeof(T) * N * N;
            sycl::local_accessor<T, 1> local_mem(sycl::range<1>(smem_possible ? N * N : 0), cgh);
            auto givens_view = givens_block.kernel_view();
            auto row_offset = i;
            cgh.parallel_for(sycl::nd_range(sycl::range(N * batch_size), sycl::range(N)), [=](sycl::nd_item<1> item) {
                auto k =      item.get_local_linear_id();
                auto bid =      item.get_group(0);
                auto G = smem_possible ? KernelMatrixView<T, MatrixFormat::Dense>(util::get_raw_ptr(local_mem), N, N) :  givens_view.batch_item(bid);
                auto Gglobal = givens_view.batch_item(bid);
                for (size_t j = 0; j < N; ++j){
                   G(k, j) = k == j ? T(1) : T(0); //Identity fill-in
                }
                for (size_t i = 0; i < Nb; ++i) {
                    for (size_t j = 0; j < Nb; ++j) {
                        auto [c, s] = rotations(row_offset + j - i, i, bid);
                        if (c == T(1) && s == T(0)) continue; //Skip identity rotations
                        auto j0 = j - i + Nb - 1;
                        auto j1 = j0 + 1;
                        //At sweep i -> i + Nb
                        T temp = c * G(k, j0) - s * G(k, j1);
                        G(k, j1) = s * G(k, j0) + c * G(k, j1);
                        G(k, j0) = temp;
                    }
                }
                if (smem_possible) {
                    //Copy back to global memory
                    for (size_t j = 0; j < N; ++j) {
                        Gglobal(k, j) = G(k, j);
                    }
                }
            });
        });

        //Post-multiply a slice of eigvects by the givens-block matrix
        auto col_slice = Slice(int64_t(i - Nb + 1), int64_t(i + Nb + 1));
        gemm<B>(ctx, eigvects(Slice{}, col_slice), givens_block, eigvects_dual, GemmOptions<T>{});

        MatrixView<T>::copy(ctx, eigvects(Slice{}, col_slice), eigvects_dual);
    }


    //Epilogue rotations (example where Nb = 4, and col_remainder = 3)
    //                                            G[0, m-3], G[0, m-2], G[0, m-1]
    //                                 G[1, m-4], G[1, m-3], G[1, m-2], G[1, m-1]
    //                      G[2, m-5], G[2, m-4], G[2, m-3], G[2, m-2], G[2, m-1]
    //           G[3, m-6], G[3, m-5], G[3, m-4], G[3, m-3], G[3, m-2], G[3, m-1]

    auto row_offset = givens_rotations.rows() - row_remainder;
    ctx -> submit([&](sycl::handler& cgh) {
        auto rotations = givens_rotations.kernel_view();
        auto smem_possible = ctx.device().get_property(DeviceProperty::LOCAL_MEM_SIZE) >= sizeof(T) * N * N;
        sycl::local_accessor<T, 1> local_mem(sycl::range<1>(smem_possible ? N * N : 0), cgh);
        auto givens_view = givens_block.kernel_view();
        
        cgh.parallel_for(sycl::nd_range(sycl::range(batch_size*(N)), sycl::range(N)), [=](sycl::nd_item<1> item) {
            auto bid = item.get_group(0);
            auto k = item.get_local_linear_id();
            auto G = smem_possible ? KernelMatrixView<T, MatrixFormat::Dense>(util::get_raw_ptr(local_mem), N, N) :  givens_view.batch_item(bid);
            auto Gglobal = givens_view.batch_item(bid);
            for (size_t j = 0; j < N; ++j){
                G(k, j) = k == j ? T(1) : T(0); //Identity fill-in
            }
            for (size_t i = 0; i < Nb; ++i) { //Apply rotations
                for (size_t j = 0; j < row_remainder + i; ++j) {
                    auto [c, s] = rotations(row_offset + j - i, i, bid);
                    if (c == T(1) && s == T(0)) continue; //Skip identity rotations
                    auto j0 = j - i + Nb - 1;
                    auto j1 = j0 + 1;
                    T temp = c * G(k, j0) - s * G(k, j1);
                    G(k, j1) = s * G(k, j0) + c * G(k, j1);
                    G(k, j0) = temp;
                }
            }
            if (smem_possible) {
                //Copy back to global memory
                for (size_t j = 0; j < N; ++j) {
                   Gglobal(k, j) = G(k, j);
                }
            }
        });
    });
    //ctx -> wait();

    auto e_slice = Slice{int64_t(row_offset - Nb + 1), SliceEnd()};
    gemm<B>(ctx,
            eigvects(Slice{},e_slice),
            givens_block({0, int64_t(Nb + row_remainder)}, {0, int64_t(Nb + row_remainder)}),
            eigvects_dual(Slice{}, {0, int64_t(Nb + row_remainder)}),
            GemmOptions<T>{});
    MatrixView<T>::copy(ctx, eigvects(Slice{},e_slice), eigvects_dual(Slice{}, {0, int64_t(Nb + row_remainder)}));
    //ctx -> wait();
    //std::cout << "Post-GEMM eigvects:\n" << eigvects << std::endl;
    return ctx.get_event();
}

template <Backend B, typename T>
Event block_rot(Queue& ctx, const MatrixView<std::array<T,2>, MatrixFormat::Dense>& givens_rotations, 
              const MatrixView<T, MatrixFormat::Dense>& eigvects, const Span<std::byte>& ws, size_t max_block_size) 
{
    //std::cout << "All rotations:\n" << givens_rotations << std::endl;
    auto sweep_remainder = givens_rotations.cols() % max_block_size;
    for (int64_t i = 0; i < givens_rotations.cols() - sweep_remainder; i+=max_block_size)
    {
        block_rot_impl<B>(ctx, givens_rotations({i, int64_t(i + max_block_size)}, Slice()), eigvects, ws, max_block_size);
    }
    if (sweep_remainder == 0) return ctx.get_event();
    return block_rot_impl<B>(ctx, givens_rotations({int64_t(givens_rotations.cols() - sweep_remainder), SliceEnd()}, Slice()), eigvects, ws, sweep_remainder);
}

template <Backend B, typename T>
Event rot(Queue& ctx, const MatrixView<std::array<T,2>, MatrixFormat::Dense>& givens_rotations, const MatrixView<T, MatrixFormat::Dense>& Q, const Span<ApplyOrder>& order_view) {
    ctx -> submit([&](sycl::handler& cgh) {
        auto Q_ = Q.kernel_view();
        auto rotations_view = givens_rotations.kernel_view();
        cgh.parallel_for(sycl::nd_range(sycl::range(Q.rows()*Q.batch_size()), sycl::range(Q.rows())), [=](sycl::nd_item<1> item) {
            auto bid = item.get_group(0);
            auto k = item.get_local_linear_id();
            for (int i = 0; i < rotations_view.cols(); ++i) {
                for (int j = 0; j < rotations_view.rows(); ++j) {
                    auto [c, s] = rotations_view(j, i, bid);
                    if (c == T(1) && s == T(0)) continue; // Skip identity rotations

                    const int ncols = static_cast<int>(Q_.cols());
                    auto col_index = [&](int v) -> int {
                        return (order_view[bid] == ApplyOrder::Forward)
                               ? v
                               : (ncols - 1 - v);
                    };
                    if (j + 1 >= ncols) continue;
                    int ix1 = col_index(j);
                    int ix2 = col_index(j + 1);

                    T temp = c * Q_(k, ix1, bid) - s * Q_(k, ix2, bid);
                    Q_(k, ix2, bid) = s * Q_(k, ix1, bid) + c * Q_(k, ix2, bid);
                    Q_(k, ix1, bid) = temp;
                }
            }
        });
    });
    return ctx.get_event();
}

template <typename T>
struct SteqrLegacyWorkspace {
    VectorView<T> d;
    VectorView<T> e;
    Span<T> apply_Q_ws;
    MatrixView<std::array<T, 2>, MatrixFormat::Dense> givens_rotations;
    Span<ApplyOrder> apply_order;
    Span<std::array<int32_t, 3>> deflation_indices;
    Span<int32_t> sweep_counts;
    // Reused, not shared: each steqr_impl pass sub-allocates from a copy of this,
    // and once the passes are done sort() takes it over. Sized for whichever
    // needs more.
    Span<std::byte> scratch;
};

// Single description of steqr_legacy's workspace; see workspace_bytes() in
// util/mempool.hh.
//
// `eigvects` is only inspected for its shape (to size the sort scratch), so the
// sizing entry point may pass a shape-only view over no memory.
template <typename T>
SteqrLegacyWorkspace<T> steqr_legacy_layout(Queue& ctx,
                                            BumpAllocator& pool,
                                            int64_t n,
                                            int64_t batch_size,
                                            JobType jobz,
                                            const SteqrParams<T>& params,
                                            const VectorView<T>& eigenvalues,
                                            const MatrixView<T, MatrixFormat::Dense>& eigvects) {
    const bool want_vectors = jobz == JobType::EigenVectors;
    const auto increment = params.transpose_working_vectors ? batch_size : 1;
    const auto d_stride = params.transpose_working_vectors ? 1 : n;
    const auto e_stride = params.transpose_working_vectors ? 1 : n - 1;

    auto d = VectorView<T>(pool.allocate<T>(ctx, VectorView<T>::required_span_length(n, increment, d_stride, batch_size)),
                           n, batch_size, increment, d_stride);
    auto e = VectorView<T>(pool.allocate<T>(ctx, VectorView<T>::required_span_length(n - 1, increment, e_stride, batch_size)),
                           n - 1, batch_size, increment, e_stride);

    // Nothing reads this any more -- it is kept only so that converting this
    // function does not change how much workspace callers must supply. Removing
    // it is a separate, deliberate change.
    auto apply_Q_ws = pool.allocate<T>(
        ctx, want_vectors ? (batch_size * params.block_size * 2 * params.block_size * 2 + batch_size * n * params.block_size * 4) : 0);

    const auto n_sweeps_to_store = (want_vectors && params.block_rotations)
                                       ? std::max(params.block_size * 2, params.max_sweeps)
                                       : params.max_sweeps;
    const auto stride = (n - 1) * n_sweeps_to_store;
    const auto max_subproblems = n / 2 + 1;

    auto givens_rotations =
        want_vectors ? MatrixView<std::array<T, 2>>(pool.allocate<std::array<T, 2>>(ctx, stride * max_subproblems * batch_size).data(),
                                                    n - 1, n_sweeps_to_store, n - 1, stride, max_subproblems * batch_size)
                     : MatrixView<std::array<T, 2>>();
    auto apply_order = pool.allocate<ApplyOrder>(ctx, batch_size * max_subproblems);
    auto deflation_indices = pool.allocate<std::array<int32_t, 3>>(ctx, batch_size * max_subproblems);
    auto sweep_counts = want_vectors ? pool.allocate<int32_t>(ctx, batch_size * max_subproblems) : Span<int32_t>();

    const size_t pass_scratch = workspace_bytes([&](BumpAllocator& p) { return steqr_impl_layout<T>(ctx, p, n, batch_size); });
    const size_t sort_scratch = params.sort ? sort_buffer_size<T>(ctx, eigenvalues.data(), eigvects, jobz) : 0;
    auto scratch = pool.allocate<std::byte>(ctx, std::max(pass_scratch, sort_scratch));

    return {d, e, apply_Q_ws, givens_rotations, apply_order, deflation_indices, sweep_counts, scratch};
}

template <Backend B, typename T>
Event steqr_legacy(Queue& ctx, const VectorView<T>& d_in, const VectorView<T>& e_in, const VectorView<T>& eigenvalues, const Span<std::byte>& ws,
                   JobType jobz, SteqrParams<T> params, const MatrixView<T, MatrixFormat::Dense>& eigvects) {
    const int64_t n = d_in.size();
    const int64_t batch_size = d_in.batch_size();

    if (jobz == JobType::EigenVectors) {
        // Ensure the eigenvector matrix is square and matches the problem size.
        if (eigvects.rows() != eigvects.cols()) {
            throw std::invalid_argument("Matrix must be square for eigenvector computation.");
        }
        if (eigvects.rows() != n || eigvects.batch_size() != batch_size) {
            throw std::invalid_argument("Eigenvector matrix has incompatible dimensions.");
        }
        if (!params.back_transform) {
            eigvects.fill_identity(ctx);
        }
    }

    auto pool = BumpAllocator(ws);
    auto wsl = steqr_legacy_layout<T>(ctx, pool, n, batch_size, jobz, params, eigenvalues, eigvects);
    auto d = wsl.d;
    auto e = wsl.e;
    //Copy inputs to working buffers
    VectorView<T>::copy(ctx, d, d_in);
    VectorView<T>::copy(ctx, e, e_in);

    auto givens_rotations = wsl.givens_rotations;
    auto apply_order = wsl.apply_order;
    auto deflation_indices = wsl.deflation_indices;
    auto sweep_counts = wsl.sweep_counts;

    // Each pass deflates at least one off-diagonal entry per active sub-problem, so n - 1
    // passes is the worst case. Random inputs typically converge long before that, and each
    // pass is O(n^2 * batch) work, so bail out as soon as the tridiagonal is fully split.
    UnifiedVector<int32_t> not_converged(1, 0);
    auto converged_flag = not_converged.to_span();
    ctx.wait();
    for (int64_t i = 0; i < n - 1; ++i) {
        not_converged[0] = 0; // Safe: the queue is idle at this point.
        steqr_impl(ctx, d, e, jobz, eigvects, givens_rotations, deflation_indices, apply_order, sweep_counts,
                   BumpAllocator(wsl.scratch), params.max_sweeps, params.zero_threshold);

        ctx -> submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range(batch_size), [=](sycl::id<1> id) {
                auto ebid = e.batch_item(id[0]);
                for (int64_t j = 0; j < n - 1; ++j) {
                    if (ebid(j) != T(0)) {
                        sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                                         sycl::memory_scope::device> flag(converged_flag[0]);
                        flag.store(1);
                        return;
                    }
                }
            });
        });
        ctx.wait();
        if (not_converged[0] == 0) break;
    }

    ctx -> submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::nd_range(sycl::range(batch_size* n), sycl::range(n)), [=](sycl::nd_item<1> item) {
            auto bid = item.get_group_linear_id();
            auto intra = item.get_local_id(0);
            eigenvalues(intra, bid) = d(intra, bid);
        });
    });

    if (params.sort){
        // The passes are finished, so their scratch is free for sort to take over.
        sort(ctx, eigenvalues, eigvects, jobz, params.sort_order, wsl.scratch);
    }
    return ctx.get_event();
}

template <typename T>
size_t steqr_legacy_buffer_size(Queue& ctx, const VectorView<T>& d, const VectorView<T>& e,
                                const VectorView<T>& eigenvalues, JobType jobz, SteqrParams<T> params) {
    static_cast<void>(e);
    const auto n = d.size();
    const auto batch_size = d.batch_size();
    // Shape-only stand-in for the eigenvector matrix; the layout reads its
    // dimensions to size the sort scratch and nothing else.
    const MatrixView<T, MatrixFormat::Dense> eigvects_shape(nullptr, n, n, n, n * n, batch_size);
    return workspace_bytes([&](BumpAllocator& pool) {
        return steqr_legacy_layout<T>(ctx, pool, n, batch_size, jobz, params, eigenvalues, eigvects_shape);
    });
}



#define STEQR_LEGACY_INSTANTIATE(back, fp) \
template Event steqr_legacy<back, BATCHLAS_UNPAREN fp>(Queue&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, const Span<std::byte>&, JobType, SteqrParams<BATCHLAS_UNPAREN fp>, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&);

#define STEQR_LEGACY_INSTANTIATE_FOR_BACKEND(back) \
    BATCHLAS_FOR_EACH_REAL_TYPE_1(STEQR_LEGACY_INSTANTIATE, back)

#if BATCHLAS_HAS_CUDA_BACKEND
STEQR_LEGACY_INSTANTIATE_FOR_BACKEND(Backend::CUDA)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
STEQR_LEGACY_INSTANTIATE_FOR_BACKEND(Backend::NETLIB)
#endif

#define STEQR_LEGACY_BUFFER_SIZE_INSTANTIATE(fp) \
template size_t steqr_legacy_buffer_size<BATCHLAS_UNPAREN fp>(Queue&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, const VectorView<BATCHLAS_UNPAREN fp>&, JobType, SteqrParams<BATCHLAS_UNPAREN fp>); 

BATCHLAS_FOR_EACH_REAL_TYPE(STEQR_LEGACY_BUFFER_SIZE_INSTANTIATE)

#undef STEQR_LEGACY_BUFFER_SIZE_INSTANTIATE
#undef STEQR_LEGACY_INSTANTIATE_FOR_BACKEND
#undef STEQR_LEGACY_INSTANTIATE

} // namespace batchlas