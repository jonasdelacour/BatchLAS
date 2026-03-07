#include "gemm_kernels.hh"

#include "../linalg-impl.hh"
#include "../queue.hh"

#include <algorithm>
#include <complex>
#include <sycl/sycl.hpp>

namespace batchlas::sycl_gemm {

namespace {

template <typename T>
class GemmDirectKernel;

template <typename T, int Tile>
class GemmTiledKernel;

template <typename T, int TileM, int TileN, int TileK, int WorkPerThread>
class GemmRegisterTiledKernel;

template <typename T>
inline int ceil_div(int value, int divisor) {
    return (value + divisor - 1) / divisor;
}

template <typename T>
inline T maybe_conj(const T& value, bool conjugate) {
    static_cast<void>(conjugate);
    return value;
}

template <typename T>
inline std::complex<T> maybe_conj(const std::complex<T>& value, bool conjugate) {
    return conjugate ? std::conj(value) : value;
}

template <typename T>
inline T operand_value(const T* ptr,
                       int ld,
                       int batch_offset,
                       int row,
                       int col,
                       Transpose trans) {
    const bool transpose = trans != Transpose::NoTrans;
    const bool conjugate = trans == Transpose::ConjTrans;
    const int source_row = transpose ? col : row;
    const int source_col = transpose ? row : col;
    return maybe_conj(ptr[batch_offset + source_col * ld + source_row], conjugate);
}

template <typename T>
Event launch_direct(Queue& ctx,
                    const MatrixView<T, MatrixFormat::Dense>& A,
                    const MatrixView<T, MatrixFormat::Dense>& B,
                    const MatrixView<T, MatrixFormat::Dense>& C,
                    T alpha,
                    T beta,
                    Transpose transA,
                    Transpose transB) {
    BATCHLAS_KERNEL_TRACE_SCOPE("gemm_sycl_direct");

    const auto [m, k] = get_effective_dims(A, transA);
    const auto [_, n] = get_effective_dims(B, transB);
    static_cast<void>(_);
    constexpr int workgroup = 8;

    const sycl::range<3> local(1, workgroup, workgroup);
    const sycl::range<3> global(static_cast<size_t>(A.batch_size()),
                                static_cast<size_t>(ceil_div<T>(m, workgroup) * workgroup),
                                static_cast<size_t>(ceil_div<T>(n, workgroup) * workgroup));

    ctx->submit([&](sycl::handler& h) {
        const T* a_ptr = A.data_ptr();
        const T* b_ptr = B.data_ptr();
        T* c_ptr = C.data_ptr();
        const int lda = A.ld();
        const int ldb = B.ld();
        const int ldc = C.ld();
        const int stride_a = A.stride();
        const int stride_b = B.stride();
        const int stride_c = C.stride();
        const int batch = A.batch_size();
        const Transpose op_a = transA;
        const Transpose op_b = transB;

        h.parallel_for<GemmDirectKernel<T>>(sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item) {
            const int bid = static_cast<int>(item.get_group(0));
            const int row = static_cast<int>(item.get_global_id(1));
            const int col = static_cast<int>(item.get_global_id(2));
            if (bid >= batch || row >= m || col >= n) {
                return;
            }

            T sum = T(0);
            const int batch_a = bid * stride_a;
            const int batch_b = bid * stride_b;
            const int batch_c = bid * stride_c;
            for (int kk = 0; kk < k; ++kk) {
                const T a_val = operand_value(a_ptr, lda, batch_a, row, kk, op_a);
                const T b_val = operand_value(b_ptr, ldb, batch_b, kk, col, op_b);
                sum += a_val * b_val;
            }
            c_ptr[batch_c + col * ldc + row] = alpha * sum + beta * c_ptr[batch_c + col * ldc + row];
        });
    });

    return ctx.get_event();
}

template <typename T, int Tile>
Event launch_tiled(Queue& ctx,
                   const MatrixView<T, MatrixFormat::Dense>& A,
                   const MatrixView<T, MatrixFormat::Dense>& B,
                   const MatrixView<T, MatrixFormat::Dense>& C,
                   T alpha,
                   T beta) {
    BATCHLAS_KERNEL_TRACE_SCOPE("gemm_sycl_tiled16");

    const auto [m, k] = get_effective_dims(A, Transpose::NoTrans);
    const auto [_, n] = get_effective_dims(B, Transpose::NoTrans);
    static_cast<void>(_);

    const sycl::range<3> local(1, Tile, Tile);
    const sycl::range<3> global(static_cast<size_t>(A.batch_size()),
                                static_cast<size_t>(ceil_div<T>(m, Tile) * Tile),
                                static_cast<size_t>(ceil_div<T>(n, Tile) * Tile));

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<T, 1> tile_a(sycl::range<1>(Tile * Tile), h);
        sycl::local_accessor<T, 1> tile_b(sycl::range<1>(Tile * Tile), h);

        const T* a_ptr = A.data_ptr();
        const T* b_ptr = B.data_ptr();
        T* c_ptr = C.data_ptr();
        const int lda = A.ld();
        const int ldb = B.ld();
        const int ldc = C.ld();
        const int stride_a = A.stride();
        const int stride_b = B.stride();
        const int stride_c = C.stride();
        const int batch = A.batch_size();

        h.parallel_for<GemmTiledKernel<T, Tile>>(sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item) {
            const int bid = static_cast<int>(item.get_group(0));
            const int local_row = static_cast<int>(item.get_local_id(1));
            const int local_col = static_cast<int>(item.get_local_id(2));
            const int row = static_cast<int>(item.get_group(1) * Tile + local_row);
            const int col = static_cast<int>(item.get_group(2) * Tile + local_col);
            if (bid >= batch) {
                return;
            }

            const int batch_a = bid * stride_a;
            const int batch_b = bid * stride_b;
            const int batch_c = bid * stride_c;
            T sum = T(0);

            for (int kk0 = 0; kk0 < k; kk0 += Tile) {
                const int a_k = kk0 + local_col;
                const int b_k = kk0 + local_row;
                tile_a[local_row * Tile + local_col] = (row < m && a_k < k)
                    ? a_ptr[batch_a + a_k * lda + row]
                    : T(0);
                tile_b[local_row * Tile + local_col] = (col < n && b_k < k)
                    ? b_ptr[batch_b + col * ldb + b_k]
                    : T(0);

                item.barrier(sycl::access::fence_space::local_space);
                for (int t = 0; t < Tile && kk0 + t < k; ++t) {
                    sum += tile_a[local_row * Tile + t] * tile_b[t * Tile + local_col];
                }
                item.barrier(sycl::access::fence_space::local_space);
            }

            if (row < m && col < n) {
                c_ptr[batch_c + col * ldc + row] = alpha * sum + beta * c_ptr[batch_c + col * ldc + row];
            }
        });
    });

    return ctx.get_event();
}

template <typename T, int TileM, int TileN, int TileK, int WorkPerThread>
Event launch_register_tiled(Queue& ctx,
                            const MatrixView<T, MatrixFormat::Dense>& A,
                            const MatrixView<T, MatrixFormat::Dense>& B,
                            const MatrixView<T, MatrixFormat::Dense>& C,
                            T alpha,
                            T beta) {
    BATCHLAS_KERNEL_TRACE_SCOPE("gemm_sycl_register_tiled");

    static_assert(TileM % WorkPerThread == 0, "TileM must divide evenly by WorkPerThread");
    static_assert(TileN % WorkPerThread == 0, "TileN must divide evenly by WorkPerThread");

    const auto [m, k] = get_effective_dims(A, Transpose::NoTrans);
    const auto [_, n] = get_effective_dims(B, Transpose::NoTrans);
    static_cast<void>(_);

    constexpr int local_rows = TileM / WorkPerThread;
    constexpr int local_cols = TileN / WorkPerThread;
    constexpr int threads_per_group = local_rows * local_cols;

    const sycl::range<3> local(1, local_rows, local_cols);
    const sycl::range<3> global(static_cast<size_t>(A.batch_size()),
                                static_cast<size_t>(ceil_div<T>(m, TileM) * local_rows),
                                static_cast<size_t>(ceil_div<T>(n, TileN) * local_cols));

    ctx->submit([&](sycl::handler& h) {
        sycl::local_accessor<T, 1> tile_a(sycl::range<1>(TileM * TileK), h);
        sycl::local_accessor<T, 1> tile_b(sycl::range<1>(TileK * TileN), h);

        const T* a_ptr = A.data_ptr();
        const T* b_ptr = B.data_ptr();
        T* c_ptr = C.data_ptr();
        const int lda = A.ld();
        const int ldb = B.ld();
        const int ldc = C.ld();
        const int stride_a = A.stride();
        const int stride_b = B.stride();
        const int stride_c = C.stride();
        const int batch = A.batch_size();

        h.parallel_for<GemmRegisterTiledKernel<T, TileM, TileN, TileK, WorkPerThread>>(
            sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item) {
                const int bid = static_cast<int>(item.get_group(0));
                const int local_row = static_cast<int>(item.get_local_id(1));
                const int local_col = static_cast<int>(item.get_local_id(2));
                if (bid >= batch) {
                    return;
                }

                const int row_base = static_cast<int>(item.get_group(1)) * TileM + local_row * WorkPerThread;
                const int col_base = static_cast<int>(item.get_group(2)) * TileN + local_col * WorkPerThread;
                const int linear_tid = local_row * local_cols + local_col;

                const int batch_a = bid * stride_a;
                const int batch_b = bid * stride_b;
                const int batch_c = bid * stride_c;
                T accum[WorkPerThread][WorkPerThread];
                for (int i = 0; i < WorkPerThread; ++i) {
                    for (int j = 0; j < WorkPerThread; ++j) {
                        accum[i][j] = T(0);
                    }
                }

                for (int kk0 = 0; kk0 < k; kk0 += TileK) {
                    const int a_row = linear_tid % TileM;
                    const int a_col = linear_tid / TileM;
                    const int b_row = linear_tid % TileK;
                    const int b_col = linear_tid / TileK;

                    const int global_a_row = static_cast<int>(item.get_group(1)) * TileM + a_row;
                    const int global_a_col = kk0 + a_col;
                    tile_a[a_col * TileM + a_row] = (a_col < TileK && global_a_row < m && global_a_col < k)
                        ? a_ptr[batch_a + global_a_col * lda + global_a_row]
                        : T(0);

                    const int global_b_row = kk0 + b_row;
                    const int global_b_col = static_cast<int>(item.get_group(2)) * TileN + b_col;
                    tile_b[b_col * TileK + b_row] = (b_col < TileN && global_b_row < k && global_b_col < n)
                        ? b_ptr[batch_b + global_b_col * ldb + global_b_row]
                        : T(0);

                    item.barrier(sycl::access::fence_space::local_space);

                    for (int t = 0; t < TileK && kk0 + t < k; ++t) {
                        T a_frag[WorkPerThread];
                        T b_frag[WorkPerThread];
                        for (int i = 0; i < WorkPerThread; ++i) {
                            a_frag[i] = tile_a[t * TileM + local_row * WorkPerThread + i];
                        }
                        for (int j = 0; j < WorkPerThread; ++j) {
                            b_frag[j] = tile_b[(local_col * WorkPerThread + j) * TileK + t];
                        }
                        for (int i = 0; i < WorkPerThread; ++i) {
                            for (int j = 0; j < WorkPerThread; ++j) {
                                accum[i][j] += a_frag[i] * b_frag[j];
                            }
                        }
                    }

                    item.barrier(sycl::access::fence_space::local_space);
                }

                for (int i = 0; i < WorkPerThread; ++i) {
                    const int row = row_base + i;
                    if (row >= m) {
                        continue;
                    }
                    for (int j = 0; j < WorkPerThread; ++j) {
                        const int col = col_base + j;
                        if (col < n) {
                            c_ptr[batch_c + col * ldc + row] = alpha * accum[i][j] + beta * c_ptr[batch_c + col * ldc + row];
                        }
                    }
                }
            });
    });

    return ctx.get_event();
}

} // namespace

template <typename T>
KernelVariant select_kernel_variant(const MatrixView<T, MatrixFormat::Dense>& A,
                                    const MatrixView<T, MatrixFormat::Dense>& B,
                                    const MatrixView<T, MatrixFormat::Dense>& C,
                                    Transpose transA,
                                    Transpose transB) {
    static_cast<void>(C);
    if (transA != Transpose::NoTrans || transB != Transpose::NoTrans) {
        return KernelVariant::Direct;
    }
    const auto [m, k] = get_effective_dims(A, transA);
    const auto [_, n] = get_effective_dims(B, transB);
    static_cast<void>(_);
    const int max_dim = std::max({m, n, k});
    if constexpr (std::is_same_v<T, float>) {
        if (max_dim >= 64) {
            return KernelVariant::Tiled32x32Register;
        }
    }
    return max_dim <= 64 ? KernelVariant::Direct : KernelVariant::Tiled16;
}

template <typename T>
Event gemm_custom(Queue& ctx,
                  const MatrixView<T, MatrixFormat::Dense>& A,
                  const MatrixView<T, MatrixFormat::Dense>& B,
                  const MatrixView<T, MatrixFormat::Dense>& C,
                  T alpha,
                  T beta,
                  Transpose transA,
                  Transpose transB,
                  ComputePrecision precision) {
    static_cast<void>(precision);
    if (A.batch_size() != B.batch_size() || A.batch_size() != C.batch_size()) {
        throw std::runtime_error("GEMM SYCL custom path requires matching batch sizes");
    }

    const auto [m, k] = get_effective_dims(A, transA);
    const auto [k_b, n] = get_effective_dims(B, transB);
    if (k != k_b || C.rows() != m || C.cols() != n) {
        throw std::runtime_error("GEMM SYCL custom path received incompatible matrix dimensions");
    }

    switch (select_kernel_variant(A, B, C, transA, transB)) {
    case KernelVariant::Direct:
        return launch_direct(ctx, A, B, C, alpha, beta, transA, transB);
    case KernelVariant::Tiled16:
        return launch_tiled<T, 16>(ctx, A, B, C, alpha, beta);
    case KernelVariant::Tiled32x32Register:
        return launch_register_tiled<T, 32, 32, 8, 2>(ctx, A, B, C, alpha, beta);
    }

    return ctx.get_event();
}

template KernelVariant select_kernel_variant<float>(const MatrixView<float, MatrixFormat::Dense>&,
                                                    const MatrixView<float, MatrixFormat::Dense>&,
                                                    const MatrixView<float, MatrixFormat::Dense>&,
                                                    Transpose,
                                                    Transpose);
template KernelVariant select_kernel_variant<double>(const MatrixView<double, MatrixFormat::Dense>&,
                                                     const MatrixView<double, MatrixFormat::Dense>&,
                                                     const MatrixView<double, MatrixFormat::Dense>&,
                                                     Transpose,
                                                     Transpose);
template KernelVariant select_kernel_variant<std::complex<float>>(const MatrixView<std::complex<float>, MatrixFormat::Dense>&,
                                                                  const MatrixView<std::complex<float>, MatrixFormat::Dense>&,
                                                                  const MatrixView<std::complex<float>, MatrixFormat::Dense>&,
                                                                  Transpose,
                                                                  Transpose);
template KernelVariant select_kernel_variant<std::complex<double>>(const MatrixView<std::complex<double>, MatrixFormat::Dense>&,
                                                                   const MatrixView<std::complex<double>, MatrixFormat::Dense>&,
                                                                   const MatrixView<std::complex<double>, MatrixFormat::Dense>&,
                                                                   Transpose,
                                                                   Transpose);

template Event gemm_custom<float>(Queue&,
                                  const MatrixView<float, MatrixFormat::Dense>&,
                                  const MatrixView<float, MatrixFormat::Dense>&,
                                  const MatrixView<float, MatrixFormat::Dense>&,
                                  float,
                                  float,
                                  Transpose,
                                  Transpose,
                                  ComputePrecision);
template Event gemm_custom<double>(Queue&,
                                   const MatrixView<double, MatrixFormat::Dense>&,
                                   const MatrixView<double, MatrixFormat::Dense>&,
                                   const MatrixView<double, MatrixFormat::Dense>&,
                                   double,
                                   double,
                                   Transpose,
                                   Transpose,
                                   ComputePrecision);
template Event gemm_custom<std::complex<float>>(Queue&,
                                                const MatrixView<std::complex<float>, MatrixFormat::Dense>&,
                                                const MatrixView<std::complex<float>, MatrixFormat::Dense>&,
                                                const MatrixView<std::complex<float>, MatrixFormat::Dense>&,
                                                std::complex<float>,
                                                std::complex<float>,
                                                Transpose,
                                                Transpose,
                                                ComputePrecision);
template Event gemm_custom<std::complex<double>>(Queue&,
                                                 const MatrixView<std::complex<double>, MatrixFormat::Dense>&,
                                                 const MatrixView<std::complex<double>, MatrixFormat::Dense>&,
                                                 const MatrixView<std::complex<double>, MatrixFormat::Dense>&,
                                                 std::complex<double>,
                                                 std::complex<double>,
                                                 Transpose,
                                                 Transpose,
                                                 ComputePrecision);

} // namespace batchlas::sycl_gemm