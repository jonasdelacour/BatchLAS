#include <blas/matrix.hh>
#include <blas/enums.hh>
#include <util/sycl-vector.hh>
#include <util/sycl-device-queue.hh>

namespace batchlas
{

    // Memory passed from outside
    template <typename T, MatrixFormat MF>
    Event norm(Queue &ctx,
              const MatrixView<T, MF> &A,
              const NormType norm_type,
              const Span<float_t<T>> norms);

    // Convenience function which allocates and returns the results stored in an array.
    template <typename T, MatrixFormat MF>
    UnifiedVector<float_t<T>> norm(Queue &ctx,
                          const MatrixView<T, MF> &A,
                          const NormType norm_type = NormType::Frobenius);

    //Memory passed from outside
    template <Backend B, typename T, MatrixFormat MF>
    Event cond(Queue &ctx,
              const MatrixView<T, MF> &A,
              const NormType norm_type,
              const Span<T> conds,
              const Span<std::byte> workspace);

    size_t cond_buffer_size(Queue &ctx,
                            const MatrixView<float, MatrixFormat::Dense> &A,
                            const NormType norm_type);

    //Convenience function which allocates memory internally
    template <Backend B, typename T, MatrixFormat MF>
    UnifiedVector<T> cond(Queue &ctx,
                          const MatrixView<T, MF> &A,
                          const NormType norm_type);

    // Batched matrix transpose into preallocated output
    template <typename T, MatrixFormat MF>
    Event transpose(Queue &ctx,
                    const MatrixView<T, MF> &A,
                    const MatrixView<T, MF> &B);

    // Convenience overload allocating the output matrix
    template <typename T, MatrixFormat MF>
    Matrix<T, MF> transpose(Queue &ctx,
                            const MatrixView<T, MF> &A);

}
