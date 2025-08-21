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

    // Forwarding overload (owning A)
    template <typename T, MatrixFormat MF>
    inline Event norm(Queue &ctx,
              const Matrix<T, MF> &A,
              const NormType norm_type,
              const Span<float_t<T>> norms) {
        return norm<T,MF>(ctx, MatrixView<T,MF>(A), norm_type, norms);
    }

    // Convenience function which allocates and returns the results stored in an array.
    template <typename T, MatrixFormat MF>
    UnifiedVector<float_t<T>> norm(Queue &ctx,
                          const MatrixView<T, MF> &A,
                          const NormType norm_type = NormType::Frobenius);

    // Forwarding overload (owning A)
    template <typename T, MatrixFormat MF>
    inline UnifiedVector<float_t<T>> norm(Queue &ctx,
                          const Matrix<T, MF> &A,
                          const NormType norm_type = NormType::Frobenius) {
        return norm<T,MF>(ctx, MatrixView<T,MF>(A), norm_type);
    }

    //Memory passed from outside
    template <Backend B, typename T, MatrixFormat MF>
    Event cond(Queue &ctx,
              const MatrixView<T, MF> &A,
              const NormType norm_type,
              const Span<T> conds,
              const Span<std::byte> workspace);

    // Forwarding overload (owning A)
    template <Backend B, typename T, MatrixFormat MF>
    inline Event cond(Queue &ctx,
              const Matrix<T, MF> &A,
              const NormType norm_type,
              const Span<T> conds,
              const Span<std::byte> workspace) {
        return cond<B,T,MF>(ctx, MatrixView<T,MF>(A), norm_type, conds, workspace);
    }

    size_t cond_buffer_size(Queue &ctx,
                            const MatrixView<float, MatrixFormat::Dense> &A,
                            const NormType norm_type);

    //Convenience function which allocates memory internally
    template <Backend B, typename T, MatrixFormat MF>
    UnifiedVector<T> cond(Queue &ctx,
                          const MatrixView<T, MF> &A,
                          const NormType norm_type);

    // Forwarding overload (owning A)
    template <Backend B, typename T, MatrixFormat MF>
    inline UnifiedVector<T> cond(Queue &ctx,
                          const Matrix<T, MF> &A,
                          const NormType norm_type) {
        return cond<B,T,MF>(ctx, MatrixView<T,MF>(A), norm_type);
    }

    // Batched matrix transpose into preallocated output
    template <typename T, MatrixFormat MF>
    Event transpose(Queue &ctx,
                    const MatrixView<T, MF> &A,
                    const MatrixView<T, MF> &B);

    // Forwarding overload (owning A and B)
    template <typename T, MatrixFormat MF>
    inline Event transpose(Queue &ctx,
                    const Matrix<T, MF> &A,
                    const Matrix<T, MF> &B) {
        return transpose<T,MF>(ctx, MatrixView<T,MF>(A), MatrixView<T,MF>(B));
    }

    // Convenience overload allocating the output matrix
    template <typename T, MatrixFormat MF>
    Matrix<T, MF> transpose(Queue &ctx,
                            const MatrixView<T, MF> &A);

    // Forwarding overload (owning A)
    template <typename T, MatrixFormat MF>
    inline Matrix<T, MF> transpose(Queue &ctx,
                            const Matrix<T, MF> &A) {
        return transpose<T,MF>(ctx, MatrixView<T,MF>(A));
    }

}
