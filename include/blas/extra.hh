#include <blas/matrix.hh>
#include <blas/enums.hh>
#include <util/sycl-vector.hh>
#include <util/sycl-device-queue.hh>

namespace batchlas
{

    // Memory passed from outside
    template <typename T, MatrixFormat MF>
    void norm(Queue &ctx,
              const MatrixView<T, MF> &A,
              const NormType norm_type,
              const Span<T> norms);

    // Convenience function which allocates and returns the results stored in an array.
    template <typename T, MatrixFormat MF>
    UnifiedVector<T> norm(Queue &ctx,
                          const MatrixView<T, MF> &A,
                          const NormType norm_type);

    // Memory passed from outside
    template <typename T, MatrixFormat MF>
    void cond(Queue &ctx,
              const MatrixView<T, MF> &A,
              const NormType norm_type,
              const Span<T> conds);

    // Convenience function which allocates and returns the results stored in an array.
    template <typename T, MatrixFormat MF>
    UnifiedVector<T> cond(Queue &ctx,
                          const MatrixView<T, MF> &A,
                          const NormType norm_type);

}