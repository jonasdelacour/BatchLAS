#include <blas/linalg.hh>
#include <util/sycl-device-queue.hh>
#include <batchlas/backend_config.h>
#include <complex>
#include <vector>
#include <iostream>


namespace batchlas {

template <Backend Ba, typename T>
Event trmm(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                const MatrixView<T, MatrixFormat::Dense>& B,
                const MatrixView<T, MatrixFormat::Dense>& C,
                T alpha,
                Side side,
                Uplo uplo,
                Transpose transA,
                Diag diag){
    constexpr auto recursion_stop_size = 256;
    auto n = A.rows();

    //Implement recursive TRMM algorithm here
    //If the size of A is less than recursion_stop_size, use a simple kernel
    if (n <= recursion_stop_size) {
        //A.triangularize(ctx, uplo, diag).wait();
        if (side == Side::Left) {
            return gemm<Ba>(ctx, A, B, C, alpha, T(1.0), transA, Transpose::NoTrans);
        } else {
            return gemm<Ba>(ctx, B, A, C, alpha, T(1.0), Transpose::NoTrans, transA);
        }
    }
    //Otherwise, split the matrix and call trmm recursively
    auto mid_row = n / 2;
    auto mid_col = n / 2;

    //Partition A into four sub-matrices: (Assuming A is lower triangular)
    // A11 | 0  
    // ----+----  
    // A21 | A22  
    
    // Create sub-matrices for the recursive calls
    auto A11 = A({0, mid_row}, {0, mid_col});
    auto A21 = A({mid_row, SliceEnd()}, {0, mid_col});
    auto A22 = A({mid_row, SliceEnd()}, {mid_col, SliceEnd()});

    auto C1 = C({0, mid_row}, Slice());
    auto C2 = C({mid_row, SliceEnd()}, Slice());

    // Partition B conformably with the split of A on the rows
    auto B1 = B({0, mid_row}, Slice());
    auto B2 = B({mid_row, SliceEnd()}, Slice());

    // Call trmm recursively on the sub-matrices
    trmm<Ba>(ctx, A11, B1, C1, alpha, side, uplo, transA, diag);
    trmm<Ba>(ctx, A22, B2, C2, alpha, side, uplo, transA, diag);
    gemm<Ba>(ctx, A21, B1, C2, alpha, T(1.0), transA, Transpose::NoTrans);

    return ctx.get_event();
}


#define TRMM_INSTANTIATE(back, fp) \
    template Event trmm<back, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        fp, \
        Side, \
        Uplo, \
        Transpose, \
        Diag); \


#define INSTANTIATE_TRMM_FOR_BACKEND(back)\
    TRMM_INSTANTIATE(back, float) \
    TRMM_INSTANTIATE(back, double)\
    TRMM_INSTANTIATE(back, std::complex<float>)\
    TRMM_INSTANTIATE(back, std::complex<double>)

#if BATCHLAS_HAS_CUDA_BACKEND
        INSTANTIATE_TRMM_FOR_BACKEND(Backend::CUDA)
    #endif
    #if BATCHLAS_HAS_ROCM_BACKEND 
        INSTANTIATE_TRMM_FOR_BACKEND(Backend::ROCM)
    #endif
    #if BATCHLAS_HAS_HOST_BACKEND 
        INSTANTIATE_TRMM_FOR_BACKEND(Backend::NETLIB)
    #endif

#undef TRMM_INSTANTIATE

} // namespace batchlas