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

    //Partition A into four sub-matrices: 
    // A11 | 0        A11 | A12
    // ----+----  or  ----+-----   depending on uplo (Lower, Upper)
    // A21 | A22       0  | A22
    
    // Create sub-matrices for the recursive calls
    auto A11 = A({0, mid_row}, {0, mid_col});
    auto A12 = A({0, mid_row}, {mid_col, SliceEnd()});
    auto A21 = A({mid_row, SliceEnd()}, {0, mid_col});
    auto A22 = A({mid_row, SliceEnd()}, {mid_col, SliceEnd()});

    
    //If side is left, we need to partition B and C into two row-blocks:
    //  B1   C1
    // -+-   -+-
    //  B2   C2
    
    // If side is right, we need to partition B and C into two column-blocks:
    // B1 | B2
    // 
    // C1 | C2

    auto C1 = side == Side::Left ? C({0, mid_row}, Slice()) : C(Slice(), {0, mid_col});
    auto C2 = side == Side::Left ? C({mid_row, SliceEnd()}, Slice()) : C(Slice(), {mid_col, SliceEnd()});

    auto B1 = side == Side::Left ? B({0, mid_row}, Slice()) : B(Slice(), {0, mid_col});
    auto B2 = side == Side::Left ? B({mid_row, SliceEnd()}, Slice()) : B(Slice(), {mid_col, SliceEnd()});

    bool is_transposed = (transA  == Transpose::ConjTrans) || (transA == Transpose::Trans);
    bool is_ll_or_ur = (uplo == Uplo::Lower && side == Side::Left) || (uplo == Uplo::Upper && side == Side::Right);
    
    // Call trmm recursively on the sub-matrices
    trmm<Ba>(ctx, A22, B2, C2, alpha, side, uplo, transA, diag);
    trmm<Ba>(ctx, A11, B1, C1, alpha, side, uplo, transA, diag);
    if (!ctx.in_order()) ctx.wait();
    auto B_block = is_ll_or_ur ?
        (is_transposed ? B2 : B1) :
        (is_transposed ? B1 : B2);
    auto C_block = is_ll_or_ur ?
        (is_transposed ? C1 : C2) :
        (is_transposed ? C2 : C1);
    auto A_block = uplo == Uplo::Lower ? A21 : A12;
    return gemm<Ba>(ctx, side == Side::Left ? A_block : B_block, 
                    side == Side::Left ? B_block : A_block, 
                    C_block, alpha, T(1.0), 
                    side == Side::Left ? transA : Transpose::NoTrans, 
                    side == Side::Left ? Transpose::NoTrans : transA);
            

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