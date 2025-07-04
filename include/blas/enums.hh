#pragma once
#include <complex>
namespace batchlas {
    template<typename T>
    struct base_type {
        using type = T;
    };

    template<typename T>
    struct base_type<std::complex<T>> {
        using type = T;
    };

    template<typename T>
    using float_t = typename base_type<T>::type;

    enum class Backend {
        AUTO,
        CUDA,
        ROCM,
        MKL,
        MAGMA,
        SYCL,
        NETLIB
        // Add more as needed
    };

    enum class BackendLibrary {
        CUBLAS,     //Belongs to CUDA backend
        CUSPARSE,   //Belongs to CUDA backend
        CUSOLVER,   //Belongs to CUDA backend
        ROCBLAS,    //Belongs to ROCM backend
        ROCSPARSE,  //Belongs to ROCM backend
        ROCSOLVER,  //Belongs to ROCM backend
        MAGMA,      //Belongs to MAGMA backend
        MKL,        //Belongs to MKL backend
        CBLAS,      //Belongs to NETLIB backend
        LAPACKE     //Belongs to NETLIB backend
    };

    enum class Transpose {
        NoTrans,
        Trans,
        ConjTrans
    };

    enum class JobType {
        EigenVectors,
        NoEigenVectors
    };

    enum class Uplo {
        Upper,
        Lower
    };

    enum class Diag {
        NonUnit,
        Unit
    };

    enum class Side {
        Left,
        Right
    };

    enum class SortOrder {
        Ascending,
        Descending
    };
    
    enum class OrthoAlgorithm {
        Chol2,          //Default
        Cholesky,       //Rarely sufficient
        ShiftChol3,     //More stable than Chol2
        Householder,    
        CGS2,           //Classical Gram-Schmidt with 2 iterations
        SVQB,       
        SVQB2,          //2 Iterations of SVQB
        NUM_ALGORITHMS  //Used to determine the number of algorithms
    };
    
    //Some of the types are not supported by all backends, compilation errors will make this apparent
    enum class ComputePrecision {
        Default, //Use same precision as input
        F32,
        F64,
        F16,
        BF16,
        TF32
    };

    enum class MatrixFormat {
        Dense,
        CSR,    // Compressed Sparse Row
        CSC,    // Compressed Sparse Column
        COO,    // Coordinate
        SELL,   // Sliced ELLPACK
        BSR,    // Blocked Sparse Row
        BLOCKED_ELL // Blocked ELLPACK
    };

    enum class NormType {
        Frobenius, //Most commonly used
        One,       //Maximum absolute column sum
        Inf,       //Maximum absolute row sum
        Max       //Maximum absolute value
    };

    enum class Layout {
        RowMajor,
        ColMajor
    };
}