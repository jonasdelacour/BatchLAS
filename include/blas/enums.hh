#pragma once

namespace batchlas {
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

    enum class BatchType {
        Single,
        Batched
    };
    
    enum class Transpose {
        NoTrans,
        Trans
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
    
    enum class OrthoAlgorithm {
        Chol2,      //Default
        Cholesky,   //Rarely sufficient
        ShiftChol3, //More stable than Chol2
        Householder, //Most numerically stable
        ModifiedGramSchmidt
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

    enum class Format {
        COO,        //Coordinate
        CSR,        //Compressed Sparse Row
        CSC,        //Compressed Sparse Column
        SELL,       //Sliced ELLPACK
        BSR,        //Blocked Sparse Row
        BLOCKED_ELL //Blocked ELLPACK
    };

    enum class Layout {
        RowMajor,
        ColMajor
    };
}