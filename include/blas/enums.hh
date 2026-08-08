#pragma once
#include <complex>
#include <concepts>
#include <cstdint>
#include <type_traits>
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

    template <typename T>
    struct is_std_complex : std::false_type {};

    template <typename T>
    struct is_std_complex<std::complex<T>> : std::true_type {};

    template <typename T>
    inline constexpr bool is_std_complex_v = is_std_complex<T>::value;

    template <typename T>
    concept RealScalar = std::floating_point<T>;

    template <typename T>
    concept ComplexScalar = is_std_complex_v<T>;

    template <typename T>
    concept FloatingOrComplexScalar = RealScalar<T> || ComplexScalar<T>;

    enum class MatrixFormat {
        Dense,
        CSR,    // Compressed Sparse Row
        CSC,    // Compressed Sparse Column
        COO,    // Coordinate
        SELL,   // Sliced ELLPACK
        BSR,    // Blocked Sparse Row
        BLOCKED_ELL // Blocked ELLPACK
    };

    template <MatrixFormat F>
    concept DenseMatrixFormat = F == MatrixFormat::Dense;

    template <MatrixFormat F>
    concept CsrMatrixFormat = F == MatrixFormat::CSR;

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

    // SVD vector output policy, LAPACK jobu/jobvt semantics.
    //
    // With k = min(m, n), for an m x n input:
    //   None  -- the factor is not computed and its MatrixView is not touched.
    //   All   -- LAPACK 'A'. U is m x m, V^H is n x n.
    //   Thin  -- LAPACK 'S'. U is m x k (the first k left singular vectors),
    //            V^H is k x n (the first k right singular vectors, conjugated).
    //
    // Thin exists because All is unusable on tall-skinny input: a 10000 x 32
    // problem has to materialise a 10000 x 10000 U, 400 MB per matrix in float,
    // so batch=4 needs 1.6 GB for a factor whose last 9968 columns are an
    // arbitrary orthonormal completion the caller did not ask for.
    //
    // The key identity, which most of the implementation rests on: Thin and All
    // DIFFER ON AT MOST ONE SIDE. For m <= n, k == m, so a thin U (m x k) is
    // exactly a full U (m x m); for m >= n, k == n, so a thin V^H is exactly a
    // full V^H. Square input has Thin == All on both sides. Entry points
    // therefore canonicalise Thin to All whenever the shapes coincide (see
    // canonical_jobu / canonical_jobvh below) and only the genuinely thinner
    // side has to be handled -- or rejected -- by any given route.
    //
    // LAPACK's 'O' (overwrite A with one of the factors) is deliberately absent;
    // add it as a further enumerator if it is ever wanted, since appending keeps
    // the existing ordinals stable for the benchmarks that pass jobs as ints.
    enum class SvdVectors {
        None,
        All,
        Thin
    };

    // Number of columns of U / rows of V^H implied by a job, given the input
    // shape and k = min(m, n). None yields 0: nothing is written.
    inline constexpr int64_t svd_u_cols(SvdVectors job, int64_t m, int64_t k) {
        return job == SvdVectors::All ? m : (job == SvdVectors::Thin ? k : 0);
    }

    inline constexpr int64_t svd_vh_rows(SvdVectors job, int64_t n, int64_t k) {
        return job == SvdVectors::All ? n : (job == SvdVectors::Thin ? k : 0);
    }

    // Rewrite Thin to All when the two request the same shape, so that a route
    // which cannot produce a genuinely thin factor still serves every request
    // where "thin" is not actually asking for anything smaller. Call these once
    // at each entry point, and pass the canonical values onward -- in
    // particular, the buffer_size and the run path must canonicalise
    // identically or the workspace is sized for a different computation than
    // the one performed.
    inline constexpr SvdVectors canonical_jobu(SvdVectors job, int64_t m, int64_t k) {
        return (job == SvdVectors::Thin && k == m) ? SvdVectors::All : job;
    }

    inline constexpr SvdVectors canonical_jobvh(SvdVectors job, int64_t n, int64_t k) {
        return (job == SvdVectors::Thin && k == n) ? SvdVectors::All : job;
    }

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

    enum class ApplyOrder {
        Forward,
        Backward
    };
    
    // Algorithm family used by `syevx` (partial symmetric eigensolve).
    //
    // `Auto` picks based on matrix format, size and the requested fraction of the
    // spectrum; see `syevx_select_algorithm`. Set per call via SyevxParams::method,
    // or globally via BATCHLAS_SYEVX_ALGORITHM
    // (auto|direct|direct_subset|filtered|lobpcg).
    //
    // Precedence: the environment variable WINS over SyevxParams::method, matching
    // the BATCHLAS_SYEV_PROVIDER convention, so that a whole application can be
    // forced onto one algorithm for diagnosis or benchmarking.
    //
    // A choice that is not available for the given scalar type or matrix format
    // degrades to the nearest implemented one rather than failing: DirectSubset
    // needs a real type and dense input, and Filtered is not implemented at all.
    enum class SyevxAlgorithm {
        Auto,           // Heuristic selection (default)
        Direct,         // Full syev + select the requested eigenpairs
        DirectSubset,   // Two-stage reduction + subset tridiagonal solve (Tier 2, not yet implemented)
        Filtered,       // Chebyshev-filtered subspace iteration (Tier 3, not yet implemented)
        LOBPCG          // Locally Optimal Block Preconditioned Conjugate Gradient
    };

    // How `syevx` chooses which part of the spectrum to return.
    //
    // `Extremal` is the historical behaviour and the default: `neigs` eigenpairs
    // from one end, chosen by SyevxParams::find_largest, returned descending for
    // the largest and ascending for the smallest. It is a special case of `Index`
    // -- [n-neigs, n-1] or [0, neigs-1] -- and is normalized to one internally by
    // `syevx_resolve_range`; it exists as a distinct value so that no existing
    // caller's behaviour depends on a default that changed meaning.
    //
    // Deliberately NOT `EigenRangeType`, whose `All` member -- the natural default
    // -- means "every eigenvalue", which is not what syevx's default does.
    // `EigenRangeType` stays the tridiagonal-layer vocabulary; `SyevxSelect` is the
    // user-facing one, and the two are converted in exactly one place.
    enum class SyevxSelect {
        Extremal,  // neigs from one end; SyevxParams::find_largest picks the end
        Index,     // SyevxParams::il .. iu inclusive, 0-based, ascending spectrum
        Value      // every eigenvalue in the half-open interval (vl, vu]
    };

    // Preconditioner family used by the LOBPCG path of `syevx`. Set per call via
    // SyevxParams::preconditioner_type; `Auto` picks ILU(k) when a factor has been
    // supplied (or requested via SyevxParams::build_preconditioner) and otherwise
    // takes the default from BATCHLAS_SYEVX_PRECONDITIONER
    // (auto|none|jacobi|jacobi_shifted|iluk), defaulting to `None`.
    //
    // The two Jacobi forms are different operators despite the shared name, and the
    // difference is not cosmetic (measured iteration counts, single precision,
    // n = 64..512, k = 2..16):
    //
    //   `Jacobi` = diag(A)^{-1} approximates A^{-1}, so -- exactly like ILU(k) -- it
    //   is only valid for the SMALLEST eigenpairs, and `find_largest` is rejected
    //   with it. On strongly graded (nearly diagonal) matrices it is worth 2-7x
    //   fewer iterations; on a random symmetric matrix, whose diagonal is neither
    //   dominant nor sign-definite, it is not a valid preconditioner at all, so the
    //   implementation falls back to the identity per batch item whose diagonal is
    //   not uniformly positive rather than diverging.
    //
    //   `JacobiShifted` = (diag(A) - lambda I)^{-1} takes its shift from the current
    //   Ritz value, which makes it valid at BOTH ends -- it targets whatever is near
    //   lambda. But it degenerates precisely where the unshifted form wins: as
    //   diag(A) -> A the operator becomes the exact inverse of (A - lambda I) and
    //   the preconditioned residual converges to X itself, so the new search
    //   direction is annihilated by the subsequent orthogonalization against X.
    //   Measured neutral (0.85-1.2x) on random symmetric input and 0.2-0.9x on
    //   graded input. It is offered because the constant-diagonal case is provably
    //   a no-op and the general case is safe, not because it was found to pay.
    //
    // Neither is chosen by `Auto`: on the matrices measured here neither is a free
    // win, so picking one implicitly would be a regression for somebody.
    //
    // Precedence differs deliberately from SyevxAlgorithm: the environment variable
    // only supplies the *default*, it does not override an explicit request. An
    // algorithm can always be substituted for another; a preconditioner cannot --
    // an ILU(k) factor a caller built and handed in has no substitute, and silently
    // ignoring it (or, worse, silently ignoring a request for it) would be a
    // correctness surprise rather than a performance one.
    //
    // Only the LOBPCG algorithm uses this; the direct and filtered paths ignore it.
    enum class SyevxPreconditioner {
        Auto,           // ILU(k) if configured, else the environment default, else None
        None,           // Unpreconditioned
        Jacobi,         // diag(A)^{-1}; dense and CSR, smallest-first only
        JacobiShifted,  // (diag(A) - lambda I)^{-1}; dense and CSR, either end
        ILUK            // Supplied or syevx-built ILU(k) factor; CSR and smallest-first only
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

    enum class VectorOrientation {
        Row,
        Column
    };

    enum class NormType {
        Frobenius, //Most commonly used
        One,       //Maximum absolute column sum
        Inf,       //Maximum absolute row sum
        Max,       //Maximum absolute value
        Spectral   //Spectral (L2) norm, symmetric/Hermitian only
    };

    enum class Layout {
        RowMajor,
        ColMajor
    };
}