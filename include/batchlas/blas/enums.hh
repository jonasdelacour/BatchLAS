#pragma once
#include <batchlas/export.hh>

#include <complex>
#include <concepts>
#include <cstdint>
#include <iosfwd>
#include <string_view>
#include <type_traits>

/// @file
/// @brief Enumerations, scalar traits and scalar concepts shared by every public header.
///
/// Every enum here has a `constexpr std::string_view to_string(E)` next to its
/// definition that returns the enumerator's own spelling, and one templated
/// `operator<<` at the end of the namespace prints all of them.

// Backend and MatrixFormat carry BATCHLAS_API because they are non-type template
// arguments: an instantiation gets the minimum visibility of its arguments, so an
// unannotated enum hides every instantiation on it and consumers fail to link.
// Annotate any further enum that becomes a template argument.
// evidence: docs/design/symbol-visibility.md#symbol-visibility-enums-used-as-template-arguments
namespace batchlas {
    /// @addtogroup enums
    /// @{

    /// @brief Real type underlying a scalar: `T` itself for a real `T`, `R` for `std::complex<R>`.
    template<typename T>
    struct base_type {
        using type = T;
    };

    /// @brief Specialisation that strips `std::complex`.
    template<typename T>
    struct base_type<std::complex<T>> {
        using type = T;
    };

    /// @brief Shorthand for `base_type<T>::type`: the precision of `T` (tolerances, norms, eigenvalues).
    template<typename T>
    using float_t = typename base_type<T>::type;

    /// @brief True for `std::complex<R>`, false otherwise.
    template <typename T>
    struct is_std_complex : std::false_type {};

    /// @brief Specialisation for `std::complex<R>`.
    template <typename T>
    struct is_std_complex<std::complex<T>> : std::true_type {};

    /// @brief Value of is_std_complex<T>.
    template <typename T>
    inline constexpr bool is_std_complex_v = is_std_complex<T>::value;

    /// @brief A real floating-point scalar (`float`, `double`).
    template <typename T>
    concept RealScalar = std::floating_point<T>;

    /// @brief A `std::complex` scalar (`std::complex<float>`, `std::complex<double>`).
    template <typename T>
    concept ComplexScalar = is_std_complex_v<T>;

    /// @brief Any scalar the numerical entry points accept: real or complex floating point.
    template <typename T>
    concept FloatingOrComplexScalar = RealScalar<T> || ComplexScalar<T>;

    /// @brief Storage format of a Matrix / MatrixView, used as a template argument.
    ///
    /// Only `Dense` and `CSR` have storage implementations; the other enumerators
    /// name formats and are not instantiated by the library.
    /// @see @ref design_matrix_model
    enum class BATCHLAS_API MatrixFormat {
        Dense,        ///< Column-major dense storage with leading dimension and batch stride.
        CSR,          ///< Compressed Sparse Row.
        CSC,          ///< Compressed Sparse Column.
        COO,          ///< Coordinate.
        SELL,         ///< Sliced ELLPACK.
        BSR,          ///< Blocked Sparse Row.
        BLOCKED_ELL   ///< Blocked ELLPACK.
    };

    /// @brief Spelling of @p v as in source.
    inline constexpr std::string_view to_string(MatrixFormat v) {
        switch (v) {
            case MatrixFormat::Dense:       return "Dense";
            case MatrixFormat::CSR:         return "CSR";
            case MatrixFormat::CSC:         return "CSC";
            case MatrixFormat::COO:         return "COO";
            case MatrixFormat::SELL:        return "SELL";
            case MatrixFormat::BSR:         return "BSR";
            case MatrixFormat::BLOCKED_ELL: return "BLOCKED_ELL";
        }
        return "MatrixFormat(?)";
    }

    /// @brief Constrains a member or overload to dense storage.
    template <MatrixFormat F>
    concept DenseMatrixFormat = F == MatrixFormat::Dense;

    /// @brief Constrains a member or overload to CSR storage.
    template <MatrixFormat F>
    concept CsrMatrixFormat = F == MatrixFormat::CSR;

    /// @brief Library family an entry point dispatches to, used as a template argument.
    ///
    /// Callers normally do not spell it: the queue-taking entry points take the
    /// backend from the batchlas::Queue (Queue::backend(), Queue::set_backend()).
    /// Only the backends compiled into the build (`BATCHLAS_HAS_*_BACKEND`) are
    /// available; Queue::backend_available() reports which.
    enum class BATCHLAS_API Backend {
        AUTO,    ///< A request, not a backend: the Queue resolves it from its device on first query.
        CUDA,    ///< NVIDIA GPU: cuBLAS / cuSOLVER / cuSPARSE and the native SYCL kernels.
        ROCM,    ///< AMD GPU: rocBLAS / rocSOLVER / rocSPARSE and the native SYCL kernels.
        MKL,     ///< Intel GPU through oneMKL.
        MAGMA,   ///< Reserved; no dispatch target, never available.
        SYCL,    ///< Reserved; no dispatch target, never available.
        NETLIB   ///< Host CBLAS / LAPACKE; the fallback for any device.
        // Add more as needed
    };

    /// @brief Spelling of @p v as in source.
    inline constexpr std::string_view to_string(Backend v) {
        switch (v) {
            case Backend::AUTO:   return "AUTO";
            case Backend::CUDA:   return "CUDA";
            case Backend::ROCM:   return "ROCM";
            case Backend::MKL:    return "MKL";
            case Backend::MAGMA:  return "MAGMA";
            case Backend::SYCL:   return "SYCL";
            case Backend::NETLIB: return "NETLIB";
        }
        return "Backend(?)";
    }

    /// @brief A vendor library inside a Backend; selects library-specific type and handle mappings.
    enum class BackendLibrary {
        CUBLAS,     ///< Backend::CUDA.
        CUSPARSE,   ///< Backend::CUDA.
        CUSOLVER,   ///< Backend::CUDA.
        ROCBLAS,    ///< Backend::ROCM.
        ROCSPARSE,  ///< Backend::ROCM.
        ROCSOLVER,  ///< Backend::ROCM.
        MAGMA,      ///< Backend::MAGMA.
        MKL,        ///< Backend::MKL.
        CBLAS,      ///< Backend::NETLIB.
        LAPACKE     ///< Backend::NETLIB.
    };

    /// @brief Spelling of @p v as in source.
    inline constexpr std::string_view to_string(BackendLibrary v) {
        switch (v) {
            case BackendLibrary::CUBLAS:    return "CUBLAS";
            case BackendLibrary::CUSPARSE:  return "CUSPARSE";
            case BackendLibrary::CUSOLVER:  return "CUSOLVER";
            case BackendLibrary::ROCBLAS:   return "ROCBLAS";
            case BackendLibrary::ROCSPARSE: return "ROCSPARSE";
            case BackendLibrary::ROCSOLVER: return "ROCSOLVER";
            case BackendLibrary::MAGMA:     return "MAGMA";
            case BackendLibrary::MKL:       return "MKL";
            case BackendLibrary::CBLAS:     return "CBLAS";
            case BackendLibrary::LAPACKE:   return "LAPACKE";
        }
        return "BackendLibrary(?)";
    }

    /// @brief The operator op() applied to an operand: op(A) = A, A^T or A^H.
    enum class Transpose {
        NoTrans,   ///< op(A) = A.
        Trans,     ///< op(A) = A^T.
        ConjTrans  ///< op(A) = A^H (the same as Trans for real scalars).
    };

    /// @brief Spelling of @p v as in source.
    inline constexpr std::string_view to_string(Transpose v) {
        switch (v) {
            case Transpose::NoTrans:   return "NoTrans";
            case Transpose::Trans:     return "Trans";
            case Transpose::ConjTrans: return "ConjTrans";
        }
        return "Transpose(?)";
    }

    /// @brief Whether an eigensolver computes eigenvectors (LAPACK `jobz`).
    enum class JobType {
        EigenVectors,    ///< Eigenvalues and eigenvectors (`jobz = 'V'`).
        NoEigenVectors   ///< Eigenvalues only (`jobz = 'N'`).
    };

    /// @brief Spelling of @p v as in source.
    inline constexpr std::string_view to_string(JobType v) {
        switch (v) {
            case JobType::EigenVectors:   return "EigenVectors";
            case JobType::NoEigenVectors: return "NoEigenVectors";
        }
        return "JobType(?)";
    }

    /// @brief Which singular vectors gesvd computes for one factor (LAPACK `jobu` / `jobvt`).
    ///
    /// For an m x n input with k = min(m, n):
    /// | Value  | LAPACK | U       | V^H     |
    /// | ------ | ------ | ------- | ------- |
    /// | `None` | 'N'    | not computed; its MatrixView is not touched | same |
    /// | `All`  | 'A'    | m x m   | n x n   |
    /// | `Thin` | 'S'    | m x k, the first k left vectors | k x n, the first k right vectors, conjugated |
    ///
    /// Thin and All differ on at most one side: for m <= n a thin U is the full U,
    /// for m >= n a thin V^H is the full V^H, and for square input they coincide on
    /// both. Entry points canonicalise with canonical_jobu() / canonical_jobvh().
    /// LAPACK's 'O' (overwrite A) is not offered.
    /// @see @ref design_gesvd
    // Append new enumerators only: benchmarks pass jobs as ints, so the ordinals are stable API.
    // evidence: docs/design/gesvd.md#gesvd-design-why-svdvectorsthin-exists
    enum class SvdVectors {
        None,  ///< Do not compute this factor.
        All,   ///< Full square factor (LAPACK 'A').
        Thin   ///< Economy factor with k = min(m, n) vectors (LAPACK 'S').
    };

    /// @brief Spelling of @p v as in source.
    inline constexpr std::string_view to_string(SvdVectors v) {
        switch (v) {
            case SvdVectors::None: return "None";
            case SvdVectors::All:  return "All";
            case SvdVectors::Thin: return "Thin";
        }
        return "SvdVectors(?)";
    }

    /// @brief Number of columns of U that @p job writes.
    /// @param job  the U job
    /// @param m    rows of the input
    /// @param k    min(m, n)
    /// @return m for All, k for Thin, 0 for None (nothing is written)
    inline constexpr int64_t svd_u_cols(SvdVectors job, int64_t m, int64_t k) {
        return job == SvdVectors::All ? m : (job == SvdVectors::Thin ? k : 0);
    }

    /// @brief Number of rows of V^H that @p job writes.
    /// @param job  the V^H job
    /// @param n    columns of the input
    /// @param k    min(m, n)
    /// @return n for All, k for Thin, 0 for None (nothing is written)
    inline constexpr int64_t svd_vh_rows(SvdVectors job, int64_t n, int64_t k) {
        return job == SvdVectors::All ? n : (job == SvdVectors::Thin ? k : 0);
    }

    /// @brief Rewrites a U job of Thin to All when both request the same shape (k == m).
    ///
    /// A route that cannot produce a genuinely thin factor then still serves every
    /// Thin request that asks for nothing smaller.
    /// @param job  the requested U job
    /// @param m    rows of the input
    /// @param k    min(m, n)
    /// @return All if @p job is Thin and k == m, otherwise @p job
    /// @trap Call once per entry point and pass the result on: a `*_buffer_size` and
    ///       its run path must canonicalise identically, or the workspace is sized
    ///       for a different computation than the one performed.
    inline constexpr SvdVectors canonical_jobu(SvdVectors job, int64_t m, int64_t k) {
        return (job == SvdVectors::Thin && k == m) ? SvdVectors::All : job;
    }

    /// @brief Rewrites a V^H job of Thin to All when both request the same shape (k == n).
    /// @param job  the requested V^H job
    /// @param n    columns of the input
    /// @param k    min(m, n)
    /// @return All if @p job is Thin and k == n, otherwise @p job
    /// @see canonical_jobu() for the calling rule.
    inline constexpr SvdVectors canonical_jobvh(SvdVectors job, int64_t n, int64_t k) {
        return (job == SvdVectors::Thin && k == n) ? SvdVectors::All : job;
    }

    /// @brief Which triangle of a symmetric, Hermitian or triangular operand is referenced.
    /// @trap In an option struct write `PotrfOptions{}`, never a bare `{}`: `{}` can
    ///       select the positional overload, whose `Uplo{}` is `Upper`.
    enum class Uplo {
        Upper,  ///< The upper triangle (i <= j).
        Lower   ///< The lower triangle (i >= j).
    };

    /// @brief Spelling of @p v as in source.
    inline constexpr std::string_view to_string(Uplo v) {
        switch (v) {
            case Uplo::Upper: return "Upper";
            case Uplo::Lower: return "Lower";
        }
        return "Uplo(?)";
    }

    /// @brief Whether a triangular operand's diagonal is read or assumed to be one.
    enum class Diag {
        NonUnit,  ///< The stored diagonal is used.
        Unit      ///< The diagonal is taken as 1 and not read.
    };

    /// @brief Spelling of @p v as in source.
    inline constexpr std::string_view to_string(Diag v) {
        switch (v) {
            case Diag::NonUnit: return "NonUnit";
            case Diag::Unit:    return "Unit";
        }
        return "Diag(?)";
    }

    /// @brief Side on which a special operand multiplies: op(A) * B or B * op(A).
    enum class Side {
        Left,   ///< The special operand is on the left.
        Right   ///< The special operand is on the right.
    };

    /// @brief Spelling of @p v as in source.
    inline constexpr std::string_view to_string(Side v) {
        switch (v) {
            case Side::Left:  return "Left";
            case Side::Right: return "Right";
        }
        return "Side(?)";
    }

    /// @brief Order in which eigenvalues (and their vectors) are returned.
    enum class SortOrder {
        Ascending,   ///< Smallest first.
        Descending   ///< Largest first.
    };

    /// @brief Spelling of @p v as in source.
    inline constexpr std::string_view to_string(SortOrder v) {
        switch (v) {
            case SortOrder::Ascending:  return "Ascending";
            case SortOrder::Descending: return "Descending";
        }
        return "SortOrder(?)";
    }

    /// @brief Order in which a sequence of plane rotations is applied (steqr sweeps).
    enum class ApplyOrder {
        Forward,   ///< First rotation first (a QR sweep).
        Backward   ///< Last rotation first (a QL sweep).
    };

    /// @brief Spelling of @p v as in source.
    inline constexpr std::string_view to_string(ApplyOrder v) {
        switch (v) {
            case ApplyOrder::Forward:  return "Forward";
            case ApplyOrder::Backward: return "Backward";
        }
        return "ApplyOrder(?)";
    }

    /// @brief Algorithm family of the partial symmetric/Hermitian eigensolver `syevx`.
    ///
    /// `Auto` picks on matrix format, size and the requested fraction of the spectrum
    /// (`syevx_select_algorithm`). Set per call through SyevxParams::method, or
    /// process-wide through `BATCHLAS_SYEVX_ALGORITHM`
    /// (`auto|direct|direct_subset|filtered|lobpcg`). **The environment variable wins**
    /// over SyevxParams::method, as a `BATCHLAS_<OP>_ROUTE` pin does for the routed
    /// ops, so a whole application can be forced onto one algorithm for diagnosis.
    ///
    /// A choice the input cannot use degrades instead of failing: DirectSubset on
    /// complex or sparse input runs Direct (dense) or LOBPCG (CSR), and Direct or
    /// DirectSubset on CSR runs LOBPCG. A non-extremal SyevxSelect range is never
    /// degraded to a different part of the spectrum: on CSR input, or with an
    /// explicit LOBPCG or Filtered, it throws batchlas::invalid_argument (under the
    /// environment override the dense case runs Direct instead). `syevx` and
    /// `syevx_buffer_size` resolve the choice identically.
    /// @see @ref perf_syevx for the routing thresholds, @ref design_syevx for the tiers.
    enum class SyevxAlgorithm {
        Auto,           ///< Heuristic selection on format, n, batch size and jobz (the default).
        Direct,         ///< Full syev on a copy of A, then select the requested eigenpairs.
        DirectSubset,   ///< Two-stage reduction, bisection + inverse iteration on the subset (real, dense).
        Filtered,       ///< Chebyshev-filtered subspace iteration (dense and CSR); only when asked for.
        LOBPCG          ///< Locally Optimal Block Preconditioned Conjugate Gradient; the CSR default.
    };

    /// @brief Spelling of @p v as in source.
    inline constexpr std::string_view to_string(SyevxAlgorithm v) {
        switch (v) {
            case SyevxAlgorithm::Auto:          return "Auto";
            case SyevxAlgorithm::Direct:        return "Direct";
            case SyevxAlgorithm::DirectSubset:  return "DirectSubset";
            case SyevxAlgorithm::Filtered:      return "Filtered";
            case SyevxAlgorithm::LOBPCG:        return "LOBPCG";
        }
        return "SyevxAlgorithm(?)";
    }

    /// @brief Which part of the spectrum `syevx` returns.
    ///
    /// `Extremal` (the default) returns `neigs` eigenpairs from one end, chosen by
    /// SyevxParams::find_largest: descending for the largest, ascending for the
    /// smallest. It is the Index range [n-neigs, n-1] or [0, neigs-1] and is
    /// normalised to one internally.
    /// @note Deliberately not `EigenRangeType`, whose natural default `All` means
    ///       "every eigenvalue"; that type stays the tridiagonal-layer vocabulary.
    /// @see @ref design_syevx_range
    // evidence: docs/design/syevx-range-selection.md#syevx-range-syevxselect-rather-than-eigenrangetype
    enum class SyevxSelect {
        Extremal,  ///< `neigs` eigenpairs from one end; SyevxParams::find_largest picks the end.
        Index,     ///< SyevxParams::il .. iu inclusive, 0-based in the ascending spectrum.
        Value      ///< Every eigenvalue in the half-open interval (vl, vu].
    };

    /// @brief Spelling of @p v as in source.
    inline constexpr std::string_view to_string(SyevxSelect v) {
        switch (v) {
            case SyevxSelect::Extremal: return "Extremal";
            case SyevxSelect::Index:    return "Index";
            case SyevxSelect::Value:    return "Value";
        }
        return "SyevxSelect(?)";
    }

    /// @brief Preconditioner of the LOBPCG path of `syevx`; the other algorithms ignore it.
    ///
    /// Set per call through SyevxParams::preconditioner_type. `Auto` picks ILU(k)
    /// when a factor was supplied (or requested through
    /// SyevxParams::build_preconditioner), otherwise the default from
    /// `BATCHLAS_SYEVX_PRECONDITIONER` (`auto|none|jacobi|jacobi_shifted|iluk`),
    /// otherwise `None`. Unlike SyevxAlgorithm, **the environment variable only
    /// supplies the default** and never overrides an explicit request.
    ///
    /// The two Jacobi forms are different operators. `Jacobi` approximates A^{-1},
    /// so, like ILU(k), it is valid only for the smallest eigenpairs and
    /// `find_largest` is rejected with it; on a batch item whose diagonal is not
    /// uniformly positive it falls back to the identity. `JacobiShifted` shifts by
    /// the current Ritz value and is valid at either end. `Auto` never picks either.
    /// @see @ref perf_syevx for the measured iteration counts.
    // evidence: docs/perf/syevx.md#lobpcg-jacobi-preconditioners
    // evidence: docs/design/syevx.md#syevx-why-the-preconditioner-environment-variable-is-only-a-default
    enum class SyevxPreconditioner {
        Auto,           ///< ILU(k) if configured, else the environment default, else None.
        None,           ///< Unpreconditioned.
        Jacobi,         ///< diag(A)^{-1}; dense and CSR, smallest eigenpairs only.
        JacobiShifted,  ///< (diag(A) - lambda I)^{-1}; dense and CSR, either end.
        ILUK            ///< Supplied or syevx-built ILU(k) factor; CSR, smallest eigenpairs only.
    };

    /// @brief Spelling of @p v as in source.
    inline constexpr std::string_view to_string(SyevxPreconditioner v) {
        switch (v) {
            case SyevxPreconditioner::Auto:          return "Auto";
            case SyevxPreconditioner::None:          return "None";
            case SyevxPreconditioner::Jacobi:        return "Jacobi";
            case SyevxPreconditioner::JacobiShifted: return "JacobiShifted";
            case SyevxPreconditioner::ILUK:          return "ILUK";
        }
        return "SyevxPreconditioner(?)";
    }

    /// @brief Algorithm used by `ortho` to orthonormalise the columns of a batch of blocks.
    enum class OrthoAlgorithm {
        Chol2,          ///< CholeskyQR applied twice; the default.
        Cholesky,       ///< One CholeskyQR pass; rarely accurate enough on its own.
        ShiftChol3,     ///< Shifted CholeskyQR3; more robust than Chol2 on ill-conditioned blocks.
        Householder,    ///< Householder QR.
        CGS2,           ///< Classical Gram-Schmidt with one reorthogonalisation pass.
        SVQB,           ///< SVQB (orthonormalisation through the Gram matrix's eigendecomposition).
        SVQB2,          ///< SVQB applied twice.
        NUM_ALGORITHMS  ///< Count of the enumerators above, not an algorithm.
    };

    /// @brief Spelling of @p v as in source.
    inline constexpr std::string_view to_string(OrthoAlgorithm v) {
        switch (v) {
            case OrthoAlgorithm::Chol2:          return "Chol2";
            case OrthoAlgorithm::Cholesky:       return "Cholesky";
            case OrthoAlgorithm::ShiftChol3:     return "ShiftChol3";
            case OrthoAlgorithm::Householder:    return "Householder";
            case OrthoAlgorithm::CGS2:           return "CGS2";
            case OrthoAlgorithm::SVQB:           return "SVQB";
            case OrthoAlgorithm::SVQB2:          return "SVQB2";
            case OrthoAlgorithm::NUM_ALGORITHMS: return "NUM_ALGORITHMS";
        }
        return "OrthoAlgorithm(?)";
    }


    /// @brief Internal compute precision of a GEMM (the cuBLAS `cublasComputeType_t`).
    ///
    /// Anything but `Default` is served only by the vendor library: the native GEMM
    /// kernels do not support it. On cuBLAS, a single-precision scalar accepts
    /// F32, F16, BF16 and TF32, a double-precision scalar only F64; any other
    /// combination throws batchlas::unsupported.
    enum class ComputePrecision {
        Default, ///< The precision of the scalar type.
        F32,     ///< 32-bit float accumulation.
        F64,     ///< 64-bit float accumulation.
        F16,     ///< Half-precision inputs, 32-bit accumulation (`CUBLAS_COMPUTE_32F_FAST_16F`).
        BF16,    ///< bfloat16 inputs, 32-bit accumulation (`CUBLAS_COMPUTE_32F_FAST_16BF`).
        TF32     ///< TensorFloat-32 inputs, 32-bit accumulation (`CUBLAS_COMPUTE_32F_FAST_TF32`).
    };

    /// @brief Spelling of @p v as in source.
    inline constexpr std::string_view to_string(ComputePrecision v) {
        switch (v) {
            case ComputePrecision::Default: return "Default";
            case ComputePrecision::F32:     return "F32";
            case ComputePrecision::F64:     return "F64";
            case ComputePrecision::F16:     return "F16";
            case ComputePrecision::BF16:    return "BF16";
            case ComputePrecision::TF32:    return "TF32";
        }
        return "ComputePrecision(?)";
    }

    /// @brief How a VectorView is reinterpreted as a MatrixView: 1 x n or n x 1.
    enum class VectorOrientation {
        Row,     ///< A 1 x n matrix; `ld` is the vector's `inc`.
        Column   ///< An n x 1 matrix; requires `inc == 1`.
    };

    /// @brief Spelling of @p v as in source.
    inline constexpr std::string_view to_string(VectorOrientation v) {
        switch (v) {
            case VectorOrientation::Row:    return "Row";
            case VectorOrientation::Column: return "Column";
        }
        return "VectorOrientation(?)";
    }

    /// @brief Matrix norm computed by `norm` and used by the condition estimators.
    enum class NormType {
        Frobenius, ///< sqrt(sum |a_ij|^2).
        One,       ///< Maximum absolute column sum.
        Inf,       ///< Maximum absolute row sum.
        Max,       ///< Maximum absolute entry (not a consistent norm).
        Spectral   ///< Largest singular value; symmetric/Hermitian input only.
    };

    /// @brief Spelling of @p v as in source.
    inline constexpr std::string_view to_string(NormType v) {
        switch (v) {
            case NormType::Frobenius: return "Frobenius";
            case NormType::One:       return "One";
            case NormType::Inf:       return "Inf";
            case NormType::Max:       return "Max";
            case NormType::Spectral:  return "Spectral";
        }
        return "NormType(?)";
    }

    /// @brief Storage order argument of the host CBLAS / LAPACKE calls.
    ///
    /// BatchLAS matrices are always column-major; this only labels host-library calls.
    /// @see @ref design_matrix_model
    enum class Layout {
        RowMajor,  ///< Row-major (CblasRowMajor).
        ColMajor   ///< Column-major (CblasColMajor).
    };

    /// @brief Spelling of @p v as in source.
    inline constexpr std::string_view to_string(Layout v) {
        switch (v) {
            case Layout::RowMajor: return "RowMajor";
            case Layout::ColMajor: return "ColMajor";
        }
        return "Layout(?)";
    }

    /// @brief Prints any BatchLAS enum as its enumerator spelling, via its to_string().
    ///
    /// Found by ADL. The printed text can be pasted back into source.
    /// @param os     output stream
    /// @param value  enumerator to print
    /// @return @p os
    // Templated on the stream so this header needs only <iosfwd>; it is pulled into device code.
    // Constrained on to_string so an enum without a printer gets no operator<< rather than a broken one.
    template <typename CharT, typename Traits, typename E>
        requires std::is_enum_v<E> && requires(E e) { to_string(e); }
    std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os, E value) {
        return os << to_string(value);
    }

    /// @}
}