// SPDX-License-Identifier: MIT
// filepath: /home/jonaslacour/BatchLAS/src/backends/cublas_matrixview.cc
//#include <batchlas/blas/linalg.hh>
#include <src/linalg-impl.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/util/sycl-span.hh>
#include <src/queue.hh>
#include <sycl/sycl.hpp>
#include <batchlas/internal/ormqr_blocked.hh>

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/functions/ormqr.hh>
#include <batchlas/error.hh>
#include <complex>

#include <src/backends/batch_launch.hh>
#include <src/backends/level3_shape.hh>
#include <src/backends/gemm_heterogeneous.hh>

// This file contains cuBLAS primitives implementation using MatrixView
#include <src/util/template-instantiations.hh>

namespace batchlas {
    namespace backend {
        template <Backend B, typename T>
        size_t ormqr_vendor_buffer_size(Queue& ctx,
                                        const MatrixView<T, MatrixFormat::Dense>& A,
                                        const MatrixView<T, MatrixFormat::Dense>& C,
                                        Side side,
                                        Transpose trans,
                                        Span<T> tau);

        template <Backend B, typename T>
        size_t orgqr_vendor_buffer_size(Queue& ctx,
                                        const MatrixView<T, MatrixFormat::Dense>& A,
                                        Span<T> tau);

    template <Backend Back, typename T>
    Event gemm_vendor_impl(Queue& ctx,
                           const MatrixView<T,MatrixFormat::Dense>& A,
                           const MatrixView<T,MatrixFormat::Dense>& B,
                           const MatrixView<T,MatrixFormat::Dense>& C,
                           T alpha,
                           T beta,
                           Transpose transA,
                           Transpose transB,
                           ComputePrecision precision) {
        static LinalgHandle<Back> handle;

        if (!gemm_batch_dimensions_compatible(A, B, C, transA, transB)) {
            throw batchlas::invalid_argument("gemm: incompatible matrix dimensions");
        }

        auto [m, k] = get_effective_dims(A, transA);
        auto [kB, n] = get_effective_dims(B, transB);
        impl::run_native<impl::Native::Cuda>(*ctx, [=](auto stream) mutable {
            handle.setStream(stream);
            alignas(16) auto alpha_h = alpha;  // cuBLASLt vector-loads it; closures are 8-aligned
            alignas(16) auto beta_h = beta;
            // Workaround: cdouble m or n == 1 through the Ex calls segfaults in cuBLASLt (known-defects #13).
            if constexpr (std::is_same_v<T, std::complex<double>>) {
                if (m == 1 || n == 1) {
                    check_status(cublasZgemmStridedBatched(handle,
                        enum_convert<BackendLibrary::CUBLAS>(transA), enum_convert<BackendLibrary::CUBLAS>(transB),
                        m, n, k, reinterpret_cast<const cuDoubleComplex*>(&alpha_h),
                        reinterpret_cast<const cuDoubleComplex*>(A.data_ptr()), A.ld(), A.stride(),
                        reinterpret_cast<const cuDoubleComplex*>(B.data_ptr()), B.ld(), B.stride(),
                        reinterpret_cast<const cuDoubleComplex*>(&beta_h),
                        reinterpret_cast<cuDoubleComplex*>(C.data_ptr()), C.ld(), C.stride(),
                        std::max(1, A.batch_size())));
                    return;
                }
            }
            if (A.batch_size() <= 1) {
                check_status(cublasGemmEx(handle,
                    enum_convert<BackendLibrary::CUBLAS>(transA), enum_convert<BackendLibrary::CUBLAS>(transB),
                    m, n, k,
                    &alpha_h,
                    A.data_ptr(), BackendScalar<T,BackendLibrary::CUBLAS>::type, A.ld(),
                    B.data_ptr(), BackendScalar<T,BackendLibrary::CUBLAS>::type, B.ld(),
                    &beta_h,
                    C.data_ptr(), BackendScalar<T,BackendLibrary::CUBLAS>::type, C.ld(),
                    enum_convert<BackendLibrary::CUBLAS, T>(precision),
                    CUBLAS_GEMM_DFALT));
            } else {
                check_status(cublasGemmStridedBatchedEx(handle,
                    enum_convert<BackendLibrary::CUBLAS>(transA), enum_convert<BackendLibrary::CUBLAS>(transB),
                    m, n, k,
                    &alpha_h,
                    A.data_ptr(), BackendScalar<T,BackendLibrary::CUBLAS>::type, A.ld(), A.stride(),
                    B.data_ptr(), BackendScalar<T,BackendLibrary::CUBLAS>::type, B.ld(), B.stride(),
                    &beta_h,
                    C.data_ptr(), BackendScalar<T,BackendLibrary::CUBLAS>::type, C.ld(), C.stride(),
                    A.batch_size(),
                    enum_convert<BackendLibrary::CUBLAS, T>(precision),
                    CUBLAS_GEMM_DFALT));
            }
        });
        return ctx.create_event_after_external_work();
    }

    template <Backend Back, typename T>
    Event gemm_vendor(Queue& ctx,
                      const MatrixView<T,MatrixFormat::Dense>& A,
                      const MatrixView<T,MatrixFormat::Dense>& B,
                      const MatrixView<T,MatrixFormat::Dense>& C,
                      T alpha,
                      T beta,
                      Transpose transA,
                      Transpose transB,
                      ComputePrecision precision) {
        // The library call only: which kernel runs is decided by the public gemm
        // (src/ops/gemm/gemm.cc), which reaches here through its `vendor` choice. A direct caller's
        // heterogeneous batch is walked member by member.
        if (gemm_has_heterogeneous_batch(A, B, C)) {
            return detail::gemm_heterogeneous_loop<T>(ctx, A, B, C, beta, transA, transB,
                [&](const auto& A_i, const auto& B_i, const auto& C_i) {
                    return gemm_vendor_impl<Back, T>(ctx, A_i, B_i, C_i, alpha, beta, transA, transB, precision);
                });
        }

        return gemm_vendor_impl<Back, T>(ctx, A, B, C, alpha, beta, transA, transB, precision);
    }

    template <Backend Back, typename T>
    Event symm_vendor_impl(Queue& ctx,
                           const MatrixView<T, MatrixFormat::Dense>& A,
                           const MatrixView<T, MatrixFormat::Dense>& B,
                           const MatrixView<T, MatrixFormat::Dense>& C,
                           T alpha,
                           T beta,
                           Side side,
                           Uplo uplo) {
        static LinalgHandle<Back> handle;

        const auto [m, n, k] = shape::validate_product<batchlas::invalid_argument>("symm", A, B, C, side);
        static_cast<void>(k);  // cublas?symm takes A's order from side, not as an argument

        // The complex slots are the complex-symmetric ?symm (no conjugate; hemm_vendor is Hermitian).
        impl::run_native<impl::Native::Cuda>(*ctx, [=](auto stream) mutable {
            handle.setStream(stream);
            alignas(16) auto alpha_h = alpha;  // cuBLASLt vector-loads it; closures are 8-aligned
            alignas(16) auto beta_h = beta;
            auto launch_single = [&](const MatrixView<T, MatrixFormat::Dense>& A_i,
                                     const MatrixView<T, MatrixFormat::Dense>& B_i,
                                     const MatrixView<T, MatrixFormat::Dense>& C_i) {
                call_backend<T, BackendLibrary::CUBLAS, Back>(cublasSsymm, cublasDsymm, cublasCsymm, cublasZsymm,
                    handle, side, uplo, m, n, &alpha_h,
                    A_i.data_ptr(), A_i.ld(), B_i.data_ptr(), B_i.ld(), &beta_h,
                    C_i.data_ptr(), C_i.ld());
            };

            for_each_batch_item(launch_single, A, B, C);
        });

        return ctx.create_event_after_external_work();
    }

    template <Backend Back, FloatingOrComplexScalar T>
    Event symm_vendor(Queue& ctx,
                      const MatrixView<T, MatrixFormat::Dense>& A,
                      const MatrixView<T, MatrixFormat::Dense>& B,
                      const MatrixView<T, MatrixFormat::Dense>& C,
                      T alpha,
                      T beta,
                      Side side,
                      Uplo uplo) {
        // The expand family is symm's own (src/ops/symm/symm.cc); this is the per-item cublas?symm loop only.
        return symm_vendor_impl<Back, T>(ctx, A, B, C, alpha, beta, side, uplo);
    }

    // The per-item cublas?hemm loop only: hemm's expand family is in src/ops/hemm/hemm.cc.
    template <Backend Back, ComplexScalar T>
    Event hemm_vendor(Queue& ctx,
                           const MatrixView<T, MatrixFormat::Dense>& A,
                           const MatrixView<T, MatrixFormat::Dense>& B,
                           const MatrixView<T, MatrixFormat::Dense>& C,
                           T alpha,
                           T beta,
                           Side side,
                           Uplo uplo) {
        static LinalgHandle<Back> handle;

        const auto [m, n, k] = shape::validate_product<batchlas::invalid_argument>("hemm", A, B, C, side);
        static_cast<void>(k);  // cublas?hemm takes A's order from side
        // The two real slots have no callee: BLAS has no real ?hemm, and T is complex.
        impl::run_native<impl::Native::Cuda>(*ctx, [=](auto stream) mutable {
            handle.setStream(stream);
            alignas(16) auto alpha_h = alpha;  // cuBLASLt vector-loads it; closures are 8-aligned
            alignas(16) auto beta_h = beta;
            auto launch_single = [&](const MatrixView<T, MatrixFormat::Dense>& A_i,
                                     const MatrixView<T, MatrixFormat::Dense>& B_i,
                                     const MatrixView<T, MatrixFormat::Dense>& C_i) {
                call_backend<T, BackendLibrary::CUBLAS, Back>(nullptr, nullptr, cublasChemm, cublasZhemm,
                    handle, side, uplo, m, n, &alpha_h,
                    A_i.data_ptr(), A_i.ld(), B_i.data_ptr(), B_i.ld(), &beta_h,
                    C_i.data_ptr(), C_i.ld());
            };

            for_each_batch_item(launch_single, A, B, C);
        });

        return ctx.create_event_after_external_work();
    }

    // The per-item cublas?herk loop only: herk's gram and fold families are in src/ops/herk/herk.cc.
    template <Backend Back, ComplexScalar T>
    Event herk_vendor(Queue& ctx,
                      const MatrixView<T, MatrixFormat::Dense>& A,
                      const MatrixView<T, MatrixFormat::Dense>& C,
                      float_t<T> alpha,
                      float_t<T> beta,
                      Uplo uplo,
                      Transpose transA) {
        static LinalgHandle<Back> handle;

        const auto [n, k] = shape::validate_rank_k<batchlas::invalid_argument>("herk", A, C, transA, /*hermitian=*/true);
        // The two real slots have no callee: BLAS has no real ?herk (that is syrk), and T is complex.
        impl::run_native<impl::Native::Cuda>(*ctx, [=](auto stream) mutable {
            handle.setStream(stream);
            alignas(16) auto alpha_h = alpha;  // cuBLASLt vector-loads it; closures are 8-aligned
            alignas(16) auto beta_h = beta;
            auto launch_single = [&](const MatrixView<T, MatrixFormat::Dense>& A_i,
                                     const MatrixView<T, MatrixFormat::Dense>& C_i) {
                call_backend<T, BackendLibrary::CUBLAS, Back>(nullptr, nullptr, cublasCherk, cublasZherk,
                    handle, uplo, transA, n, k, &alpha_h,
                    A_i.data_ptr(), A_i.ld(), &beta_h,
                    C_i.data_ptr(), C_i.ld());
            };

            for_each_batch_item(launch_single, A, C);
        });

        return ctx.create_event_after_external_work();
    }

    // The per-item cublas?her2k loop only: her2k's fold family is in src/ops/her2k/her2k.cc.
    template <Backend Back, ComplexScalar T>
    Event her2k_vendor(Queue& ctx,
                       const MatrixView<T, MatrixFormat::Dense>& A,
                       const MatrixView<T, MatrixFormat::Dense>& B,
                       const MatrixView<T, MatrixFormat::Dense>& C,
                       T alpha,
                       float_t<T> beta,
                       Uplo uplo,
                       Transpose transA) {
        static LinalgHandle<Back> handle;

        const auto [n, k] = shape::validate_rank_2k<batchlas::invalid_argument>("her2k", A, B, C, transA, /*hermitian=*/true);

        impl::run_native<impl::Native::Cuda>(*ctx, [=](auto stream) mutable {
            handle.setStream(stream);
            alignas(16) auto beta_h = beta;  // cuBLASLt vector-loads it; closures are 8-aligned
            // cublas?her2k dispatches to cublasLt, which reads the host alpha with a
            // 16-byte aligned vector load. std::complex<double> is only 8-byte
            // aligned, so handing the vendor the address of the parameter itself
            // faults whenever it happens to land 8 mod 16 -- shape-dependent, so
            // most calls survive it. Reproducible against cuBLAS 13.2 with nothing
            // of BatchLAS in the picture.
            alignas(16) T alpha_aligned = alpha;

            auto launch_single = [&](const MatrixView<T, MatrixFormat::Dense>& A_i,
                                     const MatrixView<T, MatrixFormat::Dense>& B_i,
                                     const MatrixView<T, MatrixFormat::Dense>& C_i) {
                call_backend<T, BackendLibrary::CUBLAS, Back>(nullptr, nullptr, cublasCher2k, cublasZher2k,
                    handle, uplo, transA, n, k, &alpha_aligned,
                    A_i.data_ptr(), A_i.ld(), B_i.data_ptr(), B_i.ld(), &beta_h,
                    C_i.data_ptr(), C_i.ld());
            };

            for_each_batch_item(launch_single, A, B, C);
        });

        return ctx.create_event_after_external_work();
    }

    template <Backend Back, typename T>
    Event syrk_vendor_impl(Queue& ctx,
                           const MatrixView<T, MatrixFormat::Dense>& A,
                           const MatrixView<T, MatrixFormat::Dense>& C,
                           T alpha,
                           T beta,
                           Uplo uplo,
                           Transpose transA) {
        static LinalgHandle<Back> handle;

        const auto [n, k] = shape::validate_rank_k<batchlas::invalid_argument>("syrk", A, C, transA, /*hermitian=*/false);

        // The complex slots are the complex-symmetric ?syrk (A A^T; herk_vendor is A A^H).
        impl::run_native<impl::Native::Cuda>(*ctx, [=](auto stream) mutable {
            handle.setStream(stream);
            alignas(16) auto alpha_h = alpha;  // cuBLASLt vector-loads it; closures are 8-aligned
            alignas(16) auto beta_h = beta;
            auto launch_single = [&](const MatrixView<T, MatrixFormat::Dense>& A_i,
                                     const MatrixView<T, MatrixFormat::Dense>& C_i) {
                call_backend<T, BackendLibrary::CUBLAS, Back>(cublasSsyrk, cublasDsyrk, cublasCsyrk, cublasZsyrk,
                    handle, uplo, transA, n, k, &alpha_h,
                    A_i.data_ptr(), A_i.ld(), &beta_h,
                    C_i.data_ptr(), C_i.ld());
            };

            for_each_batch_item(launch_single, A, C);
        });

        return ctx.create_event_after_external_work();
    }

    template <Backend Back, FloatingOrComplexScalar T>
    Event syrk_vendor(Queue& ctx,
                      const MatrixView<T, MatrixFormat::Dense>& A,
                      const MatrixView<T, MatrixFormat::Dense>& C,
                      T alpha,
                      T beta,
                      Uplo uplo,
                      Transpose transA) {
        // The gram and triangular tile kernels are syrk's own families (src/ops/syrk/syrk.cc);
        // this is the per-item cublas?syrk loop only.
        return syrk_vendor_impl<Back, T>(ctx, A, C, alpha, beta, uplo, transA);
    }

    template <Backend Back, typename T>
    Event syr2k_vendor_impl(Queue& ctx,
                            const MatrixView<T, MatrixFormat::Dense>& A,
                            const MatrixView<T, MatrixFormat::Dense>& B,
                            const MatrixView<T, MatrixFormat::Dense>& C,
                            T alpha,
                            T beta,
                            Uplo uplo,
                            Transpose transA) {
        static LinalgHandle<Back> handle;

        const auto [n, k] = shape::validate_rank_2k<batchlas::invalid_argument>("syr2k", A, B, C, transA, /*hermitian=*/false);

        // The complex slots are the complex-symmetric ?syr2k (her2k_vendor is the Hermitian one).
        impl::run_native<impl::Native::Cuda>(*ctx, [=](auto stream) mutable {
            handle.setStream(stream);
            alignas(16) auto alpha_h = alpha;  // cuBLASLt vector-loads it; closures are 8-aligned
            alignas(16) auto beta_h = beta;
            auto launch_single = [&](const MatrixView<T, MatrixFormat::Dense>& A_i,
                                     const MatrixView<T, MatrixFormat::Dense>& B_i,
                                     const MatrixView<T, MatrixFormat::Dense>& C_i) {
                call_backend<T, BackendLibrary::CUBLAS, Back>(cublasSsyr2k, cublasDsyr2k, cublasCsyr2k, cublasZsyr2k,
                    handle, uplo, transA, n, k, &alpha_h,
                    A_i.data_ptr(), A_i.ld(), B_i.data_ptr(), B_i.ld(), &beta_h,
                    C_i.data_ptr(), C_i.ld());
            };

            for_each_batch_item(launch_single, A, B, C);
        });

        return ctx.create_event_after_external_work();
    }

    template <Backend Back, FloatingOrComplexScalar T>
    Event syr2k_vendor(Queue& ctx,
                       const MatrixView<T, MatrixFormat::Dense>& A,
                       const MatrixView<T, MatrixFormat::Dense>& B,
                       const MatrixView<T, MatrixFormat::Dense>& C,
                       T alpha,
                       T beta,
                       Uplo uplo,
                       Transpose transA) {
        // The tile route is chosen in src/ops/syr2k/syr2k.cc; this is the library loop only.
        return syr2k_vendor_impl<Back, T>(ctx, A, B, C, alpha, beta, uplo, transA);
    }

    template <Backend Back, typename T>
    Event trmm_vendor_impl(Queue& ctx,
                           const MatrixView<T, MatrixFormat::Dense>& A,
                           const MatrixView<T, MatrixFormat::Dense>& B,
                           const MatrixView<T, MatrixFormat::Dense>& C,
                           T alpha,
                           Side side,
                           Uplo uplo,
                           Transpose transA,
                           Diag diag) {
        static LinalgHandle<Back> handle;

        const auto [m, n, k] = shape::validate_product<batchlas::invalid_argument>("trmm", A, B, C, side);
        (void)k;

        // The per-batch cublas?trmm loop only: the expansion plus gemm is trmm's own
        // `expand` family (src/ops/trmm/trmm.cc). It needs no scratch.
        impl::run_native<impl::Native::Cuda>(*ctx, [=](auto stream) mutable {
            handle.setStream(stream);
            alignas(16) auto alpha_h = alpha;  // cuBLASLt vector-loads it; closures are 8-aligned
            auto launch_single = [&](const MatrixView<T, MatrixFormat::Dense>& A_i,
                                     const MatrixView<T, MatrixFormat::Dense>& B_i,
                                     const MatrixView<T, MatrixFormat::Dense>& C_i) {
                call_backend<T, BackendLibrary::CUBLAS, Back>(cublasStrmm, cublasDtrmm, cublasCtrmm, cublasZtrmm,
                    handle, side, uplo, transA, diag, m, n, &alpha_h,
                    A_i.data_ptr(), A_i.ld(), B_i.data_ptr(), B_i.ld(), C_i.data_ptr(), C_i.ld());
            };

            for_each_batch_item(launch_single, A, B, C);
        });

        return ctx.create_event_after_external_work();
    }

    template <Backend Back, typename T>
    Event trmm_vendor(Queue& ctx,
                      const MatrixView<T, MatrixFormat::Dense>& A,
                      const MatrixView<T, MatrixFormat::Dense>& B,
                      const MatrixView<T, MatrixFormat::Dense>& C,
                      T alpha,
                      Side side,
                      Uplo uplo,
                      Transpose transA,
                      Diag diag) {
        return trmm_vendor_impl<Back, T>(ctx, A, B, C, alpha, side, uplo, transA, diag);
    }

    template <Backend B, typename T>
    Event gemv_vendor(Queue& ctx,
        const MatrixView<T,MatrixFormat::Dense>& A,
        const VectorView<T>& X,
        const VectorView<T>& Y,
        T alpha,
        T beta,
        Transpose transA) {
        static LinalgHandle<B> handle;
        auto m = A.rows();
        auto n = A.cols();
        auto batch_size = A.batch_size();
        impl::run_native<impl::Native::Cuda>(*ctx, [=](auto stream) mutable {
            handle.setStream(stream);
            alignas(16) auto alpha_h = alpha;  // cuBLASLt vector-loads it; closures are 8-aligned
            alignas(16) auto beta_h = beta;
            if (batch_size <= 1) {
                call_backend<T, BackendLibrary::CUBLAS, B>(cublasSgemv, cublasDgemv, cublasCgemv, cublasZgemv,
                    handle, transA, m, n, &alpha_h, A.data_ptr(), A.ld(), X.data_ptr(), X.inc(), &beta_h, Y.data_ptr(), Y.inc());
            } else {
                call_backend<T, BackendLibrary::CUBLAS, B>(cublasSgemvStridedBatched, cublasDgemvStridedBatched, cublasCgemvStridedBatched, cublasZgemvStridedBatched,
                    handle, transA, m, n, &alpha_h, A.data_ptr(), A.ld(), A.stride(), X.data_ptr(), X.inc(), X.stride(), &beta_h, Y.data_ptr(), Y.inc(), Y.stride(), batch_size);
            }
        });
        return ctx.create_event_after_external_work();
    }

    template <Backend Back, typename T>
    Event trsm_vendor(Queue& ctx,
                   const MatrixView<T,MatrixFormat::Dense>& A,
                   const MatrixView<T,MatrixFormat::Dense>& B,
                   Side side,
                   Uplo uplo,
                   Transpose transA,
                   Diag diag,
                   T alpha) {
        static LinalgHandle<Back> handle;
        auto [kB, n] = get_effective_dims(B, Transpose::NoTrans);
        auto batch_size = A.batch_size();
        trsm_validate_params(A, B, side, uplo, transA, diag);

        const auto side_cublas = enum_convert<BackendLibrary::CUBLAS>(side);
        const auto uplo_cublas = enum_convert<BackendLibrary::CUBLAS>(uplo);
        const auto trans_cublas = enum_convert<BackendLibrary::CUBLAS>(transA);
        const auto diag_cublas = enum_convert<BackendLibrary::CUBLAS>(diag);

        if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
            // Fallback: Implement TRSM directly for complex types.
            // cuBLAS TRSM has exhibited incorrect behavior (unchanged output / NaNs) with our
            // SYCL CUDA interop + USM complex buffers. This kernel is sequential per RHS/row,
            // but parallel across (batch, rhs) or (batch, row).
            const T* A_ptr = A.data_ptr();
            T* B_ptr = B.data_ptr();
            const int m = B.rows();
            const int nrhs = B.cols();
            // 64-bit: b * strideA passes 2^31 at cfloat order 512, batch 8193 (an int wrapped).
            const std::int64_t lda = A.ld();
            const std::int64_t ldb = B.ld();
            const std::int64_t strideA = A.stride();
            const std::int64_t strideB = B.stride();
            const int work_dim = (side == Side::Left) ? nrhs : m;

            ctx->parallel_for(sycl::range<2>(static_cast<size_t>(batch_size), static_cast<size_t>(work_dim)),
                              [=](sycl::id<2> tid) {
                                  const std::int64_t b = static_cast<std::int64_t>(tid[0]);
                                  const int p = static_cast<int>(tid[1]);

                                  const T* Ab = A_ptr + b * strideA;
                                  T* Bb = B_ptr + b * strideB;

                                  const bool do_conj = (transA == Transpose::ConjTrans);
                                  const bool do_trans = (transA != Transpose::NoTrans);
                                  const bool op_is_lower = (uplo == Uplo::Lower) ? !do_trans : do_trans;
                                  const bool unit_diag = (diag == Diag::Unit);

                                  auto conj_if = [=](T v) {
                                      if (!do_conj) return v;
                                      using std::conj;
                                      return conj(v);
                                  };

                                  // Return op(A) element at (r, c) in column-major storage.
                                  auto opA = [=](int r, int c) {
                                      if (transA == Transpose::NoTrans) {
                                          return Ab[c * lda + r];
                                      }
                                      // op(A) = A^T or A^H
                                      return conj_if(Ab[r * lda + c]);
                                  };

                                  if (side == Side::Left) {
                                      const int j = p; // RHS column
                                      if (op_is_lower) {
                                          // Forward substitution (i = 0..m-1)
                                          for (int i = 0; i < m; ++i) {
                                              T sum = T(0);
                                              for (int k = 0; k < i; ++k) {
                                                  sum += opA(i, k) * Bb[j * ldb + k];
                                              }
                                              T x = alpha * Bb[j * ldb + i] - sum;
                                              if (!unit_diag) {
                                                  x /= opA(i, i);
                                              }
                                              Bb[j * ldb + i] = x;
                                          }
                                      } else {
                                          // Backward substitution (i = m-1..0)
                                          for (int i = m - 1; i >= 0; --i) {
                                              T sum = T(0);
                                              for (int k = i + 1; k < m; ++k) {
                                                  sum += opA(i, k) * Bb[j * ldb + k];
                                              }
                                              T x = alpha * Bb[j * ldb + i] - sum;
                                              if (!unit_diag) {
                                                  x /= opA(i, i);
                                              }
                                              Bb[j * ldb + i] = x;
                                          }
                                      }
                                  } else {
                                      // Side::Right: solve X*op(A) = alpha*B, row-by-row.
                                      const int i = p; // row
                                      if (op_is_lower) {
                                          // Lower: solve backward in columns (j = nrhs-1..0)
                                          for (int j = nrhs - 1; j >= 0; --j) {
                                              T sum = T(0);
                                              for (int k = j + 1; k < nrhs; ++k) {
                                                  sum += Bb[k * ldb + i] * opA(k, j);
                                              }
                                              T x = alpha * Bb[j * ldb + i] - sum;
                                              if (!unit_diag) {
                                                  x /= opA(j, j);
                                              }
                                              Bb[j * ldb + i] = x;
                                          }
                                      } else {
                                          // Upper: solve forward in columns (j = 0..nrhs-1)
                                          for (int j = 0; j < nrhs; ++j) {
                                              T sum = T(0);
                                              for (int k = 0; k < j; ++k) {
                                                  sum += Bb[k * ldb + i] * opA(k, j);
                                              }
                                              T x = alpha * Bb[j * ldb + i] - sum;
                                              if (!unit_diag) {
                                                  x /= opA(j, j);
                                              }
                                              Bb[j * ldb + i] = x;
                                          }
                                      }
                                  }
                              });
        } else {
            T** a_ptrs = nullptr;
            T** b_ptrs = nullptr;
            if (batch_size != 1) {
                a_ptrs = A.data_ptrs(ctx).data();
                b_ptrs = B.data_ptrs(ctx).data();
            }
            impl::run_native<impl::Native::Cuda>(*ctx, [=](auto stream) mutable {
                handle.setStream(stream);
                alignas(16) auto alpha_h = alpha;  // cuBLASLt vector-loads it; closures are 8-aligned
                if (batch_size == 1) {
                    call_backend<T, BackendLibrary::CUBLAS, Back>(cublasStrsm, cublasDtrsm, cublasCtrsm, cublasZtrsm,
                        handle, side, uplo, transA, diag, kB, n, &alpha_h, A.data_ptr(), A.ld(), B.data_ptr(), B.ld());
                } else {
                    call_backend<T, BackendLibrary::CUBLAS, Back>(cublasStrsmBatched, cublasDtrsmBatched, cublasCtrsmBatched, cublasZtrsmBatched,
                        handle, side, uplo, transA, diag, kB, n, &alpha_h, a_ptrs, A.ld(), b_ptrs, B.ld(), batch_size);
                }
            });
        }
        return ctx.create_event_after_external_work();
    }

    template <Backend B, typename T>
    Event geqrf_vendor(Queue& ctx,
        const MatrixView<T,MatrixFormat::Dense>& A, //In place reflectors (Lower triangle of A)
        Span<T> tau,
        Span<std::byte> workspace) {
        static LinalgHandle<B> handle;
        auto m = A.rows();
        auto n = A.cols();
        auto k = std::min(m, n);
        auto batch_size = A.batch_size();
        auto pool = BumpAllocator(workspace);
        if (batch_size <= 1) {
            handle.setStream(ctx);  // the query below runs on whatever stream the static handle last had
            cusolverDnParams_t params;
            check_status(cusolverDnCreateParams(&params));
            try {
                size_t device_l_work = 0, host_l_work = 0;
                check_status(cusolverDnXgeqrf_bufferSize(handle, params, m, n,
                    BackendScalar<T,BackendLibrary::CUSOLVER>::type, A.data_ptr(), A.ld(),
                    BackendScalar<T,BackendLibrary::CUSOLVER>::type, tau.data(),
                    BackendScalar<T,BackendLibrary::CUSOLVER>::type, &device_l_work, &host_l_work));
                auto device_work_space = pool.allocate<std::byte>(ctx, device_l_work);
                auto d_info = pool.allocate<int>(ctx, 1);
                // bufferOnHost is host memory, not a draw from the (device) workspace.
                std::vector<std::byte> host_work_space(host_l_work);
                std::byte* host_ptr = host_work_space.data();
                impl::run_native<impl::Native::Cuda>(*ctx, [=](auto stream) mutable {
                    handle.setStream(stream);
                    check_status(cusolverDnXgeqrf(handle, params, m, n,
                        BackendScalar<T,BackendLibrary::CUSOLVER>::type, A.data_ptr(), A.ld(),
                        BackendScalar<T,BackendLibrary::CUSOLVER>::type, tau.data(),
                        BackendScalar<T,BackendLibrary::CUSOLVER>::type, device_work_space.data(),
                        device_l_work, host_ptr, host_l_work, d_info.data()));
                });
                if (host_l_work > 0) ctx.wait();  // host_work_space must outlive the call's stream work
            } catch (...) {
                cusolverDnDestroyParams(params);
                throw;
            }
            check_status(cusolverDnDestroyParams(params));
        } else {
            auto tau_data = tau.data();
            auto tau_ptrs = pool.allocate<T*>(ctx, batch_size);
            ctx->parallel_for(sycl::range<1>(batch_size), [=](sycl::id<1> item) {
                size_t i = item.get(0);
                tau_ptrs[i] = tau_data + i * k;
            });
            T** a_ptrs = A.data_ptrs(ctx).data();
            if (!ctx.in_order()) ctx.wait();  // tau_ptrs' kernel is not ordered before the native call
            // cublas<t>geqrfBatched's info is a host int: a parameter-error code, set before return.
            int info_h = 0;
            int* info_p = &info_h;
            impl::run_native<impl::Native::Cuda>(*ctx, [=](auto stream) mutable {
                handle.setStream(stream);
                call_backend<T, BackendLibrary::CUBLAS, B>(cublasSgeqrfBatched, cublasDgeqrfBatched, cublasCgeqrfBatched, cublasZgeqrfBatched,
                    handle, m, n, a_ptrs, A.ld(), tau_ptrs.data(), info_p, batch_size);
            });
            if (info_h != 0) {
                throw batchlas::invalid_argument("geqrf_vendor (CUBLAS): geqrfBatched rejected parameter " +
                                                 std::to_string(-info_h));
            }
        }
        return ctx.create_event_after_external_work();
    }

    template <Backend B, typename T>
    size_t geqrf_vendor_buffer_size(Queue& ctx,
        const MatrixView<T,MatrixFormat::Dense>& A,
        Span<T> tau) {
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        auto m = A.rows();
        auto n = A.cols();
        auto batch_size = A.batch_size();
        if (batch_size <= 1) {
            size_t device_l_work = 0, host_l_work = 0;
            cusolverDnParams_t params;
            check_status(cusolverDnCreateParams(&params));
            const cusolverStatus_t st = cusolverDnXgeqrf_bufferSize(handle, params, m, n,
                BackendScalar<T,BackendLibrary::CUSOLVER>::type, A.data_ptr(), A.ld(),
                BackendScalar<T,BackendLibrary::CUSOLVER>::type, tau.data(),
                BackendScalar<T,BackendLibrary::CUSOLVER>::type, &device_l_work, &host_l_work);
            check_status(cusolverDnDestroyParams(params));
            check_status(st);
            // The run path's draws only: its host buffer and geqrfBatched's info live on the host.
            return BumpAllocator::allocation_size<std::byte>(ctx, device_l_work) + BumpAllocator::allocation_size<int>(ctx, 1);
        } else {
            return BumpAllocator::allocation_size<T*>(ctx, batch_size);
        }
    }

    template <Backend B, typename T>
    Event ormqr_vendor(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                const MatrixView<T, MatrixFormat::Dense>& C,
                Side side,
                Transpose trans,
                Span<T> tau,
                Span<std::byte> workspace) {
        static LinalgHandle<B> handle;
        auto m = C.rows();
        auto n = C.cols();
        auto k = std::min(A.rows(), A.cols());
        auto batch_size = A.batch_size();
        BumpAllocator pool(workspace);
        if (batch_size == 1) {
            handle.setStream(ctx);
            int lwork;
            call_backend<T, BackendLibrary::CUSOLVER, B>(
                cusolverDnSormqr_bufferSize, cusolverDnDormqr_bufferSize,
                cusolverDnCunmqr_bufferSize, cusolverDnZunmqr_bufferSize,
                handle,
                enum_convert<BackendLibrary::CUSOLVER>(side),
                enum_convert<BackendLibrary::CUSOLVER>(trans),
                m, n, k,
                A.data_ptr(), A.ld(),
                tau.data(),
                C.data_ptr(), C.ld(),
                &lwork);
            auto device_ws = pool.allocate<T>(ctx, lwork);
            auto info = pool.allocate<int>(ctx, 1);
            impl::run_native<impl::Native::Cuda>(*ctx, [=](auto stream) mutable {
                handle.setStream(stream);
                call_backend<T, BackendLibrary::CUSOLVER, B>(
                    cusolverDnSormqr, cusolverDnDormqr,
                    cusolverDnCunmqr, cusolverDnZunmqr,
                    handle,
                    enum_convert<BackendLibrary::CUSOLVER>(side),
                    enum_convert<BackendLibrary::CUSOLVER>(trans),
                    m, n, k,
                    A.data_ptr(), A.ld(),
                    tau.data(),
                    C.data_ptr(), C.ld(),
                    device_ws.data(), lwork, info.data());
            });
        } else {
            size_t single_ws = ormqr_vendor_buffer_size<B>(ctx, A.batch_item(0), C.batch_item(0), side, trans, tau.subspan(0, k));
            for (int i = 0; i < batch_size; ++i) {
                auto sub_ws = pool.allocate<std::byte>(ctx, single_ws);
                (void)ormqr_vendor<B>(ctx, A.batch_item(i), C.batch_item(i), side, trans, tau.subspan(i * k, k), sub_ws);
            }
        }
        return ctx.create_event_after_external_work();
    }

    template <Backend B, typename T>
    size_t ormqr_vendor_buffer_size(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             const MatrixView<T, MatrixFormat::Dense>& C,
                             Side side,
                             Transpose trans,
                             Span<T> tau) {
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        auto m = C.rows();
        auto n = C.cols();
        auto k = std::min(A.rows(), A.cols());
        auto batch_size = A.batch_size();
        if (batch_size == 1) {
            int lwork;
            call_backend<T, BackendLibrary::CUSOLVER, B>(
                cusolverDnSormqr_bufferSize, cusolverDnDormqr_bufferSize,
                cusolverDnCunmqr_bufferSize, cusolverDnZunmqr_bufferSize,
                handle,
                enum_convert<BackendLibrary::CUSOLVER>(side),
                enum_convert<BackendLibrary::CUSOLVER>(trans),
                m, n, k,
                A.data_ptr(), A.ld(),
                tau.data(),
                C.data_ptr(), C.ld(),
                &lwork);
            return BumpAllocator::allocation_size<T>(ctx, lwork) + BumpAllocator::allocation_size<int>(ctx, 1); // +1 for info
        }

        size_t single = BumpAllocator::allocation_size<std::byte>(ctx, ormqr_vendor_buffer_size<B>(ctx, A.batch_item(0), C.batch_item(0), side, trans, tau.subspan(0, k)));
        return single * batch_size;
    }

    template <Backend B, typename T>
    Event orgqr_vendor(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& A,
                Span<T> tau,
                Span<std::byte> workspace) {
        static LinalgHandle<B> handle;
        auto m = A.rows();
        auto n = A.cols();
        auto k = std::min(m, n);
        auto batch_size = A.batch_size();
        BumpAllocator pool(workspace);
        if (batch_size == 1) {
            handle.setStream(ctx);
            int lwork;
            call_backend<T, BackendLibrary::CUSOLVER, B>(
                cusolverDnSorgqr_bufferSize, cusolverDnDorgqr_bufferSize,
                cusolverDnCungqr_bufferSize, cusolverDnZungqr_bufferSize,
                handle,
                m, n, k,
                A.data_ptr(), A.ld(),
                tau.data(),
                &lwork);
            auto device_ws = pool.allocate<T>(ctx, lwork);
            auto info = pool.allocate<int>(ctx, 1);
            impl::run_native<impl::Native::Cuda>(*ctx, [=](auto stream) mutable {
                handle.setStream(stream);
                call_backend<T, BackendLibrary::CUSOLVER, B>(
                    cusolverDnSorgqr, cusolverDnDorgqr,
                    cusolverDnCungqr, cusolverDnZungqr,
                    handle,
                    m, n, k,
                    A.data_ptr(), A.ld(),
                    tau.data(),
                    device_ws.data(), lwork, info.data());
            });
        } else {
            // On ctx, as ormqr_vendor: a fresh Queue is not ordered after ctx's pending geqrf.
            size_t single_ws = orgqr_vendor_buffer_size<B>(ctx, A.batch_item(0), tau.subspan(0, k));
            for (int i = 0; i < batch_size; ++i) {
                auto sub_ws = pool.allocate<std::byte>(ctx, single_ws);
                (void)orgqr_vendor<B>(ctx, A.batch_item(i), tau.subspan(i * k, k), sub_ws);
            }
        }
        return ctx.create_event_after_external_work();
    }

    template <Backend B, typename T>
    size_t orgqr_vendor_buffer_size(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             Span<T> tau) {
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        auto m = A.rows();
        auto n = A.cols();
        auto k = std::min(m, n);
        auto batch_size = A.batch_size();
        if (batch_size == 1) {
            int lwork;
            call_backend<T, BackendLibrary::CUSOLVER, B>(
                cusolverDnSorgqr_bufferSize, cusolverDnDorgqr_bufferSize,
                cusolverDnCungqr_bufferSize, cusolverDnZungqr_bufferSize,
                handle,
                m, n, k,
                A.data_ptr(), A.ld(),
                tau.data(),
                &lwork);
            return BumpAllocator::allocation_size<T>(ctx, lwork) + BumpAllocator::allocation_size<int>(ctx, 1);
        } else {
            size_t single = BumpAllocator::allocation_size<std::byte>(ctx, orgqr_vendor_buffer_size<B>(ctx, A.batch_item(0), tau.subspan(0, k)));
            return single * batch_size;
        }
    }

    template <Backend Back, typename T>
    Event getrs_vendor(Queue& ctx,
        const MatrixView<T,MatrixFormat::Dense>& A,
        const MatrixView<T,MatrixFormat::Dense>& B,
        Transpose transA,
        Span<int64_t> pivots,
        Span<std::byte> workspace) {
            static LinalgHandle<Back> handle;
            auto n = A.rows();
            auto nrhs = B.cols();
            auto batch_size = A.batch_size();
            auto pool = BumpAllocator(workspace);
            // ONE ARM FOR EVERY BATCH SIZE, and the deleted `batch_size <= 1`
            // special case was a CRASH, not an optimisation.
            //
            // It called cusolverDnXgetrs, whose ipiv is GENUINE int64, and handed
            // it `pivots.data()` raw. Every getrf in this tree writes PACKED
            // 1-based int32 into that same span -- getrf_vendor below at :1508
            // does `pivots.as_span<int>()`, and so do rocsolver.cc:228 and both
            // native tiers -- so at batch 1 the solve read two packed pivots per
            // int64 slot as one row index and indexed out of bounds.
            // `getrf<CUDA,float>` then `getrs<CUDA,float>` at n=40, nrhs=3,
            // batch=1 -- the exact sequence linalg::solve issues
            // (linalg-ops.hh:344-345) -- aborted with CUDA_ERROR_ILLEGAL_ADDRESS
            // (exit 134). cublas?getrsBatched reads the packed int32 the getrf
            // actually wrote and is correct at batchCount = 1, so the two-arm
            // split bought nothing and cost the only pivot format the family
            // agrees on. Found by WP6's pivot-contract survey; PRE-EXISTING, and
            // no batched test could reach it because they all use batch >= 2.
            //
            // `info` here is a HOST int (cublas?getrsBatched's info is an argument
            // validity code, not a per-item device array), which is why nothing is
            // drawn from the pool; getrs_vendor_buffer_size still reports the old
            // one-int figure, deliberately, because shrinking a workspace query is
            // the change this family has been bitten by before.
            static_cast<void>(pool);
            auto reinterpreted_pivots = pivots.as_span<int>();
            T** a_ptrs = A.data_ptrs(ctx).data();
            T** b_ptrs = B.data_ptrs(ctx).data();
            impl::run_native<impl::Native::Cuda>(*ctx, [=](auto stream) mutable {
                handle.setStream(stream);
                int info;
                call_backend<T, BackendLibrary::CUBLAS, Back>(cublasSgetrsBatched, cublasDgetrsBatched, cublasCgetrsBatched, cublasZgetrsBatched,
                    handle, enum_convert<BackendLibrary::CUBLAS>(transA), n, nrhs,
                    a_ptrs, A.ld(), reinterpreted_pivots.data(),
                    b_ptrs, B.ld(), &info, batch_size);
            });
            return ctx.create_event_after_external_work();
        }

    template <Backend Back, typename T>
    size_t getrs_vendor_buffer_size(Queue& ctx,
        const MatrixView<T,MatrixFormat::Dense>& A,
        const MatrixView<T,MatrixFormat::Dense>& B,
        Transpose transA) {
            return BumpAllocator::allocation_size<int>(ctx, A.batch_size() == 1 ? 1 : 0); //batched getrs just uses a single host integer.
        }

    template <Backend B, typename T>
    Event getrf_vendor(Queue& ctx,
        const MatrixView<T, MatrixFormat::Dense>& A,
        Span<int64_t> pivots,
        Span<std::byte> workspace,
        Span<int32_t> info_out) {
            static LinalgHandle<B> handle;
            auto n = A.rows();
            auto batch_size = A.batch_size();
            auto pool = BumpAllocator(workspace);
            // cuBLAS's infoArray is a genuine per-item device array with LAPACK
            // semantics (0 = ok, >0 = the column where U went exactly singular).
            // It used to be pool scratch that nothing ever read.
            auto info = ::batchlas::detail::info_target(ctx, pool, info_out, static_cast<size_t>(batch_size));
            auto reinterpreted_pivots = pivots.as_span<int>();
            T** a_ptrs = A.data_ptrs(ctx).data();
            impl::run_native<impl::Native::Cuda>(*ctx, [=](auto stream) mutable {
                handle.setStream(stream);
                call_backend<T, BackendLibrary::CUBLAS, B>(cublasSgetrfBatched, cublasDgetrfBatched, cublasCgetrfBatched, cublasZgetrfBatched,
                    handle, n,
                    a_ptrs, A.ld(), reinterpreted_pivots.data(), info.data(), batch_size);
            });
            return ctx.create_event_after_external_work();
        }

    template <Backend B, typename T>
    size_t getrf_vendor_buffer_size(Queue& ctx,
        const MatrixView<T, MatrixFormat::Dense>& A) {
            return BumpAllocator::allocation_size<int>(ctx, A.batch_size()); //batched getrf just uses a single host integer.
        }

    template <Backend B, typename T>
    Event getri_vendor(Queue& ctx,
        const MatrixView<T, MatrixFormat::Dense>& A,
        const MatrixView<T, MatrixFormat::Dense>& C, //C is overwritten with inverse of A
        Span<int64_t> pivots,
        Span<std::byte> workspace,
        Span<int32_t> info_out) {
            if (A.data_ptr() == C.data_ptr()) {
                throw batchlas::invalid_argument("getri_vendor (CUBLAS): cublas<t>getriBatched is out of place; C aliases A");
            }
            static LinalgHandle<B> handle;
            auto n = A.rows();
            auto batch_size = A.batch_size();
            auto pool = BumpAllocator(workspace);
            // Same story as getrf_vendor: cuBLAS writes a per-item infoArray
            // (>0 = U(i,i) is exactly zero, so this item has no inverse) and the
            // result was thrown away, leaving the caller a matrix of infinities.
            auto info_arr = ::batchlas::detail::info_target(ctx, pool, info_out, static_cast<size_t>(batch_size));
            auto reinterpreted_pivots = pivots.as_span<int>();
            T** a_ptrs = A.data_ptrs(ctx).data();  // fills and waits: never inside run_native
            T** c_ptrs = C.data_ptrs(ctx).data();
            impl::run_native<impl::Native::Cuda>(*ctx, [=](auto stream) mutable {
                handle.setStream(stream);
                call_backend<T, BackendLibrary::CUBLAS, B>(cublasSgetriBatched, cublasDgetriBatched, cublasCgetriBatched, cublasZgetriBatched,
                    handle, n,
                    a_ptrs, A.ld(), reinterpreted_pivots.data(),
                    c_ptrs, C.ld(), info_arr.data(), batch_size);
            });
            return ctx.create_event_after_external_work();
        }

    template <Backend B, typename T>
    size_t getri_vendor_buffer_size(Queue& ctx,
        const MatrixView<T, MatrixFormat::Dense>& A) {
            static LinalgHandle<B> handle;
            handle.setStream(ctx);
            auto n = A.rows();
            auto batch_size = A.batch_size();
            return BumpAllocator::allocation_size<int>(ctx, batch_size);
        }

    } // namespace backend

    // Template instantiations for cuBLAS functions (MatrixView version)
    // Explicit instantiations. Signatures live in the `sig` namespace beside each
    // public declaration (include/batchlas/blas/functions/*.hh), so changing one is a single
    // header edit rather than one edit per backend TU.
    #define B_ Backend::CUDA

    // Only the `backend::`-qualified vendor entry points are instantiated here.
    // WP0b moved every public `batchlas::<op>` out of the vendor TUs and into
    // src/ops/, so a public row in this table would be a
    // duplicate definition rather than a convenience.
    #define CUBLAS_OPS(B, fp) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, gemm_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, gemv_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, trsm_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, trmm_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, geqrf_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, geqrf_vendor_buffer_size) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, getrs_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, getrs_vendor_buffer_size) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, getrf_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, getrf_vendor_buffer_size) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, getri_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, getri_vendor_buffer_size) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, ormqr_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, ormqr_vendor_buffer_size) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, orgqr_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, orgqr_vendor_buffer_size)

    // hemm/herk/her2k are complex-only, so they get their own table.
    #define CUBLAS_SYMMETRIC_OPS(B, fp) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, symm_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, syrk_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, syr2k_vendor)

    #define CUBLAS_COMPLEX_OPS(B, fp) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, hemm_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, herk_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, her2k_vendor)

    BATCHLAS_FOR_EACH_SCALAR_TYPE_1(CUBLAS_OPS, B_)
    BATCHLAS_FOR_EACH_SCALAR_TYPE_1(CUBLAS_SYMMETRIC_OPS, B_)
    BATCHLAS_FOR_EACH_COMPLEX_TYPE_1(CUBLAS_COMPLEX_OPS, B_)

    #undef CUBLAS_OPS
    #undef CUBLAS_SYMMETRIC_OPS
    #undef CUBLAS_COMPLEX_OPS
    #undef B_
}
