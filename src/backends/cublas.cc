// filepath: /home/jonaslacour/BatchLAS/src/backends/cublas_matrixview.cc
//#include <batchlas/blas/linalg.hh>
#include "../linalg-impl.hh"
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/util/sycl-span.hh>
#include "../queue.hh"
#include <sycl/sycl.hpp>
#include <batchlas/internal/ormqr_blocked.hh>

#include <algorithm>
#include <cstdlib>
#include <string>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/functions/ormqr.hh>
#include <complex>

#include "gemm_cublasdx_dispatch.hh"
#include "batch_launch.hh"
#include "level3_shape.hh"
#include "gemm_variant.hh"
#include "gemm_heterogeneous.hh"
#include "symm_custom_dispatch.hh"
#include "syr2k_custom_dispatch.hh"
#include "syrk_custom_dispatch.hh"
#include "syrk_gram_tiles.hh"
#include "cublasdx_dispatch_common.hh"
#include "trmm_custom_dispatch.hh"
#include "trmm_triangular_tiles.hh"
#include "triangular_expand.hh"

// This file contains cuBLAS primitives implementation using MatrixView
#include "../util/template-instantiations.hh"

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
        handle.setStream(ctx);

        if (!gemm_batch_dimensions_compatible(A, B, C, transA, transB)) {
            throw batchlas::invalid_argument("GEMM: incompatible matrix dimensions");
        }

        auto [m, k] = get_effective_dims(A, transA);
        auto [kB, n] = get_effective_dims(B, transB);
        // Workaround: cdouble m or n == 1 through the Ex calls segfaults in cuBLASLt (known-defects #13).
        if constexpr (std::is_same_v<T, std::complex<double>>) {
            if (m == 1 || n == 1) {
                cublasZgemmStridedBatched(handle,
                    enum_convert<BackendLibrary::CUBLAS>(transA), enum_convert<BackendLibrary::CUBLAS>(transB),
                    m, n, k, reinterpret_cast<const cuDoubleComplex*>(&alpha),
                    reinterpret_cast<const cuDoubleComplex*>(A.data_ptr()), A.ld(), A.stride(),
                    reinterpret_cast<const cuDoubleComplex*>(B.data_ptr()), B.ld(), B.stride(),
                    reinterpret_cast<const cuDoubleComplex*>(&beta),
                    reinterpret_cast<cuDoubleComplex*>(C.data_ptr()), C.ld(), C.stride(),
                    std::max(1, A.batch_size()));
                return ctx.create_event_after_external_work();
            }
        }
        if (A.batch_size() <= 1) {
            cublasGemmEx(handle,
                enum_convert<BackendLibrary::CUBLAS>(transA), enum_convert<BackendLibrary::CUBLAS>(transB),
                m, n, k,
                &alpha,
                A.data_ptr(), BackendScalar<T,BackendLibrary::CUBLAS>::type, A.ld(),
                B.data_ptr(), BackendScalar<T,BackendLibrary::CUBLAS>::type, B.ld(),
                &beta,
                C.data_ptr(), BackendScalar<T,BackendLibrary::CUBLAS>::type, C.ld(),
                enum_convert<BackendLibrary::CUBLAS, T>(precision),
                CUBLAS_GEMM_DFALT);
        } else {
            cublasGemmStridedBatchedEx(handle,
                enum_convert<BackendLibrary::CUBLAS>(transA), enum_convert<BackendLibrary::CUBLAS>(transB),
                m, n, k,
                &alpha,
                A.data_ptr(), BackendScalar<T,BackendLibrary::CUBLAS>::type, A.ld(), A.stride(),
                B.data_ptr(), BackendScalar<T,BackendLibrary::CUBLAS>::type, B.ld(), B.stride(),
                &beta,
                C.data_ptr(), BackendScalar<T,BackendLibrary::CUBLAS>::type, C.ld(), C.stride(),
                A.batch_size(),
                enum_convert<BackendLibrary::CUBLAS, T>(precision),
                CUBLAS_GEMM_DFALT);
        }
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
        // evidence: docs/perf/gemm.md#gemm-the-heterogeneous-batch-loop
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
        handle.setStream(ctx);

        const auto [m, n, k] = shape::validate_product<std::invalid_argument>("SYMM", A, B, C, side);
        static_cast<void>(k);  // cublas?symm takes A's order from side, not as an argument

        // The two complex slots have no callee because symm is instantiated
        // only for float and double here -- a complex caller reaches the
        // Hermitian sibling hemm_vendor below instead -- so, exactly as hemm
        // does for the two real slots, they are never selected.
        auto launch_single = [&](const MatrixView<T, MatrixFormat::Dense>& A_i,
                                 const MatrixView<T, MatrixFormat::Dense>& B_i,
                                 const MatrixView<T, MatrixFormat::Dense>& C_i) {
            call_backend<T, BackendLibrary::CUBLAS, Back>(cublasSsymm, cublasDsymm, nullptr, nullptr,
                handle, side, uplo, m, n, &alpha,
                A_i.data_ptr(), A_i.ld(), B_i.data_ptr(), B_i.ld(), &beta,
                C_i.data_ptr(), C_i.ld());
        };

        for_each_batch_item(launch_single, A, B, C);

        return ctx.create_event_after_external_work();
    }

    template <Backend Back, RealScalar T>
    Event symm_vendor(Queue& ctx,
                      const MatrixView<T, MatrixFormat::Dense>& A,
                      const MatrixView<T, MatrixFormat::Dense>& B,
                      const MatrixView<T, MatrixFormat::Dense>& C,
                      T alpha,
                      T beta,
                      Side side,
                      Uplo uplo) {
        // The float custom-route gate is in the facade (src/ops/level3/level3.cc), not here.
        // evidence: docs/perf/level3.md#level-3-non-float-routes-live-only-in-cublascc
        return symm_vendor_impl<Back, T>(ctx, A, B, C, alpha, beta, side, uplo);
    }

    // cuBLAS has no batched ?hemm, so a batch is a host loop; expanding the
    // Hermitian triangle turns it into one strided-batched GEMM.
    // evidence: docs/perf/level3.md#symm-and-hemm-expansion-crossover
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
        handle.setStream(ctx);

        const auto [m, n, k] = shape::validate_product<std::invalid_argument>("HEMM", A, B, C, side);

        // Unlike trmm, the hemm loop wins on launch-bound shapes, so consult the crossover.
        const std::size_t expansion_bytes = detail::expanded_workspace_bytes<T>(ctx, k, A.batch_size());
        if (detail::expansion_fits(ctx, k, A.batch_size(), expansion_bytes) &&
            detail::expansion_preferred(std::max({m, n, k}), A.batch_size())) {
            const int ld = detail::expanded_ld<T>(k);

            auto ws = ctx.workspace(expansion_bytes);
            BumpAllocator pool(ws.span());
            auto storage = pool.allocate<T>(ctx, static_cast<std::size_t>(ld) *
                                                     static_cast<std::size_t>(k) *
                                                     static_cast<std::size_t>(A.batch_size()));

            MatrixView<T, MatrixFormat::Dense> expanded(storage.data(), k, k, ld, ld * k, A.batch_size());

            // Not the caller's A: its opposite triangle and the diagonal's
            // imaginary part are not part of the operand and may hold anything.
            Event expansion = detail::expand_mirrored<T, /*Conjugate=*/true>(ctx, expanded, A, uplo);

            // An in-order queue shares the native stream with the expansion; an
            // out-of-order one orders nothing across the SYCL/native boundary.
            if (!ctx.in_order()) {
                expansion.wait();
            }

            if (side == Side::Left) {
                return ::batchlas::gemm<Back, T>(ctx, expanded, B, C, alpha, beta,
                                            Transpose::NoTrans, Transpose::NoTrans,
                                            ComputePrecision::Default);
            }
            return ::batchlas::gemm<Back, T>(ctx, B, expanded, C, alpha, beta,
                                        Transpose::NoTrans, Transpose::NoTrans,
                                        ComputePrecision::Default);
        }

        // No-scratch fallback. The real slots are null: BLAS has no real ?hemm.
        auto launch_single = [&](const MatrixView<T, MatrixFormat::Dense>& A_i,
                                 const MatrixView<T, MatrixFormat::Dense>& B_i,
                                 const MatrixView<T, MatrixFormat::Dense>& C_i) {
            call_backend<T, BackendLibrary::CUBLAS, Back>(nullptr, nullptr, cublasChemm, cublasZhemm,
                handle, side, uplo, m, n, &alpha,
                A_i.data_ptr(), A_i.ld(), B_i.data_ptr(), B_i.ld(), &beta,
                C_i.data_ptr(), C_i.ld());
        };

        for_each_batch_item(launch_single, A, B, C);

        return ctx.create_event_after_external_work();
    }

    // BATCHLAS_EXPAND_ROUTE pin (-1 = unset). One parse, shared with sytrd_blocked.cc.
    // evidence: docs/perf/level3.md#level-3-scratch-expansions-and-their-ceilings
    inline int rankk_route_pin() {
        return ::batchlas::backend::detail::expansion_route_pin();
    }

    // GEMM-plus-fold vs the per-batch cublas?herk loop. A conjunction, unlike the
    // expansion's disjunction: this GEMM does twice the rank-k arithmetic.
    // evidence: docs/perf/level3.md#herk-and-her2k-the-gemm-plus-fold-crossovers
    inline bool herk_gemm_preferred(int n, int batch) {
        const int pin = rankk_route_pin();
        if (pin >= 0) {
            return pin != 0;
        }
        return batch >= 4 && n <= 768;
    }

    // Defined in ../expansion_budget.hh so sytrd_blocked.cc can evaluate the same predicate.
    using ::batchlas::backend::detail::her2k_gemm_preferred;

    // Fold a dense product into C's referenced triangle: C = P + beta*C, or for
    // TwoSided (her2k's second term, conj-transpose of the first) C = P + P^H + beta*C.
    // The other triangle is neither read nor written; the diagonal is real on exit.
    template <typename T, bool TwoSided>
    Event accumulate_hermitian(Queue& ctx,
                               const MatrixView<T, MatrixFormat::Dense>& C,
                               const MatrixView<T, MatrixFormat::Dense>& product,
                               float_t<T> beta,
                               Uplo uplo) {
        using real_t = float_t<T>;
        const int n = C.rows();
        const int batch = C.batch_size();
        const bool lower = uplo == Uplo::Lower;

        T* dst = C.data_ptr();
        const T* src = product.data_ptr();
        const int ldc = C.ld();
        const int ldp = product.ld();
        const std::size_t stride_c = static_cast<std::size_t>(C.stride());
        const std::size_t stride_p = static_cast<std::size_t>(product.stride());

        const auto shape = detail::expand_group_shape(n);
        const sycl::range<3> global(static_cast<std::size_t>(batch),
                                    static_cast<std::size_t>(detail::ceil_div(n, shape.cols) * shape.cols),
                                    static_cast<std::size_t>(detail::ceil_div(n, shape.rows) * shape.rows));
        const sycl::range<3> local(1,
                                   static_cast<std::size_t>(shape.cols),
                                   static_cast<std::size_t>(shape.rows));

        ctx->parallel_for(sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item) {
            const int i = static_cast<int>(item.get_global_id(2));
            const int j = static_cast<int>(item.get_global_id(1));
            if (i >= n || j >= n || (lower ? (i < j) : (i > j))) {
                return;
            }
            const int b = static_cast<int>(item.get_group(0));

            const std::size_t p_base = static_cast<std::size_t>(b) * stride_p;
            T value = src[p_base + static_cast<std::size_t>(j) * ldp + i];
            if constexpr (TwoSided) {
                const T mirrored = src[p_base + static_cast<std::size_t>(i) * ldp + j];
                value = T(value.real() + mirrored.real(), value.imag() - mirrored.imag());
            }

            T* c = dst + static_cast<std::size_t>(b) * stride_c +
                   static_cast<std::size_t>(j) * ldc + i;
            if (beta != real_t(0)) {
                // The diagonal's imaginary part is not an input: the caller need not fill it.
                const T prev = *c;
                value = T(value.real() + beta * prev.real(),
                          i == j ? value.imag() : value.imag() + beta * prev.imag());
            }
            *c = i == j ? T(value.real(), real_t(0)) : value;
        });

        return ctx.get_event();
    }

    // cuBLAS has no batched ?herk: either a host loop, or one GEMM plus the fold above.
    // evidence: docs/perf/level3.md#herk-and-her2k-the-gemm-plus-fold-crossovers
    template <Backend Back, ComplexScalar T>
    Event herk_vendor(Queue& ctx,
                      const MatrixView<T, MatrixFormat::Dense>& A,
                      const MatrixView<T, MatrixFormat::Dense>& C,
                      float_t<T> alpha,
                      float_t<T> beta,
                      Uplo uplo,
                      Transpose transA) {
        static LinalgHandle<Back> handle;
        handle.setStream(ctx);

        const auto [n, k] = shape::validate_rank_k<std::invalid_argument>("HERK", A, C, transA, /*hermitian=*/true);
        const int batch = C.batch_size();

        // syrk's Gram kernel with the ^H applied; opt-in only, it loses to the fold below.
        // evidence: docs/perf/level3.md#herk-on-the-gram-tile-kernel
        if constexpr (Back == Backend::CUDA) {
            if (detail::is_gpu_queue(ctx) && syrk_route_requests_gram() &&
                detail::syrk_gram_supported(A, C, transA, /*conjugated=*/true)) {
                return detail::syrk_gram_tiles<T, true>(ctx, A, C, T(alpha), T(beta), uplo, transA);
            }
        }

        const std::size_t product_bytes = detail::expanded_workspace_bytes<T>(ctx, n, batch);
        if (herk_gemm_preferred(n, batch) && detail::expansion_fits(ctx, n, batch, product_bytes)) {
            const int ld = detail::expanded_ld<T>(n);

            auto ws = ctx.workspace(product_bytes);
            BumpAllocator pool(ws.span());
            auto storage = pool.allocate<T>(ctx, static_cast<std::size_t>(ld) *
                                                     static_cast<std::size_t>(n) *
                                                     static_cast<std::size_t>(batch));

            MatrixView<T, MatrixFormat::Dense> product(storage.data(), n, n, ld, ld * n, batch);

            // The GEMM cannot be pointed at C: it would write both triangles,
            // and HERK owns only one of them.
            // (void) on an Event: deliberate. This Queue is in-order, so the next submission
            // is already ordered after this one and the Event carries nothing the caller needs.
            (void)::batchlas::gemm<Back, T>(ctx, A, A, product, T(alpha), T(0),
                                 transA,
                                 transA == Transpose::NoTrans ? Transpose::ConjTrans
                                                              : Transpose::NoTrans,
                                 ComputePrecision::Default);

            // Out-of-order queues order nothing across the SYCL/native boundary.
            if (!ctx.in_order()) {
                ctx.wait();
            }

            return accumulate_hermitian<T, /*TwoSided=*/false>(ctx, C, product, beta, uplo);
        }

        // No-scratch fallback. The real slots are null: the real rank-k update is syrk.
        auto launch_single = [&](const MatrixView<T, MatrixFormat::Dense>& A_i,
                                 const MatrixView<T, MatrixFormat::Dense>& C_i) {
            call_backend<T, BackendLibrary::CUBLAS, Back>(nullptr, nullptr, cublasCherk, cublasZherk,
                handle, uplo, transA, n, k, &alpha,
                A_i.data_ptr(), A_i.ld(), &beta,
                C_i.data_ptr(), C_i.ld());
        };

        for_each_batch_item(launch_single, A, C);

        return ctx.create_event_after_external_work();
    }

    // As herk_vendor; one GEMM carries both terms and the TwoSided fold mirrors it.
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
        handle.setStream(ctx);

        const auto [n, k] = shape::validate_rank_2k<std::invalid_argument>("HER2K", A, B, C, transA, /*hermitian=*/true);
        const int batch = C.batch_size();
        const bool no_trans = transA == Transpose::NoTrans;

        const std::size_t product_bytes = detail::expanded_workspace_bytes<T>(ctx, n, batch);
        if (her2k_gemm_preferred(n, batch) && detail::expansion_fits(ctx, n, batch, product_bytes)) {
            const int ld = detail::expanded_ld<T>(n);

            auto ws = ctx.workspace(product_bytes);
            BumpAllocator pool(ws.span());
            auto storage = pool.allocate<T>(ctx, static_cast<std::size_t>(ld) *
                                                     static_cast<std::size_t>(n) *
                                                     static_cast<std::size_t>(batch));

            MatrixView<T, MatrixFormat::Dense> product(storage.data(), n, n, ld, ld * n, batch);

            (void)::batchlas::gemm<Back, T>(ctx, A, B, product, alpha, T(0),
                                 transA,
                                 no_trans ? Transpose::ConjTrans : Transpose::NoTrans,
                                 ComputePrecision::Default);

            if (!ctx.in_order()) {
                ctx.wait();
            }

            return accumulate_hermitian<T, /*TwoSided=*/true>(ctx, C, product, beta, uplo);
        }

        // Trap: cublasLt reads the host alpha with a 16-byte vector load, and a
        // complex<double> parameter at 8 mod 16 faults. Keep the alignas copy.
        // evidence: docs/perf/level3.md#the-her2k-alpha-alignment-fault
        alignas(16) T alpha_aligned = alpha;

        auto launch_single = [&](const MatrixView<T, MatrixFormat::Dense>& A_i,
                                 const MatrixView<T, MatrixFormat::Dense>& B_i,
                                 const MatrixView<T, MatrixFormat::Dense>& C_i) {
            call_backend<T, BackendLibrary::CUBLAS, Back>(nullptr, nullptr, cublasCher2k, cublasZher2k,
                handle, uplo, transA, n, k, &alpha_aligned,
                A_i.data_ptr(), A_i.ld(), B_i.data_ptr(), B_i.ld(), &beta,
                C_i.data_ptr(), C_i.ld());
        };

        for_each_batch_item(launch_single, A, B, C);

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
        handle.setStream(ctx);

        const auto [n, k] = shape::validate_rank_k<std::invalid_argument>("SYRK", A, C, transA, /*hermitian=*/false);

        // The complex slots are null: complex rank-k is herk, above.
        auto launch_single = [&](const MatrixView<T, MatrixFormat::Dense>& A_i,
                                 const MatrixView<T, MatrixFormat::Dense>& C_i) {
            call_backend<T, BackendLibrary::CUBLAS, Back>(cublasSsyrk, cublasDsyrk, nullptr, nullptr,
                handle, uplo, transA, n, k, &alpha,
                A_i.data_ptr(), A_i.ld(), &beta,
                C_i.data_ptr(), C_i.ld());
        };

        for_each_batch_item(launch_single, A, C);

        return ctx.create_event_after_external_work();
    }

    template <Backend Back, RealScalar T>
    Event syrk_vendor(Queue& ctx,
                      const MatrixView<T, MatrixFormat::Dense>& A,
                      const MatrixView<T, MatrixFormat::Dense>& C,
                      T alpha,
                      T beta,
                      Uplo uplo,
                      Transpose transA) {
        if constexpr (Back == Backend::CUDA) {
            // The float gate is in the facade (src/ops/level3/level3.cc). Non-float
            // reaches the Gram kernel only from here, so it has no native route in
            // a vendor-free build.
            // evidence: docs/perf/level3.md#level-3-non-float-routes-live-only-in-cublascc
            if constexpr (!std::is_same_v<T, float>) {
                if (detail::is_gpu_queue(ctx) && !syrk_route_prefers_vendor() &&
                    detail::syrk_gram_supported(A, C, transA, /*conjugated=*/false)) {
                    return detail::syrk_gram_tiles<T, false>(ctx, A, C, alpha, beta, uplo, transA);
                }
            }
        }

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
        handle.setStream(ctx);

        const auto [n, k] = shape::validate_rank_2k<std::invalid_argument>("SYR2K", A, B, C, transA, /*hermitian=*/false);

        // The complex slots are null: complex rank-2k is her2k, above.
        auto launch_single = [&](const MatrixView<T, MatrixFormat::Dense>& A_i,
                                 const MatrixView<T, MatrixFormat::Dense>& B_i,
                                 const MatrixView<T, MatrixFormat::Dense>& C_i) {
            call_backend<T, BackendLibrary::CUBLAS, Back>(cublasSsyr2k, cublasDsyr2k, nullptr, nullptr,
                handle, uplo, transA, n, k, &alpha,
                A_i.data_ptr(), A_i.ld(), B_i.data_ptr(), B_i.ld(), &beta,
                C_i.data_ptr(), C_i.ld());
        };

        for_each_batch_item(launch_single, A, B, C);

        return ctx.create_event_after_external_work();
    }

    template <Backend Back, RealScalar T>
    Event syr2k_vendor(Queue& ctx,
                       const MatrixView<T, MatrixFormat::Dense>& A,
                       const MatrixView<T, MatrixFormat::Dense>& B,
                       const MatrixView<T, MatrixFormat::Dense>& C,
                       T alpha,
                       T beta,
                       Uplo uplo,
                       Transpose transA) {
        if constexpr (Back == Backend::CUDA) {
            if (syr2k_cuda_custom_forced()) {
                if constexpr (std::is_same_v<T, float>) {
                    return syr2k_cuda_custom(ctx, A, B, C, alpha, beta, uplo, transA);
                } else {
                    throw batchlas::unsupported("BATCHLAS_SYR2K_ROUTE=cublasdx only supports float");
                }
            }
            // The float custom-route gate is in the facade (src/ops/level3/level3.cc), not here.
            // evidence: docs/perf/level3.md#level-3-non-float-routes-live-only-in-cublascc
        }

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
        handle.setStream(ctx);

        const auto [m, n, k] = shape::validate_product<std::invalid_argument>("TRMM", A, B, C, side);

        // Expansion plus GEMM beat the cublas?trmm loop in every measured cell,
        // batch 1 included, so only the fit is checked.
        // evidence: docs/perf/level3.md#symm-and-hemm-expansion-crossover
        const std::size_t expansion_bytes = detail::expanded_workspace_bytes<T>(ctx, k, A.batch_size());
        if (detail::expansion_fits(ctx, k, A.batch_size(), expansion_bytes)) {
            const int ld = detail::expanded_ld<T>(k);

            auto ws = ctx.workspace(expansion_bytes);
            BumpAllocator pool(ws.span());
            auto storage = pool.allocate<T>(ctx, static_cast<std::size_t>(ld) *
                                                     static_cast<std::size_t>(k) *
                                                     static_cast<std::size_t>(A.batch_size()));

            MatrixView<T, MatrixFormat::Dense> expanded(storage.data(), k, k, ld, ld * k, A.batch_size());

            // Not the caller's A: its opposite triangle (and diagonal under
            // Diag::Unit) may hold anything; the expansion supplies zeros and ones.
            Event expansion = detail::expand_triangular<T>(ctx, expanded, A, uplo, diag);

            // Out-of-order queues order nothing across the SYCL/native boundary.
            if (!ctx.in_order()) {
                expansion.wait();
            }

            if (side == Side::Left) {
                return ::batchlas::gemm<Back, T>(ctx, expanded, B, C, alpha, T(0),
                                            transA, Transpose::NoTrans, ComputePrecision::Default);
            }
            return ::batchlas::gemm<Back, T>(ctx, B, expanded, C, alpha, T(0),
                                        Transpose::NoTrans, transA, ComputePrecision::Default);
        }

        // No-scratch fallback for an expansion too large for the device.
        auto launch_single = [&](const MatrixView<T, MatrixFormat::Dense>& A_i,
                                 const MatrixView<T, MatrixFormat::Dense>& B_i,
                                 const MatrixView<T, MatrixFormat::Dense>& C_i) {
            call_backend<T, BackendLibrary::CUBLAS, Back>(cublasStrmm, cublasDtrmm, cublasCtrmm, cublasZtrmm,
                handle, side, uplo, transA, diag, m, n, &alpha,
                A_i.data_ptr(), A_i.ld(), B_i.data_ptr(), B_i.ld(), C_i.data_ptr(), C_i.ld());
        };

        for_each_batch_item(launch_single, A, B, C);

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
        if constexpr (Back == Backend::CUDA) {
            if (trmm_cuda_custom_forced()) {
                if constexpr (std::is_same_v<T, float>) {
                    return trmm_cuda_custom(ctx, A, B, C, alpha, side, uplo, transA, diag);
                } else {
                    throw batchlas::unsupported("BATCHLAS_TRMM_ROUTE=cublasdx only supports float");
                }
            }
            // The float gate is in the facade (src/ops/level3/level3.cc). Non-float
            // reaches the tile kernel only from here, wherever it fits: the
            // alternative is strictly more work.
            // evidence: docs/perf/level3.md#level-3-non-float-routes-live-only-in-cublascc
            if constexpr (!std::is_same_v<T, float>) {
                if (detail::is_gpu_queue(ctx) && !trmm_route_prefers_vendor() &&
                    detail::trmm_tiles_supported(A, B, C, side)) {
                    return detail::trmm_triangular_tiles(ctx, A, B, C, alpha, uplo, transA, diag);
                }
            }
        }

        return trmm_vendor_impl<Back, T>(ctx, A, B, C, alpha, side, uplo, transA, diag);
    }

    Event symm_vendor_cuda_raw(Queue& ctx,
                               const MatrixView<float, MatrixFormat::Dense>& A,
                               const MatrixView<float, MatrixFormat::Dense>& B,
                               const MatrixView<float, MatrixFormat::Dense>& C,
                               float alpha,
                               float beta,
                               Side side,
                               Uplo uplo) {
        return symm_vendor_impl<Backend::CUDA, float>(ctx, A, B, C, alpha, beta, side, uplo);
    }

    Event syrk_vendor_cuda_raw(Queue& ctx,
                               const MatrixView<float, MatrixFormat::Dense>& A,
                               const MatrixView<float, MatrixFormat::Dense>& C,
                               float alpha,
                               float beta,
                               Uplo uplo,
                               Transpose transA) {
        return syrk_vendor_impl<Backend::CUDA, float>(ctx, A, C, alpha, beta, uplo, transA);
    }

    Event syr2k_vendor_cuda_raw(Queue& ctx,
                                const MatrixView<float, MatrixFormat::Dense>& A,
                                const MatrixView<float, MatrixFormat::Dense>& B,
                                const MatrixView<float, MatrixFormat::Dense>& C,
                                float alpha,
                                float beta,
                                Uplo uplo,
                                Transpose transA) {
        return syr2k_vendor_impl<Backend::CUDA, float>(ctx, A, B, C, alpha, beta, uplo, transA);
    }

    Event trmm_vendor_cuda_raw(Queue& ctx,
                               const MatrixView<float, MatrixFormat::Dense>& A,
                               const MatrixView<float, MatrixFormat::Dense>& B,
                               const MatrixView<float, MatrixFormat::Dense>& C,
                               float alpha,
                               Side side,
                               Uplo uplo,
                               Transpose transA,
                               Diag diag) {
        return trmm_vendor_impl<Backend::CUDA, float>(ctx, A, B, C, alpha, side, uplo, transA, diag);
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
        handle.setStream(ctx);
        auto m = A.rows();
        auto n = A.cols();
        auto batch_size = A.batch_size();
        if (batch_size <= 1) {
            call_backend<T, BackendLibrary::CUBLAS, B>(cublasSgemv, cublasDgemv, cublasCgemv, cublasZgemv,
                handle, transA, m, n, &alpha, A.data_ptr(), A.ld(), X.data_ptr(), X.inc(), &beta, Y.data_ptr(), Y.inc());
        } else {
            call_backend<T, BackendLibrary::CUBLAS, B>(cublasSgemvStridedBatched, cublasDgemvStridedBatched, cublasCgemvStridedBatched, cublasZgemvStridedBatched,
                handle, transA, m, n, &alpha, A.data_ptr(), A.ld(), A.stride(), X.data_ptr(), X.inc(), X.stride(), &beta, Y.data_ptr(), Y.inc(), Y.stride(), batch_size);
        }
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
        handle.setStream(ctx);
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
            if (batch_size == 1) {
                call_backend<T, BackendLibrary::CUBLAS, Back>(cublasStrsm, cublasDtrsm, cublasCtrsm, cublasZtrsm,
                    handle, side, uplo, transA, diag, kB, n, &alpha, A.data_ptr(), A.ld(), B.data_ptr(), B.ld());
            } else {
                call_backend<T, BackendLibrary::CUBLAS, Back>(cublasStrsmBatched, cublasDtrsmBatched, cublasCtrsmBatched, cublasZtrsmBatched,
                    handle, side, uplo, transA, diag, kB, n, &alpha, A.data_ptrs(ctx).data(), A.ld(), B.data_ptrs(ctx).data(), B.ld(), batch_size);
            }
        }
        return ctx.create_event_after_external_work();
    }

    template <Backend B, typename T>
    Event geqrf_vendor(Queue& ctx,
        const MatrixView<T,MatrixFormat::Dense>& A, //In place reflectors (Lower triangle of A)
        Span<T> tau,
        Span<std::byte> work_space) {
        static LinalgHandle<B> handle;
        handle.setStream(ctx);
        auto m = A.rows();
        auto n = A.cols();
        auto k = std::min(m, n);
        auto batch_size = A.batch_size();
        auto pool = BumpAllocator(work_space);
        if (batch_size <= 1) {
            cusolverDnParams_t params;
            cusolverDnCreateParams(&params);
            size_t device_l_work, host_l_work;
            cusolverDnXgeqrf_bufferSize(handle, params, m, n,
                BackendScalar<T,BackendLibrary::CUSOLVER>::type, A.data_ptr(), A.ld(),
                BackendScalar<T,BackendLibrary::CUSOLVER>::type, tau.data(),
                BackendScalar<T,BackendLibrary::CUSOLVER>::type, &device_l_work, &host_l_work);
            auto device_work_space = pool.allocate<std::byte>(ctx, device_l_work);
            auto host_work_space = pool.allocate<std::byte>(ctx, host_l_work);
            auto d_info = pool.allocate<int>(ctx, 1);
            cusolverDnXgeqrf(handle, params, m, n,
                BackendScalar<T,BackendLibrary::CUSOLVER>::type, A.data_ptr(), A.ld(),
                BackendScalar<T,BackendLibrary::CUSOLVER>::type, tau.data(),
                BackendScalar<T,BackendLibrary::CUSOLVER>::type, device_work_space.data(),
                device_l_work, host_work_space.data(), host_l_work, d_info.data());
        } else {
            auto tau_data = tau.data();
            auto tau_ptrs = pool.allocate<T*>(ctx, batch_size);
            ctx->parallel_for(sycl::range<1>(batch_size), [=](sycl::id<1> item) {
                size_t i = item.get(0);
                tau_ptrs[i] = tau_data + i * k;
            });
            auto info = pool.allocate<int>(ctx, batch_size);
            call_backend<T, BackendLibrary::CUBLAS, B>(cublasSgeqrfBatched, cublasDgeqrfBatched, cublasCgeqrfBatched, cublasZgeqrfBatched,
                handle, m, n, A.data_ptrs(ctx).data(), A.ld(), tau_ptrs.data(), info.data(), batch_size);
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
            size_t device_l_work, host_l_work;
            cusolverDnParams_t params;
            cusolverDnCreateParams(&params);
            cusolverDnXgeqrf_bufferSize(handle, params, m, n,
                BackendScalar<T,BackendLibrary::CUBLAS>::type, A.data_ptr(), A.ld(),
                BackendScalar<T,BackendLibrary::CUBLAS>::type, tau.data(),
                BackendScalar<T,BackendLibrary::CUBLAS>::type, &device_l_work, &host_l_work);
            return BumpAllocator::allocation_size<std::byte>(ctx, device_l_work) + BumpAllocator::allocation_size<std::byte>(ctx, host_l_work) 
                   + BumpAllocator::allocation_size<int>(ctx, 1); // +1 for info
        } else {
            return BumpAllocator::allocation_size<T*>(ctx, batch_size) + BumpAllocator::allocation_size<int>(ctx, batch_size);
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
        handle.setStream(ctx);
        auto m = C.rows();
        auto n = C.cols();
        auto k = std::min(A.rows(), A.cols());
        auto batch_size = A.batch_size();
        BumpAllocator pool(workspace);
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
            auto device_ws = pool.allocate<T>(ctx, lwork);
            auto info = pool.allocate<int>(ctx, 1);
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
        handle.setStream(ctx);
        auto m = A.rows();
        auto n = A.cols();
        auto k = std::min(m, n);
        auto batch_size = A.batch_size();
        BumpAllocator pool(workspace);
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
            auto device_ws = pool.allocate<T>(ctx, lwork);
            auto info = pool.allocate<int>(ctx, 1);
            call_backend<T, BackendLibrary::CUSOLVER, B>(
                cusolverDnSorgqr, cusolverDnDorgqr,
                cusolverDnCungqr, cusolverDnZungqr,
                handle,
                m, n, k,
                A.data_ptr(), A.ld(),
                tau.data(),
                device_ws.data(), lwork, info.data());
        } else {
            Queue sub_queue(ctx.device(), false);
            size_t single_ws = orgqr_vendor_buffer_size<B>(ctx, A.batch_item(0), tau.subspan(0, k));
            for (int i = 0; i < batch_size; ++i) {
                auto sub_ws = pool.allocate<std::byte>(sub_queue, single_ws);
                (void)orgqr_vendor<B>(sub_queue, A.batch_item(i), tau.subspan(i * k, k), sub_ws);
            }
            sub_queue.wait();
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
        Span<std::byte> work_space) {
            static LinalgHandle<Back> handle;
            handle.setStream(ctx);
            auto n = A.rows();
            auto nrhs = B.cols();
            auto batch_size = A.batch_size();
            auto pool = BumpAllocator(work_space);
            // One arm for every batch size. Pivots are packed 1-based int32
            // (every getrf writes as_span<int>); a batch-1 cusolverDnXgetrs arm
            // read them as int64 and crashed. evidence: docs/perf/lu.md#lu-correctness-findings
            //
            // `info` is a HOST int (an argument-validity code), so nothing comes
            // from the pool; the buffer-size query still reports one int on purpose.
            static_cast<void>(pool);
            int info;
            auto reinterpreted_pivots = pivots.as_span<int>();
            call_backend<T, BackendLibrary::CUBLAS, Back>(cublasSgetrsBatched, cublasDgetrsBatched, cublasCgetrsBatched, cublasZgetrsBatched,
                handle, enum_convert<BackendLibrary::CUBLAS>(transA), n, nrhs,
                A.data_ptrs(ctx).data(), A.ld(), reinterpreted_pivots.data(),
                B.data_ptrs(ctx).data(), B.ld(), &info, batch_size);
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
        Span<std::byte> work_space,
        Span<int32_t> info_out) {
            static LinalgHandle<B> handle;
            handle.setStream(ctx);
            auto n = A.rows();
            auto batch_size = A.batch_size();
            auto pool = BumpAllocator(work_space);
            // infoArray is per-item with LAPACK semantics (>0 = exactly singular column).
            auto info = ::batchlas::detail::info_target(ctx, pool, info_out, static_cast<size_t>(batch_size));
            auto reinterpreted_pivots = pivots.as_span<int>();
            call_backend<T, BackendLibrary::CUBLAS, B>(cublasSgetrfBatched, cublasDgetrfBatched, cublasCgetrfBatched, cublasZgetrfBatched,
                handle, n,
                A.data_ptrs(ctx).data(), A.ld(), reinterpreted_pivots.data(), info.data(), batch_size);
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
        Span<std::byte> work_space,
        Span<int32_t> info_out) {
            static LinalgHandle<B> handle;
            handle.setStream(ctx);
            auto n = A.rows();
            auto batch_size = A.batch_size();
            auto pool = BumpAllocator(work_space);
            // Per-item infoArray: >0 = U(i,i) is exactly zero, no inverse for this item.
            auto info_arr = ::batchlas::detail::info_target(ctx, pool, info_out, static_cast<size_t>(batch_size));
            auto reinterpreted_pivots = pivots.as_span<int>();
            call_backend<T, BackendLibrary::CUBLAS, B>(cublasSgetriBatched, cublasDgetriBatched, cublasCgetriBatched, cublasZgetriBatched,
                handle, n,
                A.data_ptrs(ctx).data(), A.ld(), reinterpreted_pivots.data(),
                C.data_ptrs(ctx).data(), C.ld(), info_arr.data(), batch_size);
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

    // Only `backend::` vendor entry points: the public ops are defined in
    // src/ops/, so a public row here would be a duplicate definition.
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

    // symm/syrk/syr2k are real-only and hemm/herk/her2k are complex-only, so the
    // narrower domains get their own tables rather than one blanket loop.
    #define CUBLAS_REAL_OPS(B, fp) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, symm_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, syrk_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, syr2k_vendor)

    #define CUBLAS_COMPLEX_OPS(B, fp) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, hemm_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, herk_vendor) \
        BATCHLAS_INSTANTIATE_BACKEND_OP(B, fp, her2k_vendor)

    BATCHLAS_FOR_EACH_SCALAR_TYPE_1(CUBLAS_OPS, B_)
    BATCHLAS_FOR_EACH_REAL_TYPE_1(CUBLAS_REAL_OPS, B_)
    BATCHLAS_FOR_EACH_COMPLEX_TYPE_1(CUBLAS_COMPLEX_OPS, B_)

    #undef CUBLAS_OPS
    #undef CUBLAS_REAL_OPS
    #undef CUBLAS_COMPLEX_OPS
    #undef B_
}
