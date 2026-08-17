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
#include <batchlas/blas/dispatch/op.hh>
#include <complex>

#include "gemm_cublasdx_dispatch.hh"
#include "gemm_variant.hh"
#include "level3_coverage.hh"
#include "symm_custom_dispatch.hh"
#include "syr2k_custom_dispatch.hh"
#include "syrk_custom_dispatch.hh"
#include "syrk_gram_tiles.hh"
#include "cublasdx_dispatch_common.hh"
#include "trmm_custom_dispatch.hh"
#include "trmm_triangular_tiles.hh"
#include "triangular_expand.hh"
#include "../sycl/gemm_kernels.hh"

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
                       ComputePrecision precision);

    template <Backend Back, typename T>
    Event gemm_heterogeneous_vendor_impl(Queue& ctx,
                                         const MatrixView<T, MatrixFormat::Dense>& A,
                                         const MatrixView<T, MatrixFormat::Dense>& B,
                                         const MatrixView<T, MatrixFormat::Dense>& C,
                                         T alpha,
                                         T beta,
                                         Transpose transA,
                                         Transpose transB,
                                         ComputePrecision precision) {
        if (!gemm_batch_dimensions_compatible(A, B, C, transA, transB)) {
            throw std::invalid_argument("GEMM: incompatible per-batch matrix dimensions for heterogeneous dispatch");
        }

        bool launched = false;
        Event last_event;
        for (int batch_index = 0; batch_index < A.batch_size(); ++batch_index) {
            const auto [m, k] = get_effective_dims(A, transA, batch_index);
            const auto [k_b, n] = get_effective_dims(B, transB, batch_index);
            static_cast<void>(k_b);
            if (m == 0 || n == 0) {
                continue;
            }
            if (k == 0) {
                last_event = scale(ctx, beta, C.batch_item(batch_index));
                launched = true;
                continue;
            }

            last_event = gemm_vendor_impl<Back, T>(ctx,
                                                   A.batch_item(batch_index),
                                                   B.batch_item(batch_index),
                                                   C.batch_item(batch_index),
                                                   alpha,
                                                   beta,
                                                   transA,
                                                   transB,
                                                   precision);
            launched = true;
        }

        if (launched) {
            return std::move(last_event);
        }
        return ctx.create_event_after_external_work();
    }

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
            throw std::invalid_argument("GEMM: incompatible matrix dimensions");
        }

        auto [m, k] = get_effective_dims(A, transA);
        auto [kB, n] = get_effective_dims(B, transB);
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
        if constexpr (Back == Backend::CUDA) {
            if constexpr (std::is_same_v<T, float>) {
                if (gemm_use_cublasdx_custom(ctx, A, B, C, transA, transB, precision)) {
                    return gemm_cublasdx(ctx, A, B, C, alpha, beta, transA, transB, precision);
                }
            }
        }

        if (gemm_has_heterogeneous_batch(A, B, C)) {
            return gemm_heterogeneous_vendor_impl<Back, T>(ctx, A, B, C, alpha, beta, transA, transB, precision);
        }

        if (gemm_use_sycl_custom(ctx, A, B, C, transA, transB, precision)) {
            return sycl_gemm::gemm_custom(ctx, A, B, C, alpha, beta, transA, transB, precision);
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

        if (A.rows() != A.cols()) {
            throw std::invalid_argument("SYMM: A must be square");
        }
        if (A.batch_size() != B.batch_size() || A.batch_size() != C.batch_size()) {
            throw std::invalid_argument(
                "SYMM: batch size mismatch (A=" + std::to_string(A.batch_size()) +
                ", B=" + std::to_string(B.batch_size()) +
                ", C=" + std::to_string(C.batch_size()) + ")");
        }

        const int m = C.rows();
        const int n = C.cols();
        const int expected_a = side == Side::Left ? B.rows() : B.cols();
        if (A.rows() != expected_a || B.rows() != m || B.cols() != n) {
            throw std::invalid_argument("SYMM: incompatible matrix dimensions");
        }

        const auto side_cublas = enum_convert<BackendLibrary::CUBLAS>(side);
        const auto uplo_cublas = enum_convert<BackendLibrary::CUBLAS>(uplo);

        auto launch_single = [&](const MatrixView<T, MatrixFormat::Dense>& A_i,
                                 const MatrixView<T, MatrixFormat::Dense>& B_i,
                                 const MatrixView<T, MatrixFormat::Dense>& C_i) {
            if constexpr (std::is_same_v<T, float>) {
                cublasSsymm(handle,
                            side_cublas,
                            uplo_cublas,
                            m,
                            n,
                            &alpha,
                            A_i.data_ptr(),
                            A_i.ld(),
                            B_i.data_ptr(),
                            B_i.ld(),
                            &beta,
                            C_i.data_ptr(),
                            C_i.ld());
            } else if constexpr (std::is_same_v<T, double>) {
                cublasDsymm(handle,
                            side_cublas,
                            uplo_cublas,
                            m,
                            n,
                            &alpha,
                            A_i.data_ptr(),
                            A_i.ld(),
                            B_i.data_ptr(),
                            B_i.ld(),
                            &beta,
                            C_i.data_ptr(),
                            C_i.ld());
            }
        };

        if (A.batch_size() <= 1) {
            launch_single(A, B, C);
        } else {
            for (int batch = 0; batch < A.batch_size(); ++batch) {
                launch_single(A[batch], B[batch], C[batch]);
            }
        }

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
        if constexpr (Back == Backend::CUDA) {
            if constexpr (std::is_same_v<T, float>) {
                if (symm_use_cuda_custom(ctx, A, B, C, side, uplo)) {
                    return symm_cuda_custom(ctx, A, B, C, alpha, beta, side, uplo);
                }
                // GATE DECLINED. Record here, because *_cuda_custom -- which
                // carries every other route row -- is never entered, and this
                // is the half a route diff needs most: a shape moving OFF a
                // native kernel onto the vendor shows up only on this side.
                // kNativeUnknown, not false: the gate conflates "nothing
                // native serves this shape" with "something does but the
                // vendor was preferred", and the call site cannot tell them
                // apart. See level3_coverage.hh.
                detail::record_level3_route(
                    dispatch::Op::symm,
                    dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto},
                    C.rows(), C.cols(), A.rows(), A.batch_size(),
                    detail::kNativeUnknown,
                    {uplo, side, Diag::NonUnit, Transpose::NoTrans});
            }
        }

        return symm_vendor_impl<Back, T>(ctx, A, B, C, alpha, beta, side, uplo);
    }

    // There is no batched or strided-batched ?hemm in cuBLAS -- only the single
    // cublasChemm/cublasZhemm -- so a batch of them is a host loop over kernel
    // launches. Expanding the Hermitian triangle into scratch turns the whole
    // batch into one strided-batched GEMM instead, which is the same trade TRMM
    // makes above for the same reason.
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

        if (A.rows() != A.cols()) {
            throw std::invalid_argument("HEMM: A must be square");
        }
        if (A.batch_size() != B.batch_size() || A.batch_size() != C.batch_size()) {
            throw std::invalid_argument(
                "HEMM: batch size mismatch (A=" + std::to_string(A.batch_size()) +
                ", B=" + std::to_string(B.batch_size()) +
                ", C=" + std::to_string(C.batch_size()) + ")");
        }

        const int m = C.rows();
        const int n = C.cols();
        // A multiplies from whichever side the caller asked for, so it is m x m
        // on the left and n x n on the right.
        const int k = side == Side::Left ? m : n;
        if (A.rows() != k || B.rows() != m || B.cols() != n) {
            throw std::invalid_argument("HEMM: incompatible matrix dimensions");
        }

        // Unlike cublas?trmm, cublas?hemm is quick enough that a per-batch loop
        // over it beats the expansion on a launch-bound call, so the shape has
        // to be worth the extra kernel before the scratch is worth allocating.
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

            // The GEMM cannot be pointed at the caller's A. HEMM reads one
            // triangle and takes the other to be its conjugate transpose, and
            // takes the diagonal to be real, so neither the opposite triangle
            // nor the diagonal's imaginary part is part of the operand -- that
            // storage may hold anything at all.
            Event expansion = detail::expand_mirrored<T, /*Conjugate=*/true>(ctx, expanded, A, uplo);

            // The GEMM runs on the queue's native stream, which an in-order
            // queue shares with the expansion kernel. An out-of-order queue
            // orders nothing across the SYCL/native boundary and offers no event
            // to hang the vendor launch off, so there the dependency has to be
            // waited out.
            if (!ctx.in_order()) {
                expansion.wait();
            }

            if (side == Side::Left) {
                return gemm_vendor<Back, T>(ctx, expanded, B, C, alpha, beta,
                                            Transpose::NoTrans, Transpose::NoTrans,
                                            ComputePrecision::Default);
            }
            return gemm_vendor<Back, T>(ctx, B, expanded, C, alpha, beta,
                                        Transpose::NoTrans, Transpose::NoTrans,
                                        ComputePrecision::Default);
        }

        // Slower, but it needs no scratch, so it is what an expansion too large
        // for the device falls back to. The two real slots have no callee
        // because BLAS has no real ?hemm; T is constrained to complex, so they
        // are never selected.
        auto launch_single = [&](const MatrixView<T, MatrixFormat::Dense>& A_i,
                                 const MatrixView<T, MatrixFormat::Dense>& B_i,
                                 const MatrixView<T, MatrixFormat::Dense>& C_i) {
            call_backend<T, BackendLibrary::CUBLAS, Back>(nullptr, nullptr, cublasChemm, cublasZhemm,
                handle, side, uplo, m, n, &alpha,
                A_i.data_ptr(), A_i.ld(), B_i.data_ptr(), B_i.ld(), &beta,
                C_i.data_ptr(), C_i.ld());
        };

        if (A.batch_size() <= 1) {
            launch_single(A, B, C);
        } else {
            for (int batch = 0; batch < A.batch_size(); ++batch) {
                launch_single(A[batch], B[batch], C[batch]);
            }
        }

        return ctx.create_event_after_external_work();
    }

    // BATCHLAS_EXPAND_ROUTE pins the route, as it does for the mirrored
    // expansion in triangular_expand.hh, so a test can reach whichever one the
    // shape would not have picked. Returns -1 when it is unset.
    // One definition, in ../expansion_budget.hh, because sytrd_blocked.cc has to
    // predict this route before calling her2k and a second parse here could
    // drift from it silently.
    inline int rankk_route_pin() {
        return ::batchlas::backend::detail::expansion_route_pin();
    }

    // Where a GEMM over the whole n x n product beats a per-batch loop over
    // cublas?herk. The GEMM computes both triangles and keeps one, so it starts
    // from twice a rank-k update's arithmetic and only wins where the loop is
    // launch-bound. Measured on sm_89 in complex64 over n in 32..1024 x batch
    // in 1..256, as loop time over GEMM-route time: 1.6x to 72x for batch >= 4
    // at n <= 512, a wash from n = 640 to 768, and 0.82x-0.93x from n = 896 up,
    // where one cublas?herk already saturates the device on its own. batch <= 2
    // is a wash or a loss at every n.
    //
    // Note that this is a conjunction where the mirrored expansion's threshold
    // is a disjunction: that one has no large-n ceiling because expanding an
    // operand costs a bandwidth-bound kernel and then does exactly the vendor's
    // work, so it never pays twice for the arithmetic.
    inline bool herk_gemm_preferred(int n, int batch) {
        const int pin = rankk_route_pin();
        if (pin >= 0) {
            return pin != 0;
        }
        return batch >= 4 && n <= 768;
    }

    // HER2K's is a far better trade and the crossover moves accordingly; the
    // threshold and its measurements now live in ../expansion_budget.hh, next to
    // the size ceiling, so that sytrd_blocked.cc can evaluate the same
    // conjunction this function is one half of before it decides to call her2k.
    using ::batchlas::backend::detail::her2k_gemm_preferred;

    // Fold a dense rank-k product into the referenced triangle of a Hermitian
    // C: C = product + beta * C, or, for TwoSided, C = product + product^H +
    // beta * C.
    //
    // TwoSided is how HER2K gets its second term for free. The two terms
    // alpha * A * B^H and conj(alpha) * B * A^H are conjugate transposes of one
    // another, so one GEMM produces both and the mirrored read below adds them.
    //
    // The unreferenced triangle is neither read nor written -- the work items
    // covering it return before touching C -- and the diagonal is real on exit,
    // because C = C^H says so whatever the caller left in its imaginary part.
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
                // beta is real and C is Hermitian, so the diagonal's
                // imaginary part is not an input either: it is storage the
                // caller never had to fill.
                const T prev = *c;
                value = T(value.real() + beta * prev.real(),
                          i == j ? value.imag() : value.imag() + beta * prev.imag());
            }
            *c = i == j ? T(value.real(), real_t(0)) : value;
        });

        return ctx.get_event();
    }

    // cuBLAS has no batched or strided-batched ?herk -- only the single
    // cublasCherk/cublasZherk -- so a batch of them is a host loop over kernel
    // launches. The alternative is one strided-batched GEMM over the whole
    // n x n product plus the fold above, which computes twice the arithmetic a
    // rank-k update needs but in two launches rather than one per batch item.
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

        if (C.rows() != C.cols()) {
            throw std::invalid_argument("HERK: C must be square");
        }
        if (A.batch_size() != C.batch_size()) {
            throw std::invalid_argument(
                "HERK: batch size mismatch (A=" + std::to_string(A.batch_size()) +
                ", C=" + std::to_string(C.batch_size()) + ")");
        }
        // Transpose::Trans would ask for A * A^T, which is complex-symmetric
        // rather than Hermitian; that operation is syrk's, and BLAS does not
        // spell it here.
        if (transA != Transpose::NoTrans && transA != Transpose::ConjTrans) {
            throw std::invalid_argument("HERK: transA must be NoTrans or ConjTrans");
        }

        const int n = C.rows();
        const int batch = C.batch_size();
        const int k = transA == Transpose::NoTrans ? A.cols() : A.rows();
        const int expected_n = transA == Transpose::NoTrans ? A.rows() : A.cols();
        if (expected_n != n || k <= 0) {
            throw std::invalid_argument("HERK: incompatible matrix dimensions");
        }

        // The same single-tile Gram kernel as syrk, with the ^H conjugating
        // whichever operand carries it. Opt-in only: see syrk_route_requests_gram
        // for the measurement that keeps it off the automatic path -- a complex
        // multiply is four real ones, so this shape is compute bound for herk
        // where it is bandwidth bound for syrk, and the GEMM-plus-fold below
        // wins on every Gram shape measured.
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
            gemm_vendor<Back, T>(ctx, A, A, product, T(alpha), T(0),
                                 transA,
                                 transA == Transpose::NoTrans ? Transpose::ConjTrans
                                                              : Transpose::NoTrans,
                                 ComputePrecision::Default);

            // The GEMM runs on the queue's native stream, which an in-order
            // queue shares with the fold below. An out-of-order queue orders
            // nothing across the SYCL/native boundary, so there the dependency
            // has to be waited out.
            if (!ctx.in_order()) {
                ctx.wait();
            }

            return accumulate_hermitian<T, /*TwoSided=*/false>(ctx, C, product, beta, uplo);
        }

        // Slower for a batch, but it needs no scratch, so it is also what a
        // product too large for the device falls back to. The two real slots
        // have no callee because BLAS has no real ?herk -- that is syrk; T is
        // constrained to complex, so they are never selected.
        auto launch_single = [&](const MatrixView<T, MatrixFormat::Dense>& A_i,
                                 const MatrixView<T, MatrixFormat::Dense>& C_i) {
            call_backend<T, BackendLibrary::CUBLAS, Back>(nullptr, nullptr, cublasCherk, cublasZherk,
                handle, uplo, transA, n, k, &alpha,
                A_i.data_ptr(), A_i.ld(), &beta,
                C_i.data_ptr(), C_i.ld());
        };

        if (batch <= 1) {
            launch_single(A, C);
        } else {
            for (int b = 0; b < batch; ++b) {
                launch_single(A[b], C[b]);
            }
        }

        return ctx.create_event_after_external_work();
    }

    // As herk_vendor, except that the single GEMM carries both of HER2K's
    // terms: alpha * A * B^H and conj(alpha) * B * A^H are conjugate transposes
    // of one another, so the fold reads the product's mirrored element instead
    // of running a second GEMM.
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

        if (C.rows() != C.cols()) {
            throw std::invalid_argument("HER2K: C must be square");
        }
        if (A.batch_size() != B.batch_size() || A.batch_size() != C.batch_size()) {
            throw std::invalid_argument(
                "HER2K: batch size mismatch (A=" + std::to_string(A.batch_size()) +
                ", B=" + std::to_string(B.batch_size()) +
                ", C=" + std::to_string(C.batch_size()) + ")");
        }
        if (transA != Transpose::NoTrans && transA != Transpose::ConjTrans) {
            throw std::invalid_argument("HER2K: transA must be NoTrans or ConjTrans");
        }

        const int n = C.rows();
        const int batch = C.batch_size();
        const bool no_trans = transA == Transpose::NoTrans;
        const int k = no_trans ? A.cols() : A.rows();
        const int expected_n = no_trans ? A.rows() : A.cols();
        const int b_n = no_trans ? B.rows() : B.cols();
        const int b_k = no_trans ? B.cols() : B.rows();
        if (expected_n != n || b_n != n || b_k != k || k <= 0) {
            throw std::invalid_argument("HER2K: incompatible matrix dimensions");
        }

        const std::size_t product_bytes = detail::expanded_workspace_bytes<T>(ctx, n, batch);
        if (her2k_gemm_preferred(n, batch) && detail::expansion_fits(ctx, n, batch, product_bytes)) {
            const int ld = detail::expanded_ld<T>(n);

            auto ws = ctx.workspace(product_bytes);
            BumpAllocator pool(ws.span());
            auto storage = pool.allocate<T>(ctx, static_cast<std::size_t>(ld) *
                                                     static_cast<std::size_t>(n) *
                                                     static_cast<std::size_t>(batch));

            MatrixView<T, MatrixFormat::Dense> product(storage.data(), n, n, ld, ld * n, batch);

            gemm_vendor<Back, T>(ctx, A, B, product, alpha, T(0),
                                 transA,
                                 no_trans ? Transpose::ConjTrans : Transpose::NoTrans,
                                 ComputePrecision::Default);

            if (!ctx.in_order()) {
                ctx.wait();
            }

            return accumulate_hermitian<T, /*TwoSided=*/true>(ctx, C, product, beta, uplo);
        }

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
                A_i.data_ptr(), A_i.ld(), B_i.data_ptr(), B_i.ld(), &beta,
                C_i.data_ptr(), C_i.ld());
        };

        if (batch <= 1) {
            launch_single(A, B, C);
        } else {
            for (int b = 0; b < batch; ++b) {
                launch_single(A[b], B[b], C[b]);
            }
        }

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

        if (C.rows() != C.cols()) {
            throw std::invalid_argument("SYRK: C must be square");
        }
        if (A.batch_size() != C.batch_size()) {
            throw std::invalid_argument(
                "SYRK: batch size mismatch (A=" + std::to_string(A.batch_size()) +
                ", C=" + std::to_string(C.batch_size()) + ")");
        }

        const int n = C.rows();
        const int k = transA == Transpose::NoTrans ? A.cols() : A.rows();
        const int expected_n = transA == Transpose::NoTrans ? A.rows() : A.cols();
        if (expected_n != n || k <= 0) {
            throw std::invalid_argument("SYRK: incompatible matrix dimensions");
        }

        const auto uplo_cublas = enum_convert<BackendLibrary::CUBLAS>(uplo);
        const auto trans_cublas = enum_convert<BackendLibrary::CUBLAS>(transA);

        auto launch_single = [&](const MatrixView<T, MatrixFormat::Dense>& A_i,
                                 const MatrixView<T, MatrixFormat::Dense>& C_i) {
            if constexpr (std::is_same_v<T, float>) {
                cublasSsyrk(handle,
                            uplo_cublas,
                            trans_cublas,
                            n,
                            k,
                            &alpha,
                            A_i.data_ptr(),
                            A_i.ld(),
                            &beta,
                            C_i.data_ptr(),
                            C_i.ld());
            } else if constexpr (std::is_same_v<T, double>) {
                cublasDsyrk(handle,
                            uplo_cublas,
                            trans_cublas,
                            n,
                            k,
                            &alpha,
                            A_i.data_ptr(),
                            A_i.ld(),
                            &beta,
                            C_i.data_ptr(),
                            C_i.ld());
            }
        };

        if (A.batch_size() <= 1) {
            launch_single(A, C);
        } else {
            for (int batch = 0; batch < A.batch_size(); ++batch) {
                launch_single(A[batch], C[batch]);
            }
        }

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
            if constexpr (std::is_same_v<T, float>) {
                if (syrk_use_cuda_custom(ctx, A, C, uplo, transA)) {
                    return syrk_cuda_custom(ctx, A, C, alpha, beta, uplo, transA);
                }
                // GATE DECLINED. Record here, because *_cuda_custom -- which
                // carries every other route row -- is never entered, and this
                // is the half a route diff needs most: a shape moving OFF a
                // native kernel onto the vendor shows up only on this side.
                // kNativeUnknown, not false: the gate conflates "nothing
                // native serves this shape" with "something does but the
                // vendor was preferred", and the call site cannot tell them
                // apart. See level3_coverage.hh.
                detail::record_level3_route(
                    dispatch::Op::syrk,
                    dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto},
                    C.rows(), C.cols(),
                    transA == Transpose::NoTrans ? A.cols() : A.rows(),
                    A.batch_size(), detail::kNativeUnknown,
                    {uplo, Side::Left, Diag::NonUnit, transA});
            } else {
                // Everything that is not float reaches the single-tile Gram
                // kernel only. It is the one route here whose staging and
                // fragment loads are not written around a 128-bit packet, so it
                // is the one that generalises; the 128x128 triangular kernel
                // stays float. Below kGramMaxTile the alternative is
                // syrk_vendor_impl's host loop over one cublasXsyrk per batch
                // member, which at large batch is two orders of magnitude off
                // anything batched, so there is no threshold to tune.
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

        if (C.rows() != C.cols()) {
            throw std::invalid_argument("SYR2K: C must be square");
        }
        if (A.batch_size() != B.batch_size() || A.batch_size() != C.batch_size()) {
            throw std::invalid_argument(
                "SYR2K: batch size mismatch (A=" + std::to_string(A.batch_size()) +
                ", B=" + std::to_string(B.batch_size()) +
                ", C=" + std::to_string(C.batch_size()) + ")");
        }

        const int n = C.rows();
        const int expected_n = transA == Transpose::NoTrans ? A.rows() : A.cols();
        const int k = transA == Transpose::NoTrans ? A.cols() : A.rows();
        const int expected_b_n = transA == Transpose::NoTrans ? B.rows() : B.cols();
        const int b_k = transA == Transpose::NoTrans ? B.cols() : B.rows();
        if (expected_n != n || expected_b_n != n || b_k != k || k <= 0) {
            throw std::invalid_argument("SYR2K: incompatible matrix dimensions");
        }

        const auto uplo_cublas = enum_convert<BackendLibrary::CUBLAS>(uplo);
        const auto trans_cublas = enum_convert<BackendLibrary::CUBLAS>(transA);

        auto launch_single = [&](const MatrixView<T, MatrixFormat::Dense>& A_i,
                                 const MatrixView<T, MatrixFormat::Dense>& B_i,
                                 const MatrixView<T, MatrixFormat::Dense>& C_i) {
            if constexpr (std::is_same_v<T, float>) {
                cublasSsyr2k(handle,
                             uplo_cublas,
                             trans_cublas,
                             n,
                             k,
                             &alpha,
                             A_i.data_ptr(),
                             A_i.ld(),
                             B_i.data_ptr(),
                             B_i.ld(),
                             &beta,
                             C_i.data_ptr(),
                             C_i.ld());
            } else if constexpr (std::is_same_v<T, double>) {
                cublasDsyr2k(handle,
                             uplo_cublas,
                             trans_cublas,
                             n,
                             k,
                             &alpha,
                             A_i.data_ptr(),
                             A_i.ld(),
                             B_i.data_ptr(),
                             B_i.ld(),
                             &beta,
                             C_i.data_ptr(),
                             C_i.ld());
            }
        };

        if (A.batch_size() <= 1) {
            launch_single(A, B, C);
        } else {
            for (int batch = 0; batch < A.batch_size(); ++batch) {
                launch_single(A[batch], B[batch], C[batch]);
            }
        }

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
                    throw std::runtime_error("BATCHLAS_SYR2K_VARIANT=cublasdx only supports float");
                }
            }
            if constexpr (std::is_same_v<T, float>) {
                if (syr2k_use_cuda_custom(ctx, A, B, C, uplo, transA)) {
                    return syr2k_cuda_custom(ctx, A, B, C, alpha, beta, uplo, transA);
                }
                // GATE DECLINED. Record here, because *_cuda_custom -- which
                // carries every other route row -- is never entered, and this
                // is the half a route diff needs most: a shape moving OFF a
                // native kernel onto the vendor shows up only on this side.
                // kNativeUnknown, not false: the gate conflates "nothing
                // native serves this shape" with "something does but the
                // vendor was preferred", and the call site cannot tell them
                // apart. See level3_coverage.hh.
                detail::record_level3_route(
                    dispatch::Op::syr2k,
                    dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto},
                    C.rows(), C.cols(),
                    transA == Transpose::NoTrans ? A.cols() : A.rows(),
                    A.batch_size(), detail::kNativeUnknown,
                    {uplo, Side::Left, Diag::NonUnit, transA});
            }
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

        if (A.rows() != A.cols()) {
            throw std::invalid_argument("TRMM: A must be square");
        }
        if (A.batch_size() != B.batch_size() || A.batch_size() != C.batch_size()) {
            throw std::invalid_argument(
                "TRMM: batch size mismatch (A=" + std::to_string(A.batch_size()) +
                ", B=" + std::to_string(B.batch_size()) +
                ", C=" + std::to_string(C.batch_size()) + ")");
        }

        const int m = C.rows();
        const int n = C.cols();
        // A multiplies from whichever side the caller asked for, so it is m x m
        // on the left and n x n on the right.
        const int k = side == Side::Left ? m : n;
        if (A.rows() != k || B.rows() != m || B.cols() != n) {
            throw std::invalid_argument("TRMM: incompatible matrix dimensions");
        }

        // One expansion plus one strided-batched GEMM beats the per-batch
        // cublas?trmm loop everywhere it fits. Measured in float on sm_89 over
        // square shapes, k in 16..1024 and batch in 1..512 (49 cells, 1.15x to
        // 162x) and over skewed ones, k in 256..2048 against 1..128 right-hand
        // sides (64 cells, 1.22x to 32x). Not one cell went the other way, not
        // even batch 1, so the only question left is whether the scratch fits.
        const std::size_t expansion_bytes = detail::expanded_workspace_bytes<T>(ctx, k, A.batch_size());
        if (detail::expansion_fits(ctx, k, A.batch_size(), expansion_bytes)) {
            const int ld = detail::expanded_ld<T>(k);

            auto ws = ctx.workspace(expansion_bytes);
            BumpAllocator pool(ws.span());
            auto storage = pool.allocate<T>(ctx, static_cast<std::size_t>(ld) *
                                                     static_cast<std::size_t>(k) *
                                                     static_cast<std::size_t>(A.batch_size()));

            MatrixView<T, MatrixFormat::Dense> expanded(storage.data(), k, k, ld, ld * k, A.batch_size());

            // The GEMM cannot be pointed at the caller's A. TRMM must not read
            // the opposite triangle, nor the diagonal under Diag::Unit, so that
            // storage is not part of the operand and may hold anything at all;
            // the expansion supplies the zeros and the ones the caller was
            // entitled to leave out.
            Event expansion = detail::expand_triangular<T>(ctx, expanded, A, uplo, diag);

            // The GEMM runs on the queue's native stream, which an in-order
            // queue shares with the expansion kernel. An out-of-order queue
            // orders nothing across the SYCL/native boundary and offers no event
            // to hang the vendor launch off, so there the dependency has to be
            // waited out.
            if (!ctx.in_order()) {
                expansion.wait();
            }

            if (side == Side::Left) {
                return gemm_vendor<Back, T>(ctx, expanded, B, C, alpha, T(0),
                                            transA, Transpose::NoTrans, ComputePrecision::Default);
            }
            return gemm_vendor<Back, T>(ctx, B, expanded, C, alpha, T(0),
                                        Transpose::NoTrans, transA, ComputePrecision::Default);
        }

        // Slower, but it needs no scratch, so it is what an expansion too large
        // for the device falls back to.
        auto launch_single = [&](const MatrixView<T, MatrixFormat::Dense>& A_i,
                                 const MatrixView<T, MatrixFormat::Dense>& B_i,
                                 const MatrixView<T, MatrixFormat::Dense>& C_i) {
            call_backend<T, BackendLibrary::CUBLAS, Back>(cublasStrmm, cublasDtrmm, cublasCtrmm, cublasZtrmm,
                handle, side, uplo, transA, diag, m, n, &alpha,
                A_i.data_ptr(), A_i.ld(), B_i.data_ptr(), B_i.ld(), C_i.data_ptr(), C_i.ld());
        };

        if (A.batch_size() <= 1) {
            launch_single(A, B, C);
        } else {
            for (int batch = 0; batch < A.batch_size(); ++batch) {
                launch_single(A[batch], B[batch], C[batch]);
            }
        }

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
                    throw std::runtime_error("BATCHLAS_TRMM_VARIANT=cublasdx only supports float");
                }
            }
            if constexpr (std::is_same_v<T, float>) {
                if (trmm_use_cuda_custom(ctx, A, B, C, side, uplo, transA, diag)) {
                    return trmm_cuda_custom(ctx, A, B, C, alpha, side, uplo, transA, diag);
                }
                // GATE DECLINED. Record here, because *_cuda_custom -- which
                // carries every other route row -- is never entered, and this
                // is the half a route diff needs most: a shape moving OFF a
                // native kernel onto the vendor shows up only on this side.
                // kNativeUnknown, not false: the gate conflates "nothing
                // native serves this shape" with "something does but the
                // vendor was preferred", and the call site cannot tell them
                // apart. See level3_coverage.hh.
                detail::record_level3_route(
                    dispatch::Op::trmm,
                    dispatch::Route{dispatch::Origin::Vendor, dispatch::Algorithm::Auto},
                    C.rows(), C.cols(), A.rows(), A.batch_size(),
                    detail::kNativeUnknown, {uplo, side, diag, transA});
            } else {
                // The tile kernel is type-generic; only its routing was ever
                // float. The alternative for double and complex is the same
                // expansion-plus-GEMM as for float, which is strictly more work
                // than the GEMM it wraps, so there is nothing to weigh here.
                if (detail::is_gpu_queue(ctx) && !trmm_route_prefers_vendor() &&
                    detail::trmm_tiles_supported(A, B, C, side)) {
                    return detail::trmm_triangular_tiles(ctx, A, B, C, alpha, uplo, transA, diag);
                }
            }
        }

        return trmm_vendor_impl<Back, T>(ctx, A, B, C, alpha, side, uplo, transA, diag);
    }

    Event gemm_vendor_cuda_raw(Queue& ctx,
                               const MatrixView<float, MatrixFormat::Dense>& A,
                               const MatrixView<float, MatrixFormat::Dense>& B,
                               const MatrixView<float, MatrixFormat::Dense>& C,
                               float alpha,
                               float beta,
                               Transpose transA,
                               Transpose transB,
                               ComputePrecision precision) {
        return gemm_vendor_impl<Backend::CUDA, float>(ctx, A, B, C, alpha, beta, transA, transB, precision);
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
            const int lda = A.ld();
            const int ldb = B.ld();
            const int strideA = A.stride();
            const int strideB = B.stride();
            const int work_dim = (side == Side::Left) ? nrhs : m;

            ctx->parallel_for(sycl::range<2>(static_cast<size_t>(batch_size), static_cast<size_t>(work_dim)),
                              [=](sycl::id<2> tid) {
                                  const int b = static_cast<int>(tid[0]);
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
        return op_external("cusolver.ormqr_vendor", [&] {
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
                    ormqr_vendor<B>(ctx, A.batch_item(i), C.batch_item(i), side, trans, tau.subspan(i * k, k), sub_ws);
                }
            }
            return ctx.create_event_after_external_work();
        });
    }

    template <Backend B, typename T>
    size_t ormqr_vendor_buffer_size(Queue& ctx,
                             const MatrixView<T, MatrixFormat::Dense>& A,
                             const MatrixView<T, MatrixFormat::Dense>& C,
                             Side side,
                             Transpose trans,
                             Span<T> tau) {
        return op_external("cusolver.ormqr_vendor_buffer_size", [&] {
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
        });
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
                orgqr_vendor<B>(sub_queue, A.batch_item(i), tau.subspan(i * k, k), sub_ws);
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
            if (batch_size <= 1) {
                auto info = pool.allocate<int>(ctx, 1);
                cusolverDnParams_t params;
                cusolverDnCreateParams(&params);
                cusolverDnXgetrs(handle, params, enum_convert<BackendLibrary::CUBLAS>(transA), n, nrhs,
                    BackendScalar<T,BackendLibrary::CUBLAS>::type, A.data_ptr(), A.ld(),
                    pivots.data(),
                    BackendScalar<T,BackendLibrary::CUBLAS>::type, B.data_ptr(), B.ld(),
                    info.data());
            } else {
                int info;
                auto reinterpreted_pivots = pivots .as_span<int>();
                call_backend<T, BackendLibrary::CUBLAS, Back>(cublasSgetrsBatched, cublasDgetrsBatched, cublasCgetrsBatched, cublasZgetrsBatched,
                    handle, enum_convert<BackendLibrary::CUBLAS>(transA), n, nrhs,
                    A.data_ptrs(ctx).data(), A.ld(), reinterpreted_pivots.data(),
                    B.data_ptrs(ctx).data(), B.ld(), &info, batch_size);
            }
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
            // cuBLAS's infoArray is a genuine per-item device array with LAPACK
            // semantics (0 = ok, >0 = the column where U went exactly singular).
            // It used to be pool scratch that nothing ever read.
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
            // Same story as getrf_vendor: cuBLAS writes a per-item infoArray
            // (>0 = U(i,i) is exactly zero, so this item has no inverse) and the
            // result was thrown away, leaving the caller a matrix of infinities.
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

    #define GEMM_INSTANTIATE(fp)                BATCHLAS_INSTANTIATE(sig::gemm_vendor<fp>, backend::gemm_vendor, B_, fp)
    #define GEMV_INSTANTIATE(fp)                BATCHLAS_INSTANTIATE(sig::gemv_vendor<fp>, backend::gemv_vendor, B_, fp)
    #define TRSM_INSTANTIATE(fp)                BATCHLAS_INSTANTIATE(sig::trsm_vendor<fp>, backend::trsm_vendor, B_, fp)
    #define TRMM_INSTANTIATE(fp)                BATCHLAS_INSTANTIATE(sig::trmm_vendor<fp>, backend::trmm_vendor, B_, fp)
    #define SYMM_INSTANTIATE(fp)                BATCHLAS_INSTANTIATE(sig::symm_vendor<fp>, backend::symm_vendor, B_, fp)
    #define HEMM_INSTANTIATE(fp)                BATCHLAS_INSTANTIATE(sig::hemm_vendor<fp>, backend::hemm_vendor, B_, fp)
    #define SYRK_INSTANTIATE(fp)                BATCHLAS_INSTANTIATE(sig::syrk_vendor<fp>, backend::syrk_vendor, B_, fp)
    #define HERK_INSTANTIATE(fp)                BATCHLAS_INSTANTIATE(sig::herk_vendor<fp>, backend::herk_vendor, B_, fp)
    #define HER2K_INSTANTIATE(fp)               BATCHLAS_INSTANTIATE(sig::her2k_vendor<fp>, backend::her2k_vendor, B_, fp)
    #define SYR2K_INSTANTIATE(fp)               BATCHLAS_INSTANTIATE(sig::syr2k_vendor<fp>, backend::syr2k_vendor, B_, fp)
    #define GEQRF_INSTANTIATE(fp)               BATCHLAS_INSTANTIATE(sig::geqrf_vendor<fp>, backend::geqrf_vendor, B_, fp)
    #define GEQRF_BUFFER_SIZE_INSTANTIATE(fp)   BATCHLAS_INSTANTIATE(sig::geqrf_vendor_buffer_size<fp>, backend::geqrf_vendor_buffer_size, B_, fp)
    #define GETRS_INSTANTIATE(fp)               BATCHLAS_INSTANTIATE(sig::getrs_vendor<fp>, backend::getrs_vendor, B_, fp)
    #define GETRS_BUFFER_SIZE_INSTANTIATE(fp)   BATCHLAS_INSTANTIATE(sig::getrs_vendor_buffer_size<fp>, backend::getrs_vendor_buffer_size, B_, fp)
    #define GETRF_INSTANTIATE(fp)               BATCHLAS_INSTANTIATE(sig::getrf_vendor<fp>, backend::getrf_vendor, B_, fp)
    #define GETRF_BUFFER_SIZE_INSTANTIATE(fp)   BATCHLAS_INSTANTIATE(sig::getrf_vendor_buffer_size<fp>, backend::getrf_vendor_buffer_size, B_, fp)
    #define GETRI_INSTANTIATE(fp)               BATCHLAS_INSTANTIATE(sig::getri_vendor<fp>, backend::getri_vendor, B_, fp)
    #define GETRI_BUFFER_SIZE_INSTANTIATE(fp)   BATCHLAS_INSTANTIATE(sig::getri_vendor_buffer_size<fp>, backend::getri_vendor_buffer_size, B_, fp)
    #define ORMQR_VENDOR_INSTANTIATE(fp)        BATCHLAS_INSTANTIATE(sig::ormqr_vendor<fp>, backend::ormqr_vendor, B_, fp)
    #define ORMQR_VENDOR_BUFFER_SIZE_INSTANTIATE(fp) BATCHLAS_INSTANTIATE(sig::ormqr_vendor_buffer_size<fp>, backend::ormqr_vendor_buffer_size, B_, fp)
    #define ORGQR_INSTANTIATE(fp)               BATCHLAS_INSTANTIATE(sig::orgqr_vendor<fp>, backend::orgqr_vendor, B_, fp)
    #define ORGQR_BUFFER_SIZE_INSTANTIATE(fp)   BATCHLAS_INSTANTIATE(sig::orgqr_vendor_buffer_size<fp>, backend::orgqr_vendor_buffer_size, B_, fp)

    #define BLAS_LEVEL3_INSTANTIATE(fp)\
        GEMM_INSTANTIATE(fp)\
        GEMV_INSTANTIATE(fp)\
        TRSM_INSTANTIATE(fp)\
        GEQRF_INSTANTIATE(fp)\
        GEQRF_BUFFER_SIZE_INSTANTIATE(fp)\
        GETRS_INSTANTIATE(fp)\
        GETRS_BUFFER_SIZE_INSTANTIATE(fp)\
        GETRF_INSTANTIATE(fp)\
        GETRF_BUFFER_SIZE_INSTANTIATE(fp)\
        GETRI_INSTANTIATE(fp)\
        GETRI_BUFFER_SIZE_INSTANTIATE(fp)\
        ORMQR_VENDOR_INSTANTIATE(fp)\
        ORMQR_VENDOR_BUFFER_SIZE_INSTANTIATE(fp)\
        ORGQR_INSTANTIATE(fp)\
        ORGQR_BUFFER_SIZE_INSTANTIATE(fp)


    BLAS_LEVEL3_INSTANTIATE(float)
    BLAS_LEVEL3_INSTANTIATE(double)
    BLAS_LEVEL3_INSTANTIATE(std::complex<float>)
    BLAS_LEVEL3_INSTANTIATE(std::complex<double>)
    TRMM_INSTANTIATE(float)
    TRMM_INSTANTIATE(double)
    TRMM_INSTANTIATE(std::complex<float>)
    TRMM_INSTANTIATE(std::complex<double>)
    SYMM_INSTANTIATE(float)
    SYMM_INSTANTIATE(double)
    HEMM_INSTANTIATE(std::complex<float>)
    HEMM_INSTANTIATE(std::complex<double>)
    HERK_INSTANTIATE(std::complex<float>)
    HERK_INSTANTIATE(std::complex<double>)
    HER2K_INSTANTIATE(std::complex<float>)
    HER2K_INSTANTIATE(std::complex<double>)
    SYRK_INSTANTIATE(float)
    SYRK_INSTANTIATE(double)
    SYR2K_INSTANTIATE(float)
    SYR2K_INSTANTIATE(double)

    #undef GEMM_INSTANTIATE
    #undef GEMV_INSTANTIATE
    #undef SYMM_INSTANTIATE
    #undef HEMM_INSTANTIATE
    #undef HERK_INSTANTIATE
    #undef HER2K_INSTANTIATE
    #undef SYRK_INSTANTIATE
    #undef SYR2K_INSTANTIATE
    #undef TRSM_INSTANTIATE
    #undef TRMM_INSTANTIATE
    #undef GEQRF_INSTANTIATE
    #undef GEQRF_BUFFER_SIZE_INSTANTIATE
    #undef GETRS_INSTANTIATE
    #undef GETRS_BUFFER_SIZE_INSTANTIATE
    #undef GETRF_INSTANTIATE
    #undef GETRF_BUFFER_SIZE_INSTANTIATE
    #undef GETRI_INSTANTIATE
    #undef GETRI_BUFFER_SIZE_INSTANTIATE
    #undef ORMQR_VENDOR_INSTANTIATE
    #undef ORMQR_VENDOR_BUFFER_SIZE_INSTANTIATE
    #undef ORGQR_INSTANTIATE
    #undef ORGQR_BUFFER_SIZE_INSTANTIATE
    #undef B_
    #undef BLAS_LEVEL3_INSTANTIATE
}
