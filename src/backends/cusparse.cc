// filepath: /home/jonaslacour/BatchLAS/src/backends/cusparse_matrixview.cc
#include "../linalg-impl.hh"
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/mempool.hh>
#include "../queue.hh"
#include <sycl/sycl.hpp>
#include <batchlas/blas/linalg.hh>
#include "backend_handle_impl.hh"
#include <algorithm>
#include <complex>
#include <cstdint>
#include <ios>
#include <type_traits>
#include <vector>

// This file contains cuSPARSE primitives implementation using MatrixView
namespace batchlas {

    namespace backend {

        namespace {
            // ConjTrans on a REAL scalar is the SAME operation as Trans: conjugation
            // of a real number is the identity, so `op(A) = conj(A)^T = A^T`. This is
            // the correct spelling of the operation, not a capability workaround for
            // a cuSPARSE limitation -- the two enum values denote one operation here,
            // and CUSPARSE_OPERATION_TRANSPOSE is the one cuSPARSE actually honours
            // for the real data types. Passing CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE
            // with CUDA_R_32F / CUDA_R_64F silently produced wrong results (the whole
            // ConjTrans family of spmm_tests, float and double only -- the complex
            // arms, where CONJUGATE_TRANSPOSE is the distinct and correct operation,
            // always passed).
            //
            // Applied to transA AND transB: cusparseSpMM takes an op for the sparse
            // operand and one for the dense operand, and the dense one was wrong on
            // its own (NineNoTransConjTrans has transA = NoTrans).
            template <typename T>
            constexpr cusparseOperation_t cusparse_op(Transpose trans) {
                constexpr bool is_complex_scalar =
                    std::is_same_v<T, std::complex<float>> ||
                    std::is_same_v<T, std::complex<double>>;
                if constexpr (!is_complex_scalar) {
                    if (trans == Transpose::ConjTrans) {
                        return static_cast<cusparseOperation_t>(CUSPARSE_OPERATION_TRANSPOSE);
                    }
                }
                return enum_convert<BackendLibrary::CUSPARSE>(trans);
            }

            // ===============================================================
            // THE STRIDED-BATCH nnz CONTRACT.
            //
            // cusparseCreateCsr takes ONE nnz, and cusparseCsrSetStridedBatch
            // adds only a batch count and two strides -- there is no per-item
            // nnz anywhere in the descriptor. cuSPARSE's CSR contract is that
            // nnz == rowOffsets[rows], so a strided batch can only describe a
            // batch whose items ALL store the same number of nonzeros.
            //
            // backend_handle_impl.hh:63 hands it A.nnz(), which is the per-item
            // CAPACITY (matrix.hh:1069-1073): convert_to<MatrixFormat::CSR>
            // sizes every item by the batch MAXIMUM (src/matrix.cc:473-478) and
            // zeroes only row_offsets (:489), so for every item that stores
            // fewer nonzeros than the maximum the descriptor claims values and
            // column indices that the conversion never wrote. cuSPARSE reads
            // them. The failure was MEASURED, not deduced:
            //   * spmm_tests HeterogeneousNnzAcrossBatch -- the over-read slots
            //     belong to the item's LAST row, and it is exactly the last row
            //     of exactly the short items that came back wrong;
            //   * spmm_tests PaddingAboveNnzIsNotRead, whose padding carries an
            //     out-of-range column index -- CUDA_ERROR_ILLEGAL_ADDRESS and a
            //     dead process.
            // A homogeneous batch has never been affected, which is why every
            // in-tree caller (lanczos, LOBPCG, the benchmark) was unharmed and
            // why this survived until a suite covered the axis.
            //
            // THE FIX IS TO STOP LYING TO THE DESCRIPTOR, in the only two ways
            // the API allows:
            //   * uniform batch -- one batched call, with nnz taken from the
            //     items' own row offsets instead of from the capacity;
            //   * non-uniform batch -- ONE cusparseSpMM PER ITEM, each with its
            //     own single-item descriptor and its own nnz. Serialising the
            //     batch is a real cost and batching is the whole reason the
            //     vendor path exists, but it is the only shape cuSPARSE offers,
            //     and correct-and-serial beats fast-and-wrong. The native route
            //     (src/sycl/spmm_native.cc) bounds its loop by each item's own
            //     row offsets inside the kernel and needs none of this.
            //
            // The fast path is UNCHANGED from what shipped before: when every
            // item's nnz already equals A.nnz() we use the cached descriptors
            // built by backend_handle_impl.hh and issue exactly one call.
            // ===============================================================
            struct SpmmCsrBatchPlan {
                std::vector<int> item_nnz;      // what each item actually stores
                bool uniform = true;            // ... and they are all equal
                bool matches_capacity = true;   // ... and equal to A.nnz()
            };

            // Reads the row offsets ON THE HOST. Two ints per batch item, and no
            // queue synchronisation on the common path -- this is the same
            // precondition MatrixView::nnz(b) already documents
            // (matrix.hh:1078-1090): whatever kernel FILLED the offsets must
            // have completed. A CSR structure is built once and then reused
            // across many spmm calls, so that is the existing caller contract
            // for a CSR view and not a new requirement invented here.
            //
            // A view over sycl::malloc_device memory is NOT host-reachable
            // (matrix.hh:1081-1083). That case is detected rather than assumed,
            // and pays one staging copy of the offsets array plus a wait. No
            // caller in this tree takes that path: every CSR matrix here is
            // backed by UnifiedVector, which is USM shared.
            template <typename T>
            SpmmCsrBatchPlan spmm_csr_batch_plan(Queue& ctx,
                                                 const MatrixView<T, MatrixFormat::CSR>& A) {
                SpmmCsrBatchPlan plan;
                const int m = A.rows();
                const int bs = A.batch_size();
                const int os = A.offset_stride();
                // A shape this degenerate is not one this file can plan for;
                // leaving matches_capacity true hands it to the pre-existing
                // single batched call, which is exactly today's behaviour.
                if (bs <= 0 || m < 0 || os < m + 1) return plan;
                const int* ro = A.row_offsets().data();
                if (ro == nullptr) return plan;

                std::vector<int> staged;
                const int* host_ro = ro;
                if (sycl::get_pointer_type(ro, ctx->get_context()) ==
                    sycl::usm::alloc::device) {
                    staged.resize(static_cast<std::size_t>(os) *
                                  static_cast<std::size_t>(bs));
                    ctx->memcpy(staged.data(), ro, staged.size() * sizeof(int)).wait();
                    host_ro = staged.data();
                }

                plan.item_nnz.resize(static_cast<std::size_t>(bs));
                for (int b = 0; b < bs; ++b) {
                    const std::size_t base = static_cast<std::size_t>(b) *
                                             static_cast<std::size_t>(os);
                    plan.item_nnz[static_cast<std::size_t>(b)] =
                        host_ro[base + static_cast<std::size_t>(m)] - host_ro[base];
                }
                plan.uniform = std::all_of(
                    plan.item_nnz.begin(), plan.item_nnz.end(),
                    [&](int v) { return v == plan.item_nnz.front(); });
                plan.matches_capacity = plan.uniform && plan.item_nnz.front() == A.nnz();
                return plan;
            }

            // Descriptors built and destroyed HERE rather than borrowed from the
            // view's cached BackendMatrixHandle. That handle's destructor is
            // `= default` (backend_handle_impl.hh:24) -- it never calls
            // cusparseDestroySpMat -- so borrowing it per batch item would leak
            // one descriptor per item per call. These own what they create.
            template <typename T>
            struct LocalCsrDescr {
                cusparseSpMatDescr_t d = nullptr;
                LocalCsrDescr(const MatrixView<T, MatrixFormat::CSR>& A,
                              int first, int count, int nnz) {
                    const std::int64_t o =
                        static_cast<std::int64_t>(first) * A.offset_stride();
                    const std::int64_t v =
                        static_cast<std::int64_t>(first) * A.matrix_stride();
                    cusparseCreateCsr(&d, A.rows(), A.cols(), nnz,
                                      A.row_offsets().data() + o,
                                      A.col_indices().data() + v,
                                      A.data_ptr() + v,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO,
                                      BackendScalar<T, BackendLibrary::CUSPARSE>::type);
                    if (count > 1) {
                        cusparseCsrSetStridedBatch(d, count, A.offset_stride(),
                                                   A.matrix_stride());
                    }
                }
                ~LocalCsrDescr() { if (d) cusparseDestroySpMat(d); }
                LocalCsrDescr(const LocalCsrDescr&) = delete;
                LocalCsrDescr& operator=(const LocalCsrDescr&) = delete;
                operator cusparseSpMatDescr_t() const { return d; }
            };

            template <typename T>
            struct LocalDnDescr {
                cusparseDnMatDescr_t d = nullptr;
                LocalDnDescr(const MatrixView<T, MatrixFormat::Dense>& M,
                             int first, int count) {
                    const std::int64_t s = static_cast<std::int64_t>(first) * M.stride();
                    cusparseCreateDnMat(&d, M.rows(), M.cols(), M.ld(),
                                        M.data_ptr() + s,
                                        BackendScalar<T, BackendLibrary::CUSPARSE>::type,
                                        CUSPARSE_ORDER_COL);
                    if (count > 1) {
                        cusparseDnMatSetStridedBatch(d, count, M.stride());
                    }
                }
                ~LocalDnDescr() { if (d) cusparseDestroyDnMat(d); }
                LocalDnDescr(const LocalDnDescr&) = delete;
                LocalDnDescr& operator=(const LocalDnDescr&) = delete;
                operator cusparseDnMatDescr_t() const { return d; }
            };

            // The raw cuSPARSE query, for whichever descriptor triple the plan
            // selects. Every sizing path in this file goes through here, so a
            // size and the call it sizes cannot describe different descriptors.
            template <Backend B, typename T>
            size_t spmm_query_buffer(LinalgHandle<B>& handle,
                                     cusparseSpMatDescr_t a,
                                     cusparseDnMatDescr_t b,
                                     cusparseDnMatDescr_t c,
                                     T alpha, T beta,
                                     Transpose transA, Transpose transB) {
                size_t size = 0;
                cusparseSpMM_bufferSize(
                    handle,
                    cusparse_op<T>(transA),
                    cusparse_op<T>(transB),
                    &alpha, a, b, &beta, c,
                    BackendScalar<T, BackendLibrary::CUSPARSE>::type,
                    CUSPARSE_SPMM_ALG_DEFAULT,
                    &size);
                return size;
            }

            // The buffer the PLANNED shape needs. For the per-item path that is
            // the MAXIMUM over items: one buffer is allocated and reused by every
            // item, and no per-item query can exceed a max taken over exactly
            // those queries.
            template <Backend B, typename T, MatrixFormat MFormat>
            size_t spmm_planned_buffer_size(LinalgHandle<B>& handle,
                                            const MatrixView<T, MFormat>& A,
                                            const MatrixView<T, MatrixFormat::Dense>& B_mat,
                                            const MatrixView<T, MatrixFormat::Dense>& C,
                                            T alpha, T beta,
                                            Transpose transA, Transpose transB,
                                            const SpmmCsrBatchPlan& plan) {
                if constexpr (MFormat == MatrixFormat::CSR) {
                    if (!plan.matches_capacity) {
                        const int bs = A.batch_size();
                        if (plan.uniform) {
                            LocalCsrDescr<T> a(A, 0, bs, plan.item_nnz.front());
                            LocalDnDescr<T> b(B_mat, 0, bs);
                            LocalDnDescr<T> c(C, 0, bs);
                            return spmm_query_buffer<B, T>(handle, a, b, c, alpha, beta,
                                                           transA, transB);
                        }
                        size_t need = 0;
                        for (int i = 0; i < bs; ++i) {
                            LocalCsrDescr<T> a(A, i, 1,
                                               plan.item_nnz[static_cast<std::size_t>(i)]);
                            LocalDnDescr<T> b(B_mat, i, 1);
                            LocalDnDescr<T> c(C, i, 1);
                            need = std::max(need,
                                            spmm_query_buffer<B, T>(handle, a, b, c, alpha,
                                                                    beta, transA, transB));
                        }
                        return need;
                    }
                }
                (void)plan;
                return spmm_query_buffer<B, T>(handle, *A, *B_mat, *C, alpha, beta,
                                               transA, transB);
            }
        }  // namespace

        template <Backend B, typename T, MatrixFormat MFormat>
        size_t spmm_vendor_buffer_size(Queue& ctx,
                                       const MatrixView<T, MFormat>& A,
                                       const MatrixView<T, MatrixFormat::Dense>& B_mat,
                                       const MatrixView<T, MatrixFormat::Dense>& C,
                                       T alpha,
                                       T beta,
                                       Transpose transA,
                                       Transpose transB);

    template <Backend B, typename T, MatrixFormat MFormat>
    Event spmm_vendor(Queue& ctx,
               const MatrixView<T, MFormat>& A,
               const MatrixView<T, MatrixFormat::Dense>& B_mat,
               const MatrixView<T, MatrixFormat::Dense>& C,
               T alpha,
               T beta,
               Transpose transA,
               Transpose transB,
               Span<std::byte> workspace) {
        // Call cuSPARSE
        static LinalgHandle<B> handle;
        handle.setStream(ctx);

        SpmmCsrBatchPlan plan;
        if constexpr (MFormat == MatrixFormat::CSR) {
            plan = spmm_csr_batch_plan(ctx, A);
        }

        BumpAllocator pool(workspace);
        // ONE plan sizes the buffer and issues the call, so the query and the
        // call cannot describe different descriptors. spmm_vendor_buffer_size
        // below builds an identical plan from the same views, which is what
        // keeps the CALLER's workspace big enough for whichever shape fires.
        auto buffer_size = BumpAllocator::allocation_size<std::byte>(
            ctx, spmm_planned_buffer_size<B, T, MFormat>(
                     handle, A, B_mat, C, alpha, beta, transA, transB, plan));
        auto buffer = pool.allocate<std::byte>(ctx, buffer_size);

        if constexpr (MFormat == MatrixFormat::CSR) {
            if (!plan.matches_capacity) {
                const int bs = A.batch_size();
                if (plan.uniform) {
                    // One batched call, with the nnz the items actually store
                    // rather than the capacity the cached descriptor carries.
                    LocalCsrDescr<T> a(A, 0, bs, plan.item_nnz.front());
                    LocalDnDescr<T> b(B_mat, 0, bs);
                    LocalDnDescr<T> c(C, 0, bs);
                    cusparseSpMM(handle, cusparse_op<T>(transA), cusparse_op<T>(transB),
                                 &alpha, a, b, &beta, c,
                                 BackendScalar<T, BackendLibrary::CUSPARSE>::type,
                                 CUSPARSE_SPMM_ALG_DEFAULT, buffer.data());
                } else {
                    // ONE CALL PER ITEM. This batch is genuinely inexpressible
                    // as a single cuSPARSE descriptor; see the note above.
                    for (int i = 0; i < bs; ++i) {
                        LocalCsrDescr<T> a(A, i, 1,
                                           plan.item_nnz[static_cast<std::size_t>(i)]);
                        LocalDnDescr<T> b(B_mat, i, 1);
                        LocalDnDescr<T> c(C, i, 1);
                        cusparseSpMM(handle, cusparse_op<T>(transA),
                                     cusparse_op<T>(transB),
                                     &alpha, a, b, &beta, c,
                                     BackendScalar<T, BackendLibrary::CUSPARSE>::type,
                                     CUSPARSE_SPMM_ALG_DEFAULT, buffer.data());
                    }
                }
                return ctx.create_event_after_external_work();
            }
        }

        cusparseSpMM(
            handle,
            cusparse_op<T>(transA),
            cusparse_op<T>(transB),
            &alpha,
            *A,
            *B_mat,
            &beta,
            *C,
            BackendScalar<T,BackendLibrary::CUSPARSE>::type,
            CUSPARSE_SPMM_ALG_DEFAULT,
            buffer.data()
        );
        return ctx.create_event_after_external_work();
    }

    template <Backend B, typename T, MatrixFormat MFormat>
    size_t spmm_vendor_buffer_size(Queue& ctx,
                          const MatrixView<T, MFormat>& A,
                          const MatrixView<T, MatrixFormat::Dense>& B_mat,
                          const MatrixView<T, MatrixFormat::Dense>& C,
                          T alpha,
                          T beta,
                          Transpose transA,
                          Transpose transB) {
        // Call cuSPARSE
        static LinalgHandle<B> handle;
        handle.setStream(ctx);

        SpmmCsrBatchPlan plan;
        if constexpr (MFormat == MatrixFormat::CSR) {
            plan = spmm_csr_batch_plan(ctx, A);
        }
        const size_t size = spmm_planned_buffer_size<B, T, MFormat>(
            handle, A, B_mat, C, alpha, beta, transA, transB, plan);
        return BumpAllocator::allocation_size<std::byte>(ctx, size);
    }

    } // namespace backend

    #define SPMM_INSTANTIATE(fp, F) \
    template Event backend::spmm_vendor<Backend::CUDA, fp, F>( \
        Queue&, \
        const MatrixView<fp, F>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        fp, fp, Transpose, Transpose, Span<std::byte>);
    
    #define SPMM_BUFFER_SIZE_INSTANTIATE(fp, F) \
    template size_t backend::spmm_vendor_buffer_size<Backend::CUDA, fp, F>( \
        Queue&, \
        const MatrixView<fp, F>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        fp, fp, Transpose, Transpose);

    #define CUSPARSE_INSTANTIATE(fp, F) \
        SPMM_INSTANTIATE(fp, F) \
        SPMM_BUFFER_SIZE_INSTANTIATE(fp, F)

    // Instantiate for all supported sparse formats
    #define CUSPARSE_INSTANTIATE_FOR_FP(fp) \
        CUSPARSE_INSTANTIATE(fp, MatrixFormat::CSR)

    // Instantiate for the floating-point types of interest
    CUSPARSE_INSTANTIATE_FOR_FP(float)
    CUSPARSE_INSTANTIATE_FOR_FP(double)
    CUSPARSE_INSTANTIATE_FOR_FP(std::complex<float>)
    CUSPARSE_INSTANTIATE_FOR_FP(std::complex<double>)

    #undef SPMM_INSTANTIATE
    #undef SPMM_BUFFER_SIZE_INSTANTIATE
    #undef CUSPARSE_INSTANTIATE
    #undef CUSPARSE_INSTANTIATE_FOR_FP
}
