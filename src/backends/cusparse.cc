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

// cuSPARSE spmm over MatrixView. Route evidence, and the vendor defects fixed
// here: docs/perf/spmm.md
namespace batchlas {

    namespace backend {

        namespace {
            // On a REAL scalar ConjTrans *is* Trans, and cuSPARSE silently returns
            // wrong results for CUDA_R_32F/CUDA_R_64F under the conjugating enum.
            // Applies to the dense operand too, not just the sparse one.
            // evidence: docs/perf/spmm.md#three-vendor-defects-found-here-and-fixed
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

            // The strided-batch nnz contract: cusparseCreateCsr takes ONE nnz and
            // cusparseCsrSetStridedBatch adds no per-item one, but A.nnz() is the
            // per-item CAPACITY, sized by the batch maximum. Trusting it makes a
            // short item's descriptor cover padding the conversion never wrote:
            // wrong last rows, or CUDA_ERROR_ILLEGAL_ADDRESS. The plan therefore
            // takes nnz from the items' own row offsets, and a non-uniform batch is
            // issued as one cusparseSpMM per item -- the only shape the API offers.
            // evidence: docs/perf/spmm.md#three-vendor-defects-found-here-and-fixed
            struct SpmmCsrBatchPlan {
                std::vector<int> item_nnz;      // what each item actually stores
                bool uniform = true;            // ... and they are all equal
                bool matches_capacity = true;   // ... and equal to A.nnz()
            };

            // Reads the row offsets ON THE HOST: whatever kernel filled them must
            // have completed, the same precondition MatrixView::nnz(b) documents.
            // Device-only memory is not host-reachable, so that case is detected and
            // staged rather than assumed; no caller in this tree takes it.
            template <typename T>
            SpmmCsrBatchPlan spmm_csr_batch_plan(Queue& ctx,
                                                 const MatrixView<T, MatrixFormat::CSR>& A) {
                SpmmCsrBatchPlan plan;
                const int m = A.rows();
                const int bs = A.batch_size();
                const int os = A.offset_stride();
                // Degenerate shape: leaving matches_capacity true hands it to the
                // single batched call, which is the pre-existing behaviour.
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

            // Owns its descriptors: BackendMatrixHandle's destructor is `= default`
            // and never calls cusparseDestroySpMat, so borrowing the view's cached
            // descriptor per batch item would leak one descriptor per item per call.
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

            // Every sizing path funnels through here, so a size and the call it
            // sizes cannot describe different descriptors.
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

            // For the per-item path this is the MAXIMUM over items: one buffer is
            // allocated once and reused by every item.
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
        static LinalgHandle<B> handle;
        handle.setStream(ctx);

        SpmmCsrBatchPlan plan;
        if constexpr (MFormat == MatrixFormat::CSR) {
            plan = spmm_csr_batch_plan(ctx, A);
        }

        BumpAllocator pool(workspace);
        // spmm_vendor_buffer_size below must build an identical plan from the same
        // views, or the caller's workspace can be too small for the shape that fires.
        auto buffer_size = BumpAllocator::allocation_size<std::byte>(
            ctx, spmm_planned_buffer_size<B, T, MFormat>(
                     handle, A, B_mat, C, alpha, beta, transA, transB, plan));
        auto buffer = pool.allocate<std::byte>(ctx, buffer_size);

        if constexpr (MFormat == MatrixFormat::CSR) {
            if (!plan.matches_capacity) {
                const int bs = A.batch_size();
                if (plan.uniform) {
                    LocalCsrDescr<T> a(A, 0, bs, plan.item_nnz.front());
                    LocalDnDescr<T> b(B_mat, 0, bs);
                    LocalDnDescr<T> c(C, 0, bs);
                    cusparseSpMM(handle, cusparse_op<T>(transA), cusparse_op<T>(transB),
                                 &alpha, a, b, &beta, c,
                                 BackendScalar<T, BackendLibrary::CUSPARSE>::type,
                                 CUSPARSE_SPMM_ALG_DEFAULT, buffer.data());
                } else {
                    // One call per item: this batch is inexpressible as a single
                    // cuSPARSE descriptor.
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

    #define CUSPARSE_INSTANTIATE_FOR_FP(fp) \
        CUSPARSE_INSTANTIATE(fp, MatrixFormat::CSR)

    CUSPARSE_INSTANTIATE_FOR_FP(float)
    CUSPARSE_INSTANTIATE_FOR_FP(double)
    CUSPARSE_INSTANTIATE_FOR_FP(std::complex<float>)
    CUSPARSE_INSTANTIATE_FOR_FP(std::complex<double>)

    #undef SPMM_INSTANTIATE
    #undef SPMM_BUFFER_SIZE_INSTANTIATE
    #undef CUSPARSE_INSTANTIATE
    #undef CUSPARSE_INSTANTIATE_FOR_FP
}
