#pragma once

// Native batched CSR SpMM -- declarations.
//
// WP8. C = alpha * op(A) * op(B) + beta * C with A sparse (batched CSR, one
// strided slab per item) and B, C dense column-major. Before this, spmm had
// exactly ONE arm on a GPU -- cusparseSpMM -- and one host arm that refuses
// every transpose (src/backends/netlib_lapack.cc:249). A vendor-free build has
// the public entry point but no route, so every caller gets NoRouteError.
//
// ===========================================================================
// FOUR INVARIANTS. Each of them is a risk class DELETED, not managed, and each
// is checkable by reading the translation unit.
//
// (1) ZERO LOCAL MEMORY. No sycl::local_accessor is constructed in any submit
//     in spmm_native.cc, and there is no sycl::group_barrier, no
//     sycl::reduce_over_group and no sycl::joint_reduce. That is what makes the
//     recorded "48 KB launch hole" (a dynamic-local-memory request in
//     (49152-static, 49152] failing at enqueue; see gemv_native.hh:169-177 and
//     src/extensions/potrf_cta.cc:259-296) STRUCTURALLY UNREACHABLE here, so
//     this file carries no pad band, no LOCAL_MEM_SIZE query, no budget
//     arithmetic and no "enqueue nothing and return false" capacity fallback.
//     The pressure on this design will be to stage op(B) in shared memory to
//     attack the gather; that is a separate work package with its own Algorithm
//     name and its own hole band, never an amendment to this TU.
//
// (2) ZERO WORKSPACE. spmm_native_csr takes no Span<std::byte> and allocates
//     nothing. spmm_buffer_size and spmm therefore agree BY CONSTRUCTION rather
//     than by replay discipline -- which is the whole of the recorded
//     under-allocation class (ormqr sized 2560 B against a call demanding
//     276480) and of the four spmm sizing-shape mismatches around it.
//
// (3) NO is_gpu GATE, AND NO SUB-GROUP. Nothing here carries
//     [[sycl::reqd_sub_group_size]] or queries Device::supports_sub_group_size,
//     so every body runs on the native_cpu image as well as on CUDA. That is
//     not decoration: SPMM_ALL(Backend::NETLIB) IS instantiated in a vendor-free
//     build (BATCHLAS_HAS_HOST_BACKEND is 1 there while BATCHLAS_HAS_LAPACKE and
//     BATCHLAS_HAS_CBLAS are 0), so the NETLIB spmm symbol exists and throws,
//     and a GPU-gated native route would leave it throwing. This is WP7's
//     lesson restated: a GPU-only native tier moves the vendor-free burn-down
//     by exactly zero on the half of the suite that runs on a CPU queue.
//
// (4) ALL NINE (transA, transB) COMBINATIONS ARE SERVED. transB in particular
//     is NOT refused. Passing the dense block row-major as transB = Trans
//     collapses the gather from `nrhs` distinct 32 B sector touches per nonzero
//     to ceil(nrhs*sizeof(T)/32), with no kernel, no workspace and no new
//     route -- and LOBPCG, syevx_filtered and lanczos all build their own dense
//     blocks, so it is available to them as a caller change. Refusing transB in
//     supports() would foreclose that lever structurally.
//
// ===========================================================================
// THE CSR INDEXING CONTRACT. Get this wrong and everything is correct at
// batch 1 and wrong at batch 2, which is the failure this library has paid for
// before. All four parts read directly out of src/matrix.cc:
//
//   * ROW OFFSETS ARE ITEM-LOCAL AND START AT 0. The builder fills the whole
//     array with 0 (:489) and then runs a joint_inclusive_scan per item with
//     init 0 (:505-513). So row_offsets[b*offset_stride + i] is an offset
//     WITHIN item b's slab, never a global one.
//
//   * TWO DIFFERENT STRIDES. row_offsets is indexed by b*offset_stride()
//     (== rows+1), values and col_indices by b*matrix_stride(). The allocating
//     constructor sets them separately (:331-333) and the population kernel
//     writes at b*max_nnz + row_offsets[b*(rows+1)+r] + pos (:561-564).
//     cuSPARSE is handed exactly these two numbers
//     (src/backends/backend_handle_impl.hh:65-66), which is an independent
//     confirmation of the contract rather than a restatement of it.
//
//   * A.nnz() IS THE BATCH MAXIMUM CAPACITY, NOT A COUNT. convert_to<CSR> sizes
//     every item by max_element over the per-item counts (:473-479); the note
//     at include/batchlas/blas/matrix.hh:1071-1074 says so. The ONLY legal
//     bound on the nonzero loop is row_offsets[ro+i+1].
//
//   * THE SLOTS ABOVE EACH ITEM'S OWN nnz ARE UNINITIALISED. Only row_offsets
//     is filled; UnifiedVector::resize (src/util/sycl-util-impl.cc:70-83)
//     mallocs and copies the old contents and never fills the tail. An
//     in-tree comment at src/matrix.cc:487-488 claims the zero-fill "also makes
//     the value/index padding deterministic" -- it does not, it makes the
//     OFFSETS deterministic, and the padding is genuinely garbage.
//
// ===========================================================================
// THREE KERNEL BODIES, ONE ROUTE. {Native, Direct} names all three and the
// launcher picks on transA -- gemv's own precedent, where {Native, Direct}
// already names GemvDirectNKernel and GemvDirectTKernel and the pick lives in
// the launcher at src/sycl/gemv_native.cc:1249-1289. Body selection is a
// DECOMPOSITION, not an algorithm, so route.hh's Algorithm enum,
// to_string(Algorithm) and route_env.hh's parse_algorithm_word need zero edits.
//
//   BODY 1  SpmmGatherKernel<T,NC>   transA == NoTrans, any transB.
//           One work-item per (batch item, row of A, block of NC output
//           columns). Each item walks its own CSR row and GATHERS op(B), so
//           every write is exclusively owned: no atomic, no collective.
//
//   BODY 0  SpmmScaleKernel<T>       transA != NoTrans, launched FIRST.
//           C = beta*C (or C = 0 at beta == 0) over the whole output. A scatter
//           cannot fold beta into its accumulation -- no work-item owns an
//           output element and there is no device-wide barrier inside a SYCL
//           kernel -- so beta*C_old must land exactly once, before any atomic.
//
//   BODY 2  SpmmScatterKernel<T>     transA != NoTrans, launched second.
//           One work-item per (batch item, row of the STORED A). A row of A is
//           a COLUMN of op(A), which is exactly what CSR hands you, so the item
//           reads one row of op(B) and SCATTERS into C with a device-scope
//           global atomic fetch_add.
//
// THE SCATTER SPENDS RUN-TO-RUN BITWISE REPRODUCIBILITY. Atomic arrival order
// varies, so summation order varies. Nothing in this tree asserts determinism
// here (convert_to<CSR>'s own column order within a row is nondeterministic,
// src/matrix.cc:552-566) and cuSPARSE guarantees none -- but any test comparing
// two runs of the transposed path for exact equality WILL be flaky, and the
// tolerance denominator must be a backward-error scale, never |expected|.
//
// THE double AND complex<double> SCATTER INSTANTIATIONS CARRY A DEVICE
// REQUIREMENT THE float ONES DO NOT: an FP64 atomic fetch_add lowers with a
// `sycl_used_aspects` of `atomic64`. Both devices on the development box have
// it (RTX 4090 and the SYCL Native CPU device both report atomic64 yes), and on
// a device without it the failure is a kernel-selection error at launch, not a
// compile error. The FP32 and FP64 atomics both lower to hardware
// (llvm.nvvm.atomic.add.global.f.f32/f64, zero cmpxchg), so the scatter's cost
// is one reduction per nonzero-column touch and not a CAS retry loop.
//
// ===========================================================================
// TWO CONTRACTS THAT ARE SILENTLY WRONG IF MIS-STATED.
//
// (1) beta == 0 MEANS C IS NOT READ. Every in-library caller passes beta = 0
//     into a BumpAllocator-allocated C, and BumpAllocator::allocate never
//     zeroes what it hands back (include/batchlas/util/mempool.hh:79-98), so a
//     kernel that computes beta*C_old unconditionally multiplies zero by
//     garbage and returns NaN. This is reference-BLAS semantics and it is the
//     one place the in-tree host arm is the OUTLIER: netlib_lapack.cc:254 does
//     `T sum = beta * C_b.at(row, col);` unconditionally. The callers escape
//     only because they all run on cuSPARSE.
//
//     The dual is equally load-bearing: alpha == 0 means A and B are not read,
//     but C = beta*C STILL HAPPENS. Do NOT copy gemv's quick return
//     (gemv_native.cc:486-489), which also fires at alpha == 0 && beta == 1 and
//     leaves y untouched -- reference ?GEMV is defined that way and reference
//     spmm is not. Copying it here would be a ROUTE-DEPENDENT wrong answer.
//
// (2) NO __restrict__ ON ANY POINTER. LOBPCG passes X, P, R as slices of ONE S
//     buffer and AX, AP, AR as slices of one AS buffer
//     (src/extensions/syevx_lobpcg.cc:331-341). They are element-disjoint but
//     they alias at the OBJECT level, and __restrict__ is a promise about the
//     object. For the same reason no body materialises a pointer array:
//     MatrixView::operator()(Slice,Slice) hands the slice the PARENT's
//     data_ptrs_ buffer (matrix.hh:1140, contradicting the comment above it),
//     so building one on LOBPCG's X would corrupt the array S shares. Every
//     operand is reached as base + b*stride, the same strided-batch contract
//     cuSPARSE gets.
//
// Conjugation is a RUNTIME bool -- one launch-uniform branch -- wrapped in an
// `if constexpr (dev_is_complex_v<D>)` so the real instantiations emit no branch
// at all. Everything arithmetic goes through the POD device scalars in
// src/sycl/device_scalar.hh, so no std::complex crosses into device code.

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

namespace batchlas::sycl_spmm {

// Is the GATHER tier (body 1, transA == NoTrans) in this build, for this scalar
// type? FALSE MEANS "NO SUCH KERNEL HERE", and RouteTable<Op::spmm,T>::
// supports() reports the native route unsupported when it is -- the
// TrsmShape::cta_max_n == 0 convention (route_trsm.hh:62-96). It is a function
// rather than a constexpr literal because the answer describes the BUILD, and
// it is answered in the same translation unit as the kernels, so a build that
// drops that TU cannot advertise a route it does not carry.
template <typename T>
bool spmm_gather_available();

// Is the SCATTER tier (bodies 0 and 2, transA != NoTrans) in this build?
// Separate from the above because they are independent capabilities: the two
// arms share no kernel, and a build that had to drop the atomic scatter for a
// target without atomic64 would still carry the gather.
//
// NEITHER FLAG SAYS ANYTHING ABOUT THE DEVICE. There is no is_gpu notion and no
// sub-group notion anywhere in this TU by construction (invariant 3); these
// answer only "was the kernel compiled".
template <typename T>
bool spmm_scatter_available();

// {Native, Direct}. Dispatches internally to body 1, or to bodies 0 + 2, on
// transA.
//
// The signature is deliberately identical to the public batchlas::spmm and to
// backend::spmm_vendor MINUS THE WORKSPACE, so the facade chooses between them
// without either side adapting. The missing parameter is invariant 2: there is
// no size to get wrong.
//
// A is the SPARSE operand and must be CSR. B_mat and C are dense column-major
// with their OWN ld and stride, both read from the view -- never derived as
// ld*cols, which is the exact bug that passed 232 gemv test cases before it was
// caught.
template <typename T>
Event spmm_native_csr(Queue& ctx,
                      const MatrixView<T, MatrixFormat::CSR>& A,
                      const MatrixView<T, MatrixFormat::Dense>& B_mat,
                      const MatrixView<T, MatrixFormat::Dense>& C,
                      T alpha,
                      T beta,
                      Transpose transA,
                      Transpose transB);

}  // namespace batchlas::sycl_spmm
