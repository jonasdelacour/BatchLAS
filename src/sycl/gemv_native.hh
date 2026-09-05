#pragma once

// Native batched GEMV -- declarations.
//
// WP7. The point of this kernel is NOT speed. The recon phase measured
// cuBLAS `gemvStridedBatched` at 94-105% of the ~950 GB/s achievable DRAM roof
// on 90 of 92 reproducing cells, across all four scalar types and both transA
// values (experiments/wp7_gemv/baseline/README.md). A batched gemv is pure
// streaming: it reads A once and does two flops per element, so on this device
// there is no bandwidth left to take. WP7 is a PARITY + VENDOR-FREEDOM
// exercise, and anyone reporting a large speedup over cuBLAS on a DRAM-resident
// gemv cell has made a measurement error.
//
// THE ONE EXCEPTION, reproduced to four significant figures in two independent
// sessions: complex<double> with transA=Trans runs at 310-380 GB/s (33-40% of
// roof) for m in [64,320] with n >= 128 once A exceeds L2, while float, double
// and complex<float> run 936-967 GB/s at IDENTICAL bytes and IDENTICAL (m,n).
// That region is what the CTA body below goes after.
//
// ===========================================================================
// THE FACT THAT DRIVES THE WHOLE DESIGN: THIS OP IS NOT GPU-ONLY.
//
// Every other native tier in this campaign is gated on `is_gpu`. gemv must not
// be. tests/gemv_tests.cc instantiates EIGHT suites: GemvMatrixViewTest/0..3
// are Backend::NETLIB on a native_cpu `Device("cpu")` queue and /4..7 are
// Backend::CUDA. Both halves fail in a vendor-free build, so a GPU-only native
// gemv closes 20 of 40 failures, the suite stays red, and the vendor-free
// burn-down moves by ZERO. The whole deliverable is the other 20.
//
// So: the Direct algorithm carries NO GPU GATE in RouteTable<Op::gemv,T>::
// supports(), and this translation unit is compiled for the native_cpu target
// as well as for CUDA -- it goes in src/sycl/CMakeLists.txt's ordinary source
// list, never into a NO_CPU_TARGETS object library. Moving it to save device
// link time would forfeit the native_cpu image and with it the burn-down.
//
// ===========================================================================
// THE LAYOUT PREMISE, AND WHY THE USUAL INTUITION INVERTS.
//
// Column-major: A(i,j) lives at i + j*ld + b*stride. Assign ONE WORK-ITEM PER
// OUTPUT ELEMENT and:
//
//   transA = NoTrans   y_i = alpha*sum_j A[i + j*ld]*x[j] + beta*y_i.
//                      At reduction step j, work-items i and i+1 read
//                      A[i + j*ld] and A[i+1 + j*ld] -- ADJACENT. Fully
//                      coalesced with no collective at all.
//
//   transA = Trans     y_j = alpha*sum_i A[i + j*ld]*x[i] + beta*y_j.
//                      Work-items j and j+1 read addresses `ld` apart. THIS is
//                      the direction that wants a reduction.
//
// i.e. it is the TRANSPOSED case that needs the sub-group, not the NoTrans one.
// That is the single most important fact in WP7 and it is the opposite of the
// intuition carried over from a row-major reading.
//
// ===========================================================================
// THREE KERNEL BODIES.
//
//   BODY 1  GemvDirectNKernel<T>  {Native, Direct}, transA == NoTrans.
//           One work-item per output ROW. No reduction, zero local memory.
//
//   BODY 2  GemvDirectTKernel<T>  {Native, Direct}, transA != NoTrans.
//           One work-item per output COLUMN, each walking a whole contiguous
//           column. THE PORTABLE ARM: this is what runs on native_cpu (closing
//           the 20 NETLIB rows) and on any device that does not enumerate
//           sub-group size 32.
//
//   BODY 3  GemvCtaTKernel<T>     {Native, CTA}, transA != NoTrans, GPU with
//           an ENUMERATED sub-group size of 32. One 32-lane sub-group per
//           output element, lanes striding the reduction index, hand-rolled
//           shift_group_left ladder, total in lane 0. Fully coalesced.
//
// ALL THREE DECLARE ZERO BYTES OF LOCAL MEMORY. That is a property to verify,
// not to assume -- it is what makes the recorded "48 KB launch hole"
// (a dynamic-local-memory request in (49152-static, 49152] failing at enqueue;
// 48896 passes, 49152 FAILS, 49664 passes; see potrf_cta.cc:259-296)
// structurally unreachable here, so this file carries no hole-padding band.
// The reduction is therefore a HAND-ROLLED sub-group shuffle ladder and
// explicitly NOT sycl::reduce_over_group, which WP6 measured 1.5-4.7x slower
// for double/complex<double> AND which is the construct that puts static shared
// into a kernel and reopens the hole.
//
// ===========================================================================
// TWO CONTRACTS THAT ARE SILENTLY WRONG IF MIS-STATED.
//
// (1) ZERO REDUCTION LENGTH. Reference BLAS ?GEMV quick-returns on
//     `m == 0 || n == 0 || (alpha == 0 && beta == 1)` and leaves y COMPLETELY
//     UNTOUCHED -- it does NOT compute y = beta*y. Both vendors match it. A
//     native path that scaled y in that case would return a ROUTE-DEPENDENT
//     wrong answer, visible only where the route differs. Matched exactly.
//     Reference BLAS also never reads A when alpha == 0; that is matched too,
//     with a launch-uniform branch, so a NaN in A cannot leak into a pure
//     y = beta*y.
//
// (2) NO __restrict__ ON ANY POINTER. ortho.cc:227-232 passes A_i and A_next as
//     views into the SAME allocation. They are element-disjoint but they alias
//     at the object level, and __restrict__ is a promise about the OBJECT.
//
// Conjugation is a RUNTIME bool -- one launch-uniform branch -- wrapped in an
// `if constexpr (dev_is_complex_v<D>)` so the real instantiations emit no
// branch at all. Everything arithmetic goes through the POD device scalars in
// src/sycl/device_scalar.hh, so no std::complex crosses into device code: its
// operator* is Annex-G conformant, i.e. an isnan branch and a call to
// __mulsc3/__muldc3 in the inner loop, worth 1.2-1.3x in a hot loop.

#include <batchlas/blas/enums.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>

namespace batchlas::sycl_gemv {

// Is the Direct tier (bodies 1 and 2) in this build, for this scalar type?
//
// FALSE MEANS "NO NATIVE GEMV KERNEL HERE", and RouteTable<Op::gemv,T>::
// supports() reports the native route unsupported when it is -- the
// TrsmShape::cta_max_n == 0 convention. It is a function rather than a
// constexpr literal for trsm_native.hh:47-60's reason: the answer describes the
// BUILD, and it is answered in the same translation unit as the kernels, so a
// build that drops this TU cannot advertise a route it does not carry.
template <typename T>
bool gemv_direct_available();

// Is the CTA tier (body 3) in this build? Separate from the above because they
// are independent capabilities: a device with no enumerated sub-group size 32
// still gets the Direct tier, and that is the whole point of body 2.
//
// NOTE that this flag says nothing about the DEVICE. `has_sg32` and `is_gpu`
// are the shape builder's job (src/backends/gemv_route.hh); this answers only
// "was the kernel compiled".
template <typename T>
bool gemv_cta_available();

// {Native, Direct}. Dispatches internally to body 1 or body 2 on transA.
//
// The signature is deliberately identical to the public batchlas::gemv and to
// backend::gemv_vendor, so the facade chooses between them without either side
// adapting.
template <typename T>
Event gemv_native_direct(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& A,
                         const VectorView<T>& X,
                         const VectorView<T>& Y,
                         T alpha,
                         T beta,
                         Transpose transA);

// {Native, CTA}. transA MUST NOT be NoTrans -- there is no NoTrans body here,
// because NoTrans is already fully coalesced with one work-item per row and a
// sub-group reduction would only add a shuffle ladder to a loop that does not
// need one. supports() enforces it; a direct caller that violates it gets a
// throw rather than a wrong answer, the trsm_native_v1_buckets convention.
template <typename T>
Event gemv_native_cta(Queue& ctx,
                      const MatrixView<T, MatrixFormat::Dense>& A,
                      const VectorView<T>& X,
                      const VectorView<T>& Y,
                      T alpha,
                      T beta,
                      Transpose transA);

}  // namespace batchlas::sycl_gemv
