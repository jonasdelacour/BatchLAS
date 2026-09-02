#pragma once

// Native batched GEMV -- declarations.
//
// WP7. The point of this kernel is NOT speed. The recon phase measured
// cuBLAS `gemvStridedBatched` at 94-105% of the ~950 GB/s achievable DRAM roof
// on 90 of 92 reproducing cells, across all four scalar types and both transA
// values (docs/perf/gemv.md#the-vendor-baseline). A batched gemv is pure
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
//   BODY 4  GemvSegNKernel<T,W>   {Native, Direct}, transA == NoTrans,
//           out_len <= 16. W = 32/out_len lanes per output, one sub-group per
//           BATCH ITEM, fold at stride out_len.
//
//   BODY 5  GemvSegTKernel<T,W>   {Native, CTA}, transA != NoTrans, SHORT
//           REDUCTION -- red_len <= 32 for float, <= 16 for complex<float>,
//           <= 48 for double and <= 64 for complex<double> -- AND a launch of at
//           least 16*CU outputs (64*CU in the W = 4 band). W outputs per
//           sub-group and L = 32/W lanes each, fold at stride 1 in log2(L)
//           steps. Body 4's idea TRANSPOSED -- and transposed is the operative
//           word: the lane index varies fastest in the REDUCTION here, because
//           under Trans that is the contiguous direction.
//
//           {Native, CTA} THEREFORE NAMES TWO KERNELS. A resolved route column
//           reading `native:cta` no longer says which one ran, and the campaign's
//           usual "linked is not reachable" instrument is blind to the
//           difference. gemv_seg_trans_width_debug below is what separates them,
//           and route_gemv.hh's note on what the route column cannot tell you
//           needs extending to say so.
//
// ---- WHY BODY 5 EXISTS, AND WHAT THE RECORDED MECHANISM GOT WRONG ---------
// route_gemv.hh blamed body 3's short-reduction collapse on the shuffle LADDER
// being "a fixed cost per output (5 steps, doubled to 10 for a complex
// scalar)". That reading predicts `double` and `complex<float>` are hurt
// EQUALLY -- both issue 10 hardware shuffles per fold. Measured on body 3 at
// out_len = 2048, batch = 512, transA = Trans, DRAM-resident, GB/s:
//
//     red_len        32      64     128    2048 (each type's own roof)
//     float       833.8   924.2   931.0    952.3
//     double      547.5   928.6   932.2    953.8
//     cfloat      921.2   925.4   932.7    954.0
//     cdouble     434.5   708.5   932.5    952.2
//
// double and cfloat are 1.68x apart at red_len = 32 at identical bytes and
// identical shuffle count. ncu on those launches gives the discriminator:
// sm__pipe_fp64_cycles_active is 85.6/86.1/84.7% for cdouble and 85.0/82.7/58.3%
// for double across red_len 32/64/128, and EXACTLY 0.00% for cfloat and float,
// while occupancy holds at 79-93% and sectors-per-load is ideal throughout. The
// fold is FP64 WORK on a 1/64-rate GeForce part: 32*5 = 160 double-adds per
// output for `double` and 320 for `complex<double>`, against only red_len useful
// FMAs. Body 5 cuts that to L*log2(L) -- 8 at L = 4.
//
// ---- WHAT BODY 5 IS WORTH, MEASURED --------------------------------------
// docs/perf/gemv.md#the-body-5-gates: body 5 (the shipped `auto` decision)
// against body 3, interleaved REP BY REP inside one process, 11 reps, median,
// warm JIT, two independent passes, foreign compute-process count 0 on every
// row, both arms checked against the same in-process host oracle. The worse of
// the two passes is quoted. Ratio = body3_ms / body5_ms.
//
//   83 ADMITTED cells (body 5 ran):  geomean 3.277x, MIN 1.073x, MAX 10.49x,
//                                    ZERO cells below 1.00x, zero below 1.05x
//   53 DECLINED cells (gate sent both arms to body 3): 0.985x .. 1.006x
//   cross-pass spread: worst 1.153, all but three cells under 1.05
//   ConjTrans, separately: 36 admitted cells, geomean 2.734x, MIN 1.072x
//
// AND THE SKINNY REGIME, which the plane above could not reach and which is
// where this pass found its own trap-8 defect
// (docs/perf/gemv.md#the-body-5-gates, out_len walked from 1 to 64):
//
//   30 ADMITTED cells: geomean 1.566x, MIN 1.037x, MAX 3.043x, ZERO below 1.00x
//   74 DECLINED cells: 0.977x .. 1.009x
//
// Body 3's own GB/s and body 5's, at out_len = 2048, batch = 512, Trans:
//
//   red_len        1      4      8     16     24     32     48     64    128
//   float b3    59.6  148.2  263.7  484.3  655.6  832.3  914.6  924.0  927.0
//   float b5   349.3  874.7 1557.3 2382.5  878.0  893.4      -      -      -
//   cfloat b3  105.6  260.9  451.9  777.9  905.5  923.8  927.9  926.5  931.0
//   cfloat b5  659.9 1629.1 1837.3  889.1      -      -      -      -      -
//   double b3   33.4   83.2  149.8  282.1  415.0  545.9  818.6  930.2  930.8
//   double b5  343.8  873.7 1512.7  882.9  897.6  901.6  915.9      -      -
//   cdouble b3  26.6   66.0  118.5  224.2  329.6  434.1  456.4  708.5  930.8
//   cdouble b5 269.0  663.5  861.7  891.4  904.1  910.5  919.3  923.8      -
//
// (a dash is where the gate declines and body 3 serves the call.)
//
// AN ODD ld COSTS BODY 5 SOMETHING AND NEVER INVERTS THE SIGN. A run starts at
// (b*stride + j*ld + s), so an ld that is not a multiple of the run length
// straddles an extra 32-byte sector. Measured at out_len = 2048, batch = 512
// (docs/perf/gemv.md#the-body-5-gates), packed ld vs odd ld:
// cdouble red_len 8: 7.32x -> 6.65x; double red_len 8: 9.92x -> 5.59x;
// cfloat red_len 8: 4.06x -> 2.13x; float red_len 32: 1.076x -> 1.062x.
// Every admitted odd-ld cell stays at or above 1.06x. tests/gemv_tests.cc
// exercises ld = 79 at m = 70, so this is a live layout and not a hypothetical.
//
// WHAT IS **NOT** CLAIMED, and it is a real window left open. The gates are on
// red_len alone, and in the L2-RESIDENT regime they are too tight: at
// out_len = 256, batch = 512 -- where A is 33-67 MB against this device's 72 MB
// L2 -- body 5 at W = 4 measures 1.40x-2.09x for cfloat at red_len 24..64,
// 2.62x for double at red_len 64 and 1.22x-1.71x for float at red_len 48..128,
// all ABOVE their gates (docs/perf/gemv.md#open-debts). The same
// red_len at out_len = 2048 measures 0.986x-0.996x. Separating them needs a
// FOOTPRINT term, which is the L2-residency reasoning route_gemv.hh:279-284
// forbids in preferred() and which would be no better founded in a launcher.
// The gate is therefore set where it never loses, and the L2 window is stated
// rather than taken.
//
// ALL FIVE DECLARE ZERO BYTES OF LOCAL MEMORY. That is a property to verify,
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

// ---------------------------------------------------------------------------
// WHICH CTA KERNEL gemv_native_cta WOULD LAUNCH for (this queue, this red_len):
// 1 = body 3 (one sub-group per output), W >= 2 = body 5 at that W.
// TEST-ONLY.
//
// {Native, CTA} NAMES TWO KERNELS AS OF WP8, so the resolved route column --
// the campaign's usual "linked is not reachable" instrument -- can no longer
// tell which one ran. A GATE-D break against body 5 that executes at a shape
// the gate sends to body 3 is VACUOUS and passes green. This query is what makes
// such a break provably non-vacuous, and it is also what a test uses to reach
// body 3 at a shape where body 5 is the default.
//
// It resolves through the SAME gate function the launcher calls, with the SAME
// sub-group query and the SAME BATCHLAS_GEMV_SEGT reading; there is no second
// copy of the decision. That defect -- two copies of one boundary, the driver's
// flipped by a break while the test-visible copy kept the old sense and the
// whole suite stayed GREEN -- is recorded at src/extensions/getrs_native.cc:410.
// ---------------------------------------------------------------------------
//
// out_len_times_batch is the third gate's input: body 5 launches
// (out_len*batch)/W sub-groups, i.e. W TIMES FEWER than body 3, and below
// 8*MAX_COMPUTE_UNITS outputs it is giving away parallelism the shape cannot
// spare. Pass A.cols() * A.batch_size(), which is what the launcher passes.
template <typename T>
int gemv_seg_trans_width_debug(Queue& ctx, int red_len, int64_t out_len_times_batch);

}  // namespace batchlas::sycl_gemv
