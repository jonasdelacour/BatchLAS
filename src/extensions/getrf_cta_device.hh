#pragma once

// The native batched GETRF panel factorisation -- ALL of its device code.
//
// ONE algorithm, TWO residencies, exactly geqrf_cta_device.hh's shape.
// `getf2_panel_device` is LAPACK ?GETF2 -- unblocked right-looking LU with
// partial pivoting -- written against a `Tile` abstraction that supplies
// `at(r, c)`, and instantiated twice:
//
//   * against a local_accessor  -> the CTA tier (the whole n x n matrix staged
//                                  into local memory once, factorised, stored);
//   * against a raw global ptr  -> the BLOCKED tier's panel leaf, for panels
//                                  too tall to hold resident.
//
// One body rather than two is the point: the residencies differ by a pointer
// type, so a correctness fix cannot land in one and miss the other.
//
// ===========================================================================
// THE PIVOT SEARCH IS AN EXPLICIT SUB-GROUP BUTTERFLY PLUS A 32-SLOT SLM SCAN,
// AND NOT sycl::reduce_over_group. THREE INDEPENDENT REASONS, ALL MEASURED
// (experiments/wp6_lu/baseline/, summary_pivot.txt / hole2.csv):
//
//   1. THE 48 KB LAUNCH HOLE IS ATTRIBUTABLE TO reduce_over_group ALONE. With a
//      PAD= knob holding kernel, shape and work-group fixed and moving only the
//      declared byte count, one process per point: 49,024 B launches, 49,152 B
//      FAILS ("unknown internal error"), 49,280 B launches -- 5/5 deterministic
//      across five processes, at every work-group width (32/64/128/256/512).
//      The same shape at the same byte count with an explicit SLM tree launches
//      fine. The band is WIDER for wide scalars: the collective also fails at
//      48,896 B for double and cdouble.
//   2. IT IS ALSO SLOWER. Against the unpivoted lower bound, the collective
//      costs 1.5-4.7x more than an explicit tree for double and cdouble
//      (double n=16: 7.07x the bound with the collective, 2.00x with the tree)
//      and is a wash for float/cfloat (0.87-1.25x).
//   3. THE SCRATCH IS A CAPACITY HAZARD. A per-work-item SLM tree needs
//      wg*(sizeof(real) + sizeof(int)) on top of the tile -- 2040 B at wg 256
//      for float, 3060 B for cdouble -- and at cdouble n=78 that took the
//      request from 98,608 B to 101,668 B, past this device's 101,376 B hard
//      cap: "Excessive allocation of local memory on the device". It also moves
//      the blocks-per-SM cliff (slm + 1024 crossing 50,688 B) DOWN BY TWO
//      ORDERS OF n and costs 1.73x exactly there.
//
// The form used here fixes (3) as well as (1) and (2): the per-item tree is
// replaced by a sub-group XOR butterfly (registers, no local memory at all)
// followed by a scan over ONE SLOT PER SUB-GROUP. That is at most 16 slots at
// the widest work-group this file will launch and is allocated at a CONSTANT 32,
// so the scratch is 384 B for cdouble instead of 3,060 B and -- more
// importantly -- IT DOES NOT DEPEND ON THE WORK-GROUP WIDTH. That is what lets
// the capacity query, the fit predicate and the launcher agree on the SLM
// footprint without any of them knowing which wg the others picked, which is the
// potrf_cta_launch_params defect (potrf_cta.cc:442-454) closed by construction
// rather than by a comment.
//
// The butterfly spelling is geqrf_sg_sum's (geqrf_cta_device.hh:319-341), for
// the same reason it gives: DPC++'s CUDA path for non-uniform group collectives
// has had limitations, and permute_group_by_xor is a register shuffle.
//
// ===========================================================================
// THE PIVOT METRIC IS LAPACK's cabs1, |Re| + |Im|, NOT THE TRUE MODULUS -- AND
// cuBLAS DISAGREES WITH LAPACK HERE, WHICH IS A MEASURED FACT WITH CONSEQUENCES
// FOR EVERY TEST THIS OP WILL EVER HAVE.
//
// ?GETF2 selects with I?AMAX, whose complex form is |Re| + |Im| (cabs1). The
// modulus is an equally valid selection rule and gives an equally valid
// factorisation, so NO RESIDUAL CAN TELL THEM APART; only the elementwise pivot
// sequence can. Measured with a matrix built to separate them -- column 0 holding
// (3 + 0i) in row 0 and (2 + 2i) in row 1, so cabs1 reads 3 vs 4 and the modulus
// reads 3 vs 2.828 (experiments/wp6_lu/kernels/luverify.cpp's `pivmetric` mode,
// run_pivmetric.sh):
//
//     native:cta      ipiv[0] = 2   ==  host LAPACKE_?getrf
//     native:blocked  ipiv[0] = 2   ==  host LAPACKE_?getrf
//     cublas?getrfBatched  ipiv[0] = 1  !=  host, for BOTH cfloat and cdouble
//
// i.e. cuBLAS pivots on the MODULUS for complex. Replacing cabs1 with the modulus
// in this file reproduces cuBLAS's answer exactly, which is what identifies the
// cause rather than merely observing a difference.
//
// THREE CONSEQUENCES, all of them for whoever writes the tests:
//   1. This kernel is LAPACK-faithful and the vendor is not. That is the right
//      side to be on -- LAPACK is the published contract and NETLIB is one of
//      this library's own backends -- but it means "matching the vendor bit for
//      bit", which the pivot-contract note below asks for, holds for the FORMAT
//      (packed 1-based int32, an interchange list) and NOT for the VALUES on
//      complex input.
//   2. A test that compares native pivots to VENDOR pivots elementwise is wrong
//      and will go red on complex. The oracle must be the HOST.
//   3. Mixing arms remains safe. getrs and getri consume ipiv together with the
//      factor produced by the SAME getrf call, and any valid pivot sequence is
//      self-consistent with its own factor; what would break is only a caller
//      that factored with one implementation and permuted with another's ipiv,
//      which no API in this tree permits.
//
// The tie-break is I?AMAX's too: the LOWEST index attaining the maximum, carried
// explicitly through the butterfly and the slot scan because a work-item's
// strided row set does not order its candidates by work-item id.
//
// ===========================================================================
// info IS EXACT-ZERO SEMANTICS, 1-BASED, GLOBAL, FIRST-FAILURE-WINS, AND THE
// FACTORISATION CONTINUES. Measured against cuBLAS and host LAPACKE through the
// public API (see getrf_native.hh's PIVOT CONTRACT note 4): on a matrix whose
// step 2 cancels to a true binary zero, device info == host info == 2; on a
// float matrix with a duplicated column the device produced U(3,3) = -1.375e-08
// and reported info = 0 while the host got a true 0.0 and reported 3.
//
// SO THERE IS NO EPSILON FLOOR HERE, and that is a deliberate divergence from
// one line of the WP6 brief ("a pivot floor rather than an exact-zero test --
// see stein.cc:177-190"). stein.cc's floor is a tridiagonal solver's stability
// hack on its OWN output; getrf's info is a PUBLIC CONTRACT shared with cuBLAS,
// rocSOLVER and LAPACK, and a kernel that flags |pivot| < eps reports non-zero
// where the vendor reports zero. That divergence is invisible to every
// native-vs-native test.
//
// What IS taken from LAPACK's own safe-scaling discipline -- and is the useful
// half of a "floor" -- is the reciprocal test below: ?GETF2 multiplies by
// 1/pivot when the pivot is comfortably above the underflow threshold and
// DIVIDES element by element otherwise. Here the test is on the reciprocal
// itself (finite and non-zero), which is the spelling geqrf_cta_device.hh
// already uses for v = x/(alpha-beta) and device_scalar.hh:200-209 describes.
// A zero pivot takes NEITHER path: the column is left untouched, exactly as
// ?GETF2 does, so a failed item stays FINITE instead of filling with Inf/NaN.
//
// ===========================================================================
// THE nd_range, because "parallel over batch only" is this repository's
// recurring performance defect: ONE WORK-GROUP PER MATRIX of 64..512 work-items
// (see getrf_leaf_wg in getrf_cta.cc), every phase of the step parallel over the
// work-group -- the search strides rows, the row exchange strides columns, the
// column scale strides rows, and the rank-1 update strides the whole trailing
// block. Work-group width DOMINATES this kernel and is NOT inheritable from
// potrf or geqrf: measured (experiments/wp6_lu/baseline/wg.csv), float n=128
// batch 4096 is 39.72 ms at wg=32 and 4.77 ms at wg=512, an 8.3x spread.
//
// BARRIER AUDIT -- every site is at the top level of the k loop, whose trip
// count (kmax) is kernel-uniform, and none is inside a divergent region:
//   B0  after the tile load                       (CTA tier only, in the launcher)
//   B1  after the per-sub-group argmax slots are published, before they are read
//   B2  after the row exchange, before the pivot is read as a scalar
//   B3  after the column scale, before the rank-1 update reads L(:,k)
//   B4  after the rank-1 update, before the next step's search reads column k+1
//        -- and it is also what makes B1's slots safe to overwrite
//   B5  after the loop, before the store-back            (CTA tier, in launcher)
// The row-exchange loop needs no barrier of its own: work-item t reads and
// writes exactly the columns it strides over. Neither does the scale loop.

#include "../sycl/device_scalar.hh"

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>

namespace batchlas::getrf_native {

using batchlas::sycl_device::Cx;

// The real component type of a DEVICE scalar. DevMap answers the question for a
// SOURCE type; a kernel body templated on D cannot see T. Same reason
// IsDevComplex exists (device_scalar.hh:55-60), and the same spelling geqrf uses.
template <typename D> struct RealOf        { using type = D; };
template <typename R> struct RealOf<Cx<R>> { using type = R; };
template <typename D> using real_of = typename RealOf<D>::type;

// The ONE spelling device_scalar.hh does not carry that this file needs. Every
// multiply, divide, reciprocal and finiteness test comes from there, so this
// file contains NO private complex multiply -- the thing the campaign forbids
// and which latrd_lower_panel.cc:148 and seven other TUs each carry a copy of.
template <typename D>
inline D lu_zero() {
    if constexpr (batchlas::sycl_device::dev_is_complex_v<D>) {
        return D{real_of<D>(0), real_of<D>(0)};
    } else {
        return D(0);
    }
}

// LAPACK's cabs1: |Re| + |Im|. See the note above for why not the modulus.
template <typename D>
inline real_of<D> lu_cabs1(D a) {
    if constexpr (batchlas::sycl_device::dev_is_complex_v<D>) {
        return sycl::fabs(a.re) + sycl::fabs(a.im);
    } else {
        return sycl::fabs(a);
    }
}

// ---------------------------------------------------------------------------
// Tile accessors. The ONLY difference between the two residencies.
// ---------------------------------------------------------------------------
template <typename D>
struct LuGlobalTile {
    D* p;
    int ld;
    D& at(int r, int c) const {
        return p[static_cast<std::ptrdiff_t>(r) + static_cast<std::ptrdiff_t>(c) * ld];
    }
};

template <typename D, typename LocalAcc>
struct LuLocalTile {
    LocalAcc a;
    int ld;
    // ld IS PADDED TO AN ODD NUMBER by the launcher, unlike geqrf's tile which
    // is packed at ld = m. LU has an access pattern geqrf does not: the ROW
    // EXCHANGE walks a row, i.e. `wg` work-items at stride ld, and an even ld
    // puts every one of them in the same local-memory bank. geqrf's header can
    // say a pad "would only cost capacity" precisely because it never touches a
    // row.
    auto& at(int r, int c) const {
        return a[static_cast<std::size_t>(r) +
                 static_cast<std::size_t>(c) * static_cast<std::size_t>(ld)];
    }
};

// The number of per-sub-group argmax slots the launchers allocate. CONSTANT, not
// a function of the work-group width -- see the header note: it is what lets the
// capacity query, the fit predicate and the launcher agree on the SLM footprint.
// 32 covers a work-group of 1024 at sub-group size 32; this file never launches
// wider than 512.
inline constexpr int kLuRedSlots = 32;

// ---------------------------------------------------------------------------
// THE PANEL FACTORISATION. LAPACK ?GETF2 on an m x n tile, in place.
//
//   for k = 0 .. kmax-1
//       p       = argmax_{i >= k} cabs1(A(i,k))          (lowest index wins)
//       ipiv[k] = piv_base + p + 1                       (GLOBAL, 1-based)
//       swap rows k and p across ALL n columns of the tile
//       if A(k,k) != 0   A(k+1:m, k) /= A(k,k)
//       else             info = piv_base + k + 1  (first failure only)
//       A(k+1:m, k+1:n) -= A(k+1:m, k) * A(k, k+1:n)
//
// `piv_base` is the panel's first GLOBAL row/column index, so the blocked driver
// gets LAPACK's global ipiv and global info for free rather than by a fix-up
// pass. Both callers pass their own j0 (the CTA tier passes 0).
//
// `kmax` is passed rather than derived so a caller can factor fewer columns than
// min(m, n); both callers pass min(m, n).
//
// `piv_item` and `info_item` are GLOBAL pointers already offset to this matrix
// (and, for `piv_item`, to this panel). `info_item` is READ as well as written
// -- first-failure-wins across the blocked driver's panels -- so both launchers
// zero it before the first panel.
// ---------------------------------------------------------------------------
template <typename D, typename Tile, typename ValAcc, typename IdxAcc>
inline void getf2_panel_device(sycl::nd_item<1> it, Tile A, int m, int n, int kmax,
                               int* piv_item, int piv_base, int32_t* info_item,
                               ValAcc rval, IdxAcc ridx) {
    using R = real_of<D>;

    const auto g = it.get_group();
    const auto sg = it.get_sub_group();
    const int wg = static_cast<int>(it.get_local_range(0));
    const int tid = static_cast<int>(it.get_local_linear_id());
    const int lane = static_cast<int>(sg.get_local_linear_id());
    const int nlanes = static_cast<int>(sg.get_local_linear_range());
    const int team = static_cast<int>(sg.get_group_linear_id());
    const int nteams = static_cast<int>(sg.get_group_linear_range());

    // Held in ONE work-item rather than re-read from global every step. It is
    // still read from global once, because the blocked driver's later panels
    // must not clear an earlier panel's failure.
    int32_t info_local = (tid == 0) ? *info_item : 0;

    for (int k = 0; k < kmax; ++k) {
        // ---- the pivot search --------------------------------------------
        // Per work-item scan first: strided by row, so consecutive work-items
        // read consecutive elements of column k and the global (or local) read
        // is coalesced. The scan visits increasing i, so a plain `>` already
        // keeps the LOWEST index within one work-item.
        R bv = R(-1);
        int bi = m;                       // "no candidate" -- never wins a tie
        for (int i = k + tid; i < m; i += wg) {
            const R v = lu_cabs1<D>(A.at(i, k));
            if (v > bv) { bv = v; bi = i; }
        }

        // Sub-group butterfly. Registers only -- no local memory, no group
        // collective. The index rides along so I?AMAX's lowest-index tie-break
        // survives the reduction.
        for (uint32_t off = static_cast<uint32_t>(nlanes) / 2; off > 0; off >>= 1) {
            const R ov = sycl::permute_group_by_xor(sg, bv, off);
            const int oi = sycl::permute_group_by_xor(sg, bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) {
            rval[static_cast<std::size_t>(team)] = bv;
            ridx[static_cast<std::size_t>(team)] = bi;
        }
        sycl::group_barrier(g);                                        // B1

        // Every work-item scans the (at most 16) slots redundantly rather than
        // one leader broadcasting: the answer must be work-group-uniform for the
        // exchange and the scale below, and a leader-plus-broadcast would add a
        // barrier and a slot to save at most sixteen loads from local memory.
        R fv = R(-1);
        int p = m;
        for (int t = 0; t < nteams; ++t) {
            const R v = rval[static_cast<std::size_t>(t)];
            const int ii = ridx[static_cast<std::size_t>(t)];
            if (v > fv || (v == fv && ii < p)) { fv = v; p = ii; }
        }
        if (p >= m) p = k;   // unreachable: k < m, so column k always has a row

        if (tid == 0) {
            piv_item[k] = piv_base + p + 1;      // GLOBAL, 1-BASED, LAPACK ipiv
        }

        // ---- the row exchange ---------------------------------------------
        // Across ALL n columns of the tile -- including the columns to the LEFT
        // of k, which already hold finished L. ?GETF2 swaps the whole row and so
        // must this: L's rows are permuted by every later pivot, and applying
        // the exchange only to columns >= k is the classic silently-wrong LU.
        if (p != k) {
            for (int c = tid; c < n; c += wg) {
                D& x = A.at(k, c);
                D& y = A.at(p, c);
                const D t = x;
                x = y;
                y = t;
            }
        }
        sycl::group_barrier(g);                                        // B2

        // ---- the column scale ---------------------------------------------
        const D d = A.at(k, k);
        if (batchlas::sycl_device::dev_is_zero(d)) {
            // EXACT zero, no epsilon -- see the header note. LAPACK records the
            // failure and leaves the column ALONE, which is what keeps a failed
            // item finite; a kernel that divides unconditionally produces
            // Inf/NaN where the vendor gives finite garbage.
            if (tid == 0 && info_local == 0) {
                info_local = static_cast<int32_t>(piv_base + k + 1);
            }
        } else {
            const D r = batchlas::sycl_device::dev_recip(d);
            const bool use_mul = batchlas::sycl_device::dev_isfinite(r) &&
                                 !batchlas::sycl_device::dev_is_zero(r);
            if (use_mul) {
                for (int i = k + 1 + tid; i < m; i += wg) {
                    A.at(i, k) = batchlas::sycl_device::dev_mul(A.at(i, k), r);
                }
            } else {
                // ?GETF2's small-pivot arm: divide element by element rather
                // than multiply by a reciprocal that has overflowed.
                for (int i = k + 1 + tid; i < m; i += wg) {
                    A.at(i, k) = batchlas::sycl_device::dev_div(A.at(i, k), d);
                }
            }
        }
        sycl::group_barrier(g);                                        // B3

        // ---- the rank-1 update --------------------------------------------
        // A(k+1:m, k+1:n) -= L(k+1:m, k) * U(k, k+1:n). Flattened over the whole
        // trailing block so the work-group is saturated even when the block is
        // short in one extent, and indexed row-fastest so consecutive work-items
        // touch consecutive rows of one column.
        //
        // THE TWO RUNTIME DIVISIONS ARE DELIBERATE, AND THE ALTERNATIVE WAS
        // MEASURED SLOWER. `mm` is loop-variant, so `e % mm` / `e / mm` really are
        // runtime 32-bit divisions on a device with no integer-divide unit, and a
        // review priced them at ~15% of this kernel. Replacing them with an
        // equivalent power-of-two (row, column) split -- division-free, still
        // coalesced, still saturating, and with U(k,j) hoisted into a register so
        // per-element traffic drops from four accesses to three -- LOST 6.4% of
        // getrf float across the 39-cell saturation sweep and 17% at the shape it
        // was aimed at (float n=128: 3.938 ms -> 4.746 ms at batch 2048, and
        // 0.829-0.833x across every batch on that rung; cdouble 0.976x geomean;
        // route unchanged, 0 cells discarded).
        //
        // The reason is the trip counts, not the arithmetic. At the resident
        // tier's shape mm is CLOSE TO wg (127 against 512 at n=128), so the split
        // gives every work-item an inner loop of trip count ONE and an outer loop
        // of ~32 -- all loop overhead and an un-amortised U(k,j) load per element,
        // against 31 iterations of one tight flat loop. The division is only worth
        // removing where mm >> wg, which is the blocked driver's global panel leaf
        // alone, and paying for that everywhere else costs more than it saves.
        // If this is revisited, it must be a HYBRID keyed on mm/wg and it must be
        // measured on the resident rung, not only on the panel.
        const int mm = m - k - 1;
        const int nn = n - k - 1;
        if (mm > 0 && nn > 0) {
            const int total = mm * nn;
            for (int e = tid; e < total; e += wg) {
                const int i = k + 1 + (e % mm);
                const int j = k + 1 + (e / mm);
                A.at(i, j) = batchlas::sycl_device::dev_sub(
                    A.at(i, j),
                    batchlas::sycl_device::dev_mul(A.at(i, k), A.at(k, j)));
            }
        }
        sycl::group_barrier(g);                                        // B4
    }

    if (tid == 0) {
        *info_item = info_local;
    }
}

}  // namespace batchlas::getrf_native
