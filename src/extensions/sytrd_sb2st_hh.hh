#pragma once

// Stage-2 band -> tridiagonal reduction by Householder bulge chasing, with the
// reflectors *retained* so that the eigenvector back-transform Z := Q2 Z is
// possible.
//
// The shipped sytrd_sb2st is a Givens (LAPACK DSBTRD/ZHBTRD) chase and writes
// tau_out = 0 -- Q2 is discarded. That is why syev_two_stage clamps kd to 1
// whenever eigenvectors are requested, which in turn degenerates stage 1 into an
// unblocked BLAS-2 reduction. This routine exists to remove that clamp.
//
// Schedule: the plain sequential one (for each sweep, eliminate then chase the
// bulge to the bottom), not LAPACK's pipelined THGRSIZ/GRSIZ/SHIFT=3 order. The
// pipelining exists to give multi-core CPU parallelism; here a work-group owns
// one problem and parallelism comes from the batch dimension and from lanes
// inside each <= kd x kd window. Validated in
// playground/sb2st_hh_sequential.py against tridiagonality, d, signed e,
// orthogonality of Q and the reference spectrum.
//
// Reflector k acts on rows [start_k, start_k + len_k) and the overall
// similarity is
//
//     Q = H_1 H_2 ... H_m   (generation order),   Q^H A Q = T
//
// so the back-transform applies them in *reverse* generation order.

#include <blas/enums.hh>
#include <blas/matrix.hh>
#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>

#include <cstdint>
#include <vector>

namespace batchlas {
namespace internal {

// One stored reflector. `sweep` is retained because all reflectors belonging to
// a single sweep act on mutually disjoint row ranges (starts stride by kd, each
// length <= kd), so a whole sweep can be applied concurrently in the
// back-transform.
struct Sb2stHhRefl {
    int32_t start;
    int32_t len;
    int32_t sweep;
};

// Replays the sequential chase schedule on the host. The schedule depends only
// on (n, kd) -- never on the matrix values -- so it is identical for every item
// in the batch, which is what lets the back-transform use uniform batched work.
inline std::vector<Sb2stHhRefl> build_sb2st_hh_schedule(int32_t n, int32_t kd) {
    std::vector<Sb2stHhRefl> out;
    if (n <= 2 || kd <= 1) return out;

    for (int32_t st = 0; st + 2 < n; ++st) {
        int32_t r0 = st + 1;
        int32_t r1 = (st + kd < n - 1) ? (st + kd) : (n - 1);
        if (r1 <= r0) continue;

        // TYPE 1: annihilate column st below the subdiagonal.
        out.push_back(Sb2stHhRefl{r0, r1 - r0 + 1, st});

        // Chase the resulting bulge to the bottom of the band.
        while (true) {
            const int32_t p0 = r1 + 1;
            const int32_t p1 = (r1 + kd < n - 1) ? (r1 + kd) : (n - 1);
            if (p0 > p1) break;
            out.push_back(Sb2stHhRefl{p0, p1 - p0 + 1, st});
            r0 = p0;
            r1 = p1;
        }
    }
    return out;
}

inline int32_t sb2st_hh_num_reflectors(int32_t n, int32_t kd) {
    return static_cast<int32_t>(build_sb2st_hh_schedule(n, kd).size());
}

// Working half-bandwidth needed to hold transient bulge fill. A length-kd
// reflector applied symmetrically pushes fill up to kd rows below the band.
inline int32_t sb2st_hh_work_bandwidth(int32_t n, int32_t kd) {
    const int32_t want = 2 * kd;
    const int32_t cap = (n > 0) ? (n - 1) : 0;
    return (want < cap) ? want : cap;
}

// Band -> tridiagonal, retaining the reflectors.
//
//   ab_in      (kd+1) x n   lower band, read-only
//   ab_tri_out 2 x n        row 0 = diagonal, row 1 = *signed* subdiagonal,
//                           so build_phase_from_kd1_band consumes it unchanged
//   d_out/e_out             real diagonal and |subdiagonal|
//   v_out      kd x nrefl   reflector k in column k, v[0] = 1, zero-padded;
//                           nrefl == build_sb2st_hh_schedule(n, kd).size()
//   tau_out    nrefl
template <Backend B, typename T>
Event sytrd_sb2st_hh(Queue& ctx,
                     const MatrixView<T, MatrixFormat::Dense>& ab_in,
                     const MatrixView<T, MatrixFormat::Dense>& ab_tri_out,
                     const VectorView<typename base_type<T>::type>& d_out,
                     const VectorView<typename base_type<T>::type>& e_out,
                     const MatrixView<T, MatrixFormat::Dense>& v_out,
                     const VectorView<T>& tau_out,
                     Uplo uplo,
                     int32_t kd,
                     const Span<std::byte>& ws);

template <Backend B, typename T>
size_t sytrd_sb2st_hh_buffer_size(Queue& ctx, int32_t n, int32_t kd, int32_t batch);

} // namespace internal
} // namespace batchlas
