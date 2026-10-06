#pragma once

// The canonical fold shared by every native TRSM kernel: the 24 (side, uplo, transA,
// diag) combinations become ONE recurrence over a canonical unit-lower Lc.
// evidence: docs/perf/trsm.md#design-v1-v2-and-the-canonical-fold

#include <batchlas/blas/enums.hh>

namespace batchlas::sycl_trsm {

struct Canonical {
    bool do_trans;
    bool do_conj;
    bool op_is_lower;
    bool unit;
    bool fwd;
};

inline Canonical canonicalise(Side side, Uplo uplo, Transpose transA, Diag diag) {
    Canonical c{};
    c.do_trans = (transA != Transpose::NoTrans);
    c.do_conj = (transA == Transpose::ConjTrans);
    c.op_is_lower = (uplo == Uplo::Lower) ? !c.do_trans : c.do_trans;
    c.unit = (diag == Diag::Unit);
    // Backwards fwd is silent: it solves a different triangle and still returns.
    c.fwd = (side == Side::Left) ? c.op_is_lower : !c.op_is_lower;
    return c;
}

}  // namespace batchlas::sycl_trsm
