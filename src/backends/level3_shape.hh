#pragma once

#include "../linalg-impl.hh"

#include <string>

// These live in their own namespace rather than in batchlas::backend::detail,
// where the neighbouring expansion helpers live, because netlib_lapack.cc
// writes most of its file in namespace batchlas and reaches its own
// batchlas::detail helpers unqualified. Introducing a
// batchlas::backend::detail into that translation unit makes the unqualified
// `detail::` inside its `namespace backend` blocks resolve to the wrong one,
// and submit_host_task stops being found. A distinct name cannot shadow
// anything.
namespace batchlas::backend::shape {

// The Level-3 shape checks, written once instead of seventeen times.
//
// symm/hemm/trmm, syrk/herk and syr2k/her2k each impose the same three shape
// contracts whatever backend is about to serve them, and every one of the three
// backends wrote all three out by hand: cuBLAS and rocBLAS at the top of the
// vendor wrapper, netlib inside the deferred host task. The checks are not
// merely similar, they are the same predicates in the same order, so the only
// thing worth keeping per-backend is the exception type -- hence E.
//
// E is a template parameter rather than a fixed type because netlib's checks
// run inside submit_host_task, i.e. at a different point in time from the
// caller's own stack, and it throws std::runtime_error there; cuBLAS and
// rocBLAS throw std::invalid_argument from the call itself, which
// options_api_tests pins with EXPECT_THROW. Collapsing the two would change
// what a caller catches, so it is not collapsed.
//
// Each validator returns the dimensions it had to derive in order to check
// them, so the caller does not recompute what the check already knows. That is
// the reason these are functions rather than a void assert-block: the derived
// m/n/k were the other half of the duplication.

// What symm/hemm/trmm derive: C is m x n and A is square of order k.
struct ProductShape {
    int m;
    int n;
    int k;
};

// What syrk/herk/syr2k/her2k derive: C is n x n, formed from a k-wide operand.
struct RankShape {
    int n;
    int k;
};

// The two-operand and three-operand batch checks. Naming every batch size is
// the whole diagnostic -- which operand is the odd one out is not otherwise
// visible to the caller -- so the short "batch size mismatch" that rocBLAS and
// netlib used to emit now says what cuBLAS already said.
template <class E>
inline void check_batch(const char* op, int batch_a, int batch_c) {
    if (batch_a != batch_c) {
        throw E(std::string(op) + ": batch size mismatch (A=" + std::to_string(batch_a) +
                ", C=" + std::to_string(batch_c) + ")");
    }
}

template <class E>
inline void check_batch(const char* op, int batch_a, int batch_b, int batch_c) {
    if (batch_a != batch_b || batch_a != batch_c) {
        throw E(std::string(op) + ": batch size mismatch (A=" + std::to_string(batch_a) +
                ", B=" + std::to_string(batch_b) +
                ", C=" + std::to_string(batch_c) + ")");
    }
}

// symm / hemm / trmm: C <- alpha * A * B or alpha * B * A, with A square and B
// and C both m x n.
//
// The order of the checks is the order all seven call sites used: squareness
// first, because a non-square A makes "order k" meaningless; then the batch,
// because a mismatch there says nothing about the shapes; then the shapes.
template <class E, typename T>
inline ProductShape validate_product(const char* op,
                                     const MatrixView<T, MatrixFormat::Dense>& A,
                                     const MatrixView<T, MatrixFormat::Dense>& B,
                                     const MatrixView<T, MatrixFormat::Dense>& C,
                                     Side side) {
    if (A.rows() != A.cols()) {
        throw E(std::string(op) + ": A must be square");
    }
    check_batch<E>(op, A.batch_size(), B.batch_size(), C.batch_size());

    const int m = C.rows();
    const int n = C.cols();
    // A multiplies from whichever side the caller asked for, so it is m x m on
    // the left and n x n on the right.
    const int k = side == Side::Left ? m : n;
    if (A.rows() != k || B.rows() != m || B.cols() != n) {
        throw E(std::string(op) + ": incompatible matrix dimensions");
    }

    return {m, n, k};
}

// syrk / herk: C <- alpha * op(A) * op(A)^T-or-^H + beta * C, C square.
//
// `hermitian` additionally rejects Transpose::Trans, which would ask for
// A * A^T -- complex-symmetric rather than Hermitian. That operation is syrk's,
// and BLAS does not spell it for a complex type.
template <class E, typename T>
inline RankShape validate_rank_k(const char* op,
                                 const MatrixView<T, MatrixFormat::Dense>& A,
                                 const MatrixView<T, MatrixFormat::Dense>& C,
                                 Transpose transA,
                                 bool hermitian) {
    if (C.rows() != C.cols()) {
        throw E(std::string(op) + ": C must be square");
    }
    check_batch<E>(op, A.batch_size(), C.batch_size());
    if (hermitian && transA != Transpose::NoTrans && transA != Transpose::ConjTrans) {
        throw E(std::string(op) + ": transA must be NoTrans or ConjTrans");
    }

    const bool no_trans = transA == Transpose::NoTrans;
    const int n = C.rows();
    const int k = no_trans ? A.cols() : A.rows();
    const int expected_n = no_trans ? A.rows() : A.cols();
    if (expected_n != n || k <= 0) {
        throw E(std::string(op) + ": incompatible matrix dimensions");
    }

    return {n, k};
}

// syr2k / her2k: as the rank-k form, with B carrying the second term and
// required to have op(A)'s shape.
template <class E, typename T>
inline RankShape validate_rank_2k(const char* op,
                                  const MatrixView<T, MatrixFormat::Dense>& A,
                                  const MatrixView<T, MatrixFormat::Dense>& B,
                                  const MatrixView<T, MatrixFormat::Dense>& C,
                                  Transpose transA,
                                  bool hermitian) {
    if (C.rows() != C.cols()) {
        throw E(std::string(op) + ": C must be square");
    }
    check_batch<E>(op, A.batch_size(), B.batch_size(), C.batch_size());
    if (hermitian && transA != Transpose::NoTrans && transA != Transpose::ConjTrans) {
        throw E(std::string(op) + ": transA must be NoTrans or ConjTrans");
    }

    const bool no_trans = transA == Transpose::NoTrans;
    const int n = C.rows();
    const int k = no_trans ? A.cols() : A.rows();
    const int expected_n = no_trans ? A.rows() : A.cols();
    const int b_n = no_trans ? B.rows() : B.cols();
    const int b_k = no_trans ? B.cols() : B.rows();
    if (expected_n != n || b_n != n || b_k != k || k <= 0) {
        throw E(std::string(op) + ": incompatible matrix dimensions");
    }

    return {n, k};
}

} // namespace batchlas::backend::shape
