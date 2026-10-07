#pragma once

#include "../linalg-impl.hh"

#include <string>

// Trap: NOT batchlas::backend::detail. netlib_lapack.cc reaches batchlas::detail
// unqualified, and a backend::detail in that TU makes its `detail::` resolve wrong.
namespace batchlas::backend::shape {

// The level-3 shape checks shared by every backend. E is the exception type,
// and it must stay per-backend: netlib throws std::runtime_error from its host
// task, cuBLAS/rocBLAS std::invalid_argument (pinned by options_api_tests).
// evidence: docs/perf/level3.md#level-3-one-set-of-shape-validators

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

// Batch checks. The message names every batch size: that is the whole diagnostic.
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
// and C both m x n. Check order (square, batch, shapes) is part of the contract.
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
    const int k = side == Side::Left ? m : n;
    if (A.rows() != k || B.rows() != m || B.cols() != n) {
        throw E(std::string(op) + ": incompatible matrix dimensions");
    }

    return {m, n, k};
}

// syrk / herk: C <- alpha * op(A) * op(A)^T-or-^H + beta * C, C square.
//
// `hermitian` rejects Transpose::Trans: A * A^T would be complex-symmetric.
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
