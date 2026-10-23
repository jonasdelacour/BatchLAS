// syev_align (fullerenes plan P2.5): eigenvector continuity across a degenerate spectrum. The
// acceptance case is the Hueckel (adjacency) matrix of Ih C60, whose levels are 1-, 3-, 4-, 5- and
// 9-fold: a basis rotated at random inside every degenerate subspace must come back as the
// previous basis. Every operand has a padded ld and a stride gap holding a large finite poison.
#include <gtest/gtest.h>

#include <batchlas/blas/functions/syev.hh>
#include <batchlas/blas/functions/syev_align.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>

#include <tests/test_utils.hh>

#include <batchlas/verify/tolerance.hh>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

using namespace batchlas;

namespace {

using verify::Check;

template <typename T>
using Real = batchlas::float_t<T>;

template <typename T, Backend B>
struct SyevAlignConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

using SyevAlignTestTypes = typename test_utils::backend_types<SyevAlignConfig>::type;

template <typename Config>
class SyevAlignTest : public test_utils::BatchLASTest<Config> {};

TYPED_TEST_SUITE(SyevAlignTest, SyevAlignTestTypes);

constexpr double kPoison = 1.0e30;

// Ih C60 is the truncated icosahedron (e.g. Fowler & Manolopoulos, "An Atlas of Fullerenes",
// 1995, ch. 2): its 60 atoms are the directed edges (v, w) of the icosahedron, (v, w) bonds to
// (w, v) (hexagon-hexagon bond) and to (v, w') for the two neighbours w' of v adjacent to w (the
// pentagon around v). The icosahedron's vertices are the cyclic permutations of (0, +-1, +-phi),
// edge length 2.
std::vector<int> c60_adjacency() {
    const double phi = (1.0 + std::sqrt(5.0)) / 2.0;
    std::vector<std::array<double, 3>> ico;
    for (double s1 : {-1.0, 1.0})
        for (double s2 : {-1.0, 1.0}) {
            ico.push_back({0.0, s1, s2 * phi});
            ico.push_back({s1, s2 * phi, 0.0});
            ico.push_back({s2 * phi, 0.0, s1});
        }
    auto adj = [&](int a, int b) {
        double d2 = 0;
        for (int k = 0; k < 3; ++k) d2 += (ico[a][k] - ico[b][k]) * (ico[a][k] - ico[b][k]);
        return std::fabs(d2 - 4.0) < 1e-9;
    };
    std::vector<std::array<int, 12>> id(12);
    int count = 0;
    for (int v = 0; v < 12; ++v)
        for (int w = 0; w < 12; ++w) id[v][w] = adj(v, w) ? count++ : -1;
    EXPECT_EQ(count, 60);
    std::vector<int> A(60 * 60, 0);
    for (int v = 0; v < 12; ++v)
        for (int w = 0; w < 12; ++w) {
            if (id[v][w] < 0) continue;
            A[id[v][w] * 60 + id[w][v]] = 1;
            for (int u = 0; u < 12; ++u)
                if (id[v][u] >= 0 && adj(u, w)) A[id[v][w] * 60 + id[v][u]] = 1;
        }
    for (int i = 0; i < 60; ++i) EXPECT_EQ(std::accumulate(A.begin() + i * 60, A.begin() + i * 60 + 60, 0), 3);
    return A;
}

// Columns j-1, j share a cluster when lambda_j - lambda_{j-1} <= tol * max(|lambda_0|, |lambda_{m-1}|).
template <typename R>
std::vector<int> cluster_starts(const R* lam, int m, double tol) {
    const double thr = tol * std::max(std::fabs(double(lam[0])), std::fabs(double(lam[m - 1])));
    std::vector<int> s{0};
    for (int j = 1; j < m; ++j)
        if (double(lam[j]) - double(lam[j - 1]) > thr) s.push_back(j);
    s.push_back(m);
    return s;
}

template <typename T>
MatrixView<T> padded(Matrix<T>& storage, int rows, int cols, int ld, int batch) {
    return MatrixView<T>(storage.data_ptr(), rows, cols, ld, ld * cols + 7, batch);
}

// A batch whose items have ld = rows + pad and stride ld * cols + 7, every pad element poisoned.
template <typename T>
Matrix<T> poisoned_storage(int rows, int cols, int ld, int batch) {
    Matrix<T> s(ld * cols + 7, batch);
    std::fill(s.data_ptr(), s.data_ptr() + int64_t(ld * cols + 7) * batch, T(Real<T>(kPoison)));
    return s;
}

// The item-local poison is intact outside the rows x cols block.
template <typename T>
bool pad_intact(const MatrixView<T>& V) {
    const T* p = V.data_ptr();
    for (int b = 0; b < V.batch_size(); ++b)
        for (int k = 0; k < V.stride(); ++k) {
            const int j = k / V.ld(), i = k % V.ld();
            if (j < V.cols() && i < V.rows()) continue;
            if (p[int64_t(b) * V.stride() + k] != T(Real<T>(kPoison))) return false;
        }
    return true;
}

template <typename T>
void copy_items(Queue& q, const MatrixView<T>& dst, const MatrixView<T>& src) {
    MatrixView<T>::copy(q, dst, src).wait();
}

// Block-unitary Q per item: eigenvectors of a random Hermitian block-diagonal H whose block c is
// shifted by 10 c (so sorted eigenvectors keep the block order), times a random sign or phase per
// column.
template <typename T, Backend B>
Matrix<T> block_unitary(Queue& q, const std::vector<int>& starts, int n, int batch, unsigned seed) {
    Matrix<T> R = Matrix<T>::Random(n, n, true, batch, seed);
    Matrix<T> H = Matrix<T>::Zeros(n, n, batch);
    for (int b = 0; b < batch; ++b)
        for (size_t c = 0; c + 1 < starts.size(); ++c)
            for (int j = starts[c]; j < starts[c + 1]; ++j) {
                for (int i = starts[c]; i < starts[c + 1]; ++i) H(i, j, b) = R(i, j, b);
                H(j, j, b) += T(Real<T>(10.0 * double(c)));
            }
    UnifiedVector<Real<T>> w(size_t(n) * batch);
    UnifiedVector<std::byte> ws(syev_buffer_size<B, T>(q, H, w.to_span(), JobType::EigenVectors, Uplo::Lower));
    syev<B, T>(q, H, w.to_span(), JobType::EigenVectors, Uplo::Lower, ws.to_span()).wait();
    std::mt19937 gen(seed + 1);
    std::uniform_real_distribution<double> ang(0.0, 2.0 * M_PI);
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < n; ++j) {
            T ph;
            const double a = ang(gen);
            if constexpr (is_std_complex_v<T>) ph = T(Real<T>(std::cos(a)), Real<T>(std::sin(a)));
            else ph = T(a < M_PI ? 1 : -1);
            for (int i = 0; i < n; ++i) H(i, j, b) *= ph;
        }
    return H;
}

template <typename T, Backend B>
void align(Queue& q, const MatrixView<T>& Vold, UnifiedVector<Real<T>>& lam, const MatrixView<T>& V, double tol) {
    UnifiedVector<std::byte> ws(syev_align_buffer_size<B, T>(q, Vold, lam.to_span(), V));
    syev_align<B, T>(q, Vold, lam.to_span(), V, Real<T>(tol), ws.to_span()).wait();
}

// ||Ref - X||_F / ||Ref||_F per item (qr_reconstruction with R = I), the max over all items.
template <typename T>
double distance(Queue& q, const MatrixView<T>& Ref, const MatrixView<T>& X) {
    Matrix<T> I = Matrix<T>::Identity(X.cols(), X.batch_size());
    std::vector<int> items(X.batch_size());
    std::iota(items.begin(), items.end(), 0);
    return MatrixView<T>::qr_reconstruction(q, Ref, X, I, items);
}

// The C60 eigenbasis (V_old, ld 65) and its eigenvalues, `batch` identical items.
template <typename T, Backend B>
struct C60 {
    static constexpr int n = 60;
    Matrix<T> vold_store;
    MatrixView<T> Vold;
    UnifiedVector<Real<T>> lam;
    C60(Queue& q, int batch) : vold_store(poisoned_storage<T>(n, n, n + 5, batch)), lam(size_t(n) * batch) {
        const std::vector<int> adj = c60_adjacency();
        Matrix<T> A(n, n, batch);
        for (int b = 0; b < batch; ++b)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) A(i, j, b) = T(Real<T>(adj[i * n + j]));
        UnifiedVector<std::byte> ws(syev_buffer_size<B, T>(q, A, lam.to_span(), JobType::EigenVectors, Uplo::Lower));
        syev<B, T>(q, A, lam.to_span(), JobType::EigenVectors, Uplo::Lower, ws.to_span()).wait();
        Vold = padded(vold_store, n, n, n + 5, batch);
        copy_items(q, Vold, MatrixView<T>(A));
    }
};

TYPED_TEST(SyevAlignTest, C60HueckelSpectrumIsIh) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    C60<T, B> c(*this->ctx, 1);
    const std::vector<int> s = cluster_starts(c.lam.data(), 60, 1e-3);
    auto cluster_of = [&](int j) {
        size_t k = 0;
        while (s[k + 1] <= j) ++k;
        return std::array<int, 2>{s[k], s[k + 1]};
    };
    // Ih C60 Hueckel levels: the Perron root 3 is simple, the HOMO (orbital 30 of 60 from the top)
    // is the fivefold h_u at (sqrt 5 - 1)/2 and the LUMO a threefold t_1u near -0.139.
    EXPECT_NEAR(double(c.lam[59]), 3.0, 1e-4);
    EXPECT_EQ(cluster_of(59), (std::array<int, 2>{59, 60}));
    EXPECT_EQ(cluster_of(30), (std::array<int, 2>{30, 35}));
    EXPECT_NEAR(double(c.lam[30]), (std::sqrt(5.0) - 1.0) / 2.0, 1e-4);
    EXPECT_EQ(cluster_of(29), (std::array<int, 2>{27, 30}));
    EXPECT_NEAR(double(c.lam[29]), -0.1386, 1e-3);
    // The 15 levels ascending; x = 1 is the accidental ninefold g_g + h_g level.
    std::vector<int> deg;
    for (size_t k = 0; k + 1 < s.size(); ++k) deg.push_back(s[k + 1] - s[k]);
    EXPECT_EQ(deg, (std::vector<int>{3, 4, 4, 5, 3, 5, 3, 3, 5, 9, 4, 3, 5, 3, 1}));
    EXPECT_NEAR(double(c.lam[35]), 1.0, 1e-4);
}

// Every item's V is V_old Q_b with a different random block-unitary Q_b: all of them align back
// to V_old. Before the call they are far from it, so the check cannot pass vacuously.
TYPED_TEST(SyevAlignTest, C60RotatedDegenerateSubspacesAlignToPrevious) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    constexpr int n = 60, batch = 5;
    C60<T, B> c(*this->ctx, batch);
    const double tol = 1e-3;
    Matrix<T> Q = block_unitary<T, B>(*this->ctx, cluster_starts(c.lam.data(), n, tol), n, batch, 2024);
    Matrix<T> vs = poisoned_storage<T>(n, n, n + 3, batch);
    MatrixView<T> V = padded(vs, n, n, n + 3, batch);
    MatrixView<T>::product(*this->ctx, c.Vold, Transpose::NoTrans, Q, Transpose::NoTrans, V).wait();
    EXPECT_GT(distance<T>(*this->ctx, c.Vold, V), 0.5);

    align<T, B>(*this->ctx, c.Vold, c.lam, V, tol);
    EXPECT_VERIFY(T, Check::factorization, n, distance<T>(*this->ctx, c.Vold, V));
    EXPECT_VERIFY(T, Check::orthogonality, n, MatrixView<T>::orthogonality(*this->ctx, V, std::vector<int>{0, 2, 4}));
    EXPECT_TRUE(pad_intact(V));
    EXPECT_TRUE(pad_intact(c.Vold));
}

// Only signs (complex: phases) flipped, tol = 0: whether rounding splits a level or not, every
// column comes back to V_old.
TYPED_TEST(SyevAlignTest, C60SignAndPhaseFlipsAreUndone) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    constexpr int n = 60;
    C60<T, B> c(*this->ctx, 1);
    const std::vector<int> singles = [] { std::vector<int> s(n + 1); std::iota(s.begin(), s.end(), 0); return s; }();
    Matrix<T> Q = block_unitary<T, B>(*this->ctx, singles, n, 1, 77);
    Matrix<T> vs = poisoned_storage<T>(n, n, n + 3, 1);
    MatrixView<T> V = padded(vs, n, n, n + 3, 1);
    MatrixView<T>::product(*this->ctx, c.Vold, Transpose::NoTrans, Q, Transpose::NoTrans, V).wait();
    align<T, B>(*this->ctx, c.Vold, c.lam, V, 0.0);
    EXPECT_VERIFY(T, Check::factorization, n, distance<T>(*this->ctx, c.Vold, V));
}

// 1024 identical items: every result bit-identical to item 0, and item 0 aligned.
TYPED_TEST(SyevAlignTest, SaturatingBatchIsBitIdentical) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    constexpr int n = 60, batch = 1024;
    C60<T, B> c(*this->ctx, batch);
    Matrix<T> Q1 = block_unitary<T, B>(*this->ctx, cluster_starts(c.lam.data(), n, 1e-3), n, 1, 99);
    Matrix<T> Q(n, n, batch);
    for (int b = 0; b < batch; ++b) copy_items(*this->ctx, Q.batch_item(b), Q1);
    Matrix<T> vs = poisoned_storage<T>(n, n, n + 3, batch);
    MatrixView<T> V = padded(vs, n, n, n + 3, batch);
    MatrixView<T>::product(*this->ctx, c.Vold, Transpose::NoTrans, Q, Transpose::NoTrans, V).wait();
    align<T, B>(*this->ctx, c.Vold, c.lam, V, 1e-3);
    int differing = 0;
    for (int b = 1; b < batch; ++b)
        for (int j = 0; j < n; ++j)
            if (std::memcmp(&V(0, j, b), &V(0, j, 0), sizeof(T) * n) != 0) ++differing;
    EXPECT_EQ(differing, 0);
    EXPECT_VERIFY(T, Check::factorization, n, distance<T>(*this->ctx, c.Vold.batch_item(0), V.batch_item(0)));
}

// A pair with gap 0.9 thr (one cluster: the rotation inside it is undone) and 1.1 thr (two
// clusters: only signs or phases are fixed, the rotation stays). thr = tol * max|lambda|.
TYPED_TEST(SyevAlignTest, NearDegeneratePairStraddlesTolerance) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    constexpr int n = 9, m = 8;
    const double tol = 1e-2, thr = tol * 7.0;
    Matrix<T> H = Matrix<T>::Random(n, n, true, 2, 5);
    UnifiedVector<Real<T>> w(size_t(n) * 2);
    UnifiedVector<std::byte> sws(syev_buffer_size<B, T>(*this->ctx, H, w.to_span(), JobType::EigenVectors, Uplo::Lower));
    syev<B, T>(*this->ctx, H, w.to_span(), JobType::EigenVectors, Uplo::Lower, sws.to_span()).wait();
    Matrix<T> os = poisoned_storage<T>(n, m, n + 2, 2), vs = poisoned_storage<T>(n, m, n + 4, 2);
    MatrixView<T> Vold = padded(os, n, m, n + 2, 2), V = padded(vs, n, m, n + 4, 2);
    copy_items(*this->ctx, Vold, MatrixView<T>(H, n, m));
    UnifiedVector<Real<T>> lam(size_t(m) * 2);
    const double gap[2] = {0.9 * thr, 1.1 * thr};
    for (int b = 0; b < 2; ++b)
        for (int j = 0; j < m; ++j) lam[b * m + j] = Real<T>(j < 4 ? j : (j == 4 ? 3.0 + gap[b] : j));
    lam[m - 1] = lam[2 * m - 1] = Real<T>(7.0);
    // G rotates columns 3 and 4 by 0.3 rad; P puts a sign or phase on each of them on top.
    Matrix<T> G = Matrix<T>::Identity(m, 2), GP = Matrix<T>::Identity(m, 2);
    T ph3 = T(-1), ph4 = T(1);
    if constexpr (is_std_complex_v<T>) ph3 = T(std::polar(Real<T>(1), Real<T>(2))), ph4 = T(std::polar(Real<T>(1), Real<T>(-1)));
    for (int b = 0; b < 2; ++b) {
        const Real<T> cs = Real<T>(std::cos(0.3)), sn = Real<T>(std::sin(0.3));
        G(3, 3, b) = cs, G(4, 3, b) = -sn, G(3, 4, b) = sn, G(4, 4, b) = cs;
        GP(3, 3, b) = cs * ph3, GP(4, 3, b) = -sn * ph3, GP(3, 4, b) = sn * ph4, GP(4, 4, b) = cs * ph4;
    }
    MatrixView<T>::product(*this->ctx, Vold, Transpose::NoTrans, GP, Transpose::NoTrans, V).wait();
    Matrix<T> Expect(n, m, 2);
    MatrixView<T>::product(*this->ctx, Vold, Transpose::NoTrans, G, Transpose::NoTrans, Expect).wait();
    align<T, B>(*this->ctx, Vold, lam, V, tol);
    EXPECT_VERIFY(T, Check::factorization, n, distance<T>(*this->ctx, Vold.batch_item(0), V.batch_item(0)));
    EXPECT_VERIFY(T, Check::factorization, n, distance<T>(*this->ctx, Expect.batch_item(1), V.batch_item(1)));
    EXPECT_GT(distance<T>(*this->ctx, Vold.batch_item(1), V.batch_item(1)), 0.1);
    EXPECT_TRUE(pad_intact(V));
}

// V_old^H V is exactly zero on the cluster: every column of U is completed, R = I and V is
// returned unchanged and orthonormal. m < n.
TYPED_TEST(SyevAlignTest, OrthogonalSubspacesKeepV) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    constexpr int n = 8, m = 2;
    Matrix<T> os = poisoned_storage<T>(n, m, n + 1, 1), vs = poisoned_storage<T>(n, m, n + 3, 1);
    MatrixView<T> Vold = padded(os, n, m, n + 1, 1), V = padded(vs, n, m, n + 3, 1);
    for (int j = 0; j < m; ++j)
        for (int i = 0; i < n; ++i) Vold(i, j, 0) = T(i == j ? 1 : 0), V(i, j, 0) = T(i == j + 2 ? 1 : 0);
    Matrix<T> V0(n, m, 1);
    copy_items(*this->ctx, V0, V);
    UnifiedVector<Real<T>> lam(m, Real<T>(1));
    align<T, B>(*this->ctx, Vold, lam, V, 1e-3);
    EXPECT_VERIFY(T, Check::factorization, n, distance<T>(*this->ctx, V0, V));
    EXPECT_VERIFY(T, Check::orthogonality, n, MatrixView<T>::orthogonality(*this->ctx, V));
    EXPECT_TRUE(pad_intact(V));
}

TYPED_TEST(SyevAlignTest, RejectsBadArguments) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    Matrix<T> A(6, 4, 2), Bm(6, 3, 2), C(4, 6, 2), D(6, 4, 3);
    UnifiedVector<Real<T>> lam(8), shortlam(7);
    UnifiedVector<std::byte> ws(1 << 16);
    Queue& q = *this->ctx;
    auto call = [&](const Matrix<T>& X, UnifiedVector<Real<T>>& w, const Matrix<T>& Y, double tol, size_t bytes) {
        (void)syev_align<B, T>(q, X, w.to_span(), Y, Real<T>(tol), ws.to_span().subspan(0, bytes));
    };
    EXPECT_THROW(call(A, lam, Bm, 0.0, ws.size()), batchlas::invalid_argument);
    EXPECT_THROW(call(C, lam, C, 0.0, ws.size()), batchlas::invalid_argument);
    EXPECT_THROW(call(A, lam, D, 0.0, ws.size()), batchlas::invalid_argument);
    EXPECT_THROW(call(A, shortlam, A, 0.0, ws.size()), batchlas::invalid_argument);
    EXPECT_THROW(call(A, lam, A, -1.0, ws.size()), batchlas::invalid_argument);
    EXPECT_THROW(call(A, lam, A, 0.0, 16), batchlas::workspace_error);
}

}  // namespace
