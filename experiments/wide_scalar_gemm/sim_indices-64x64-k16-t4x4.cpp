// sim_indices.cpp -- host-only simulation of the exact index arithmetic in
// tile-64x64-k16-t4x4.cpp, so the tiling can be proved correct without a GPU.
//
// It replays, verbatim, the staging coordinates, the two-band fragment loads,
// the accumulate mapping and the epilogue mapping, for all three band
// configurations the real kernel instantiates (VecN = 4, 2, 1), on both the
// fast and the predicated path, and compares against a naive GEMM.
//
// Build: g++ -O2 -std=c++20 sim_indices.cpp -o sim_indices && ./sim_indices

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>

static constexpr int TileM = 64, TileN = 64, TileK = 16;
static constexpr int TT = 4;
static constexpr int LocalRows = TileM / TT;   // 16
static constexpr int LocalCols = TileN / TT;   // 16
static constexpr int Threads = LocalRows * LocalCols;
static constexpr int AStride = TileM, BStride = TileN;
static constexpr int PerThreadA = TileM * TileK / Threads;  // 4
static constexpr int PerThreadB = TileK * TileN / Threads;  // 4

// One work-group of the kernel, for a given granule width VecN.
template <int VecN, bool Fast>
void group(int m, int n, int k, int m0, int n0,
           const std::vector<double>& A, int lda,
           const std::vector<double>& B, int ldb,
           std::vector<double>& C, int ldc,
           double alpha, double beta,
           std::vector<char>& touched) {
    constexpr int Wb = (VecN < TT) ? VecN : TT;
    constexpr int NB = TT / Wb;
    constexpr int MSep = TileM / NB;
    constexpr int NSep = TileN / NB;
    constexpr int Chunks = PerThreadA / VecN;
    constexpr int AGranPerRow = TileM / VecN;
    static_assert(AGranPerRow * TileK == Chunks * Threads, "A granule handout");

    std::vector<double> sa(TileK * AStride), sb(TileK * BStride);
    // accum[thread][i][j]
    std::vector<double> accum(std::size_t(Threads) * TT * TT, 0.0);

    // Track that staging writes every shared cell exactly once per k0 step.
    std::vector<int> hits_a(TileK * AStride, 0), hits_b(TileK * BStride, 0);

    for (int k0 = 0; k0 < k; k0 += TileK) {
        std::fill(hits_a.begin(), hits_a.end(), 0);
        std::fill(hits_b.begin(), hits_b.end(), 0);

        for (int tid = 0; tid < Threads; ++tid) {
            int a_gm[Chunks], a_gk[Chunks];
            for (int c = 0; c < Chunks; ++c) {
                const int g = tid + c * Threads;
                a_gm[c] = (g % AGranPerRow) * VecN;
                a_gk[c] = g / AGranPerRow;
            }
            const int b_k = (tid % (TileK / PerThreadB)) * PerThreadB;
            const int b_n = tid / (TileK / PerThreadB);

            for (int c = 0; c < Chunks; ++c) {
                const int gk_a = k0 + a_gk[c];
                for (int e = 0; e < VecN; ++e) {
                    const int gm = m0 + a_gm[c] + e;
                    const int si = a_gk[c] * AStride + a_gm[c] + e;
                    ++hits_a[si];
                    if constexpr (Fast) {
                        sa[si] = A[std::size_t(gk_a) * lda + gm];
                    } else {
                        sa[si] = (gm < m && gk_a < k)
                                     ? A[std::size_t(gk_a) * lda + gm]
                                     : 0.0;
                    }
                }
            }
            const int gn_b = n0 + b_n;
            for (int i = 0; i < PerThreadB; ++i) {
                const int gk = k0 + b_k + i;
                const int si = (b_k + i) * BStride + b_n;
                ++hits_b[si];
                if constexpr (Fast) {
                    sb[si] = B[std::size_t(gn_b) * ldb + gk];
                } else {
                    sb[si] = (gk < k && gn_b < n) ? B[std::size_t(gn_b) * ldb + gk] : 0.0;
                }
            }
        }

        for (int i = 0; i < TileK * AStride; ++i)
            if (hits_a[i] != 1) { std::printf("  A stage coverage BAD at %d: %d\n", i, hits_a[i]); std::exit(1); }
        for (int i = 0; i < TileK * BStride; ++i)
            if (hits_b[i] != 1) { std::printf("  B stage coverage BAD at %d: %d\n", i, hits_b[i]); std::exit(1); }

        for (int tid = 0; tid < Threads; ++tid) {
            const int ty = tid % LocalRows;  // linear id = tx*LocalRows + ty
            const int tx = tid / LocalRows;
            for (int kk = 0; kk < TileK; ++kk) {
                double af[TT], bf[TT];
                for (int band = 0; band < NB; ++band)
                    for (int w = 0; w < Wb; ++w)
                        af[band * Wb + w] = sa[kk * AStride + band * MSep + ty * Wb + w];
                for (int band = 0; band < NB; ++band)
                    for (int w = 0; w < Wb; ++w)
                        bf[band * Wb + w] = sb[kk * BStride + band * NSep + tx * Wb + w];
                for (int i = 0; i < TT; ++i)
                    for (int j = 0; j < TT; ++j)
                        accum[(std::size_t(tid) * TT + i) * TT + j] += af[i] * bf[j];
            }
        }
    }

    // Epilogue
    for (int tid = 0; tid < Threads; ++tid) {
        const int ty = tid % LocalRows;
        const int tx = tid / LocalRows;
        for (int bm = 0; bm < NB; ++bm) {
            const int gm = m0 + bm * MSep + ty * Wb;
            for (int j = 0; j < TT; ++j) {
                const int bn = j / Wb;
                const int w_n = j % Wb;
                const int gn = n0 + bn * NSep + tx * Wb + w_n;
                if constexpr (!Fast) {
                    if (gn >= n) continue;
                }
                for (int w = 0; w < Wb; ++w) {
                    const int row = gm + w;
                    if constexpr (!Fast) {
                        if (row >= m) continue;
                    }
                    const std::size_t ci = std::size_t(gn) * ldc + row;
                    const double a = accum[(std::size_t(tid) * TT + bm * Wb + w) * TT + j];
                    C[ci] = alpha * a + beta * C[ci];
                    ++touched[ci];
                }
            }
        }
    }
}

template <int VecN, bool Fast>
int one(int m, int n, int k) {
    const int lda = m, ldb = k, ldc = m;
    std::vector<double> A(std::size_t(m) * k), B(std::size_t(k) * n), C(std::size_t(m) * n),
        C0;
    unsigned s = 12345;
    auto rnd = [&]() { s = s * 1103515245u + 12345u; return double((s >> 16) & 1023) / 1024.0 - 0.5; };
    for (auto& x : A) x = rnd();
    for (auto& x : B) x = rnd();
    for (auto& x : C) x = rnd();
    C0 = C;

    const double alpha = 1.25, beta = 0.75;
    std::vector<char> touched(C.size(), 0);
    for (int m0 = 0; m0 < m; m0 += TileM)
        for (int n0 = 0; n0 < n; n0 += TileN)
            group<VecN, Fast>(m, n, k, m0, n0, A, lda, B, ldb, C, ldc, alpha, beta, touched);

    for (std::size_t i = 0; i < C.size(); ++i)
        if (touched[i] != 1) {
            std::printf("  C coverage BAD at %zu: %d\n", i, int(touched[i]));
            return 1;
        }

    double maxerr = 0, maxref = 0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            double acc = 0;
            for (int p = 0; p < k; ++p) acc += A[std::size_t(p) * lda + i] * B[std::size_t(j) * ldb + p];
            const double ref = alpha * acc + beta * C0[std::size_t(j) * ldc + i];
            maxerr = std::max(maxerr, std::fabs(C[std::size_t(j) * ldc + i] - ref));
            maxref = std::max(maxref, std::fabs(ref));
        }
    const double rel = maxerr / std::max(maxref, 1e-300);
    std::printf("  VecN=%d %-5s %dx%dx%d  coverage OK  relerr=%.3e  %s\n", VecN,
                Fast ? "fast" : "pred", m, n, k, rel, rel < 1e-12 ? "PASS" : "FAIL");
    return rel < 1e-12 ? 0 : 1;
}

int main() {
    int bad = 0;
    std::printf("fast path (m%%64==0, n%%64==0, k%%16==0):\n");
    bad |= one<4, true>(128, 128, 32);   // float geometry:            NB=1
    bad |= one<2, true>(128, 128, 32);   // double / complex<float>:   NB=2
    bad |= one<1, true>(128, 128, 32);   // complex<double>:           NB=4
    bad |= one<2, true>(64, 64, 16);
    bad |= one<2, true>(192, 320, 128);
    std::printf("predicated path (ragged):\n");
    bad |= one<4, false>(70, 53, 37);
    bad |= one<2, false>(70, 53, 37);
    bad |= one<1, false>(70, 53, 37);
    bad |= one<2, false>(1, 1, 1);
    bad |= one<2, false>(65, 129, 17);
    std::printf("%s\n", bad ? "SOME CHECKS FAILED" : "ALL INDEX CHECKS PASS");
    return bad;
}
