// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief Eigenvectors of a real symmetric tridiagonal by inverse iteration, `stein` (xSTEIN),
/// inside a kernel, for ascending eigenvalues such as `stebz` returns.
///
/// One matrix on one executor, any n, in one tier, `tier::Blocked<1>`: one item per vector runs a
/// serial LU of the shifted tridiagonal per step, then the vectors of each cluster of close
/// eigenvalues are reorthogonalised by modified Gram-Schmidt. The LU rows live in the caller's
/// global workspace. `with_tier` checks a `WorkGroup<256>` at sub-group size 32. The factor and
/// solve kernels `lagtf` and `lagts` are in @ref api_device_lowlevel.
/// @ingroup api_device_lapack_eig
//   detail::iterate    phase 1: one item per vector, a serial LU per shift (the host
//                            route's SteinIterateKernel body)
//   detail::reorthogonalize  phase 2: modified Gram-Schmidt inside clusters of close
//                            eigenvalues, the executor per dot / axpy (SteinOrthoKernel)
// Tier: one body for every n, tier::Blocked<1>: the LU rows live in the caller's global
// workspace (five k x n arrays and the pivot flags), local memory holds one broadcast scalar.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <batchlas/device/arith.hh>
#include <batchlas/device/core.hh>
#include <batchlas/device/exec.hh>
#include <batchlas/device/lapack/lagtf.hh>
#include <batchlas/device/lapack/lagts.hh>
#include <batchlas/device/lapack_aux.hh>
#include <batchlas/device/level1.hh>
#include <batchlas/device/tier.hh>
#include <batchlas/device/workspace.hh>

namespace batchlas::device {

/// @brief Run-time options of `stein`; the defaults are the host SteinParams'.
/// @tparam Real  the tridiagonal's real type
/// @ingroup api_device_lapack_eig
template <class Real> struct SteinOptions {
    /// Inverse-iteration steps per vector, >= 1; there is no convergence test.
    std::int32_t max_iter = 3;
    /// Neighbouring eigenvalues closer than ortho_threshold times the infinity norm of T form a
    /// cluster, whose vectors are reorthogonalised.
    Real ortho_threshold = Real(1e-3);
    /// This matrix's seed: vector j starts from the LCG seeded with seed + j * 104729 (the host
    /// route passes params.seed + b * 7919 for matrix b).
    std::uint32_t seed = 0x5eed1234u;
};

namespace detail {
/// The start vector's generator (dstein uses dlarnv): a 32-bit LCG, so a fixed per-(item,
/// vector) seed reproduces a run bit for bit. Advances state and returns it.
BATCHLAS_DEVICE_HELPER std::uint32_t lcg(std::uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

// The workspace plan (design C.5): one local broadcast slot; per vector j the shifted
// tridiagonal's LU rows dl, dd, du, du2, the iterate x and the pivot flags at j * n.
template <class Real> struct SteinLayout {
    WsSlot<Real> norm;
    WsSlot<Real> dl, dd, du, du2, x;
    WsSlot<unsigned char> piv;
    WorkspaceSize size;
};
template <class Real> constexpr SteinLayout<Real> stein_layout(int n, int k) {
    Plan p;
    const std::size_t nk = std::size_t(n) * std::size_t(k);
    const auto norm = p.take<Real>(WsSpace::Local, 1);
    const auto dl = p.take<Real>(WsSpace::Global, nk);
    const auto dd = p.take<Real>(WsSpace::Global, nk);
    const auto du = p.take<Real>(WsSpace::Global, nk);
    const auto du2 = p.take<Real>(WsSpace::Global, nk);
    const auto x = p.take<Real>(WsSpace::Global, nk);
    const auto piv = p.take<unsigned char>(WsSpace::Global, nk);
    return {norm, dl, dd, du, du2, x, piv, p.size()};
}

// The phases' scratch as pointers: the broadcast slot, and vector j's LU rows and pivot flags
// at (first + j) * n. device::stein carves it from its workspace (scratch(ws, L), first = 0); the
// host route points it into batch-wide arrays, first = b * k, so its addressing is the legacy one.
template <class Real> struct SteinScratch {
    local_mptr<Real> norm;
    Real* dl;
    Real* dd;
    Real* du;
    Real* du2;
    Real* x;
    unsigned char* piv;
    std::int64_t first;  // this matrix's first vector slot in the arrays
};
template <class Real> BATCHLAS_DEVICE_INLINE SteinScratch<Real> scratch(const Workspace& ws,
                                                                        const SteinLayout<Real>& L) {
    return {carve_local(ws, L.norm), carve_global(ws, L.dl), carve_global(ws, L.dd), carve_global(ws, L.du),
            carve_global(ws, L.du2), carve_global(ws, L.x), carve_global(ws, L.piv), 0};
}

// Phase 1. Vectors j < kb: inverse iteration on T - w(j) I from a pseudo-random start, an
// exactly repeated eigenvalue nudged by eps ||T|| j; z(:, j) = the normalised result. Columns
// [kb, k) of z are written as zero (callers back-transform all k). No barrier inside the
// vector loop: items diverge freely. Needs only z(i, j) and z.cols() of z.
template <class Ex, class D, class E, class W, class Z, class Real>
BATCHLAS_DEVICE_INLINE void iterate(const Ex& ex, const D& d, const E& e, const W& w, int kb, const Z& z, const SteinScratch<Real>& s,
                                 const SteinOptions<Real>& p) {
    using T = Real;
    const std::int64_t n = d.size();
    const std::int64_t k = z.cols();
    const auto norm_local = s.norm;

    // ||T||_inf, used for the pivot floor and the perturbation scale. The broadcast through
    // local memory is the legacy one (the reduction is already uniform).
    const T tnorm_g = lanst<NormType::Inf>(ex, d, e);
    if (ex.leader()) norm_local[0] = tnorm_g;
    sync(ex, Mem::Global);
    const T tnorm = sycl::max(T(norm_local[0]), std::numeric_limits<T>::min());

    const T eps = std::numeric_limits<T>::epsilon();
    const T pivot_floor = eps * tnorm;
    const std::int64_t step = detail::size_of(ex);

    for (std::int64_t j = ex.lid(); j < k; j += step) {
        if (j >= kb) {
            for (std::int64_t i = 0; i < n; ++i) z(i, j) = T(0);
            continue;
        }

        const std::int64_t base = (s.first + j) * n;
        T* my_dl = s.dl + base;
        T* my_dd = s.dd + base;
        T* my_du = s.du + base;
        T* my_du2 = s.du2 + base;
        T* x = s.x + base;
        unsigned char* my_piv = s.piv + base;

        // Separate exactly-degenerate eigenvalues so their shifted systems differ; phase 2
        // then has independent vectors to work with.
        T lambda = w(j);
        if (j > 0) {
            const T prev = w(j - 1);
            if (sycl::fabs(lambda - prev) < eps * tnorm) lambda += eps * tnorm * static_cast<T>(j);
        }

        std::uint32_t rng = p.seed + static_cast<std::uint32_t>(j * 104729);
        for (std::int64_t i = 0; i < n; ++i) {
            const std::uint32_t r = lcg(rng);
            x[i] = T(2) * (static_cast<T>(r >> 8) / static_cast<T>(1u << 24)) - T(1);
        }

        for (std::int32_t iter = 0; iter < p.max_iter; ++iter) {
            T nrm = T(0);
            for (std::int64_t i = 0; i < n; ++i) nrm += x[i] * x[i];
            nrm = sycl::sqrt(nrm);
            if (nrm <= T(0)) {
                x[0] = T(1);
                nrm = T(1);
            }
            const T inv = T(1) / nrm;
            for (std::int64_t i = 0; i < n; ++i) x[i] *= inv;

            // Rebuild the shifted tridiagonal: the factorization overwrites it every iteration.
            for (std::int64_t i = 0; i < n; ++i) my_dd[i] = d(i) - lambda;
            for (std::int64_t i = 0; i < n - 1; ++i) {
                my_dl[i] = e(i);
                my_du[i] = e(i);
            }
            for (std::int64_t i = 0; i < n; ++i) my_du2[i] = T(0);

            const int ni = static_cast<int>(n);
            lagtf(Thread{}, vec(my_dl, ni), vec(my_dd, ni), vec(my_du, ni), vec(my_du2, ni), vec(my_piv, ni), pivot_floor);
            lagts(Thread{}, vec(my_dl, ni), vec(my_dd, ni), vec(my_du, ni), vec(my_du2, ni), vec(my_piv, ni), vec(x, ni));

            // Rescale eagerly: the solve amplifies by roughly 1/|lambda - lambda_exact|, which
            // overflows otherwise.
            T amax = T(0);
            for (std::int64_t i = 0; i < n; ++i) amax = sycl::max(amax, sycl::fabs(x[i]));
            if (amax > T(0)) {
                const T s = T(1) / amax;
                for (std::int64_t i = 0; i < n; ++i) x[i] *= s;
            }
        }

        T nrm = T(0);
        for (std::int64_t i = 0; i < n; ++i) nrm += x[i] * x[i];
        nrm = sycl::sqrt(nrm);
        const T inv = (nrm > T(0)) ? (T(1) / nrm) : T(1);
        for (std::int64_t i = 0; i < n; ++i) z(i, j) = x[i] * inv;
    }
}

// Phase 2. Walk the ascending w(0..kb); a gap wider than ortho_threshold * ||T||_inf starts a
// new cluster, and each vector is made orthogonal to its cluster's earlier ones (modified
// Gram-Schmidt) and renormalised. Stopping at kb, not k, is a cost bound. Needs z.col(j).
template <class Ex, class D, class E, class W, class Z, class Real>
BATCHLAS_DEVICE_INLINE void reorthogonalize(const Ex& ex, const D& d, const E& e, const W& w, int kb, const Z& z,
                                         const SteinScratch<Real>& s, const SteinOptions<Real>& p) {
    using T = Real;
    const auto scratch = s.norm;
    const T tnorm = sycl::max(lanst<NormType::Inf>(ex, d, e), std::numeric_limits<T>::min());
    const T gap_tol = p.ortho_threshold * tnorm;

    int cluster_start = 0;
    for (int j = 0; j < kb; ++j) {
        if (j > 0 && (w(j) - w(j - 1)) > gap_tol) cluster_start = j;
        const auto zj = z.col(j);
        for (int i = cluster_start; i < j; ++i) {
            const T dt = dot<Scope::Own, arith::Native>(ex, z.col(i), zj);
            axpy<arith::Native>(ex, -dt, z.col(i), zj);
            sync(ex, Mem::Global);
        }
        if (j > cluster_start) {
            const T ss = sumsq<alg::Fused>(ex, zj);
            if (ex.leader()) scratch[0] = sycl::sqrt(ss);
            sync(ex, Mem::Global);
            const T nrm = scratch[0];
            if (nrm > T(0)) scal<arith::Native>(ex, T(1) / nrm, zj);
            sync(ex, Mem::Global);
        }
    }
}
}  // namespace detail

namespace routine {
/// @brief The routine tag of `stein` (xSTEIN) for `with_tier`, `launch` and `buffer_size`.
/// @ingroup api_device_lapack_eig
struct stein {
    /// The name `with_tier` puts in its error message.
    static constexpr const char* name = "stein";
};
}  // namespace routine

/// @brief The tiers `with_tier` tries for `stein`: `tier::Blocked<1>` only.
/// @ingroup api_device_lapack_eig
template <class Real> struct tiers<routine::stein, Real> {
    using type = tier::list<tier::Blocked<1>>;
};
/// @brief The one tier of `stein`, `tier::Blocked<1>`: real data, any n >= 1, global workspace
/// for the LU rows and one local slot.
///
/// `buffer_size(n, k)` is the workspace for k vectors; `buffer_size(n)` is the k = n bound that
/// `with_tier` queries. `wg_size` 256 and `sg_size` 32 are what `with_tier` checks; the op itself
/// runs on any executor width. Real types only: xSTEIN's complex form takes a real tridiagonal,
/// and the complex back-transform is the caller's.
/// @ingroup api_device_lapack_eig
template <class Real> struct tier_info<routine::stein, tier::Blocked<1>, Real> {
    static_assert(std::is_floating_point_v<Real>, "stein: a real T");
    static constexpr int wg_size = 256, sg_size = 32;
    using executor = WorkGroup<wg_size, sg_size>;
    static constexpr int per_group = 1;
    static constexpr bool admits(int n) { return n >= 1; }
    static constexpr WorkspaceSize buffer_size(int n, int k) { return detail::stein_layout<Real>(n, k).size; }
    static constexpr WorkspaceSize buffer_size(int n) { return buffer_size(n, n); }
};

/// @brief xSTEIN on one real symmetric tridiagonal (d, e) inside a kernel: the eigenvectors of the
/// ascending eigenvalues w(0) to w(kb - 1), by inverse iteration.
///
/// Arguments in the host order (d, e, w, kb, z, ws, opts). z is n x k with k = `z.cols()` >= kb;
/// column j < kb receives the unit vector of w(j), and the columns from kb on are written as
/// zero. An exactly repeated eigenvalue is nudged so that its vectors differ. COLLECTIVE over
/// `ex`; the inputs must be visible to every item on entry, and every output is visible on
/// return.
/// @tparam Tier  `tier::Blocked<1>`
/// @param[in]  ex  the executor, any width
/// @param[in]  d   n reals, the diagonal
/// @param[in]  e   n - 1 reals, the off-diagonal
/// @param[in]  w   at least kb eigenvalues, ascending (from `stebz`)
/// @param[in]  kb  the number of vectors wanted
/// @param[out] z   n x k, the vectors
/// @param      ws  this matrix's slices of `buffer_size<routine::stein, Tier, T>(n, k)`
/// @param[in]  p   the step count, the cluster threshold and the seed
/// @retval 0  always: inverse iteration runs a fixed number of steps and xSTEIN's info is not
///            computed
/// @ingroup api_device_lapack_eig
template <class Tier, class Ex, class D, class E, class W, class Z, class Real>
BATCHLAS_DEVICE_INLINE int stein(const Ex& ex, const D& d, const E& e, const W& w, int kb, const Z& z, const Workspace& ws,
                                 const SteinOptions<Real>& p) {
    static_assert(std::is_same_v<elem_t<D>, Real>, "stein: options of the matrix's real type");
    static_assert(std::is_same_v<resolve_tier_t<routine::stein, Real, Tier>, tier::Blocked<1>>,
                  "stein: tier::Blocked<1>");
    const auto s = detail::scratch(ws, detail::stein_layout<Real>(d.size(), z.cols()));
    detail::iterate(ex, d, e, w, kb, z, s, p);
    sync(ex, Mem::Global);  // phase 2 reads the vectors other items wrote
    detail::reorthogonalize(ex, d, e, w, kb, z, s, p);
    return 0;
}

}  // namespace batchlas::device
