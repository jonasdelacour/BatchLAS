// SPDX-License-Identifier: MIT
#pragma once
// Device LA v2 ops private to the ortho SVQB kernels (krylov-ortho-ritz family). Everything
// else those kernels call is v2 core (batchlas/blas/device/v2/dla.hh).

#include <type_traits>
#include <utility>

#include <batchlas/blas/device/v2/dla.hh>

namespace batchlas::krylov_v2 {
namespace dla = device::v2;

// x(i) = f(i), computed on read: never stored, so any item may read any index unpublished.
template <class F> struct Generated {
    using value_type = std::remove_cvref_t<decltype(std::declval<const F&>()(0))>;
    F f;
    int n;
    BATCHLAS_DLA_INLINE value_type operator()(int i) const { return f(i); }
    BATCHLAS_DLA_INLINE int size() const { return n; }
};
template <class F> BATCHLAS_DLA_INLINE Generated<F> generated(int n, F f) { return {f, n}; }

template <class V> inline constexpr bool is_generated_v = false;
template <class F> inline constexpr bool is_generated_v<Generated<F>> = true;

// A := diag(r) A diag(c) (LAPACK laqge): a(i, j) * (r(i) c(j)), r and c real. One row per item:
// the launch width equals a.rows() (the SVQB kernels launch WG = k), so there is no row loop.
// c is read by every item, so it is Published or Generated.
template <class Ex, class Rv, class Cv, class MA> BATCHLAS_DLA_INLINE void laqge(const Ex& ex, const Rv& r, const Cv& cp, const MA& a) {
    using Ar = dla::Arith<typename MA::value_type>;
    static_assert(dla::is_published_v<Cv> || is_generated_v<Cv>, "laqge: c is read across owners; publish it");
    const auto& c = dla::detail::unpub(cp);
    const int n = a.cols(), ld = int(a.ld);
    dla::for_each_owned_exact(ex, a.rows(), [&](int i) {
        const auto ri = r(i);
        auto* const ai = a.p + a.off + i;  // row i moved once; a per-element i + j*ld blocks the unroll
        for (int j = 0; j < n; ++j) {
            const auto s = ri * c(j);
            auto& e = ai[j * ld];
            if constexpr (Ar::is_complex) e = Ar::make(Ar::re(e) * s, Ar::im(e) * s);
            else e = e * s;
        }
    });
}

}  // namespace batchlas::krylov_v2
