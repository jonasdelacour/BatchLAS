// SPDX-License-Identifier: MIT
#pragma once
// The ortho SVQB kernels' one op core lacks (core laqge tests LAPACK's ratios first).

#include <batchlas/device.hh>

namespace batchlas::krylov_v2 {
namespace dla = device;

// A core fn_vec is computed on read, never stored: any item may read any index unpublished.
template <class V> inline constexpr bool is_fn_vec_v = false;
template <class F> inline constexpr bool is_fn_vec_v<dla::FnVec<F>> = true;

// A := diag(r) A diag(c), r and c real. One row per item (launch width == a.rows(), WG = k);
// c is read by every item, so it is Published or an fn_vec.
template <class Ex, class Rv, class Cv, class MA> BATCHLAS_DLA_INLINE void laqge(const Ex& ex, const Rv& r, const Cv& cp, const MA& a) {
    using Ar = dla::Arith<typename MA::value_type>;
    static_assert(dla::is_published_v<Cv> || is_fn_vec_v<Cv>, "laqge: c is read across owners; publish it");
    const auto& c = dla::detail::unpub(cp);
    const int n = a.cols(), ld = int(a.ld);
    dla::for_each_owned_exact(ex, a.rows(), [&](int i) {
        const auto ri = r(i);
        auto* const ai = a.p + a.off + i;  // row i moved once; a per-element i + j*ld blocks the unroll
        for (int j = 0; j < n; ++j) {
            const auto s = ri * c(j);
            auto& e = ai[j * ld];
            e = Ar::scale_real(s, e);
        }
    });
}

}  // namespace batchlas::krylov_v2
