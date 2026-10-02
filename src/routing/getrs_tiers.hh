#pragma once

// getrs on the same engine as potrf: a second op is a schema (features {n, nrhs, batch}, key
// {dtype, trans}), three descriptors and a generated RuleSet. No chooser code is copied.
// evidence: docs/design/routing-rules-as-data.md

#include <batchlas/routing/rules.hh>

#include <batchlas/blas/dispatch/coverage.hh>
#include <batchlas/blas/dispatch/vendor_available.hh>
#include <batchlas/blas/functions/getrs.hh>
#include <batchlas/blas/functions/trsm.hh>
#include <batchlas/error.hh>

#include "../backends/getrs_route.hh"
#include "../extensions/getrs_native.hh"

#include "generated/getrs_rules.inc"

namespace batchlas::routing::getrs {

using dispatch::Algorithm;
using dispatch::Origin;
template <class T>
using Tbl = dispatch::RouteTable<dispatch::Op::getrs, T>;
template <class T>
using View = MatrixView<T, MatrixFormat::Dense>;

template <class T>
constexpr int dtype_index() {
    if constexpr (std::is_same_v<T, float>) return 0;
    else if constexpr (std::is_same_v<T, double>) return 1;
    else if constexpr (std::is_same_v<T, std::complex<float>>) return 2;
    else return 3;
}

template <class T>
struct Ctx {
    dispatch::GetrsShape s;
    bool vendor_legal = true;
    Queue* q = nullptr;
    const View<T>* A = nullptr;
    const View<T>* B = nullptr;
    std::uint16_t key() const {
        const int t = s.transA == Transpose::NoTrans ? 0 : (s.transA == Transpose::Trans ? 1 : 2);
        return static_cast<std::uint16_t>(dtype_index<T>() * 3 + t);
    }
    Features features() const { return {s.order(), s.nrhs(), s.batch, 0}; }
};

template <class T>
struct Call {
    Queue& q;
    const View<T>& A;
    const View<T>& B;
    Transpose trans;
    Span<int64_t> piv;
    Span<std::byte> ws;
};

template <class T>
struct FusedTier {
    static constexpr std::string_view id = "native:fused";
    static constexpr dispatch::Route route{Origin::Native, Algorithm::CTA};
    struct Plan { bool fits = true; };
    static bool legal(const Ctx<T>& c) { return Tbl<T>::supports(route, c.s); }
    static Plan plan(const Ctx<T>&, Knobs) { return {}; }
    static std::size_t workspace(const Plan&, const Ctx<T>& c) {
        return c.q ? sycl_getrs::getrs_fused_buffer_size<T>(*c.q, *c.A, *c.B, c.s.transA) : 0;
    }
    static Event launch(const Plan&, Call<T>& k) {
        return sycl_getrs::getrs_fused_dispatch<T>(k.q, k.A, k.B, k.trans, k.piv, k.ws);
    }
};

template <Backend Bk, class T>
struct BlockedTier {
    static constexpr std::string_view id = "native:blocked";
    static constexpr dispatch::Route route{Origin::Native, Algorithm::Blocked};
    struct Plan { bool fits = true; };
    static bool legal(const Ctx<T>& c) { return Tbl<T>::supports(route, c.s); }
    static Plan plan(const Ctx<T>&, Knobs) { return {}; }
    static std::size_t workspace(const Plan&, const Ctx<T>& c) {
        return c.q ? sycl_getrs::getrs_blocked_buffer_size<T>(*c.q, *c.A, *c.B, c.s.transA) : 0;
    }
    static Event launch(const Plan&, Call<T>& k) {
        return sycl_getrs::getrs_blocked_dispatch<T>(
            k.q, k.A, k.B, k.trans, k.piv, k.ws,
            [](Queue& c, const View<T>& ta, const View<T>& tb, T al, Side sd, Uplo ul,
               Transpose tr, Diag dg) { return trsm<Bk, T>(c, ta, tb, al, sd, ul, tr, dg); });
    }
};

template <Backend Bk, class T>
struct VendorTier {
    static constexpr std::string_view id = "vendor";
    static constexpr dispatch::Route route{Origin::Vendor, Algorithm::Auto};
    struct Plan { bool fits = true; };
    static bool legal(const Ctx<T>& c) {
        return dispatch::factorization_vendor_available<Bk> && c.vendor_legal;
    }
    static Plan plan(const Ctx<T>&, Knobs) { return {}; }
    static std::size_t workspace(const Plan&, const Ctx<T>& c) {
        if constexpr (dispatch::factorization_vendor_available<Bk>) {
            if (c.q) return backend::getrs_vendor_buffer_size<Bk, T>(*c.q, *c.A, *c.B, c.s.transA);
        }
        return 0;
    }
    static Event launch(const Plan&, Call<T>& k) {
        if constexpr (dispatch::factorization_vendor_available<Bk>) {
            return backend::getrs_vendor<Bk, T>(k.q, k.A, k.B, k.trans, k.piv, k.ws);
        } else {
            throw batchlas::internal_error("getrs: vendor launched although legal() is false");
        }
    }
};

template <Backend Bk, class T>
using Tiers = TierList<FusedTier<T>, BlockedTier<Bk, T>, VendorTier<Bk, T>>;

template <Backend Bk, class T>
inline constexpr auto kBound = bind<Tiers<Bk, T>>(getrs_rules::kNames);

template <Backend Bk, class T>
Selection<Tiers<Bk, T>> select_ctx(const Ctx<T>& c, const Pin& pin = {}) {
    return routing::select<Tiers<Bk, T>>(getrs_rules::rules_for_profile(c.s.profile),
                                         kBound<Bk, T>, c, pin);
}

template <Backend Bk, class T>
Selection<Tiers<Bk, T>> getrs_select(Queue& q, const View<T>& A, const View<T>& B,
                                     Transpose trans,
                                     bool vendor_legal = dispatch::factorization_vendor_available<Bk>) {
    Ctx<T> c;
    c.s = *backend::getrs_op_shape<Bk, T>(q, A, B, trans);
    c.vendor_legal = vendor_legal;
    c.q = &q;
    c.A = &A;
    c.B = &B;
    return select_ctx<Bk, T>(c);
}

template <Backend Bk, class T>
Event getrs_rules_run(Queue& q, const View<T>& A, const View<T>& B, Transpose trans,
                      Span<int64_t> piv, Span<std::byte> ws) {
    getrs_validate_params<T>(A, B);
    const auto sel = getrs_select<Bk, T>(q, A, B, trans);
    Call<T> call{q, A, B, trans, piv, ws};
    return Tiers<Bk, T>::template launch<Event>(sel.plan, call);
}

}  // namespace batchlas::routing::getrs
