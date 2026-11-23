// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @ingroup api_device_launch
/// @brief routine_call for every LAPACK routine tag, so `launch<R, Tier, T>::run(args...)` calls
/// the routine R names in tier Tier.
///
/// Each specialization forwards to the routine's free function with the tag's modes, then the
/// tier, in the function's template order: `run<Tier>(ex, A, ws)` for `routine::potrf<UL>` is
/// `potrf<UL, Tier>(ex, A, ws)`. A tag's parameters the function does not take (sy2sb's and
/// sb2st's `KDMax`, which only size the workspace) are dropped. `<batchlas/device/lapack.hh>`
/// includes this header; the sparse routines have no routine_call.

#include <batchlas/device/lapack.hh>

namespace batchlas::device {

// ---- no modes: name<Tier> ----------------------------------------------------------------------
template <> struct routine_call<routine::bdsdc> {
    template <class Tier, class... A> static BATCHLAS_DEVICE_INLINE decltype(auto) run(A&&... a) {
        return device::bdsdc<Tier>(static_cast<A&&>(a)...);
    }
};
template <> struct routine_call<routine::bdsqr> {
    template <class Tier, class... A> static BATCHLAS_DEVICE_INLINE decltype(auto) run(A&&... a) {
        return device::bdsqr<Tier>(static_cast<A&&>(a)...);
    }
};
template <> struct routine_call<routine::gebrd> {
    template <class Tier, class... A> static BATCHLAS_DEVICE_INLINE decltype(auto) run(A&&... a) {
        return device::gebrd<Tier>(static_cast<A&&>(a)...);
    }
};
template <> struct routine_call<routine::geqrf> {
    template <class Tier, class... A> static BATCHLAS_DEVICE_INLINE decltype(auto) run(A&&... a) {
        return device::geqrf<Tier>(static_cast<A&&>(a)...);
    }
};
template <> struct routine_call<routine::gesv> {
    template <class Tier, class... A> static BATCHLAS_DEVICE_INLINE decltype(auto) run(A&&... a) {
        return device::gesv<Tier>(static_cast<A&&>(a)...);
    }
};
template <> struct routine_call<routine::getrf> {
    template <class Tier, class... A> static BATCHLAS_DEVICE_INLINE decltype(auto) run(A&&... a) {
        return device::getrf<Tier>(static_cast<A&&>(a)...);
    }
};
template <> struct routine_call<routine::getri> {
    template <class Tier, class... A> static BATCHLAS_DEVICE_INLINE decltype(auto) run(A&&... a) {
        return device::getri<Tier>(static_cast<A&&>(a)...);
    }
};
template <> struct routine_call<routine::inv> {
    template <class Tier, class... A> static BATCHLAS_DEVICE_INLINE decltype(auto) run(A&&... a) {
        return device::inv<Tier>(static_cast<A&&>(a)...);
    }
};
template <> struct routine_call<routine::orgqr> {
    template <class Tier, class... A> static BATCHLAS_DEVICE_INLINE decltype(auto) run(A&&... a) {
        return device::orgqr<Tier>(static_cast<A&&>(a)...);
    }
};
template <> struct routine_call<routine::stebz> {
    template <class Tier, class... A> static BATCHLAS_DEVICE_INLINE decltype(auto) run(A&&... a) {
        return device::stebz<Tier>(static_cast<A&&>(a)...);
    }
};
template <> struct routine_call<routine::stein> {
    template <class Tier, class... A> static BATCHLAS_DEVICE_INLINE decltype(auto) run(A&&... a) {
        return device::stein<Tier>(static_cast<A&&>(a)...);
    }
};

// ---- Uplo: name<UL, Tier> ----------------------------------------------------------------------
template <Uplo UL> struct routine_call<routine::potrf<UL>> {
    template <class Tier, class... A> static BATCHLAS_DEVICE_INLINE decltype(auto) run(A&&... a) {
        return device::potrf<UL, Tier>(static_cast<A&&>(a)...);
    }
};
template <Uplo UL> struct routine_call<routine::potrs<UL>> {
    template <class Tier, class... A> static BATCHLAS_DEVICE_INLINE decltype(auto) run(A&&... a) {
        return device::potrs<UL, Tier>(static_cast<A&&>(a)...);
    }
};
template <Uplo UL> struct routine_call<routine::posv<UL>> {
    template <class Tier, class... A> static BATCHLAS_DEVICE_INLINE decltype(auto) run(A&&... a) {
        return device::posv<UL, Tier>(static_cast<A&&>(a)...);
    }
};
template <Uplo UL> struct routine_call<routine::potri<UL>> {
    template <class Tier, class... A> static BATCHLAS_DEVICE_INLINE decltype(auto) run(A&&... a) {
        return device::potri<UL, Tier>(static_cast<A&&>(a)...);
    }
};
template <Uplo UL> struct routine_call<routine::lauum<UL>> {
    template <class Tier, class... A> static BATCHLAS_DEVICE_INLINE decltype(auto) run(A&&... a) {
        return device::lauum<UL, Tier>(static_cast<A&&>(a)...);
    }
};
template <Uplo UL> struct routine_call<routine::sytrf<UL>> {
    template <class Tier, class... A> static BATCHLAS_DEVICE_INLINE decltype(auto) run(A&&... a) {
        return device::sytrf<UL, Tier>(static_cast<A&&>(a)...);
    }
};
template <Uplo UL> struct routine_call<routine::sytrs<UL>> {
    template <class Tier, class... A> static BATCHLAS_DEVICE_INLINE decltype(auto) run(A&&... a) {
        return device::sytrs<UL, Tier>(static_cast<A&&>(a)...);
    }
};
template <Uplo UL> struct routine_call<routine::sysv<UL>> {
    template <class Tier, class... A> static BATCHLAS_DEVICE_INLINE decltype(auto) run(A&&... a) {
        return device::sysv<UL, Tier>(static_cast<A&&>(a)...);
    }
};
template <Uplo UL> struct routine_call<routine::sytrd<UL>> {
    template <class Tier, class... A> static BATCHLAS_DEVICE_INLINE decltype(auto) run(A&&... a) {
        return device::sytrd<UL, Tier>(static_cast<A&&>(a)...);
    }
};
template <Uplo UL, int KDMax> struct routine_call<routine::sy2sb<UL, KDMax>> {
    template <class Tier, class... A> static BATCHLAS_DEVICE_INLINE decltype(auto) run(A&&... a) {
        return device::sy2sb<UL, Tier>(static_cast<A&&>(a)...);
    }
};
template <Uplo UL, int KDMax> struct routine_call<routine::sb2st<UL, KDMax>> {
    template <class Tier, class... A> static BATCHLAS_DEVICE_INLINE decltype(auto) run(A&&... a) {
        return device::sb2st<UL, Tier>(static_cast<A&&>(a)...);
    }
};

// ---- other modes, in the function's order ------------------------------------------------------
template <Uplo UL, Diag DG> struct routine_call<routine::trtri<UL, DG>> {
    template <class Tier, class... A> static BATCHLAS_DEVICE_INLINE decltype(auto) run(A&&... a) {
        return device::trtri<UL, DG, Tier>(static_cast<A&&>(a)...);
    }
};
template <Transpose TR> struct routine_call<routine::getrs<TR>> {
    template <class Tier, class... A> static BATCHLAS_DEVICE_INLINE decltype(auto) run(A&&... a) {
        return device::getrs<TR, Tier>(static_cast<A&&>(a)...);
    }
};
template <Side SD, Transpose TR> struct routine_call<routine::ormqr<SD, TR>> {
    template <class Tier, class... A> static BATCHLAS_DEVICE_INLINE decltype(auto) run(A&&... a) {
        return device::ormqr<SD, TR, Tier>(static_cast<A&&>(a)...);
    }
};
template <Vect VT, Side SD, Transpose TR> struct routine_call<routine::ormbr<VT, SD, TR>> {
    template <class Tier, class... A> static BATCHLAS_DEVICE_INLINE decltype(auto) run(A&&... a) {
        return device::ormbr<VT, SD, TR, Tier>(static_cast<A&&>(a)...);
    }
};
template <Itype IT, Uplo UL> struct routine_call<routine::sygst<IT, UL>> {
    template <class Tier, class... A> static BATCHLAS_DEVICE_INLINE decltype(auto) run(A&&... a) {
        return device::sygst<IT, UL, Tier>(static_cast<A&&>(a)...);
    }
};
template <JobType J, Uplo UL> struct routine_call<routine::syev<J, UL>> {
    template <class Tier, class... A> static BATCHLAS_DEVICE_INLINE decltype(auto) run(A&&... a) {
        return device::syev<J, UL, Tier>(static_cast<A&&>(a)...);
    }
};
template <JobType J> struct routine_call<routine::stedc<J>> {
    template <class Tier, class... A> static BATCHLAS_DEVICE_INLINE decltype(auto) run(A&&... a) {
        return device::stedc<J, Tier>(static_cast<A&&>(a)...);
    }
};
template <JobType J> struct routine_call<routine::steqr<J>> {
    template <class Tier, class... A> static BATCHLAS_DEVICE_INLINE decltype(auto) run(A&&... a) {
        return device::steqr<J, Tier>(static_cast<A&&>(a)...);
    }
};
template <SvdVectors JU, SvdVectors JVT> struct routine_call<routine::gesvd<JU, JVT>> {
    template <class Tier, class... A> static BATCHLAS_DEVICE_INLINE decltype(auto) run(A&&... a) {
        return device::gesvd<JU, JVT, Tier>(static_cast<A&&>(a)...);
    }
};

// ---- the tier not right after the modes ---------------------------------------------------------
template <Transpose TR, class Cfg, OrthoAlgorithm Alg> struct routine_call<routine::ortho<TR, Cfg, Alg>> {
    template <class Tier, class... A> static BATCHLAS_DEVICE_INLINE decltype(auto) run(A&&... a) {
        return device::ortho<TR, Tier, Cfg, Alg>(static_cast<A&&>(a)...);
    }
};
template <class Cfg> struct routine_call<routine::ritz_values<Cfg>> {
    template <class Tier, class... A> static BATCHLAS_DEVICE_INLINE decltype(auto) run(A&&... a) {
        return device::ritz_values<Tier, Cfg>(static_cast<A&&>(a)...);
    }
};

}  // namespace batchlas::device
