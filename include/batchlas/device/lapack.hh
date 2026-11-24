// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @ingroup api_device
/// @brief Every LAPACK-level device routine in one include, on top of `<batchlas/device.hh>`.
///
/// Including this header gives every routine of @ref api_device_lapack_chol,
/// @ref api_device_lapack_lu, @ref api_device_lapack_ldl, @ref api_device_lapack_qr,
/// @ref api_device_lapack_eig and @ref api_device_lapack_svd, together with the core
/// (`<batchlas/device.hh>`) and the in-group level 3 that the blocked tiers use
/// (`<batchlas/device/level3/group.hh>`). To compile less, include one routine's header
/// instead, e.g. `<batchlas/device/lapack/potrf.hh>`; each one includes what it needs, but
/// `launch<Routine, Tier, T>::run` needs this header (its `routine_call`s are in `lapack_run.hh`).
/// `<batchlas/device/sparse.hh>` is not included.

#include <batchlas/device.hh>

#include <batchlas/device/lapack/bdsdc.hh>
#include <batchlas/device/lapack/bdsqr.hh>
#include <batchlas/device/lapack/gebrd.hh>
#include <batchlas/device/lapack/geqrf.hh>
#include <batchlas/device/lapack/gesv.hh>
#include <batchlas/device/lapack/gesvd.hh>
#include <batchlas/device/lapack/getrf.hh>
#include <batchlas/device/lapack/getri.hh>
#include <batchlas/device/lapack/getrs.hh>
#include <batchlas/device/lapack/inv.hh>
#include <batchlas/device/lapack/jacobi.hh>
#include <batchlas/device/lapack/laed2.hh>
#include <batchlas/device/lapack/laed3.hh>
#include <batchlas/device/lapack/laed4.hh>
#include <batchlas/device/lapack/lagtf.hh>
#include <batchlas/device/lapack/lagts.hh>
#include <batchlas/device/lapack/larfb.hh>
#include <batchlas/device/lapack/larft.hh>
#include <batchlas/device/lapack/orgqr.hh>
#include <batchlas/device/lapack/ormbr.hh>
#include <batchlas/device/lapack/ormqr.hh>
#include <batchlas/device/lapack/ortho.hh>
#include <batchlas/device/lapack/posv.hh>
#include <batchlas/device/lapack/potrf.hh>
#include <batchlas/device/lapack/potri.hh>
#include <batchlas/device/lapack/potrs.hh>
#include <batchlas/device/lapack/ritz_values.hh>
#include <batchlas/device/lapack/sb2st.hh>
#include <batchlas/device/lapack/stebz.hh>
#include <batchlas/device/lapack/stedc.hh>
#include <batchlas/device/lapack/stein.hh>
#include <batchlas/device/lapack/steqr.hh>
#include <batchlas/device/lapack/sy2sb.hh>
#include <batchlas/device/lapack/syev.hh>
#include <batchlas/device/lapack/syevx.hh>
#include <batchlas/device/lapack/sygst.hh>
#include <batchlas/device/lapack/sysv.hh>
#include <batchlas/device/lapack/sytrd.hh>
#include <batchlas/device/lapack/sytrf.hh>
#include <batchlas/device/lapack/sytrs.hh>

#include <batchlas/device/lapack_run.hh>
