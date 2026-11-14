// SPDX-License-Identifier: MIT
#pragma once
// Every LAPACK-level device op (include/batchlas/device/lapack/*.hh) over the core umbrella.

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
