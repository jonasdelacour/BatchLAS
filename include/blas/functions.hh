#ifndef BATCHLAS_BLAS_CUBLAS_MATRIXVIEW_HH
#define BATCHLAS_BLAS_CUBLAS_MATRIXVIEW_HH

#include <util/sycl-device-queue.hh>
#include <blas/matrix.hh>
#include <util/sycl-span.hh>
#include <complex>
#include <sycl/sycl.hpp>

// Include all function headers
#include <blas/functions/gemm.hh>
#include <blas/functions/gemv.hh>
#include <blas/functions/geqrf.hh>
#include <blas/functions/getrf.hh>
#include <blas/functions/getri.hh>
#include <blas/functions/getrs.hh>
#include <blas/functions/orgqr.hh>
#include <blas/functions/ormqr.hh>
#include <blas/functions/potrf.hh>
#include <blas/functions/spmm.hh>
#include <blas/functions/syev.hh>
#include <blas/functions/trmm.hh>
#include <blas/functions/trsm.hh>

namespace batchlas {


} // namespace batchlas

#endif // BATCHLAS_BLAS_CUBLAS_MATRIXVIEW_HH
