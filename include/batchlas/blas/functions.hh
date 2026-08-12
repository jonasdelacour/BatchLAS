#ifndef BATCHLAS_BLAS_CUBLAS_MATRIXVIEW_HH
#define BATCHLAS_BLAS_CUBLAS_MATRIXVIEW_HH

#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-span.hh>
#include <complex>

// Include all function headers
#include <batchlas/blas/functions/gemm.hh>
#include <batchlas/blas/functions/gemv.hh>
#include <batchlas/blas/functions/geqrf.hh>
#include <batchlas/blas/functions/getrf.hh>
#include <batchlas/blas/functions/getri.hh>
#include <batchlas/blas/functions/getrs.hh>
#include <batchlas/blas/functions/gesvd.hh>
#include <batchlas/blas/functions/hemm.hh>
#include <batchlas/blas/functions/her2k.hh>
#include <batchlas/blas/functions/herk.hh>
#include <batchlas/blas/functions/orgqr.hh>
#include <batchlas/blas/functions/ormqr.hh>
#include <batchlas/blas/functions/potrf.hh>
#include <batchlas/blas/functions/spmm.hh>
#include <batchlas/blas/functions/symm.hh>
#include <batchlas/blas/functions/syrk.hh>
#include <batchlas/blas/functions/syr2k.hh>
#include <batchlas/blas/functions/iluk.hh>
#include <batchlas/blas/functions/syev.hh>
#include <batchlas/blas/functions/trmm.hh>
#include <batchlas/blas/functions/trsm.hh>

namespace batchlas {


} // namespace batchlas

#endif // BATCHLAS_BLAS_CUBLAS_MATRIXVIEW_HH

// Option-struct spellings of everything declared above.
#include <batchlas/blas/options.hh>
