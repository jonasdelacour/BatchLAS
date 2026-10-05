// The public ormqr entry point's instantiations, outside the vendor TUs.
//
// ormqr differs from every other op S5 moves. Its public template is already
// DEFINED in a header -- functions/ormqr.hh, forwarding to its *_dispatch,
// which resolves a Route and may call a native kernel instead of the vendor.
// So there was never a definition to relocate.
//
// What did live in the vendor TUs was their explicit INSTANTIATION, which is
// just as binding: with the instantiation in cusolver.cc, `syev<Backend::CUDA,
// float>` had no out-of-line symbol in a build without cuSOLVER, even though
// every line of code implementing it was vendor-independent. Moving the
// instantiation here is the whole change.
//
// The backend::*_vendor instantiations stay behind in the vendor TUs, as for
// every other op. syev has since moved to src/ops/syev/syev.cc (flat selection).

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/ormqr.hh>

#include "../../util/template-instantiations.hh"

#include <complex>

namespace batchlas {

#define OP_INSTANTIATE(OP, B_, fp) BATCHLAS_INSTANTIATE(sig::OP<fp>, OP, B_, fp)

#define EIGEN_ONE(B_, fp)                          \
    OP_INSTANTIATE(ormqr, B_, fp)                  \
    OP_INSTANTIATE(ormqr_buffer_size, B_, fp)

#define EIGEN_ALL(B_)                              \
    EIGEN_ONE(B_, float)                           \
    EIGEN_ONE(B_, double)                          \
    EIGEN_ONE(B_, std::complex<float>)             \
    EIGEN_ONE(B_, std::complex<double>)

// Keyed on the DEVICE FAMILY, not on the vendor library. The bodies above
// compile to a throw when the library is absent, so the public entry point
// exists as a symbol in every build that has the device -- which is exactly what
// stopped being true when the definitions lived in the vendor TUs.
#if BATCHLAS_HAS_CUDA_BACKEND
EIGEN_ALL(Backend::CUDA)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
EIGEN_ALL(Backend::ROCM)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
EIGEN_ALL(Backend::NETLIB)
#endif

#undef EIGEN_ALL
#undef EIGEN_ONE
#undef OP_INSTANTIATE

}  // namespace batchlas
