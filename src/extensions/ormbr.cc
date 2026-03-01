#include <blas/extensions.hh>
#include <internal/ormbr.hh>
#include <internal/ormqr_blocked.hh>
#include <batchlas/backend_config.h>

#include "../math-helpers.hh"
#include "../queue.hh"

#include <cctype>
#include <complex>
#include <cstdint>
#include <stdexcept>

namespace batchlas {

namespace {

inline char upper_ascii(char c) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
}

template <typename T>
inline T conj_if_needed(const T& x) {
    if constexpr (internal::is_complex<T>::value) {
        return T(x.real(), -x.imag());
    } else {
        return x;
    }
}

template <typename T>
class OrmbrPKernel;

template <typename T>
Event ormbr_apply_p_unblocked(Queue& ctx,
                              const MatrixView<T, MatrixFormat::Dense>& a,
                              const VectorView<T>& tau,
                              const MatrixView<T, MatrixFormat::Dense>& c,
                              Side side,
                              Transpose trans) {
    auto& c_mut = const_cast<MatrixView<T, MatrixFormat::Dense>&>(c);
    const int32_t n = static_cast<int32_t>(a.rows());
    const int32_t k = std::max<int32_t>(0, std::min<int32_t>(a.rows(), a.cols()) - 1);

    const bool apply_conj_trans =
        (trans == Transpose::ConjTrans) ||
        (trans == Transpose::Trans && !internal::is_complex<T>::value);
    const bool forward = !apply_conj_trans;

    ctx->submit([&](sycl::handler& cgh) {
        auto A = a.kernel_view();
        auto C = c_mut.kernel_view();
        auto TAU = tau;

        cgh.parallel_for<OrmbrPKernel<T>>(sycl::range<1>(static_cast<size_t>(a.batch_size())), [=](sycl::id<1> tid) {
            const int32_t b = static_cast<int32_t>(tid[0]);

            auto apply_reflector = [&](int32_t i) {
                if (i < 0 || i >= k) return;
                const int32_t start = i + 1;
                const int32_t len = n - start;
                if (len <= 0) return;

                T tau_i = TAU(i, b);
                if (apply_conj_trans) {
                    tau_i = conj_if_needed(tau_i);
                }
                if (tau_i == T(0)) return;

                if (side == Side::Right) {
                    const int32_t m = static_cast<int32_t>(C.rows());
                    const int32_t ncols = static_cast<int32_t>(C.cols());
                    if (ncols < n) return;

                    for (int32_t r = 0; r < m; ++r) {
                        T dot = T(0);
                        for (int32_t t = 0; t < len; ++t) {
                            const int32_t col = start + t;
                            const T v = (t == 0) ? T(1) : A(i, col, b);
                            dot += C(r, col, b) * conj_if_needed(v);
                        }
                        dot *= tau_i;
                        for (int32_t t = 0; t < len; ++t) {
                            const int32_t col = start + t;
                            const T v = (t == 0) ? T(1) : A(i, col, b);
                            C(r, col, b) -= dot * v;
                        }
                    }
                } else {
                    const int32_t mrows = static_cast<int32_t>(C.rows());
                    const int32_t ncols = static_cast<int32_t>(C.cols());
                    if (mrows < n) return;

                    for (int32_t col = 0; col < ncols; ++col) {
                        T dot = T(0);
                        for (int32_t t = 0; t < len; ++t) {
                            const int32_t row = start + t;
                            const T v = (t == 0) ? T(1) : A(i, row, b);
                            dot += conj_if_needed(v) * C(row, col, b);
                        }
                        dot *= tau_i;
                        for (int32_t t = 0; t < len; ++t) {
                            const int32_t row = start + t;
                            const T v = (t == 0) ? T(1) : A(i, row, b);
                            C(row, col, b) -= v * dot;
                        }
                    }
                }
            };

            if (forward) {
                for (int32_t i = 0; i < k; ++i) {
                    apply_reflector(i);
                }
            } else {
                for (int32_t i = k - 1; i >= 0; --i) {
                    apply_reflector(i);
                }
            }
        });
    });

    return ctx.get_event();
}

template <typename T>
inline void validate_ormbr_dims(const MatrixView<T, MatrixFormat::Dense>& a,
                                const VectorView<T>& tau,
                                const MatrixView<T, MatrixFormat::Dense>& c,
                                char vect,
                                Side side) {
    if (a.rows() != a.cols()) {
        throw std::invalid_argument("ormbr: current implementation supports square A only");
    }
    if (a.batch_size() != c.batch_size() || tau.batch_size() != a.batch_size()) {
        throw std::invalid_argument("ormbr: batch size mismatch");
    }
    if (a.batch_size() < 1) {
        throw std::invalid_argument("ormbr: invalid batch size");
    }

    const int32_t n = static_cast<int32_t>(a.rows());
    const int32_t k = std::min<int32_t>(a.rows(), a.cols());
    const int32_t nq = (side == Side::Left) ? static_cast<int32_t>(c.rows()) : static_cast<int32_t>(c.cols());
    if (nq != n) {
        throw std::invalid_argument("ormbr: expected nq == A.rows() for square-path staging");
    }

    const char v = upper_ascii(vect);
    if (v != 'Q' && v != 'P') {
        throw std::invalid_argument("ormbr: vect must be 'Q' or 'P'");
    }

    const int32_t need_tau = (v == 'Q') ? k : std::max<int32_t>(0, k - 1);
    if (tau.inc() != 1) {
        throw std::invalid_argument("ormbr: tau must be unit-stride");
    }
    if (tau.stride() != tau.size()) {
        throw std::invalid_argument("ormbr: tau must be tightly packed by batch");
    }
    if (tau.size() < static_cast<size_t>(need_tau)) {
        throw std::invalid_argument("ormbr: tau span too small");
    }
}

} // namespace

template <Backend B, typename T>
Event ormbr(Queue& ctx,
            const MatrixView<T, MatrixFormat::Dense>& a,
            const VectorView<T>& tau,
            const MatrixView<T, MatrixFormat::Dense>& c,
            char vect,
            Side side,
            Transpose trans,
            const Span<std::byte>& ws,
            int32_t block_size) {
    validate_ormbr_dims(a, tau, c, vect, side);

    const char v = upper_ascii(vect);
    if (v == 'Q') {
        const size_t tau_elems = tau.data().size();
        Span<T> tau_span(const_cast<T*>(tau.data_ptr()), tau_elems);
        Span<std::byte> ws_mut(const_cast<std::byte*>(ws.data()), ws.size());
        return ormqr_blocked<B, T>(ctx, a, c, side, trans, tau_span, ws_mut, block_size);
    }

    return ormbr_apply_p_unblocked<T>(ctx, a, tau, c, side, trans);
}

template <Backend B, typename T>
size_t ormbr_buffer_size(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& a,
                         const VectorView<T>& tau,
                         const MatrixView<T, MatrixFormat::Dense>& c,
                         char vect,
                         Side side,
                         Transpose trans,
                         int32_t block_size) {
    validate_ormbr_dims(a, tau, c, vect, side);

    const char v = upper_ascii(vect);
    if (v == 'Q') {
        const size_t tau_elems = tau.data().size();
        Span<T> tau_span(const_cast<T*>(tau.data_ptr()), tau_elems);
        return ormqr_blocked_buffer_size<B, T>(ctx, a, c, side, trans, tau_span, block_size);
    }

    return 0;
}

#define ORMBR_INSTANTIATE(back, fp) \
    template Event ormbr<back, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const VectorView<fp>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        char, \
        Side, \
        Transpose, \
        const Span<std::byte>&, \
        int32_t); \
    template size_t ormbr_buffer_size<back, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const VectorView<fp>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        char, \
        Side, \
        Transpose, \
        int32_t);

#if BATCHLAS_HAS_CUDA_BACKEND
ORMBR_INSTANTIATE(Backend::CUDA, float)
ORMBR_INSTANTIATE(Backend::CUDA, double)
ORMBR_INSTANTIATE(Backend::CUDA, std::complex<float>)
ORMBR_INSTANTIATE(Backend::CUDA, std::complex<double>)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
ORMBR_INSTANTIATE(Backend::ROCM, float)
ORMBR_INSTANTIATE(Backend::ROCM, double)
ORMBR_INSTANTIATE(Backend::ROCM, std::complex<float>)
ORMBR_INSTANTIATE(Backend::ROCM, std::complex<double>)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
ORMBR_INSTANTIATE(Backend::NETLIB, float)
ORMBR_INSTANTIATE(Backend::NETLIB, double)
ORMBR_INSTANTIATE(Backend::NETLIB, std::complex<float>)
ORMBR_INSTANTIATE(Backend::NETLIB, std::complex<double>)
#endif

#undef ORMBR_INSTANTIATE

} // namespace batchlas
