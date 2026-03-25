#include <blas/extensions.hh>
#include <blas/device.hh>
#include <blas/functions.hh>
#include <blas/matrix.hh>
#include <internal/sytrd_blocked.hh>
#include <util/mempool.hh>

#include <batchlas/backend_config.h>
#include <batchlas/tuning_params.hh>

#include "../math-helpers.hh"
#include "../queue.hh"
#include "../util/template-instantiations.hh"

#include <algorithm>
#include <atomic>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <stdexcept>
#include <type_traits>

namespace batchlas {

namespace {

template <typename T>
struct HouseholderScalars {
    T tau{};
    T beta{};
    T scale{};
};

inline bool env_truthy(const char* v) {
    if (!v) return false;
    const std::string s(v);
    return (s == "1" || s == "true" || s == "TRUE" || s == "on" || s == "ON");
}

inline bool env_falsy(const char* v) {
    if (!v) return false;
    const std::string s(v);
    return (s == "0" || s == "false" || s == "FALSE" || s == "off" || s == "OFF");
}

enum class SytrdTrailingUpdateMode {
    Gemm,
    Syr2k,
};

inline SytrdTrailingUpdateMode sytrd_trailing_update_mode() {
    const char* v = std::getenv("BATCHLAS_SYTRD_TRAILING_UPDATE");
    if (!v) return SytrdTrailingUpdateMode::Gemm;

    const std::string s(v);
    if (s == "syr2k" || s == "SYR2K") {
        return SytrdTrailingUpdateMode::Syr2k;
    }
    return SytrdTrailingUpdateMode::Gemm;
}

inline int32_t latrd_lower_panel_wg_hint_override() {
    const char* v = std::getenv("BATCHLAS_LATRD_LOWER_PANEL_WG_HINT");
    if (!v || *v == '\0') {
        return tuning::latrd_lower_panel_wg_hint();
    }

    const int value = std::atoi(v);
    if (value == 64 || value == 128 || value == 256) {
        return value;
    }
    return tuning::latrd_lower_panel_wg_hint();
}

template <typename U>
inline U conj_if_needed(const U& x) {
    return batchlas::device::detail::conjugate_if_needed(x);
}

template <typename T>
inline typename base_type<T>::type abs2_if_complex(const T& x) {
    using Real = typename base_type<T>::type;
    if constexpr (internal::is_complex<T>::value) {
        const Real re = x.real();
        const Real im = x.imag();
        return re * re + im * im;
    } else {
        return x * x;
    }
}

template <typename Real>
inline Real sign_nonzero_real(Real x) {
    return (sycl::signbit(x) ? Real(-1) : Real(1));
}

template <typename Real>
inline Real sign_nonzero(const Real& x) {
    return sign_nonzero_real(x);
}

template <typename Real>
inline std::complex<Real> sign_nonzero(const std::complex<Real>& x) {
    const Real magnitude = sycl::hypot(x.real(), x.imag());
    return (magnitude == Real(0)) ? std::complex<Real>(1) : (x / magnitude);
}

template <typename Real>
inline HouseholderScalars<Real> compute_householder_scalars(const Real& alpha, Real xnorm) {
    HouseholderScalars<Real> result;
    if (xnorm == Real(0)) {
        result.beta = alpha;
        return result;
    }

    result.beta = -sign_nonzero(alpha) * Real(sycl::hypot(alpha, xnorm));
    result.tau = (result.beta - alpha) / result.beta;
    result.scale = Real(1) / (alpha - result.beta);
    return result;
}

template <typename Real>
inline HouseholderScalars<std::complex<Real>> compute_householder_scalars(const std::complex<Real>& alpha, Real xnorm) {
    using Complex = std::complex<Real>;

    HouseholderScalars<Complex> result;
    if (xnorm == Real(0) && alpha.imag() == Real(0)) {
        result.beta = alpha;
        return result;
    }

    const Real alpha_abs = sycl::hypot(alpha.real(), alpha.imag());
    const Real beta_abs = sycl::hypot(alpha_abs, xnorm);
    result.beta = -sign_nonzero(alpha) * Complex(beta_abs);
    result.tau = (result.beta - alpha) / result.beta;
    result.scale = Complex(1) / (alpha - result.beta);
    return result;
}

template <typename T>
inline T reduce_sum_group(const sycl::group<1>& g, T x) {
    if constexpr (internal::is_complex<T>::value) {
        using R = typename T::value_type;
        const R re = sycl::reduce_over_group(g, x.real(), sycl::plus<R>());
        const R im = sycl::reduce_over_group(g, x.imag(), sycl::plus<R>());
        return T(re, im);
    } else {
        return sycl::reduce_over_group(g, x, sycl::plus<T>());
    }
}

template <typename T>
inline typename base_type<T>::type reduce_sum_group_real(const sycl::group<1>& g,
                                                        typename base_type<T>::type x) {
    using R = typename base_type<T>::type;
    return sycl::reduce_over_group(g, x, sycl::plus<R>());
}

template <typename T>
class UpdateVWLowerSmallKernel;

template <typename T>
class SytrdLowerLocalSmallKernel;

template <typename T>
inline std::size_t sytrd_rank2k_local_size(const Queue& q) {
    const std::size_t max_work_group_size = q.device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE);
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>>) {
        return std::min<std::size_t>(256, max_work_group_size);
    }
    return std::min<std::size_t>(128, max_work_group_size);
}

template <typename T>
Event update_vw_lower_small(Queue& q,
                            const MatrixView<T, MatrixFormat::Dense>& v2,
                            const MatrixView<T, MatrixFormat::Dense>& w2,
                            const MatrixView<T, MatrixFormat::Dense>& a22) {
    const int batch = a22.batch_size();
    const std::size_t local_size = sytrd_rank2k_local_size<T>(q);

    (void)q->submit([&](sycl::handler& h) {
        const auto v_view = v2.kernel_view();
        const auto w_view = w2.kernel_view();
        const auto a_view = a22.kernel_view();

        h.parallel_for<UpdateVWLowerSmallKernel<T>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(batch) * local_size), sycl::range<1>(local_size)),
            [=](sycl::nd_item<1> item) {
                const int b = static_cast<int>(item.get_group(0));
                if (b >= batch) {
                    return;
                }

                const auto vb = v_view.batch_item(b);
                const auto wb = w_view.batch_item(b);
                const auto ab = a_view.batch_item(b);
                batchlas::device::her2k(item,
                                        vb,
                                        wb,
                                        ab,
                                        T(-1),
                                        T(1),
                                        Uplo::Lower,
                                        Transpose::NoTrans,
                                        batchlas::device::DeviceBlasPolicy::Auto);
            });
    });

    return q.get_event();
}

template <typename T>
Event sytrd_lower_local_small(Queue& q,
                              const MatrixView<T, MatrixFormat::Dense>& a,
                              const VectorView<T>& e,
                              const VectorView<T>& tau,
                              int n) {
    constexpr int WG = 64;
    const int lda = a.ld();
    const int stride_a = a.stride();
    T* a_ptr = a.data_ptr();
    T* e_ptr = e.data_ptr();
    T* tau_ptr = tau.data_ptr();
    const int stride_e = e.stride();
    const int stride_tau = tau.stride();
    const int batch = a.batch_size();

    (void)q->submit([&](sycl::handler& h) {
        // Allocate just enough local memory for the active n x n tile (plus w vector).
        auto A_local = sycl::local_accessor<T, 1>(sycl::range<1>(static_cast<size_t>(n) * static_cast<size_t>(n)), h);
        auto W_local = sycl::local_accessor<T, 1>(sycl::range<1>(static_cast<size_t>(n)), h);

        h.parallel_for<SytrdLowerLocalSmallKernel<T>>(
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(batch) * WG), sycl::range<1>(WG)),
            [=](sycl::nd_item<1> it) {
                const int b = static_cast<int>(it.get_group_linear_id());
                if (b >= batch) return;

                const int lane = static_cast<int>(it.get_local_linear_id());
                const sycl::group<1> g = it.get_group();
                T* Al_ptr = A_local.template get_multi_ptr<sycl::access::decorated::no>().get();
                T* W_ptr = W_local.template get_multi_ptr<sycl::access::decorated::no>().get();

                T* A = a_ptr + b * stride_a;
                T* Eb = e_ptr + b * stride_e;
                T* Taub = tau_ptr + b * stride_tau;

                const int ld_loc = n;
                auto Al = [&](int r, int c) -> T& { return A_local[r + c * ld_loc]; };

                if (lane < n) {
                    for (int c = 0; c < n; ++c) {
                        Al(lane, c) = A[lane + c * lda];
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);

                for (int k = 0; k < n - 1; ++k) {
                    using Real = typename base_type<T>::type;

                    const int alpha_row = k + 1;
                    const int x0 = k + 2;
                    const int tail = n - alpha_row;
                    const bool lane_in_tail = (lane >= alpha_row && lane < n);
                    const Real sumsq_partial = (lane >= x0 && lane < n) ? abs2_if_complex(Al(lane, k)) : Real(0);
                    const Real sumsq = reduce_sum_group_real<T>(g, sumsq_partial);

                    T tau_k = T(0);
                    T scale = T(0);
                    if (lane == alpha_row) {
                        const T alpha = Al(alpha_row, k);
                        const auto scalars = compute_householder_scalars(alpha, sycl::sqrt(sumsq));
                        Eb[k] = scalars.beta;
                        Taub[k] = scalars.tau;
                        Al(alpha_row, k) = T(1);
                        tau_k = scalars.tau;
                        scale = scalars.scale;
                    }
                    tau_k = sycl::group_broadcast(g, tau_k, sycl::id<1>(alpha_row));
                    scale = sycl::group_broadcast(g, scale, sycl::id<1>(alpha_row));

                    if (tau_k != T(0) && x0 < n) {
                        batchlas::device::scal(g, VectorView<T>(Al_ptr + x0 + k * ld_loc, n - x0), scale);
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    if constexpr (!internal::is_complex<T>::value) {
                        auto trailing_view = KernelMatrixView<T, MatrixFormat::Dense>(
                            Al_ptr + alpha_row + alpha_row * ld_loc,
                            tail,
                            tail,
                            ld_loc,
                            ld_loc * n);
                        auto v_view = VectorView<T>(Al_ptr + alpha_row + k * ld_loc, tail);
                        auto w_view = VectorView<T>(W_ptr + alpha_row, tail);
                        batchlas::device::hemv(it,
                                               trailing_view,
                                               v_view,
                                               w_view,
                                               tau_k,
                                               T(0),
                                               Uplo::Lower,
                                               batchlas::device::DeviceBlasPolicy::Auto);
                    } else if (lane < n) {
                        T w = T(0);
                        if (lane_in_tail) {
                            for (int c = alpha_row; c < n; ++c) {
                                const T vc = (c == alpha_row) ? T(1) : Al(c, k);
                                w += Al(lane, c) * vc;
                            }
                            w *= tau_k;
                        }
                        W_local[lane] = w;
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    const T vr = lane_in_tail ? ((lane == alpha_row) ? T(1) : Al(lane, k)) : T(0);
                    const T dot_partial = lane_in_tail ? (conj_if_needed(vr) * W_local[lane]) : T(0);
                    const T dot = reduce_sum_group(g, dot_partial);

                    const T alpha2 = T(-0.5) * tau_k * dot;
                    if (lane_in_tail) {
                        W_local[lane] += alpha2 * vr;
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    auto trailing_view = KernelMatrixView<T, MatrixFormat::Dense>(
                        Al_ptr + alpha_row + alpha_row * ld_loc,
                        tail,
                        tail,
                        ld_loc,
                        ld_loc * n);
                    auto v_matrix = KernelMatrixView<T, MatrixFormat::Dense>(
                        Al_ptr + alpha_row + k * ld_loc,
                        tail,
                        1,
                        ld_loc,
                        ld_loc);
                    auto w_matrix = KernelMatrixView<T, MatrixFormat::Dense>(
                        W_ptr + alpha_row,
                        tail,
                        1,
                        tail,
                        tail);
                    batchlas::device::her2k(it,
                                            v_matrix,
                                            w_matrix,
                                            trailing_view,
                                            T(-1),
                                            T(1),
                                            Uplo::Lower,
                                            Transpose::NoTrans,
                                            batchlas::device::DeviceBlasPolicy::Auto);
                    if constexpr (internal::is_complex<T>::value) {
                        it.barrier(sycl::access::fence_space::local_space);
                        if (lane_in_tail) {
                            for (int c = alpha_row; c < lane; ++c) {
                                Al(c, lane) = conj_if_needed(Al(lane, c));
                            }
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }

                if (lane < n) {
                    for (int c = 0; c < n; ++c) {
                        A[lane + c * lda] = Al(lane, c);
                    }
                }
            });
    });

    return q.get_event();
}


template <typename T>
class RestoreTridiagKernel;

template <typename T>
Event restore_tridiag_lower(Queue& q,
                                 const MatrixView<T, MatrixFormat::Dense>& a,
                                 const VectorView<T>& d,
                                 const VectorView<T>& e,
                                 int n) {
    const int lda = a.ld();
    const int stride_a = a.stride();
    T* a_ptr = a.data_ptr();
    T* d_ptr = d.data_ptr();
    T* e_ptr = e.data_ptr();
    const int stride_d = d.stride();
    const int stride_e = e.stride();
    const int batch = a.batch_size();

    (void)q->submit([&](sycl::handler& h) {
        h.parallel_for<RestoreTridiagKernel<T>>(sycl::range<2>(static_cast<size_t>(batch), static_cast<size_t>(n)),
                                               [=](sycl::id<2> idx) {
                                                   const int b = static_cast<int>(idx[0]);
                                                   const int i = static_cast<int>(idx[1]);
                                                   T* A = a_ptr + b * stride_a;
                                                   T* Db = d_ptr + b * stride_d;
                                                   T* Eb = e_ptr + b * stride_e;

                                                   if (i < n) {
                                                       Db[i] = A[i + i * lda];
                                                   }
                                                   if (i < n - 1) {
                                                       const T ei = Eb[i];
                                                       A[(i + 1) + i * lda] = ei;
                                                       A[i + (i + 1) * lda] = conj_if_needed(ei);
                                                   }
                                               });
    });

    return q.get_event();
}

template <typename T>
inline void validate_sytrd_dims(const MatrixView<T, MatrixFormat::Dense>& a,
                               const VectorView<T>& d,
                               const VectorView<T>& e,
                               const VectorView<T>& tau) {
    if (a.rows() != a.cols()) {
        throw std::invalid_argument("sytrd_blocked: A must be square");
    }
    const int n = a.rows();
    if (d.size() != n || e.size() != std::max(0, n - 1) || tau.size() != std::max(0, n - 1)) {
        throw std::invalid_argument("sytrd_blocked: invalid d/e/tau sizes");
    }
    if (a.batch_size() != d.batch_size() || a.batch_size() != e.batch_size() || a.batch_size() != tau.batch_size()) {
        throw std::invalid_argument("sytrd_blocked: batch size mismatch");
    }
    if (a.batch_size() < 1) {
        throw std::invalid_argument("sytrd_blocked: invalid batch size");
    }
}

} // namespace

inline int resolved_nb_sytrd(int32_t block_size, int compiled_nb) {
    if (compiled_nb > 0) {
        return compiled_nb;
    }
    return std::max<int>(1, block_size);
}

template <Backend B, typename T, int NB>
size_t sytrd_blocked_buffer_size_impl(Queue& ctx,
                                      const MatrixView<T, MatrixFormat::Dense>& a,
                                      const VectorView<T>& d,
                                      const VectorView<T>& e,
                                      const VectorView<T>& tau,
                                      Uplo uplo,
                                      int32_t block_size) {
    (void)uplo;
    (void)d;
    (void)e;
    (void)tau;

    const int n = a.rows();
    const int batch = a.batch_size();
    const int nb = resolved_nb_sytrd(block_size, NB);

    size_t size = 0;
    size += BumpAllocator::allocation_size<T>(ctx, static_cast<size_t>(n) * static_cast<size_t>(nb) * static_cast<size_t>(batch));
    return size;
}

template <Backend B, typename T, int NB>
Event sytrd_blocked_impl(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& a_in,
                         const VectorView<T>& d_out,
                         const VectorView<T>& e_out,
                         const VectorView<T>& tau_out,
                         Uplo uplo,
                         const Span<std::byte>& ws,
                         int32_t block_size) {
    const int n = a_in.rows();
    const int batch = a_in.batch_size();
    const int nb = resolved_nb_sytrd(block_size, NB);

    if (uplo != Uplo::Lower) {
        throw std::runtime_error("sytrd_blocked: only Uplo::Lower is implemented");
    }

    if (n <= 32) {
        bool has_sg32 = false;
        try {
            const auto sizes = ctx->get_device().get_info<sycl::info::device::sub_group_sizes>();
            has_sg32 = std::find(sizes.begin(), sizes.end(), 32) != sizes.end();
        } catch (...) {
            has_sg32 = false;
        }
        if (has_sg32) {
            return sytrd_cta<B, T>(ctx, a_in, d_out, e_out, tau_out, uplo, Span<std::byte>(), /*cta_wg_size_multiplier=*/1);
        }
    }

    if (n <= 64) {
        const bool force_local = env_truthy(std::getenv("BATCHLAS_SYTRD_FORCE_LOCAL_SMALL"));
        const bool debug_small = env_truthy(std::getenv("BATCHLAS_DEBUG_SYTRD_SMALL"));

        const size_t local_mem_bytes = ctx.device().get_property(DeviceProperty::LOCAL_MEM_SIZE);
        const size_t max_wg_size = ctx.device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE);

        constexpr size_t WG = 64;
        const size_t elems = static_cast<size_t>(n) * static_cast<size_t>(n) + static_cast<size_t>(n);
        const size_t bytes_needed = elems * sizeof(T);

        const bool props_ok = (max_wg_size >= WG && (local_mem_bytes == 0 || local_mem_bytes >= bytes_needed));
        if (debug_small) {
            static std::atomic<bool> printed{false};
            if (!printed.exchange(true)) {
                std::cerr << "[sytrd_blocked] n=" << n << " batch=" << batch << " nb=" << nb
                          << " max_wg_size=" << max_wg_size << " local_mem_bytes=" << local_mem_bytes
                          << " bytes_needed=" << bytes_needed << " props_ok=" << (props_ok ? 1 : 0)
                          << " force_local=" << (force_local ? 1 : 0) << "\n";
            }
        }

        constexpr bool allow_local_small = (B != Backend::CUDA);
        const bool auto_local = allow_local_small && std::is_same_v<T, float>;
        if ((auto_local && props_ok) || (allow_local_small && force_local && props_ok)) {
            {
                BATCHLAS_KERNEL_TRACE_SCOPE("sytrd_blocked.local_small");
                (void)sytrd_lower_local_small<T>(ctx, a_in, e_out, tau_out, n);
            }
            {
                BATCHLAS_KERNEL_TRACE_SCOPE("sytrd_blocked.restore_tridiag");
                (void)restore_tridiag_lower<T>(ctx, a_in, d_out, e_out, n);
            }
            return ctx.get_event();
        }
    }

    MatrixView<T, MatrixFormat::Dense> A = a_in;
    VectorView<T> D = d_out;
    VectorView<T> E = e_out;
    VectorView<T> TAU = tau_out;

    BumpAllocator pool(ws);
    auto Wbuf = pool.allocate<T>(ctx, static_cast<size_t>(n) * static_cast<size_t>(nb) * static_cast<size_t>(batch));
    MatrixView<T, MatrixFormat::Dense> Wmat(Wbuf.data(), n, nb, n, n * nb, batch);

    const int k = n - 1;
    const char* fuse_env = std::getenv("BATCHLAS_SYTRD_FUSE_PANEL_UPDATE");
    const bool fuse_override_on = env_truthy(fuse_env);
    const bool fuse_override_off = env_falsy(fuse_env);
    const bool fuse_default = false;
    const bool enable_fused_panel_update = fuse_override_on || (!fuse_override_off && fuse_default);
    constexpr bool allow_syr2k_experiment = (B == Backend::CUDA) && std::is_same_v<T, float>;
    const bool use_syr2k_trailing_update = allow_syr2k_experiment &&
                                           (sytrd_trailing_update_mode() == SytrdTrailingUpdateMode::Syr2k);

    for (int j0 = 0; j0 < k; j0 += nb) {
        const int ib = std::min(nb, k - j0);

        const int j2 = j0 + ib;
        const int n2 = n - j2;
        const bool fuse_this_panel = n2 > 0 &&
                         (fuse_override_on ||
                          (enable_fused_panel_update && n2 <= 128));

        {
            BATCHLAS_KERNEL_TRACE_SCOPE(fuse_this_panel ? "sytrd_blocked.panel_fused"
                                                        : "sytrd_blocked.panel_only");
            auto A_panel = A({j0, SliceEnd()}, {j0, SliceEnd()});
            auto E_panel = E(Slice(j0, j0 + ib));
            auto TAU_panel = TAU(Slice(j0, j0 + ib));
            auto W_panel = Wmat({j0, SliceEnd()}, {0, ib});
            (void)latrd_lower_panel<B, T>(ctx,
                                          A_panel,
                                          E_panel,
                                          TAU_panel,
                                          W_panel,
                                          latrd_lower_panel_wg_hint_override(),
                                          fuse_this_panel);
        }

        if (n2 > 0 && !fuse_this_panel) {
            auto A22 = A({j2, SliceEnd()}, {j2, SliceEnd()});
            auto V2 = A({j2, SliceEnd()}, {j0, j0 + ib});
            auto W2 = Wmat({j2, SliceEnd()}, {0, ib});

            if (n2 <= 128) {
                BATCHLAS_KERNEL_TRACE_SCOPE("sytrd_blocked.update_vw_small");
                (void)update_vw_lower_small<T>(ctx, V2, W2, A22);
            } else {
                if constexpr (allow_syr2k_experiment) {
                    if (use_syr2k_trailing_update) {
                        {
                            BATCHLAS_KERNEL_TRACE_SCOPE("sytrd_blocked.update_vw_syr2k");
                            syr2k<B>(ctx, V2, W2, A22, T(-1), T(1), Uplo::Lower, Transpose::NoTrans);
                        }
                        {
                            BATCHLAS_KERNEL_TRACE_SCOPE("sytrd_blocked.update_vw_syr2k_symmetrize");
                            A22.symmetrize(ctx, Uplo::Lower);
                        }
                    } else {
                        {
                            BATCHLAS_KERNEL_TRACE_SCOPE("sytrd_blocked.update_vw_gemm_vw");
                            gemm<B>(ctx, V2, W2, A22, T(-1), T(1), Transpose::NoTrans, Transpose::ConjTrans);
                        }
                        {
                            BATCHLAS_KERNEL_TRACE_SCOPE("sytrd_blocked.update_vw_gemm_wv");
                            gemm<B>(ctx, W2, V2, A22, T(-1), T(1), Transpose::NoTrans, Transpose::ConjTrans);
                        }
                    }
                } else {
                    {
                        BATCHLAS_KERNEL_TRACE_SCOPE("sytrd_blocked.update_vw_gemm_vw");
                        gemm<B>(ctx, V2, W2, A22, T(-1), T(1), Transpose::NoTrans, Transpose::ConjTrans);
                    }
                    {
                        BATCHLAS_KERNEL_TRACE_SCOPE("sytrd_blocked.update_vw_gemm_wv");
                        gemm<B>(ctx, W2, V2, A22, T(-1), T(1), Transpose::NoTrans, Transpose::ConjTrans);
                    }
                }
            }
        }
    }

    {
        BATCHLAS_KERNEL_TRACE_SCOPE("sytrd_blocked.restore_tridiag");
        (void)restore_tridiag_lower<T>(ctx, A, D, E, n);
    }

    return ctx.get_event();
}

template <Backend B, typename T>
size_t sytrd_blocked_buffer_size(Queue& ctx,
                                 const MatrixView<T, MatrixFormat::Dense>& a,
                                 const VectorView<T>& d,
                                 const VectorView<T>& e,
                                 const VectorView<T>& tau,
                                 Uplo uplo,
                                 int32_t block_size) {
    validate_sytrd_dims(a, d, e, tau);

    const int nb = std::max<int>(1, block_size);
    switch (nb) {
        case 8:
            return sytrd_blocked_buffer_size_impl<B, T, 8>(ctx, a, d, e, tau, uplo, block_size);
        case 16:
            return sytrd_blocked_buffer_size_impl<B, T, 16>(ctx, a, d, e, tau, uplo, block_size);
        case 32:
            return sytrd_blocked_buffer_size_impl<B, T, 32>(ctx, a, d, e, tau, uplo, block_size);
        case 64:
            return sytrd_blocked_buffer_size_impl<B, T, 64>(ctx, a, d, e, tau, uplo, block_size);
        default:
            return sytrd_blocked_buffer_size_impl<B, T, -1>(ctx, a, d, e, tau, uplo, block_size);
    }
}

template <Backend B, typename T>
Event sytrd_blocked(Queue& ctx,
                    const MatrixView<T, MatrixFormat::Dense>& a_in,
                    const VectorView<T>& d_out,
                    const VectorView<T>& e_out,
                    const VectorView<T>& tau_out,
                    Uplo uplo,
                    const Span<std::byte>& ws,
                    int32_t block_size) {
    validate_sytrd_dims(a_in, d_out, e_out, tau_out);

    if (!ctx.in_order()) {
        throw std::runtime_error("sytrd_blocked: requires an in-order Queue");
    }

    const int nb = std::max<int>(1, block_size);
    switch (nb) {
        case 8:
            return sytrd_blocked_impl<B, T, 8>(ctx, a_in, d_out, e_out, tau_out, uplo, ws, block_size);
        case 16:
            return sytrd_blocked_impl<B, T, 16>(ctx, a_in, d_out, e_out, tau_out, uplo, ws, block_size);
        case 32:
            return sytrd_blocked_impl<B, T, 32>(ctx, a_in, d_out, e_out, tau_out, uplo, ws, block_size);
        case 64:
            return sytrd_blocked_impl<B, T, 64>(ctx, a_in, d_out, e_out, tau_out, uplo, ws, block_size);
        default:
            return sytrd_blocked_impl<B, T, -1>(ctx, a_in, d_out, e_out, tau_out, uplo, ws, block_size);
    }
}

#define SYTRD_BLOCKED_INSTANTIATE(back, fp) \
    template Event sytrd_blocked<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        Uplo, \
        const Span<std::byte>&, \
        int32_t); \
    template size_t sytrd_blocked_buffer_size<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        Uplo, \
        int32_t);

#define SYTRD_BLOCKED_INSTANTIATE_FOR_BACKEND(back) \
    BATCHLAS_FOR_EACH_SCALAR_TYPE_1(SYTRD_BLOCKED_INSTANTIATE, back)

#if BATCHLAS_HAS_CUDA_BACKEND
SYTRD_BLOCKED_INSTANTIATE_FOR_BACKEND(Backend::CUDA)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
SYTRD_BLOCKED_INSTANTIATE_FOR_BACKEND(Backend::ROCM)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
SYTRD_BLOCKED_INSTANTIATE_FOR_BACKEND(Backend::NETLIB)
#endif

#undef SYTRD_BLOCKED_INSTANTIATE_FOR_BACKEND
#undef SYTRD_BLOCKED_INSTANTIATE

} // namespace batchlas
