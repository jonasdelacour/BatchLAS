#include <blas/extensions.hh>
#include <batchlas/backend_config.h>
#include <util/group-invoke.hh>

#include "../math-helpers.hh"
#include "../queue.hh"
#include "../util/template-instantiations.hh"

#include <array>
#include <numeric>
#include <stdexcept>

using namespace sycl::ext::oneapi;

namespace batchlas {

namespace {

template <typename T, typename Group>
inline T group_reduce_sum_select_from_group(const Group& g, T v) {
    const uint32_t lanes = static_cast<uint32_t>(g.get_local_linear_range());
    (void)lanes;

    if constexpr (internal::is_complex<T>::value) {
        using Real = typename base_type<T>::type;
        Real re = v.real();
        Real im = v.imag();
        for (uint32_t offset = lanes / 2; offset > 0; offset >>= 1) {
            re += sycl::permute_group_by_xor(g, re, offset);
            im += sycl::permute_group_by_xor(g, im, offset);
        }
        return T(re, im);
    } else {
        for (uint32_t offset = lanes / 2; offset > 0; offset >>= 1) {
            v += sycl::permute_group_by_xor(g, v, offset);
        }
        return v;
    }
}

template <typename T, typename Partition>
inline T larfg_small(const Partition& part,
                     int32_t len,
                     int32_t lane,
                     int32_t alpha_lane,
                     T& alpha,
                     T& x,
                     bool x_active) {
    using Real = typename base_type<T>::type;

    const Real xsq = x_active ? (x * x) : Real(0);
    const Real sumsq = group_reduce_sum_select_from_group(part, xsq);
    const Real xnorm = invoke_one_broadcast(part, [&]() {
        return sycl::sqrt(sumsq);
    });

    const T alpha_leader = sycl::select_from_group(part, alpha, static_cast<uint32_t>(alpha_lane));

    const auto [beta_b, tau_b, scale_b] = invoke_one_broadcast(part, [&]() {
        if (len <= 1) {
            return std::array<T, 3>{alpha_leader, T(0), T(0)};
        }

        const auto scalars = internal::larfg(alpha_leader, xnorm, len);
        return std::array<T, 3>{scalars.beta, scalars.tau, scalars.scale};
    });

    if (lane == alpha_lane) {
        alpha = beta_b;
    } else if (x_active && tau_b != T(0)) {
        x *= scale_b;
    }

    return tau_b;
}

template <typename T, size_t P>
class GebrdCTAKernel;

template <typename T, size_t P>
inline void gebrd_cta_impl(Queue& ctx,
                           MatrixView<T, MatrixFormat::Dense>& a,
                           const VectorView<typename base_type<T>::type>& d,
                           const VectorView<typename base_type<T>::type>& e,
                           const VectorView<T>& tauq,
                           const VectorView<T>& taup,
                           int32_t n,
                           size_t cta_wg_size_multiplier) {
    using Real = typename base_type<T>::type;

    const auto batch_size = a.batch_size();
    if (n < 1 || n > static_cast<int32_t>(P) || a.rows() != n || a.cols() != n) {
        throw std::runtime_error("gebrd_cta_impl: invalid n or matrix sizes for CTA partition");
    }
    if (d.size() != n || e.size() != (n - 1) || tauq.size() != n || taup.size() != n) {
        throw std::runtime_error("gebrd_cta_impl: invalid d/e/tau sizes");
    }
    if (d.batch_size() != batch_size || e.batch_size() != batch_size ||
        tauq.batch_size() != batch_size || taup.batch_size() != batch_size) {
        throw std::runtime_error("gebrd_cta_impl: batch size mismatch");
    }

    ctx->submit([&](sycl::handler& cgh) {
        auto A_view = a.kernel_view();
        auto D_view = d;
        auto E_view = e;
        auto TAUQ_view = tauq;
        auto TAUP_view = taup;

        const auto dev = ctx->get_device();
        const int32_t sg_size = 32;
        const int32_t base_wg_size = std::lcm<int32_t>(static_cast<int32_t>(P), sg_size);
        int32_t wg_mul = std::max<int32_t>(1, static_cast<int32_t>(cta_wg_size_multiplier));
        int32_t wg_size = base_wg_size * wg_mul;

        const int32_t max_wg_size = static_cast<int32_t>(dev.get_info<sycl::info::device::max_work_group_size>());
        if (wg_size > max_wg_size) {
            const int32_t max_mul = std::max<int32_t>(1, max_wg_size / base_wg_size);
            wg_mul = std::min(wg_mul, max_mul);
            wg_size = base_wg_size * wg_mul;
        }

        {
            const std::size_t local_mem_bytes = dev.get_info<sycl::info::device::local_mem_size>();
            const std::size_t elems_per_prob = static_cast<std::size_t>(P) * static_cast<std::size_t>(P);
            const std::size_t bytes_per_prob = elems_per_prob * sizeof(T);
            const int32_t max_probs = (bytes_per_prob == 0)
                ? int32_t(1)
                : std::max<int32_t>(1, static_cast<int32_t>(local_mem_bytes / bytes_per_prob));
            wg_mul = std::min(wg_mul, max_probs);
            wg_size = base_wg_size * wg_mul;
        }

        const int32_t probs_per_wg = wg_size / static_cast<int32_t>(P);
        const int32_t num_wg = (static_cast<int32_t>(batch_size) + probs_per_wg - 1) / probs_per_wg;
        const int32_t global_size = num_wg * wg_size;

        auto A_local = sycl::local_accessor<T, 1>(sycl::range<1>(probs_per_wg * P * P), cgh);

        cgh.parallel_for<GebrdCTAKernel<T, P>>(
            sycl::nd_range<1>(global_size, wg_size),
            [=](sycl::nd_item<1> it) {
                const auto sg = it.get_sub_group();
                const auto part = sycl::ext::oneapi::experimental::chunked_partition<P>(sg);

                const int32_t sg_id = static_cast<int32_t>(sg.get_group_linear_id());
                const int32_t parts_per_sg = static_cast<int32_t>(part.get_group_linear_range());
                const int32_t part_id = sg_id * parts_per_sg + static_cast<int32_t>(part.get_group_linear_id());
                const int32_t lane = static_cast<int32_t>(part.get_local_linear_id());

                const int32_t wg_id = static_cast<int32_t>(it.get_group().get_group_linear_id());
                const int32_t prob_id = wg_id * probs_per_wg + part_id;
                if (prob_id >= static_cast<int32_t>(batch_size)) return;

                auto A_prob = A_view.batch_item(prob_id);
                auto D_prob = D_view.batch_item(prob_id);
                auto E_prob = E_view.batch_item(prob_id);
                auto TAUQ_prob = TAUQ_view.batch_item(prob_id);
                auto TAUP_prob = TAUP_view.batch_item(prob_id);

                const int32_t base_a = part_id * static_cast<int32_t>(P) * static_cast<int32_t>(P);
                for (int32_t c = 0; c < n; ++c) {
                    if (lane < n) {
                        A_local[base_a + lane + c * static_cast<int32_t>(P)] = A_prob(lane, c);
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);

                for (int32_t i = 0; i < n; ++i) {
                    T alpha_l = T(0);
                    if (lane == i) {
                        alpha_l = A_local[base_a + i + i * static_cast<int32_t>(P)];
                    }
                    T x_l = T(0);
                    const bool x_l_active = (lane > i && lane < n);
                    if (x_l_active) {
                        x_l = A_local[base_a + lane + i * static_cast<int32_t>(P)];
                    }

                    const T tau_l = larfg_small(part, n - i, lane, i, alpha_l, x_l, x_l_active);
                    if (lane == i) {
                        A_local[base_a + i + i * static_cast<int32_t>(P)] = alpha_l;
                        D_prob(i) = static_cast<Real>(alpha_l);
                        TAUQ_prob(i) = tau_l;
                    } else if (x_l_active) {
                        A_local[base_a + lane + i * static_cast<int32_t>(P)] = x_l;
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    if (i < n - 1 && tau_l != T(0) && lane > i && lane < n) {
                        const int32_t j = lane;
                        T dot = A_local[base_a + i + j * static_cast<int32_t>(P)];
                        for (int32_t r = i + 1; r < n; ++r) {
                            dot += A_local[base_a + r + i * static_cast<int32_t>(P)] *
                                   A_local[base_a + r + j * static_cast<int32_t>(P)];
                        }
                        const T gamma = tau_l * dot;
                        A_local[base_a + i + j * static_cast<int32_t>(P)] -= gamma;
                        for (int32_t r = i + 1; r < n; ++r) {
                            A_local[base_a + r + j * static_cast<int32_t>(P)] -=
                                A_local[base_a + r + i * static_cast<int32_t>(P)] * gamma;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    if (i >= n - 1) continue;

                    T alpha_r = T(0);
                    if (lane == i + 1) {
                        alpha_r = A_local[base_a + i + (i + 1) * static_cast<int32_t>(P)];
                    }
                    T x_r = T(0);
                    const bool x_r_active = (lane > i + 1 && lane < n);
                    if (x_r_active) {
                        x_r = A_local[base_a + i + lane * static_cast<int32_t>(P)];
                    }

                    const T tau_r = larfg_small(part, n - (i + 1), lane, i + 1, alpha_r, x_r, x_r_active);
                    if (lane == i + 1) {
                        A_local[base_a + i + (i + 1) * static_cast<int32_t>(P)] = alpha_r;
                        E_prob(i) = static_cast<Real>(alpha_r);
                        TAUP_prob(i) = tau_r;
                    } else if (x_r_active) {
                        A_local[base_a + i + lane * static_cast<int32_t>(P)] = x_r;
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    if (tau_r != T(0) && lane > i && lane < n) {
                        const int32_t r = lane;
                        T dot = A_local[base_a + r + (i + 1) * static_cast<int32_t>(P)];
                        for (int32_t c = i + 2; c < n; ++c) {
                            dot += A_local[base_a + r + c * static_cast<int32_t>(P)] *
                                   A_local[base_a + i + c * static_cast<int32_t>(P)];
                        }
                        const T gamma = tau_r * dot;
                        A_local[base_a + r + (i + 1) * static_cast<int32_t>(P)] -= gamma;
                        for (int32_t c = i + 2; c < n; ++c) {
                            A_local[base_a + r + c * static_cast<int32_t>(P)] -=
                                gamma * A_local[base_a + i + c * static_cast<int32_t>(P)];
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }

                if (lane == n - 1) {
                    TAUP_prob(n - 1) = T(0);
                }

                for (int32_t c = 0; c < n; ++c) {
                    if (lane < n) {
                        A_prob(lane, c) = A_local[base_a + lane + c * static_cast<int32_t>(P)];
                    }
                }
            });
    });
}

} // namespace

template <Backend B, typename T>
Event gebrd_cta(Queue& ctx,
                const MatrixView<T, MatrixFormat::Dense>& a_in,
                const VectorView<typename base_type<T>::type>& d_out,
                const VectorView<typename base_type<T>::type>& e_out,
                const VectorView<T>& tauq_out,
                const VectorView<T>& taup_out,
                size_t cta_wg_size_multiplier) {
    if (a_in.rows() != a_in.cols()) {
        throw std::invalid_argument("gebrd_cta: A must be square");
    }
    if constexpr (internal::is_complex<T>::value) {
        throw std::runtime_error("gebrd_cta: complex types are not implemented");
    } else {
        const int64_t n64 = a_in.rows();
        if (n64 < 1 || n64 > 32) {
            throw std::invalid_argument("gebrd_cta currently supports 1 <= n <= 32");
        }

        const int64_t batch_size = a_in.batch_size();
        if (batch_size != d_out.batch_size() || batch_size != e_out.batch_size() ||
            batch_size != tauq_out.batch_size() || batch_size != taup_out.batch_size()) {
            throw std::invalid_argument("gebrd_cta: batch size mismatch");
        }

        const auto dev = ctx->get_device();
        const auto sg_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
        bool has32 = false;
        for (auto sgs : sg_sizes) {
            if (static_cast<int32_t>(sgs) == 32) {
                has32 = true;
                break;
            }
        }
        if (!has32) {
            throw std::runtime_error("gebrd_cta: device does not support subgroup size 32 required for CTA kernels");
        }

        auto& a = const_cast<MatrixView<T, MatrixFormat::Dense>&>(a_in);
        auto launch = [&](auto P_tag) {
            constexpr int32_t P = decltype(P_tag)::value;
            gebrd_cta_impl<T, P>(ctx, a, d_out, e_out, tauq_out, taup_out, static_cast<int32_t>(n64), cta_wg_size_multiplier);
        };

        if (n64 <= 4) {
            launch(std::integral_constant<int32_t, 4>{});
        } else if (n64 <= 8) {
            launch(std::integral_constant<int32_t, 8>{});
        } else if (n64 <= 16) {
            launch(std::integral_constant<int32_t, 16>{});
        } else {
            launch(std::integral_constant<int32_t, 32>{});
        }

        return ctx.get_event();
    }
}

#define GEBRD_CTA_INSTANTIATE(back, fp) \
    template Event gebrd_cta<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const VectorView<typename base_type<BATCHLAS_UNPAREN fp>::type>&, \
        const VectorView<typename base_type<BATCHLAS_UNPAREN fp>::type>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        size_t);

#define GEBRD_CTA_INSTANTIATE_FOR_BACKEND(back) \
    BATCHLAS_FOR_EACH_SCALAR_TYPE_1(GEBRD_CTA_INSTANTIATE, back)

#if BATCHLAS_HAS_CUDA_BACKEND
GEBRD_CTA_INSTANTIATE_FOR_BACKEND(Backend::CUDA)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
GEBRD_CTA_INSTANTIATE_FOR_BACKEND(Backend::ROCM)
#endif

#undef GEBRD_CTA_INSTANTIATE_FOR_BACKEND
#undef GEBRD_CTA_INSTANTIATE

} // namespace batchlas
