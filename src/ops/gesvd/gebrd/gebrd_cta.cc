// SPDX-License-Identifier: MIT
#include <batchlas/blas/extensions.hh>
#include <batchlas/backend_config.h>
#include <batchlas/device.hh>
#include <batchlas/device/lapack/gebrd.hh>
#include <batchlas/device/sg_partition.hh>
#include <batchlas/device/detail/sg_partition/sg_partition.hh>

#include <src/math-helpers.hh>
#include <src/queue.hh>
#include <src/util/template-instantiations.hh>
#include <src/sycl/local_mem.hh>
#include <src/util/batch_mask.hh>

#include <numeric>
#include <stdexcept>


namespace batchlas {

namespace {

namespace v2 = batchlas::device;

template <typename T, size_t P>
class GebrdCTAKernel;

// The hand kernel's operand (A_view.batch_item(b), addressed by its own operator()) with the host
// n as its order: v2::item would re-form the batch term per element (+2% SASS here).
template <typename T> struct ItemOfOrder {
    KernelMatrixView<T, MatrixFormat::Dense> v;
    int32_t n;
    T& operator()(int i, int j) const { return v(i, j); }
    int rows() const { return n; }
};

template <typename T, size_t P>
inline void gebrd_cta_impl(Queue& ctx,
                           MatrixView<T, MatrixFormat::Dense>& a,
                           const VectorView<typename base_type<T>::type>& d,
                           const VectorView<typename base_type<T>::type>& e,
                           const VectorView<T>& tauq,
                           const VectorView<T>& taup,
                           int32_t n,
                           size_t cta_wg_size_multiplier,
                           const std::uint8_t* mask) {
    const auto batch_size = a.batch_size();
    if (n < 1 || n > static_cast<int32_t>(P) || a.rows() != n || a.cols() != n) {
        throw batchlas::invalid_argument("gebrd_cta_impl: invalid n or matrix sizes for CTA partition");
    }
    if (d.size() != n || e.size() != (n - 1) || tauq.size() != n || taup.size() != n) {
        throw batchlas::invalid_argument("gebrd_cta_impl: invalid d/e/tau sizes");
    }
    if (d.batch_size() != batch_size || e.batch_size() != batch_size ||
        tauq.batch_size() != batch_size || taup.batch_size() != batch_size) {
        throw batchlas::invalid_argument("gebrd_cta_impl: batch size mismatch");
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
            const std::size_t local_mem_bytes = impl::local_mem_bytes(dev);
            const std::size_t elems_per_prob = static_cast<std::size_t>(P) * static_cast<std::size_t>(P);
            const std::size_t bytes_per_prob = elems_per_prob * sizeof(T);
            wg_mul = impl::cta_wg_multiplier("gebrd_cta", wg_mul, base_wg_size / static_cast<int32_t>(P),
                                              bytes_per_prob, local_mem_bytes);
            wg_size = base_wg_size * wg_mul;
        }

        const int32_t probs_per_wg = wg_size / static_cast<int32_t>(P);
        const int32_t num_wg = (static_cast<int32_t>(batch_size) + probs_per_wg - 1) / probs_per_wg;
        const int32_t global_size = num_wg * wg_size;

        using Tier = v2::tier::Cta<static_cast<int>(P)>;
        // One matrix's local slice (P x P of T); the accessor holds probs_per_wg of them.
        constexpr int32_t kSlice = static_cast<int32_t>(v2::gebrd_buffer_size<Tier, T>(static_cast<int>(P)).local_bytes);
        static_assert(kSlice == static_cast<int32_t>(P * P * sizeof(T)), "gebrd_cta: the tile slice is P x P of T");
        auto A_local = sycl::local_accessor<T, 1>(sycl::range<1>(probs_per_wg * P * P), cgh);

        // A thin launcher over device::gebrd<tier::Cta<P>> (lapack/gebrd.hh).
        impl::parallel_for<GebrdCTAKernel<T, P>>(cgh,
            sycl::nd_range<1>(global_size, wg_size),
            [=](sycl::nd_item<1> it) {
                const auto sg = it.get_sub_group();
                const v2::SgpPartition<static_cast<int>(P)> ex{sg};

                const int32_t sg_id = static_cast<int32_t>(sg.get_group_linear_id());
                const int32_t parts_per_sg = static_cast<int32_t>(ex.part.get_group_linear_range());
                const int32_t part_id = sg_id * parts_per_sg + static_cast<int32_t>(ex.part.get_group_linear_id());

                const int32_t wg_id = static_cast<int32_t>(it.get_group().get_group_linear_id());
                const int32_t prob_id = wg_id * probs_per_wg + part_id;
                // NO EARLY RETURN (I2): every barrier in gebd2_tile is work-group wide, so a dead
                // partition (batch % probs_per_wg != 0, or an inactive item) runs a zero tile
                // (tau = 0 throughout) and stores nothing.
                const bool live = prob_id < static_cast<int32_t>(batch_size) &&
                                  (mask == nullptr || mask[prob_id] != 0);
                const int32_t pid = live ? prob_id : 0;

                // The operands of the hand kernel this replaced (batch_item views, the host n), in
                // its capture order (A_view .. TAUP_view, A_local, n): the kernel parameters match.
                const auto A_prob = A_view.batch_item(pid);
                const auto D_prob = D_view.batch_item(pid);
                const auto E_prob = E_view.batch_item(pid);
                const auto TAUQ_prob = TAUQ_view.batch_item(pid);
                const auto TAUP_prob = TAUP_view.batch_item(pid);
                // The work-group's accessor; gebrd indexes this partition's slice by part_id.
                const v2::Workspace ws{v2::local_mptr<std::byte>(reinterpret_cast<v2::local_mptr<std::byte>::pointer>(
                                           A_local.template get_multi_ptr<sycl::access::decorated::yes>().get_decorated())),
                                       nullptr};
                v2::gebrd<Tier>(ex, it, ItemOfOrder<T>{A_prob, n}, D_prob, E_prob, TAUQ_prob, TAUP_prob, ws, live);
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
    // The packed kernel folds the mask into `live` (I2): an inactive item's A, d, e and taus are untouched.
    const std::uint8_t* const mask =
        ::batchlas::detail::mask_ptr(::batchlas::detail::call_mask("gebrd_cta", a_in, d_out, e_out, tauq_out, taup_out));
    if (a_in.rows() != a_in.cols()) {
        throw batchlas::invalid_argument("gebrd_cta: A must be square");
    }
    if constexpr (detail::is_complex<T>::value) {
        throw batchlas::unsupported("gebrd_cta: complex types are not implemented");
    } else {
        const int64_t n64 = a_in.rows();
        if (n64 < 1 || n64 > 32) {
            throw batchlas::invalid_argument("gebrd_cta currently supports 1 <= n <= 32");
        }

        const int64_t batch_size = a_in.batch_size();
        if (batch_size != d_out.batch_size() || batch_size != e_out.batch_size() ||
            batch_size != tauq_out.batch_size() || batch_size != taup_out.batch_size()) {
            throw batchlas::invalid_argument("gebrd_cta: batch size mismatch");
        }

        if (!ctx.device().supports_sub_group_size(32)) {
            throw batchlas::unsupported("gebrd_cta: device does not support subgroup size 32 required for CTA kernels");
        }

        auto& a = const_cast<MatrixView<T, MatrixFormat::Dense>&>(a_in);
        auto launch = [&](auto P_tag) {
            constexpr int32_t P = decltype(P_tag)::value;
            gebrd_cta_impl<T, P>(ctx, a, d_out, e_out, tauq_out, taup_out, static_cast<int32_t>(n64), cta_wg_size_multiplier, mask);
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

BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(GEBRD_CTA_INSTANTIATE)

#undef GEBRD_CTA_INSTANTIATE

} // namespace batchlas
