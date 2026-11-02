// SPDX-License-Identifier: MIT
// The gemm kernels' launchers, one per family of src/ops/gemm/choice.hh. Which family runs is
// decided only in src/ops/gemm/gemm.cc; nothing here selects, and nothing falls back to another
// kernel: a launcher handed a form or config it does not instantiate throws. The reg and wide rows
// are instantiated in the shard TUs beside this one (gemm_rows.hh).

#include <src/ops/gemm/kernels/gemm_launch.hh>

namespace batchlas::sycl_gemm {

namespace {

static_assert(og::wide_configs.size() == 9 && og::reg_configs.size() == 13,
              "a new wide or reg row needs a shard in BATCHLAS_GEMM_SHARDS (src/ops/CMakeLists.txt)");

template <typename T>
class GemmSplitKReduceKernel;

// C = alpha (W[z s] + ... + W[z s + s - 1]) + beta C per item z, the slices summed in that fixed
// order (deterministic, no atomics); beta = 0 never reads C. One work-item per element, m fastest.
// An item the mask switches off is skipped here as in the slices, which left its partials unwritten.
template <typename T>
Event splitk_reduce(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& W, int splits,
                    const MatrixView<T, MatrixFormat::Dense>& C, T alpha, T beta, const std::uint8_t* mask) {
    using D = typename sycl_device::DevMap<T>::type;
    constexpr int kWg = 256;
    // mn <= ldw n <= INT_MAX (can_run), but the padded range and the global id can pass it.
    const int m = C.rows(), mn = C.rows() * C.cols();
    const sycl::range<2> global(static_cast<size_t>(C.batch_size()), (static_cast<size_t>(mn) + kWg - 1) / kWg * kWg);
    ctx->submit([&](sycl::handler& h) {
        const D* w_ptr = reinterpret_cast<const D*>(W.data_ptr());
        D* c_ptr = reinterpret_cast<D*>(C.data_ptr());
        const int ldw = W.ld(), ldc = C.ld();
        const std::ptrdiff_t stride_w = W.stride(), stride_c = C.stride();
        D alpha_d, beta_d;
        std::memcpy(&alpha_d, &alpha, sizeof(D));
        std::memcpy(&beta_d, &beta, sizeof(D));
        batchlas::impl::parallel_for<GemmSplitKReduceKernel<T>>(
            h, sycl::nd_range<2>(global, sycl::range<2>(1, kWg)), [=](sycl::nd_item<2> item) {
                const size_t g = item.get_global_id(1);
                const auto b = static_cast<std::ptrdiff_t>(item.get_group(0));
                if (g >= static_cast<size_t>(mn) || item_off(mask, b)) return;
                const int e = static_cast<int>(g);
                const int j = e / m, i = e - j * m;
                const D* p = w_ptr + b * splits * stride_w + i + static_cast<std::ptrdiff_t>(j) * ldw;
                D acc = p[0];
                for (int q = 1; q < splits; ++q) {
                    const D v = p[q * stride_w];
                    if constexpr (sycl_device::dev_is_complex_v<D>) {
                        acc.re += v.re;
                        acc.im += v.im;
                    } else {
                        acc += v;
                    }
                }
                D* dst = c_ptr + b * stride_c + i + static_cast<std::ptrdiff_t>(j) * ldc;
                const bool beta_zero = sycl_device::dev_is_zero(beta_d);
                *dst = wide_scalar::lin_epi(alpha_d, beta_zero ? D{} : beta_d, acc, beta_zero ? D{} : *dst);
            });
    });
    return ctx.get_event();
}

[[noreturn]] void refuse(const std::string& what) {
    throw batchlas::unsupported("gemm: " + what + " (the choice's can_run should have refused it)");
}

// can_run's slm_budget_for(wg) term, so a pin it refuses throws here rather than at submit.
// Only a request above 48 KiB less the reserve can miss a non-Metal budget, so those skip the query.
void refuse_over_budget(Queue& ctx, const char* who, std::int64_t bytes, int wg) {
    if (!BATCHLAS_HAS_SYCL_BACKEND && bytes <= std::int64_t(resident::device_slm_budget(48 * 1024))) return;
    const auto budget = resident::device_slm_budget(impl::local_mem_bytes(ctx->get_device(), wg));
    if (bytes > static_cast<std::int64_t>(budget))
        refuse(std::string(who) + " needs " + std::to_string(bytes) + " B of local memory; the device budgets " +
               std::to_string(budget));
}

} // namespace


template <typename T>
Event gemm_direct(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& B,
                  const MatrixView<T, MatrixFormat::Dense>& C, T alpha, T beta, Transpose transA, Transpose transB) {
    return launch_direct<T>(ctx, A, B, C, alpha, beta, transA, transB);
}

template <typename T>
Event gemm_tiled(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& B,
                 const MatrixView<T, MatrixFormat::Dense>& C, T alpha, T beta, Transpose transA, Transpose transB) {
    return launch_tiled<T, 16>(ctx, A, B, C, alpha, beta, transA, transB);
}

template <typename T>
Event gemm_small(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, const MatrixView<T, MatrixFormat::Dense>& B,
                 const MatrixView<T, MatrixFormat::Dense>& C, T alpha, T beta, Transpose transA, Transpose transB) {
    const auto [m, k] = get_effective_dims(A, transA);
    const auto [k_b, n] = get_effective_dims(B, transB);
    static_cast<void>(k_b);
    if (std::max({m, n, k}) > sycl_gemm_small::kSmallMaxDimFor<T>)
        refuse("small serves max(m, n, k) <= " + std::to_string(sycl_gemm_small::kSmallMaxDimFor<T>));
    const bool nn = transA == Transpose::NoTrans && transB == Transpose::NoTrans;
    const int mx = static_cast<int>(std::max({m, n, k}));
    refuse_over_budget(ctx, "small", og::small_slm_bytes<T>(nn, mx), og::small_wg<T>(nn, mx));
    BATCHLAS_KERNEL_TRACE_SCOPE("gemm_sycl_small_batched");
    return sycl_gemm_small::small_batched<T>(ctx, A, B, C, alpha, beta, transA, transB, m, n, k);
}

Event gemm_reg(Queue& ctx, int tm, int tn, int tk, int u, const MatrixView<float, MatrixFormat::Dense>& A,
               const MatrixView<float, MatrixFormat::Dense>& B, const MatrixView<float, MatrixFormat::Dense>& C,
               float alpha, float beta, Transpose transA, Transpose transB) {
    // A real scalar's ConjTrans is its Trans, so the forms fold C -> T.
    const bool ta = transA != Transpose::NoTrans, tb = transB != Transpose::NoTrans;
    std::optional<Event> out;
    static_for<static_cast<int>(og::reg_configs.size())>([&](auto I) {
        constexpr og::RegCfg c = og::reg_configs[I];
        if (out || c.m != tm || c.n != tn || c.k != tk || c.u != u) return;
        // Only tiles above 48 KiB less the reserve can miss a budget, so the others skip the query;
        // a Metal build checks every tile (~28 KiB at this work-group).
        if constexpr (BATCHLAS_HAS_SYCL_BACKEND ||
                      c.slm_bytes() > std::int64_t(resident::device_slm_budget(48 * 1024))) {
            const auto budget = resident::device_slm_budget(impl::local_mem_bytes(ctx->get_device(), c.threads()));
            if (c.slm_bytes() > static_cast<std::int64_t>(budget))
                refuse("reg needs " + std::to_string(c.slm_bytes()) + " B of local memory; the device budgets " +
                       std::to_string(budget));
        }
        gemm_reg_row<decltype(I)::value>(out, ctx, A, B, C, alpha, beta, ta, tb);
    });
    if (!out)
        refuse("reg:m=" + std::to_string(tm) + ":n=" + std::to_string(tn) + ":k=" + std::to_string(tk) +
               ":u=" + std::to_string(u) + " has no instantiation for this transpose form");
    return std::move(*out);
}

template <typename T>
Event gemm_wide(Queue& ctx, int tm, int tn, int tk, const MatrixView<T, MatrixFormat::Dense>& A,
                const MatrixView<T, MatrixFormat::Dense>& B, const MatrixView<T, MatrixFormat::Dense>& C, T alpha,
                T beta, Transpose transA, Transpose transB) {
    std::optional<Event> out;
    static_for<static_cast<int>(og::wide_configs.size())>([&](auto I) {
        constexpr og::WideCfg c = og::wide_configs[I];
        if (out || c.m != tm || c.n != tn || c.k != tk) return;
        refuse_over_budget(ctx, "wide", og::wide_slm_bytes<T>(c), c.threads());
        gemm_wide_row<T, decltype(I)::value, false, 0>(out, ctx, A, B, C, alpha, beta, transA, transB, WideSplit{});
        gemm_wide_row<T, decltype(I)::value, false, 1>(out, ctx, A, B, C, alpha, beta, transA, transB, WideSplit{});
    });
    if (!out)
        refuse("wide:m=" + std::to_string(tm) + ":n=" + std::to_string(tn) + ":k=" + std::to_string(tk) +
               " has no instantiation for this transpose form");
    return std::move(*out);
}

template <typename T>
Event gemm_splitk(Queue& ctx, int tm, int tn, int tk, int splits, const MatrixView<T, MatrixFormat::Dense>& A,
                  const MatrixView<T, MatrixFormat::Dense>& B, const MatrixView<T, MatrixFormat::Dense>& C, T alpha,
                  T beta, Transpose transA, Transpose transB) {
    const std::string spelling = "splitk:m=" + std::to_string(tm) + ":n=" + std::to_string(tn) + ":k=" +
                                 std::to_string(tk) + ":s=" + std::to_string(splits);
    if (std::find(og::kSplitKSlices.begin(), og::kSplitKSlices.end(), splits) == og::kSplitKSlices.end())
        refuse(spelling + " names no compiled slice count");
    const auto [m, k] = get_effective_dims(A, transA);
    const auto [k_b, n] = get_effective_dims(B, transB);
    static_cast<void>(k_b);
    const std::int64_t ldw = og::splitk_ld(m);
    if (ldw * n > INT_MAX) refuse(spelling + ": one item's partials pass int indexing");
    const int batch = A.batch_size();
    if (std::int64_t(batch) * splits > INT_MAX) refuse(spelling + ": batch * s passes the int group id");
    const auto budget = batchlas::select::describe(ctx.device(), ctx.backend(), false).scratch_budget;
    if (batch > 0 && og::splitk_item_bytes<T>(m, n, splits) > budget / batch)
        refuse(spelling + ": the partials pass the scratch budget of " + std::to_string(budget) + " B");
    const int kc = ((k + splits - 1) / splits + tk - 1) / tk * tk;
    const std::size_t elems = static_cast<std::size_t>(ldw * n) * static_cast<std::size_t>(splits) * batch;
    auto sizer = BumpAllocator::measuring();
    sizer.allocate<T>(ctx, elems);
    auto lease = ctx.workspace(sizer.required_bytes());
    BumpAllocator pool(lease.span());
    const auto partials = pool.allocate<T>(ctx, elems);
    const MatrixView<T, MatrixFormat::Dense> W(partials.data(), m, n, int(ldw), int(ldw * n), batch * splits);
    const std::uint8_t* mask = item_mask(A, B, C);
    std::optional<Event> out;
    static_for<static_cast<int>(og::wide_configs.size())>([&](auto I) {
        constexpr og::WideCfg c = og::wide_configs[I];
        if (out || !og::splitk_host(c) || c.m != tm || c.n != tn || c.k != tk) return;
        refuse_over_budget(ctx, "splitk", og::wide_slm_bytes<T>(c), c.threads());
        gemm_wide_row<T, decltype(I)::value, true, 0>(out, ctx, A, B, W, T(1), T(0), transA, transB,
                                                      WideSplit{splits, kc, mask});
        gemm_wide_row<T, decltype(I)::value, true, 1>(out, ctx, A, B, W, T(1), T(0), transA, transB,
                                                      WideSplit{splits, kc, mask});
    });
    if (!out) refuse(spelling + " has no instantiation for this transpose form");
    // An out-of-order queue would let the reduction overtake the slices.
    if (!ctx.in_order()) out->wait();
    BATCHLAS_KERNEL_TRACE_SCOPE("gemm_splitk_reduce");
    return splitk_reduce<T>(ctx, W, splits, C, alpha, beta, mask);
}

#define GEMM_LAUNCHERS(T)                                                                                       \
    template Event gemm_direct<T>(Queue&, const MatrixView<T, MatrixFormat::Dense>&,                           \
                                  const MatrixView<T, MatrixFormat::Dense>&, const MatrixView<T, MatrixFormat::Dense>&, \
                                  T, T, Transpose, Transpose);                                                  \
    template Event gemm_tiled<T>(Queue&, const MatrixView<T, MatrixFormat::Dense>&,                            \
                                 const MatrixView<T, MatrixFormat::Dense>&, const MatrixView<T, MatrixFormat::Dense>&, \
                                 T, T, Transpose, Transpose);                                                   \
    template Event gemm_wide<T>(Queue&, int, int, int, const MatrixView<T, MatrixFormat::Dense>&,             \
                                const MatrixView<T, MatrixFormat::Dense>&, const MatrixView<T, MatrixFormat::Dense>&, \
                                T, T, Transpose, Transpose);                                                    \
    template Event gemm_splitk<T>(Queue&, int, int, int, int, const MatrixView<T, MatrixFormat::Dense>&,       \
                                  const MatrixView<T, MatrixFormat::Dense>&, const MatrixView<T, MatrixFormat::Dense>&, \
                                  T, T, Transpose, Transpose);

GEMM_LAUNCHERS(float)
GEMM_LAUNCHERS(double)
GEMM_LAUNCHERS(std::complex<float>)
GEMM_LAUNCHERS(std::complex<double>)
#undef GEMM_LAUNCHERS

template Event gemm_small<float>(Queue&, const MatrixView<float, MatrixFormat::Dense>&,
                                 const MatrixView<float, MatrixFormat::Dense>&,
                                 const MatrixView<float, MatrixFormat::Dense>&, float, float, Transpose, Transpose);
template Event gemm_small<double>(Queue&, const MatrixView<double, MatrixFormat::Dense>&,
                                  const MatrixView<double, MatrixFormat::Dense>&,
                                  const MatrixView<double, MatrixFormat::Dense>&, double, double, Transpose,
                                  Transpose);
template Event gemm_small<std::complex<float>>(Queue&, const MatrixView<std::complex<float>, MatrixFormat::Dense>&,
                                               const MatrixView<std::complex<float>, MatrixFormat::Dense>&,
                                               const MatrixView<std::complex<float>, MatrixFormat::Dense>&,
                                               std::complex<float>, std::complex<float>, Transpose, Transpose);
template Event gemm_small<std::complex<double>>(Queue&, const MatrixView<std::complex<double>, MatrixFormat::Dense>&,
                                                const MatrixView<std::complex<double>, MatrixFormat::Dense>&,
                                                const MatrixView<std::complex<double>, MatrixFormat::Dense>&,
                                                std::complex<double>, std::complex<double>, Transpose, Transpose);

} // namespace batchlas::sycl_gemm