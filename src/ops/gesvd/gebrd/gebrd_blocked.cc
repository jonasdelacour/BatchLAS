// SPDX-License-Identifier: MIT
#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/backend_config.h>
#include <batchlas/util/mempool.hh>
#include <batchlas/util/group-collectives.hh>
#include <batchlas/blas/device/v2/dla.hh>

#include <src/math-helpers.hh>
#include <src/ops/gesvd/gebrd/gebrd_v2.hh>
#include <src/ops/gesvd/gebrd/larfg_real_beta.hh>
#include <src/queue.hh>
#include <src/util/template-instantiations.hh>
#include <src/select/vendor.hh>
#include <src/util/batch_mask.hh>

#include <algorithm>
#include <stdexcept>

namespace batchlas {

namespace {

inline int32_t gebrd_blocked_resolved_nb(int32_t block_size) {
    return std::max<int32_t>(1, block_size);
}

template <typename T>
struct GebrdBlockedWorkspace {
    MatrixView<T, MatrixFormat::Dense> x;
    MatrixView<T, MatrixFormat::Dense> y;
};

// Single description of gebrd_blocked's workspace; see workspace_bytes() in
// util/mempool.hh.
template <typename T>
GebrdBlockedWorkspace<T> gebrd_blocked_layout(Queue& ctx,
                                              BumpAllocator& pool,
                                              int32_t m,
                                              int32_t n,
                                              int32_t nb,
                                              int32_t batch) {
    auto x_buf = pool.allocate<T>(ctx, static_cast<size_t>(m) * static_cast<size_t>(nb) * static_cast<size_t>(batch));
    auto y_buf = pool.allocate<T>(ctx, static_cast<size_t>(n) * static_cast<size_t>(nb) * static_cast<size_t>(batch));
    return {
        MatrixView<T, MatrixFormat::Dense>(x_buf.data(), m, nb, m, static_cast<int64_t>(m) * static_cast<int64_t>(nb), batch),
        MatrixView<T, MatrixFormat::Dense>(y_buf.data(), n, nb, n, static_cast<int64_t>(n) * static_cast<int64_t>(nb), batch),
    };
}

template <typename T>
inline void validate_gebrd_dims(const MatrixView<T, MatrixFormat::Dense>& a,
                                const VectorView<typename base_type<T>::type>& d,
                                const VectorView<typename base_type<T>::type>& e,
                                const VectorView<T>& tauq,
                                const VectorView<T>& taup,
                                const char* where) {
    if (a.rows() < a.cols()) {
        throw batchlas::invalid_argument(std::string(where) + ": A must satisfy rows >= cols");
    }

    const int32_t k = static_cast<int32_t>(a.cols());
    const int32_t need_e = std::max<int32_t>(0, k - 1);
    if (d.size() != k || e.size() != need_e || tauq.size() != k || taup.size() != k) {
        throw batchlas::invalid_argument(std::string(where) + ": invalid d/e/tau sizes");
    }
    if (a.batch_size() != d.batch_size() || a.batch_size() != e.batch_size() ||
        a.batch_size() != tauq.batch_size() || a.batch_size() != taup.batch_size()) {
        throw batchlas::invalid_argument(std::string(where) + ": batch size mismatch");
    }
    if (a.batch_size() < 1) {
        throw batchlas::invalid_argument(std::string(where) + ": invalid batch size");
    }
}

template <typename T>
Event gebrd_restore_bidiag_upper(Queue& ctx,
                                 const MatrixView<T, MatrixFormat::Dense>& a,
                                 const VectorView<typename base_type<T>::type>& d,
                                 const VectorView<typename base_type<T>::type>& e,
                                 const std::uint8_t* mask) {
    auto& a_mut = const_cast<MatrixView<T, MatrixFormat::Dense>&>(a);
    const int32_t k = static_cast<int32_t>(d.size());
    const int32_t batch = static_cast<int32_t>(a.batch_size());

    ctx->submit([&](sycl::handler& h) {
        auto A = a_mut.kernel_view();
        auto D = d;
        auto E = e;

        h.parallel_for(
            sycl::range<2>(static_cast<size_t>(batch), static_cast<size_t>(std::max<int32_t>(1, k))),
            [=](sycl::id<2> idx) {
                const int32_t b = static_cast<int32_t>(idx[0]);
                const int32_t i = static_cast<int32_t>(idx[1]);
                if (i >= k || (mask != nullptr && mask[b] == 0)) return;

                A(i, i, b) = static_cast<T>(D(i, b));
                if (i < k - 1) {
                    A(i, i + 1, b) = static_cast<T>(E(i, b));
                }
            });
    });

    return ctx.get_event();
}

namespace v2 = batchlas::device::v2;

// The complex labrd panel's work-group: a type, so every Cyclic loop strides by a constant.
constexpr int32_t kPanelWG = 128;
using PanelEx = v2::WorkGroup<kPanelWG>;

// zlabrd on panel columns j0 .. j0+ib-1 of item b (one work-group). Rows hold conj(u) on exit,
// d and e are real (ZLARFG's real beta). wa / wx hold the negated small products, so every
// update below is one gemv_acc: y := y0 + alpha (sum of the terms). A function, not the kernel
// lambda's body, so the lambda calls no view operator() (SemaSYCL entry quirk).
template <typename T, typename AV, typename RV, typename TV>
BATCHLAS_DLA_INLINE void labrd_panel_v2(const PanelEx& ex, int32_t b, const AV& A, const AV& X, const AV& Y, const RV& D,
                                        const RV& E, const TV& TAUQ, const TV& TAUP, const sycl::local_accessor<T, 1>& work,
                                        int32_t m, int32_t n, int32_t nb, int32_t j0, int32_t ib) {
    using Real = typename base_type<T>::type;
    using gebrd_v2::conj_view;
    using gebrd_v2::herm;
    constexpr auto C = Transpose::ConjTrans;
    const auto Ab = v2::mat(A.data() + static_cast<int64_t>(b) * A.stride(), m, n, A.ld());
    const auto Xb = v2::mat(X.data() + static_cast<int64_t>(b) * X.stride(), m, nb, X.ld());
    const auto Yb = v2::mat(Y.data() + static_cast<int64_t>(b) * Y.stride(), n, nb, Y.ld());
    const auto wa = v2::vec(work, nb, 0);   // -A^H v, then -Y^H u
    const auto wx = v2::vec(work, nb, nb);  // -X^H v, then -A u
    const auto pub = [](auto x) { return v2::assume_published(x); };  // written before the last barrier
    const gebrd_v2::NegZero<T> zero{};

    for (int32_t ii = 0; ii < ib; ++ii) {
        const int32_t gi = j0 + ii;
        const int32_t mt = m - gi, nt = n - gi - 1;  // rows from gi, columns after gi

        // A(gi:m, gi) -= A(gi:m, j0:gi) Y(gi, 0:ii)^H + X(gi:m, 0:ii) A(j0:gi, gi)
        const auto acol = Ab.col(gi).tail(gi);
        v2::gemv_acc(ex, v2::NegOne{}, acol, acol, v2::term(Ab.sub(gi, j0, mt, ii), pub(conj_view(Yb.row(gi).head(ii)))),
                     v2::term(Xb.sub(gi, 0, mt, ii), pub(Ab.col(gi).tail(j0).head(ii))));
        ex.barrier(v2::Mem::Global);

        // H(gi) annihilates A(gi+1:m, gi); v overwrites it, A(gi, gi) = 1. Every item reads alpha
        // before the reduction and forms the same scalars; the leader overwrites it after.
        const auto xq = Ab.col(gi).tail(gi + 1);
        const T alpha_q = Ab(gi, gi);
        const Real sigma = v2::sumsq<v2::alg::Unfused>(ex, xq);
        const auto hq = v2::larfg_scalars<v2::conv::RealBeta>(alpha_q, sycl::sqrt(sigma), mt);
        if (ex.leader()) {
            D(gi, b) = detail::real_part(hq.beta);
            TAUQ(gi, b) = hq.tau;
            Ab(gi, gi) = T(1);
        }
        if (hq.tau != T(0)) v2::scal(ex, hq.scale, xq);
        ex.barrier(v2::Mem::Global);

        if (gi >= n - 1) {
            continue;
        }

        // Y(gi+1:n, ii) = tau_q (A(gi:m, gi+1:n)^H v - Y(gi+1:n, 0:ii) A(gi:m, j0:gi)^H v
        //                        - A(j0:gi, gi+1:n)^H X(gi:m, 0:ii)^H v)
        const auto v = pub(Ab.col(gi).tail(gi));
        v2::gemv<C, v2::alg::GroupReducePerOutput>(ex, v2::NegOne{}, Ab.sub(gi, j0, mt, ii), v, v2::Zero{}, wa);
        v2::gemv<C, v2::alg::GroupReducePerOutput>(ex, v2::NegOne{}, Xb.sub(gi, 0, mt, ii), v, v2::Zero{}, wx);
        ex.barrier(v2::Mem::Local);
        v2::gemv_acc(ex, hq.tau, zero, Yb.col(ii).tail(gi + 1), v2::term(herm(Ab.sub(gi, gi + 1, mt, nt)), v),
                     v2::term(Yb.sub(gi + 1, 0, nt, ii), pub(wa)), v2::term(herm(Ab.sub(j0, gi + 1, ii, nt)), pub(wx)));
        ex.barrier(v2::Mem::Global);

        // Row gi, conjugated (zlabrd's lacgv before the right reflector):
        // A(gi, gi+1:n) = conj(A(gi, gi+1:n)) - Y(gi+1:n, 0:ii+1) conj(A(gi, j0:gi+1)) - A(j0:gi, gi+1:n)^H conj(X(gi, 0:ii))
        const auto arow = Ab.row(gi).tail(gi + 1);
        v2::gemv_acc(ex, v2::NegOne{}, conj_view(arow), arow,
                     v2::term(Yb.sub(gi + 1, 0, nt, ii + 1), pub(conj_view(Ab.row(gi).tail(j0).head(ii + 1)))),
                     v2::term(herm(Ab.sub(j0, gi + 1, ii, nt)), pub(conj_view(Xb.row(gi).head(ii)))));
        ex.barrier(v2::Mem::Global);

        // G(gi) annihilates A(gi, gi+2:n); u overwrites it, A(gi, gi+1) = 1.
        const auto xp = Ab.row(gi).tail(gi + 2);
        const T alpha_p = Ab(gi, gi + 1);
        const Real sigma_r = v2::sumsq<v2::alg::Unfused>(ex, xp);
        const auto hp = v2::larfg_scalars<v2::conv::RealBeta>(alpha_p, sycl::sqrt(sigma_r), nt);
        if (ex.leader()) {
            E(gi, b) = detail::real_part(hp.beta);
            TAUP(gi, b) = hp.tau;
            Ab(gi, gi + 1) = T(1);
        }
        if (hp.tau != T(0)) v2::scal(ex, hp.scale, xp);
        ex.barrier(v2::Mem::Global);

        // X(gi+1:m, ii) = tau_p (A(gi+1:m, gi+1:n) u - A(gi+1:m, j0:gi+1) Y(gi+1:n, 0:ii+1)^H u
        //                        - X(gi+1:m, 0:ii) A(j0:gi, gi+1:n) u)
        const auto u = pub(Ab.row(gi).tail(gi + 1));
        v2::gemv<C, v2::alg::GroupReducePerOutput>(ex, v2::NegOne{}, Yb.sub(gi + 1, 0, nt, ii + 1), u, v2::Zero{}, wa);
        v2::gemv<Transpose::NoTrans>(ex, v2::NegOne{}, Ab.sub(j0, gi + 1, ii, nt), u, v2::Zero{}, wx);
        ex.barrier(v2::Mem::Local);
        v2::gemv_acc(ex, hp.tau, zero, Xb.col(ii).tail(gi + 1), v2::term(Ab.sub(gi + 1, gi + 1, mt - 1, nt), u),
                     v2::term(Ab.sub(gi + 1, j0, mt - 1, ii + 1), pub(wa)), v2::term(Xb.sub(gi + 1, 0, mt - 1, ii), pub(wx)));
        ex.barrier(v2::Mem::Global);
        gebrd_v2::lacgv(ex, Ab.row(gi).tail(gi + 1));
        ex.barrier(v2::Mem::Global);
    }

    if (ex.leader() && n > 0) {
        TAUP(n - 1, b) = T(0);
    }
}

// One panel launch of the complex path (one work-group of kPanelWG per item).
template <typename T>
void labrd_panel_complex(Queue& ctx,
                         MatrixView<T, MatrixFormat::Dense>& a,
                         const VectorView<typename base_type<T>::type>& d_out,
                         const VectorView<typename base_type<T>::type>& e_out,
                         const VectorView<T>& tauq_out,
                         const VectorView<T>& taup_out,
                         const MatrixView<T, MatrixFormat::Dense>& x_mat,
                         const MatrixView<T, MatrixFormat::Dense>& y_mat,
                         const std::uint8_t* mask,
                         int32_t nb, int32_t j0, int32_t ib) {
    if (static_cast<int64_t>(ctx->get_device().get_info<sycl::info::device::max_work_group_size>()) < kPanelWG) {
        throw batchlas::unsupported("gebrd_blocked: the complex panel kernel needs a work-group of 128");
    }
    const int32_t m = static_cast<int32_t>(a.rows());
    const int32_t n = static_cast<int32_t>(a.cols());
    const int32_t batch = static_cast<int32_t>(a.batch_size());
    ctx->submit([&](sycl::handler& h) {
        auto A = a.kernel_view();
        auto D = d_out;
        auto E = e_out;
        auto TAUQ = tauq_out;
        auto TAUP = taup_out;
        auto X = x_mat.kernel_view();
        auto Y = y_mat.kernel_view();
        auto work_local = sycl::local_accessor<T, 1>(sycl::range<1>(static_cast<size_t>(2 * nb)), h);

        batchlas::impl::parallel_for(h, sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(batch) * kPanelWG), sycl::range<1>(kPanelWG)),
                                     [=](sycl::nd_item<1> item) {
            const int32_t b = static_cast<int32_t>(item.get_group_linear_id());
            if (b >= batch || (mask != nullptr && mask[b] == 0)) return;  // I1: one work-group per item
            labrd_panel_v2<T>(PanelEx{item}, b, A, X, Y, D, E, TAUQ, TAUP, work_local, m, n, nb, j0, ib);
        });
    });
}

// LAPACK ?labrd + the trailing gemm. Complex follows zlabrd: rows hold conj(u) on exit (unmbr 'P'
// order), d and e are real, and between generating and storing a right reflector its row is
// conjugated. For real T every conj() below is the identity.
template <Backend B, typename T>
Event gebrd_blocked_impl(Queue& ctx,
                         const MatrixView<T, MatrixFormat::Dense>& a_in,
                         const VectorView<typename base_type<T>::type>& d_out,
                         const VectorView<typename base_type<T>::type>& e_out,
                         const VectorView<T>& tauq_out,
                         const VectorView<T>& taup_out,
                         const Span<std::byte>& ws,
                         int32_t block_size,
                         Span<const std::uint8_t> mask_span) {
    using Real = typename base_type<T>::type;
    using gebrd_detail::conj;
    constexpr bool kComplex = detail::is_complex<T>::value;

    validate_gebrd_dims(a_in, d_out, e_out, tauq_out, taup_out, "gebrd_blocked");

    if (!ctx.in_order()) {
        throw batchlas::invalid_argument("gebrd_blocked: requires an in-order Queue");
    }

    constexpr int32_t MaxPanelNB = 64;
    const int32_t nb = gebrd_blocked_resolved_nb(block_size);
    if (nb > MaxPanelNB) {
        throw batchlas::invalid_argument("gebrd_blocked: block_size > 64 is not supported");
    }

    auto& a = const_cast<MatrixView<T, MatrixFormat::Dense>&>(a_in);
    const int32_t m = static_cast<int32_t>(a.rows());
    const int32_t n = static_cast<int32_t>(a.cols());
    const int32_t k_total = std::min(m, n);
    const int32_t batch = static_cast<int32_t>(a.batch_size());

    Span<std::byte> ws_mut(const_cast<std::byte*>(ws.data()), ws.size());
    BumpAllocator pool(ws_mut);

    auto ws_layout = gebrd_blocked_layout<T>(ctx, pool, m, n, nb, batch);
    // Rebuilt from raw pointers: re-attach the call's mask, so the trailing gemm sees one span.
    MatrixView<T, MatrixFormat::Dense> x_mat = ws_layout.x.with_batch_mask(mask_span);
    MatrixView<T, MatrixFormat::Dense> y_mat = ws_layout.y.with_batch_mask(mask_span);
    const std::uint8_t* const mask = ::batchlas::detail::mask_ptr(mask_span);
    const int32_t panel_wg = std::min<int32_t>(128, static_cast<int32_t>(ctx->get_device().get_info<sycl::info::device::max_work_group_size>()));

    for (int32_t j0 = 0; j0 < k_total; j0 += nb) {
        const int32_t ib = std::min(nb, k_total - j0);
        auto x_panel = x_mat({j0, SliceEnd()}, {0, ib});
        auto y_panel = y_mat({j0, SliceEnd()}, {0, ib});
        // (void) on an Event: deliberate. This Queue is in-order, so the next submission
        // is already ordered after this one and the Event carries nothing the caller needs.
        (void)x_panel.fill_zeros(ctx);
        (void)y_panel.fill_zeros(ctx);

        // Complex panels run on the v2 core (explicit complex arithmetic, 1.3x faster); real
        // panels keep the legacy kernel below (the v2 port cost float REG 64 -> 80 and 4%).
        if constexpr (kComplex)
            labrd_panel_complex<T>(ctx, a, d_out, e_out, tauq_out, taup_out, x_mat, y_mat, mask, nb, j0, ib);
        else ctx->submit([&](sycl::handler& h) {
            auto A = a.kernel_view();
            auto D = d_out;
            auto E = e_out;
            auto TAUQ = tauq_out;
            auto TAUP = taup_out;
            auto X = x_mat.kernel_view();
            auto Y = y_mat.kernel_view();
            auto work_local = sycl::local_accessor<T, 1>(sycl::range<1>(static_cast<size_t>(2 * nb)), h);

            batchlas::impl::parallel_for(h, sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(batch) * static_cast<size_t>(panel_wg)),
                                             sycl::range<1>(static_cast<size_t>(panel_wg))),
                           [=](sycl::nd_item<1> item) {
                const int32_t b = static_cast<int32_t>(item.get_group_linear_id());
                if (b >= batch || (mask != nullptr && mask[b] == 0)) return;  // I1: one work-group per item

                const int32_t lid = static_cast<int32_t>(item.get_local_linear_id());
                const int32_t local_size = static_cast<int32_t>(item.get_local_range(0));
                const auto g = item.get_group();

                for (int32_t ii = 0; ii < ib; ++ii) {
                    const int32_t gi = j0 + ii;

                    for (int32_t r = gi + lid; r < m; r += local_size) {
                        T val = A(r, gi, b);
                        for (int32_t k = 0; k < ii; ++k) {
                            const int32_t gk = j0 + k;
                            val -= A(r, gk, b) * conj(Y(gi, k, b));
                            val -= X(r, k, b) * A(gk, gi, b);
                        }
                        A(r, gi, b) = val;
                    }
                    item.barrier(sycl::access::fence_space::global_space);

                    Real sigma_partial = Real(0);
                    for (int32_t r = gi + 1 + lid; r < m; r += local_size) {
                        const T ari = A(r, gi, b);
                        sigma_partial += detail::norm_squared(ari);
                    }
                    const Real sigma = batchlas::portable::reduce_over_group(g, sigma_partial, sycl::plus<Real>());

                    T alpha = T(0);
                    if (lid == 0) {
                        alpha = A(gi, gi, b);
                    }
                    alpha = batchlas::portable::group_broadcast(g, alpha);
                    T tau_q = T(0);
                    T beta_q = alpha;
                    T scale_q = T(0);
                    if (lid == 0) {
                        const auto scalars = gebrd_detail::larfg_real_beta(alpha, sycl::sqrt(sigma), m - gi);
                        beta_q = scalars.beta;
                        tau_q = scalars.tau;
                        scale_q = scalars.scale;
                    }
                    tau_q = batchlas::portable::group_broadcast(g, tau_q);
                    beta_q = batchlas::portable::group_broadcast(g, beta_q);
                    scale_q = batchlas::portable::group_broadcast(g, scale_q);

                    if (lid == 0) {
                        D(gi, b) = detail::real_part(beta_q);
                        TAUQ(gi, b) = tau_q;
                    }

                    if (lid == 0) {
                        A(gi, gi, b) = T(1);
                    }
                    if (tau_q != T(0)) {
                        for (int32_t r = gi + 1 + lid; r < m; r += local_size) {
                            A(r, gi, b) *= scale_q;
                        }
                    }
                    item.barrier(sycl::access::fence_space::global_space);

                    if (gi >= n - 1) {
                        continue;
                    }

                    for (int32_t k = 0; k < ii; ++k) {
                        const int32_t gk = j0 + k;
                        T sum_a_partial = T(0);
                        T sum_x_partial = T(0);
                        for (int32_t r = gi + lid; r < m; r += local_size) {
                            const T argi = A(r, gi, b);
                            sum_a_partial += conj(A(r, gk, b)) * argi;
                            sum_x_partial += conj(X(r, k, b)) * argi;
                        }
                        const T sum_a = batchlas::portable::reduce_over_group(g, sum_a_partial, sycl::plus<T>());
                        const T sum_x = batchlas::portable::reduce_over_group(g, sum_x_partial, sycl::plus<T>());
                        if (lid == 0) {
                            work_local[k] = sum_a;
                            work_local[nb + k] = sum_x;
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    for (int32_t c = gi + 1 + lid; c < n; c += local_size) {
                        T y = T(0);
                        for (int32_t r = gi; r < m; ++r) {
                            y += conj(A(r, c, b)) * A(r, gi, b);
                        }
                        for (int32_t k = 0; k < ii; ++k) {
                            y -= Y(c, k, b) * work_local[k];
                            y -= conj(A(j0 + k, c, b)) * work_local[nb + k];
                        }
                        Y(c, ii, b) = tau_q * y;
                    }
                    item.barrier(sycl::access::fence_space::global_space);

                    for (int32_t c = gi + 1 + lid; c < n; c += local_size) {
                        // The conjugated row: zlabrd's lacgv before the right reflector.
                        T val = conj(A(gi, c, b));
                        for (int32_t k = 0; k <= ii; ++k) {
                            val -= Y(c, k, b) * conj(A(gi, j0 + k, b));
                        }
                        for (int32_t k = 0; k < ii; ++k) {
                            val -= conj(A(j0 + k, c, b)) * conj(X(gi, k, b));
                        }
                        A(gi, c, b) = val;
                    }
                    item.barrier(sycl::access::fence_space::global_space);

                    Real sigma_r_partial = Real(0);
                    for (int32_t c = gi + 2 + lid; c < n; c += local_size) {
                        const T aic = A(gi, c, b);
                        sigma_r_partial += detail::norm_squared(aic);
                    }
                    const Real sigma_r = batchlas::portable::reduce_over_group(g, sigma_r_partial, sycl::plus<Real>());

                    T alpha_r = T(0);
                    if (lid == 0) {
                        alpha_r = A(gi, gi + 1, b);
                    }
                    alpha_r = batchlas::portable::group_broadcast(g, alpha_r);
                    T tau_p = T(0);
                    T beta_p = alpha_r;
                    T scale_p = T(0);
                    if (lid == 0) {
                        const auto scalars = gebrd_detail::larfg_real_beta(alpha_r, sycl::sqrt(sigma_r), n - (gi + 1));
                        beta_p = scalars.beta;
                        tau_p = scalars.tau;
                        scale_p = scalars.scale;
                    }
                    tau_p = batchlas::portable::group_broadcast(g, tau_p);
                    beta_p = batchlas::portable::group_broadcast(g, beta_p);
                    scale_p = batchlas::portable::group_broadcast(g, scale_p);

                    if (lid == 0) {
                        E(gi, b) = detail::real_part(beta_p);
                        TAUP(gi, b) = tau_p;
                        A(gi, gi + 1, b) = T(1);
                    }
                    if (tau_p != T(0)) {
                        for (int32_t c = gi + 2 + lid; c < n; c += local_size) {
                            A(gi, c, b) *= scale_p;
                        }
                    }
                    item.barrier(sycl::access::fence_space::global_space);

                    for (int32_t k = 0; k <= ii; ++k) {
                        T sum_y_partial = T(0);
                        for (int32_t c = gi + 1 + lid; c < n; c += local_size) {
                            sum_y_partial += conj(Y(c, k, b)) * A(gi, c, b);
                        }
                        const T sum_y = batchlas::portable::reduce_over_group(g, sum_y_partial, sycl::plus<T>());
                        if (lid == 0) {
                            work_local[k] = sum_y;
                        }
                    }
                    for (int32_t k = 0; k < ii; ++k) {
                        T sum_a_partial = T(0);
                        for (int32_t c = gi + 1 + lid; c < n; c += local_size) {
                            sum_a_partial += A(j0 + k, c, b) * A(gi, c, b);
                        }
                        const T sum_a = batchlas::portable::reduce_over_group(g, sum_a_partial, sycl::plus<T>());
                        if (lid == 0) {
                            work_local[nb + k] = sum_a;
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    for (int32_t r = gi + 1 + lid; r < m; r += local_size) {
                        T x = T(0);
                        for (int32_t c = gi + 1; c < n; ++c) {
                            x += A(r, c, b) * A(gi, c, b);
                        }
                        for (int32_t k = 0; k <= ii; ++k) {
                            x -= A(r, j0 + k, b) * work_local[k];
                        }
                        for (int32_t k = 0; k < ii; ++k) {
                            x -= X(r, k, b) * work_local[nb + k];
                        }
                        X(r, ii, b) = tau_p * x;
                    }
                    item.barrier(sycl::access::fence_space::global_space);
                    if constexpr (kComplex) {
                        for (int32_t c = gi + 1 + lid; c < n; c += local_size) {
                            A(gi, c, b) = conj(A(gi, c, b));
                        }
                        item.barrier(sycl::access::fence_space::global_space);
                    }
                }

                if (lid == 0 && n > 0) {
                    TAUP(n - 1, b) = T(0);
                }
            });
        });

        const int32_t j2 = j0 + ib;
        if (j2 < n) {
            auto a22 = a({j2, SliceEnd()}, {j2, SliceEnd()});
            auto v2 = a({j2, SliceEnd()}, {j0, j2});
            auto y2 = y_mat({j2, SliceEnd()}, {0, ib});
            auto x2 = x_mat({j2, SliceEnd()}, {0, ib});
            auto u2 = a({j0, j2}, {j2, SliceEnd()});

            (void)gemm<B>(ctx, v2, y2, a22, {.alpha = T(-1), .beta = T(1), .transB = Transpose::ConjTrans});
            (void)gemm<B>(ctx, x2, u2, a22, {.alpha = T(-1), .beta = T(1)});
        }
    }

    (void)gebrd_restore_bidiag_upper(ctx, a, d_out, e_out, mask);
    return ctx.get_event();
}

} // namespace

template <Backend B, typename T>
Event gebrd_blocked(Queue& ctx,
                    const MatrixView<T, MatrixFormat::Dense>& a_in,
                    const VectorView<typename base_type<T>::type>& d_out,
                    const VectorView<typename base_type<T>::type>& e_out,
                    const VectorView<T>& tauq_out,
                    const VectorView<T>& taup_out,
                    const Span<std::byte>& ws,
                    int32_t block_size) {
    // NETLIB's gemm is its host BLAS, which cannot skip an item: refuse before the panel writes A.
    if constexpr (!select::native_gpu_backend<B>)
        ::batchlas::detail::refuse_batch_mask("gebrd_blocked", a_in, d_out, e_out, tauq_out, taup_out);
    // A carries the call's mask whichever operand brought it, so the trailing gemm sees it.
    const auto mask = ::batchlas::detail::call_mask("gebrd_blocked", a_in, d_out, e_out, tauq_out, taup_out);
    auto a = mask.empty() || a_in.has_batch_mask() ? a_in : a_in.with_batch_mask(mask);
    return gebrd_blocked_impl<B, T>(ctx, a, d_out, e_out, tauq_out, taup_out, ws, block_size, mask);
}

template <Backend B, typename T>
size_t gebrd_blocked_buffer_size(Queue& ctx,
                                 const MatrixView<T, MatrixFormat::Dense>& a,
                                 const VectorView<typename base_type<T>::type>& d,
                                 const VectorView<typename base_type<T>::type>& e,
                                 const VectorView<T>& tauq,
                                 const VectorView<T>& taup,
                                 int32_t block_size) {
    validate_gebrd_dims(a, d, e, tauq, taup, "gebrd_blocked_buffer_size");

    const int32_t m = static_cast<int32_t>(a.rows());
    const int32_t n = static_cast<int32_t>(a.cols());
    const int32_t batch = static_cast<int32_t>(a.batch_size());
    const int32_t nb = gebrd_blocked_resolved_nb(block_size);

    return workspace_bytes([&](BumpAllocator& pool) {
        return gebrd_blocked_layout<T>(ctx, pool, m, n, nb, batch);
    });
}

#define GEBRD_BLOCKED_INSTANTIATE(back, fp) \
    template Event gebrd_blocked<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const VectorView<typename base_type<BATCHLAS_UNPAREN fp>::type>&, \
        const VectorView<typename base_type<BATCHLAS_UNPAREN fp>::type>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const Span<std::byte>&, \
        int32_t); \
    template size_t gebrd_blocked_buffer_size<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const VectorView<typename base_type<BATCHLAS_UNPAREN fp>::type>&, \
        const VectorView<typename base_type<BATCHLAS_UNPAREN fp>::type>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        int32_t);

BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(GEBRD_BLOCKED_INSTANTIATE)

#undef GEBRD_BLOCKED_INSTANTIATE

} // namespace batchlas
