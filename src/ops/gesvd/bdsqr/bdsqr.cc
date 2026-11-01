// SPDX-License-Identifier: MIT
#include <batchlas/blas/extensions.hh>
#include <batchlas/backend_config.h>

#include <batchlas/util/mempool.hh>
#include <batchlas/blas/device/v2/dla.hh>
#include <batchlas/blas/device/v2/ext/bdsqr-bdsdc.hh>

#include <src/math-helpers.hh>
#include <src/queue.hh>
#include <src/util/template-instantiations.hh>
#include <src/kernels/common/info_span.hh>
#include <src/util/batch_mask.hh>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <limits>

namespace batchlas {

namespace {

namespace v2 = ::batchlas::device::v2;

// Serial per-item LAPACK on the v2 Thread executor: lartg / las2 scalars, rot on U columns and
// Vh rows, scal / swap / select_sort for the sign fix and the sort.
template <typename Real>
bool bdsqr_implicit_qr_attempt(Queue& ctx,
                               const VectorView<Real>& d,
                               const VectorView<Real>& e,
                               Span<Real> singular_values_out,
                               const MatrixView<Real, MatrixFormat::Dense>& u,
                               const MatrixView<Real, MatrixFormat::Dense>& vh,
                               int32_t n,
                               int32_t batch,
                               bool sort_desc,
                               Span<int32_t> fail_flags,
                               const std::uint8_t* mask) {
    constexpr Real eps = std::numeric_limits<Real>::epsilon();
    const Real tolmul = sycl::fmax(Real(10), sycl::fmin(Real(100), sycl::pow(eps, Real(-0.125))));
    const Real tol = tolmul * eps;
    const int32_t maxitr = 6; // LAPACK DBDSQR default
    const int32_t maxit = std::max<int32_t>(32, maxitr * n * n);

    ctx->submit([&](sycl::handler& cgh) {
        auto D = d;
        auto E = e;
        Real* out = singular_values_out.data();
        auto U = u.kernel_view();
        auto Vh = vh.kernel_view();
        int32_t* fail = fail_flags.data();
        const int32_t nn = n;
        const int32_t nb = batch;
        const bool descending = sort_desc;
        const bool accumulate_u = u.rows() > 0 && u.cols() == n;
        const bool accumulate_vh = vh.rows() == n && vh.cols() > 0;

        // Explicit captures in the legacy kernel's argument order (the order sets the PTX
        // parameter layout, which the SASS-parity gate compares against).
        cgh.parallel_for(sycl::range<1>(static_cast<size_t>(nb)), [mask, fail, D, nn, E, maxit, tol, accumulate_u, U,
                                                                   accumulate_vh, Vh, out, descending](sycl::id<1> tid) {
            const int32_t b = static_cast<int32_t>(tid[0]);
            // I3. The host fold below reads every flag, so an inactive item writes the
            // neutral "converged" and nothing else.
            if (mask != nullptr && mask[b] == 0) {
                fail[b] = 0;
                return;
            }

            const v2::Thread th{};
            // d and e are read at unit stride (their inc is not consulted).
            Real* db = D.data_ptr() + static_cast<size_t>(b) * static_cast<size_t>(D.stride());
            Real* eb = (nn > 1) ? (E.data_ptr() + static_cast<size_t>(b) * static_cast<size_t>(E.stride())) : nullptr;
            const auto dv = v2::vec(db, nn);
            const auto ev = v2::vec(eb, nn - 1);
            const auto Um = v2::mat_item(U, b);
            const auto Vm = v2::mat_item(Vh, b);

            int32_t iters = 0;
            bool converged = (nn <= 1);

            while (!converged && iters < maxit) {
                const Real abs_thresh = tol * v2::amax_fmax(th, ev, v2::amax_fmax(th, dv, Real(0)));
                converged = v2::zero_if(th, ev, [&](int i) {
                    const Real rel_thresh = eps * (sycl::fabs(dv(i)) + sycl::fabs(dv(i + 1)));
                    return sycl::fabs(ev(i)) <= sycl::fmax(rel_thresh, abs_thresh);
                });
                if (converged) break;

                int32_t l = 0;
                while (l < nn - 1) {
                    while (l < nn - 1 && ev(l) == Real(0)) ++l;
                    if (l >= nn - 1) break;

                    int32_t m = l;
                    while (m < nn - 1 && ev(m) != Real(0)) ++m;

                    if (m == l) {
                        ev(l) = Real(0);
                        l = m + 1;
                        continue;
                    }

                    // ---- Zero-diagonal deflation (DBDSQR's zero-shift chase).
                    // Required: with db[l] == 0, g = 0, lartg returns the
                    // identity and the sweep spins to maxit. Chase the entry
                    // off the block with zero-shift rotations to deflate.
                    // evidence: docs/perf/gesvd.md#gesvd-bdsqr-the-missing-zero-shift-chase
                    const Real dthresh = sycl::fmax(abs_thresh, std::numeric_limits<Real>::min());
                    const int32_t z = v2::ifirst(th, dv.tail(l).head(m - l + 1),
                                                 [&](Real x) { return sycl::fabs(x) <= dthresh; });
                    if (z >= 0) {
                        const int32_t zrow = l + z;
                        dv(zrow) = Real(0);
                        if (zrow < m) {
                            // Zero strictly inside the block: chase e[zrow] RIGHTWARDS with
                            // LEFT rotations, which empties row zrow.
                            Real f2 = ev(zrow);
                            ev(zrow) = Real(0);
                            for (int32_t j = zrow + 1; j <= m; ++j) {
                                const auto rz = detail::lartg<Real>(dv(j), f2);
                                dv(j) = rz.r;
                                if (accumulate_u) v2::rot(th, Um.col(j), Um.col(zrow), rz.c, rz.s);
                                if (j < m) {
                                    f2 = -rz.s * ev(j);
                                    ev(j) = rz.c * ev(j);
                                }
                            }
                        } else if (m > l) {
                            // Zero at the BOTTOM: the rightward chase does nothing here and the
                            // sweep never converges. Chase LEFTWARDS with RIGHT rotations,
                            // emptying column m.
                            Real f2 = ev(m - 1);
                            ev(m - 1) = Real(0);
                            for (int32_t j = m - 1; j >= l; --j) {
                                const auto rz = detail::lartg<Real>(dv(j), f2);
                                dv(j) = rz.r;
                                if (accumulate_vh) v2::rot(th, Vm.row(j), Vm.row(m), rz.c, rz.s);
                                if (j > l) {
                                    f2 = -rz.s * ev(j - 1);
                                    ev(j - 1) = rz.c * ev(j - 1);
                                }
                            }
                        }
                        iters += (m - l + 1);
                        l = m + 1;
                        continue;
                    }

                    // DBDSQR's shift: the smaller singular value of the trailing 2x2 of B (?LAS2,
                    // no squares, so no overflow near sqrt(max)), dropped when negligible.
                    const int32_t p = m - 1;
                    Real shift = detail::las2(dv(p), ev(p), dv(m))[0];
                    const Real sll = sycl::fabs(dv(l));
                    if (sll > Real(0) && (shift / sll) * (shift / sll) < std::numeric_limits<Real>::epsilon()) {
                        shift = Real(0);
                    }
                    Real f = (sll - shift) * (sycl::copysign(Real(1), dv(l)) + shift / dv(l));
                    Real g = ev(l);

                    // One implicit-shift QR sweep: a right rotation (into Vh), then a left one
                    // (into U), chasing the bulge down the block.
                    for (int32_t k = l; k <= m - 1; ++k) {
                        const auto r1 = detail::lartg<Real>(f, g);
                        const Real cs = r1.c;
                        const Real sn = r1.s;
                        if (accumulate_vh) v2::rot(th, Vm.row(k), Vm.row(k + 1), cs, sn);
                        if (k > l) ev(k - 1) = r1.r;

                        const Real dk = dv(k);
                        const Real ek = ev(k);
                        const Real dk1 = dv(k + 1);

                        f = cs * dk + sn * ek;
                        ev(k) = cs * ek - sn * dk;
                        g = sn * dk1;
                        dv(k + 1) = cs * dk1;

                        const auto r2 = detail::lartg<Real>(f, g);
                        const Real cs2 = r2.c;
                        const Real sn2 = r2.s;
                        if (accumulate_u) v2::rot(th, Um.col(k), Um.col(k + 1), cs2, sn2);
                        dv(k) = r2.r;

                        const Real dk1b = dv(k + 1);
                        const Real ekb = ev(k);
                        f = cs2 * ekb + sn2 * dk1b;
                        dv(k + 1) = cs2 * dk1b - sn2 * ekb;

                        if (k < m - 1) {
                            g = sn2 * ev(k + 1);
                            ev(k + 1) = cs2 * ev(k + 1);
                        } else {
                            g = Real(0);
                        }
                        ev(k) = f;
                    }

                    iters += (m - l + 1);
                    l = m + 1;
                }
            }

            // sigma := |d| with the sign moved into Vh, then dbdsqr's selection sort.
            const auto sv = v2::vec(out + static_cast<size_t>(b) * static_cast<size_t>(nn), nn);
            for (int32_t i = 0; i < nn; ++i) {
                if (dv(i) < Real(0)) {
                    dv(i) = -dv(i);
                    if (accumulate_vh) v2::scal(th, v2::NegOne{}, Vm.row(i));
                }
                sv(i) = dv(i);
            }
            if (descending) {
                v2::select_sort<SortOrder::Descending>(th, sv, [&](int i, int j) {
                    if (accumulate_u) v2::swap(th, Um.col(i), Um.col(j));
                    if (accumulate_vh) v2::swap(th, Vm.row(i), Vm.row(j));
                });
            }

            const bool bad = v2::any_nan(th, sv);
            fail[b] = (converged && !bad) ? 0 : 1;
        });
    });

    ctx.wait();
    bool ok = true;
    for (int32_t b = 0; b < batch; ++b) {
        if (fail_flags[static_cast<size_t>(b)] != 0) {
            ok = false;
            break;
        }
    }
    return ok;
}

} // namespace

// Single description of bdsqr's workspace; see workspace_bytes() in
// util/mempool.hh.
template <typename T>
Span<int32_t> bdsqr_layout(Queue& ctx, BumpAllocator& pool, int32_t batch) {
    return pool.allocate<int32_t>(ctx, static_cast<size_t>(batch));
}

template <Backend B, typename T>
Event bdsqr(Queue& ctx,
            const VectorView<T>& d,
            const VectorView<T>& e,
            Span<T> singular_values_out,
            const Span<std::byte>& ws,
            bool sort_desc,
            Span<int32_t> info) {
    const auto empty = MatrixView<T, MatrixFormat::Dense>(nullptr, 0, 0, 1, 1, d.batch_size());
    return bdsqr<B, T>(ctx,
                       d,
                       e,
                       singular_values_out,
                       ws,
                       empty,
                       empty,
                       sort_desc,
                       info);
}

template <Backend B, typename T>
Event bdsqr(Queue& ctx,
            const VectorView<T>& d,
            const VectorView<T>& e,
            Span<T> singular_values_out,
            const Span<std::byte>& ws,
            const MatrixView<T, MatrixFormat::Dense>& u,
            const MatrixView<T, MatrixFormat::Dense>& vh,
            bool sort_desc,
            Span<int32_t> info) {
    const auto mask = ::batchlas::detail::call_mask("bdsqr", d, e, u, vh);
    static_cast<void>(B);

    const int32_t n = static_cast<int32_t>(d.size());
    const int32_t batch = static_cast<int32_t>(d.batch_size());

    if (batch < 1 || n < 1) {
        throw batchlas::invalid_argument("bdsqr: invalid dimensions");
    }
    if (e.size() != std::max<int32_t>(0, n - 1) || e.batch_size() != batch) {
        throw batchlas::invalid_argument("bdsqr: e must have length n-1 and matching batch size");
    }
    const size_t need_s = static_cast<size_t>(n) * static_cast<size_t>(batch);
    if (singular_values_out.size() < need_s) {
        throw batchlas::invalid_argument("bdsqr: singular_values_out span too small");
    }
    if (u.rows() > 0 || u.cols() > 0) {
        if (u.cols() != n || u.batch_size() != batch) {
            throw batchlas::invalid_argument("bdsqr: U must have n columns and matching batch size");
        }
    }
    if (vh.rows() > 0 || vh.cols() > 0) {
        if (vh.rows() != n || vh.batch_size() != batch) {
            throw batchlas::invalid_argument("bdsqr: Vh must have n rows and matching batch size");
        }
    }

    if constexpr (detail::is_complex<T>::value) {
        throw batchlas::unsupported("bdsqr: complex types are not implemented yet");
    } else {
        Span<std::byte> ws_mut(const_cast<std::byte*>(ws.data()), ws.size());
        BumpAllocator pool(ws_mut);
        // fail_flags stays a pool draw and stays UNCONDITIONAL: it is the whole of
        // bdsqr's workspace (see bdsqr_layout / bdsqr_buffer_size), the kernel
        // STORES into it rather than raising, and `info` is an accumulator that
        // several producers may share. Folding afterwards keeps both contracts and
        // leaves bdsqr_buffer_size untouched.
        auto fail_flags = bdsqr_layout<T>(ctx, pool, batch);
        const bool ok = bdsqr_implicit_qr_attempt<T>(ctx,
                                                     d,
                                                     e,
                                                     singular_values_out,
                                                     u,
                                                     vh,
                                                     n,
                                                     batch,
                                                     sort_desc,
                                                     fail_flags,
                                                     ::batchlas::detail::mask_ptr(mask));
        // 0 already means converged here, so no polarity flip -- unlike
        // syevx_lobpcg / syevx_filtered, whose flags mean the opposite.
        detail::info_from_flags(ctx, info, fail_flags.data(), batch, /*one_means_converged=*/false, mask);
        // The batch-wide throw is what a caller who asked for NO per-item status
        // still needs: without it a non-converged item is silent again. A caller who
        // supplied `info` gets the per-item answer instead of an exception that says
        // nothing about which item failed.
        //
        // The condition is info_ptr, not info.empty(): a SHORT non-empty span is
        // "not requested" everywhere else in this mechanism, and testing empty()
        // here would drop both the span AND the throw for one.
        if (!ok && detail::info_ptr(info, batch) == nullptr) {
            throw batchlas::convergence_error("bdsqr: native implicit bidiagonal QR did not converge");
        }
        return ctx.get_event();
    }
}

template <typename T>
size_t bdsqr_buffer_size(Queue& ctx,
                         const VectorView<T>& d,
                         const VectorView<T>& e,
                         Span<T> singular_values_out) {
    static_cast<void>(e);
    static_cast<void>(singular_values_out);
    return workspace_bytes([&](BumpAllocator& pool) {
        return bdsqr_layout<T>(ctx, pool, static_cast<int32_t>(d.batch_size()));
    });
}

#define BDSQR_INSTANTIATE(back, fp) \
    template Event bdsqr<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        Span<BATCHLAS_UNPAREN fp>, \
        const Span<std::byte>&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        bool, Span<int32_t>); \
    template Event bdsqr<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        Span<BATCHLAS_UNPAREN fp>, \
        const Span<std::byte>&, \
        bool, Span<int32_t>);

#define BDSQR_BUFFER_INSTANTIATE(fp) \
    template size_t bdsqr_buffer_size<BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        Span<BATCHLAS_UNPAREN fp>);

BATCHLAS_INSTANTIATE_REAL_ALL_BACKENDS(BDSQR_INSTANTIATE)

BATCHLAS_FOR_EACH_REAL_TYPE(BDSQR_BUFFER_INSTANTIATE)

#undef BDSQR_BUFFER_INSTANTIATE
#undef BDSQR_INSTANTIATE

#undef BDSQR_INSTANTIATE
#undef BDSQR_BUFFER_INSTANTIATE

} // namespace batchlas
