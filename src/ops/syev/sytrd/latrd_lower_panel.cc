// SPDX-License-Identifier: MIT
#include <batchlas/device.hh>
#include <batchlas/device/lapack/sytrd.hh>
#include <batchlas/util/group-collectives.hh>
#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/matrix.hh>

#include <batchlas/backend_config.h>

#include <src/util/template-instantiations.hh>

#include <src/math-helpers.hh>
#include <src/queue.hh>
#include <src/util/batch_mask.hh>

#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-local-accessor-helpers.hh>
#include <batchlas/util/group-invoke.hh>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <batchlas/settings.hh>
#include <batchlas/tuning_params.hh>

namespace batchlas {

namespace {


enum class LatrdImpl { Legacy, Grid };

// BATCHLAS_LATRD_IMPL: unset/other -> Legacy, "grid" -> Grid.
// Re-read on every call, never cached: a static would pin the first answer and
// make an in-process A/B vacuous.
// evidence: docs/perf/sytrd.md#sytrd-the-latrd-implementation-selector
inline LatrdImpl latrd_impl() {
    const char* v = batchlas::settings().selection.latrd_impl.get();
    if (!v) return LatrdImpl::Legacy;
    const std::string s(v);
    if (s == "grid") return LatrdImpl::Grid;
    return LatrdImpl::Legacy;
}

// Smallest n at which the multi-work-group panel is worth its grid barrier (five
// device-scope syncs per panel column). A knob, not a formula: the crossover is a
// barrier-latency property of the device. Applied in both jobz modes.
// evidence: docs/perf/sytrd.md#sytrd-the-latrd-grid-path-threshold
inline int64_t latrd_grid_min_n() {
    return batchlas::settings().geometry.latrd_grid_min_n;
}

// Grid path is the default above latrd_grid_min_n(); BATCHLAS_LATRD_IMPL still
// forces either path explicitly at any size, which is what makes the two an
// intra-run A/B (=legacy restores the old behaviour everywhere).
inline bool use_grid_latrd(int64_t n) {
    const char* v = batchlas::settings().selection.latrd_impl.get();
    if (v && *v) return latrd_impl() == LatrdImpl::Grid;
    return n >= latrd_grid_min_n();
}

// ---------------------------------------------------------------------------
// Kernel name tags — "Legacy" variants so both paths can coexist.
// ---------------------------------------------------------------------------
template <typename T, int WG, bool FuseTrailingUpdate> class LatrdLowerPanelKernelLegacy;
template <typename T, int WG, bool FuseTrailingUpdate> class LatrdLowerPanelKernelGrid;

// ---------------------------------------------------------------------------
// Grid-path helpers
// ---------------------------------------------------------------------------

template <typename T>
inline T pack_real(typename base_type<T>::type x) {
    if constexpr (detail::is_complex<T>::value) {
        return T(x, typename base_type<T>::type(0));
    } else {
        return T(x);
    }
}

template <typename T>
inline typename base_type<T>::type unpack_real(const T& x) {
    if constexpr (detail::is_complex<T>::value) {
        return x.real();
    } else {
        return static_cast<typename base_type<T>::type>(x);
    }
}

using GridBarrierWord = uint32_t;

// Sense-reversing software grid barrier over the `groups` work-groups that share
// `bar` (bar[0] = arrival counter, bar[1] = generation).
//
// SAFETY: terminates only if every participating work-group is co-resident.
// choose_grid_launch guarantees it (<= MAX_COMPUTE_UNITS work-groups in total, one
// block per SM); BATCHLAS_LATRD_GRID_FORCE_UNSAFE deliberately breaks it.
// evidence: docs/perf/sytrd.md#sytrd-the-latrd-grid-kernel-and-its-co-residency-cap
inline void grid_barrier(const sycl::nd_item<1>& it, GridBarrierWord* bar, int groups) {
    if (groups <= 1) {
        it.barrier(sycl::access::fence_space::global_and_local);
        return;
    }
    it.barrier(sycl::access::fence_space::global_and_local);
    if (it.get_local_linear_id() == 0) {
        sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);
        sycl::atomic_ref<GridBarrierWord, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                         sycl::access::address_space::global_space> cnt(bar[0]);
        sycl::atomic_ref<GridBarrierWord, sycl::memory_order::acq_rel, sycl::memory_scope::device,
                         sycl::access::address_space::global_space> gen(bar[1]);
        const GridBarrierWord g0 = gen.load();
        if (cnt.fetch_add(GridBarrierWord(1)) ==
            static_cast<GridBarrierWord>(groups) - GridBarrierWord(1)) {
            cnt.store(GridBarrierWord(0));
            gen.store(g0 + GridBarrierWord(1));
        } else {
            while (gen.load() == g0) {
                // spin
            }
        }
        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);
    }
    it.barrier(sycl::access::fence_space::global_and_local);
}

// Process-lifetime device scratch for the grid path (barrier words + partial
// reduction slots). Deliberately NOT taken from the caller's BumpAllocator so
// that no *_buffer_size query anywhere in the codebase changes meaning.
inline void* acquire_grid_scratch(Queue& q, size_t bytes) {
    static std::mutex m;
    static void* ptr = nullptr;
    static size_t cap = 0;
    static bool has_ctx = false;
    static sycl::context cached_ctx{};

    std::lock_guard<std::mutex> lk(m);
    const sycl::context cur_ctx = q->get_context();
    const bool ctx_changed = has_ctx && (cached_ctx != cur_ctx);
    if (ptr && (bytes > cap || ctx_changed)) {
        q->wait();
        sycl::free(ptr, cached_ctx);
        ptr = nullptr;
        cap = 0;
    }
    if (!ptr) {
        ptr = sycl::malloc_device(bytes, q->get_device(), cur_ctx);
        if (!ptr) {
            throw batchlas::device_error("latrd_lower_panel(grid): device scratch allocation failed");
        }
        cap = bytes;
        cached_ctx = cur_ctx;
        has_ctx = true;
    }
    return ptr;
}

// ---------------------------------------------------------------------------
// Legacy kernel: manual group-reduction inner loops
// ---------------------------------------------------------------------------
template <typename T, int WG, bool FuseTrailingUpdate>
Event latrd_lower_panel_batched_wg_legacy(Queue& q,
                                          const MatrixView<T, MatrixFormat::Dense>& a,
                                          const VectorView<T>& e,
                                          const VectorView<T>& tau,
                                          const MatrixView<T, MatrixFormat::Dense>& w,
                                          const std::uint8_t* mask) {
    constexpr int wg = WG;
    // Bytes per work-item, so the cap scales with WG (the panels shrink from WG 256 to 64).
    // evidence: docs/perf/sytrd.md#sytrd-the-latrd-lower-panel-on-device-la-v2-an-accepted-miss
    const std::ptrdiff_t slm_floor =
        static_cast<std::ptrdiff_t>(tuning::latrd_lower_panel_slm_floor<T>(a.rows(), a.batch_size())) * wg;

    (void)q->submit([&](sycl::handler& h) {
        KernelMatrixView<T, MatrixFormat::Dense> A_view(a.data_ptr(), a.rows(), a.cols(), a.ld(), a.stride(), a.batch_size());
        KernelMatrixView<T, MatrixFormat::Dense> W_view(w.data_ptr(), w.rows(), w.cols(), w.ld(), w.stride(), w.batch_size());

        VectorView<T> E_view = e;
        VectorView<T> TAU_view = tau;

        const int n = A_view.rows();
        const int batch = A_view.batch_size();
        const int ib = W_view.cols();

        auto v_local   = sycl::local_accessor<T, 1>(sycl::range<1>(static_cast<size_t>(n)), h);
        // Pad wcol_local so the four accessors total >= slm_floor bytes (occupancy cap; tuning_params.hh).
        const std::ptrdiff_t pad = (slm_floor + static_cast<std::ptrdiff_t>(sizeof(T)) - 1) /
                                       static_cast<std::ptrdiff_t>(sizeof(T)) - n - 2 * ib;
        const size_t wcol_elems = static_cast<size_t>(std::max<std::ptrdiff_t>(n, pad));
        auto wcol_local = sycl::local_accessor<T, 1>(sycl::range<1>(wcol_elems), h);
        auto vip_local  = sycl::local_accessor<T, 1>(sycl::range<1>(static_cast<size_t>(ib)), h);
        auto wip_local  = sycl::local_accessor<T, 1>(sycl::range<1>(static_cast<size_t>(ib)), h);

        batchlas::impl::parallel_for<LatrdLowerPanelKernelLegacy<T, WG, FuseTrailingUpdate>>(h,
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(batch) * wg), sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) {
                const int b = static_cast<int>(it.get_group_linear_id());
                if (b >= batch || (mask != nullptr && mask[b] == 0)) return;  // I1: before the first barrier

                const device::WorkGroup<WG> ex{it};

                auto Ab = A_view.batch_item(b);
                auto Wb = W_view.batch_item(b);
                // Thin launcher: the panel body of device::sytrd<Lower, tier::Blocked<NB>> (latrd_lower_run).
                device::detail::latrd_lower_run<WG>(ex, Ab, Wb, n, ib, util::get_raw_ptr(v_local), util::get_raw_ptr(wcol_local),
                                                      vip_local, wip_local, [&](int i, const T& beta, const T& tau_i) BATCHLAS_DEVICE_BODY {
                                                          E_view(i, b) = beta;
                                                          TAU_view(i, b) = tau_i;
                                                      });

                if constexpr (FuseTrailingUpdate) {
                    const int lid = static_cast<int>(it.get_local_linear_id());
                    const int j2 = ib;
                    const int n2 = n - j2;
                    for (int lin = lid; lin < n2 * n2; lin += wg) {
                        const int r = lin % n2;
                        const int c = lin / n2;
                        if (r < c) continue;

                        const int rr = j2 + r;
                        const int cc = j2 + c;
                        T acc = T(0);
                        for (int k = 0; k < ib; ++k) {
                            const T vrk = (rr == k + 1) ? T(1) : ((rr > k + 1) ? Ab(rr, k) : T(0));
                            const T vck = (cc == k + 1) ? T(1) : ((cc > k + 1) ? Ab(cc, k) : T(0));
                            const T wrk = Wb(rr, k);
                            const T wck = Wb(cc, k);
                            acc += vrk * batchlas::device::conj(wck) + wrk * batchlas::device::conj(vck);
                        }

                        T a_rc = Ab(rr, cc) - acc;
                        if constexpr (detail::is_complex<T>::value) {
                            if (rr == cc) {
                                a_rc = T(a_rc.real(), typename T::value_type(0));
                            }
                        }
                        Ab(rr, cc) = a_rc;
                    }
                }
            });
    });

    return q.get_event();
}

// ---------------------------------------------------------------------------
// Grid kernel: G work-groups per matrix, each owning a contiguous block of the
// trailing rows [i+1, n). Whole-range reductions (sumsq, gamma/delta, the final
// dot) are combined through global scratch in a FIXED group order, so the result is
// run-to-run deterministic. Five grid barriers per panel column:
//   1. after the column update + sumsq partials  (reflector needs the whole col)
//   2. after scaling the reflector               (v must be fully updated)
//   3. after gamma/delta partials
//   4. after the dot partials
//   5. at the end of the column                  (next column reads row i+1 of W)
// evidence: docs/perf/sytrd.md#sytrd-the-latrd-grid-kernel-and-its-co-residency-cap
// ---------------------------------------------------------------------------
template <typename T, int WG, bool FuseTrailingUpdate>
Event latrd_lower_panel_batched_wg_grid(Queue& q,
                                        const MatrixView<T, MatrixFormat::Dense>& a,
                                        const VectorView<T>& e,
                                        const VectorView<T>& tau,
                                        const MatrixView<T, MatrixFormat::Dense>& w,
                                        int groups,
                                        const std::uint8_t* mask) {
    constexpr int wg = WG;
    const int n_host = a.rows();
    const int batch_host = a.batch_size();
    const int ib_host = w.cols();
    const int G_host = groups;

    // Scratch layout, per matrix:
    //   gamma partials : [G * ib]   index g*ib + p
    //   delta partials : [G * ib]
    //   sumsq partials : [G]
    //   dot partials   : [G]
    const size_t red_per_matrix =
        static_cast<size_t>(G_host) * (2u * static_cast<size_t>(ib_host) + 2u);
    const size_t bar_words = static_cast<size_t>(batch_host) * 2u;
    size_t bar_bytes = bar_words * sizeof(GridBarrierWord);
    bar_bytes = (bar_bytes + 255u) & ~static_cast<size_t>(255u);
    const size_t red_bytes = static_cast<size_t>(batch_host) * red_per_matrix * sizeof(T);

    std::byte* scratch = static_cast<std::byte*>(acquire_grid_scratch(q, bar_bytes + red_bytes));
    GridBarrierWord* bar_base = reinterpret_cast<GridBarrierWord*>(scratch);
    T* red_base_ptr = reinterpret_cast<T*>(scratch + bar_bytes);

    auto zero_ev = q->memset(bar_base, 0, bar_words * sizeof(GridBarrierWord));

    (void)q->submit([&](sycl::handler& h) {
        h.depends_on(zero_ev);

        KernelMatrixView<T, MatrixFormat::Dense> A_view(a.data_ptr(), a.rows(), a.cols(), a.ld(), a.stride(), a.batch_size());
        KernelMatrixView<T, MatrixFormat::Dense> W_view(w.data_ptr(), w.rows(), w.cols(), w.ld(), w.stride(), w.batch_size());

        VectorView<T> E_view = e;
        VectorView<T> TAU_view = tau;

        const int n = n_host;
        const int batch = batch_host;
        const int ib = ib_host;
        const int G = G_host;
        const size_t red_stride = red_per_matrix;
        GridBarrierWord* bar_all = bar_base;
        T* red_all = red_base_ptr;

        auto v_local    = sycl::local_accessor<T, 1>(sycl::range<1>(static_cast<size_t>(n)), h);
        auto wcol_local = sycl::local_accessor<T, 1>(sycl::range<1>(static_cast<size_t>(n)), h);
        auto vip_local  = sycl::local_accessor<T, 1>(sycl::range<1>(static_cast<size_t>(std::max(ib_host, 1))), h);
        auto wip_local  = sycl::local_accessor<T, 1>(sycl::range<1>(static_cast<size_t>(std::max(ib_host, 1))), h);
        auto gam_local  = sycl::local_accessor<T, 1>(sycl::range<1>(static_cast<size_t>(std::max(ib_host, 1))), h);
        auto del_local  = sycl::local_accessor<T, 1>(sycl::range<1>(static_cast<size_t>(std::max(ib_host, 1))), h);

        batchlas::impl::parallel_for<LatrdLowerPanelKernelGrid<T, WG, FuseTrailingUpdate>>(h,
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(batch) * static_cast<size_t>(G) * wg),
                              sycl::range<1>(wg)),
            [=](sycl::nd_item<1> it) {
                using Real = typename base_type<T>::type;

                const int glid = static_cast<int>(it.get_group_linear_id());
                const int b = glid / G;
                // I1 per item: all G work-groups of an inactive item leave before its grid barrier.
                if (mask != nullptr && mask[b] == 0) return;
                const int gg = glid - b * G;
                const int lid = static_cast<int>(it.get_local_linear_id());
                const sycl::group<1> g = it.get_group();
                const batchlas::device::WorkGroup<WG> ex{it};

                auto Ab = A_view.batch_item(b);
                auto Wb = W_view.batch_item(b);

                GridBarrierWord* bar = bar_all + 2 * b;
                T* red = red_all + static_cast<size_t>(b) * red_stride;
                T* red_gamma = red;
                T* red_delta = red + static_cast<size_t>(G) * static_cast<size_t>(ib);
                T* red_sumsq = red + 2u * static_cast<size_t>(G) * static_cast<size_t>(ib);
                T* red_dot   = red_sumsq + G;

                for (int i = 0; i < ib; ++i) {
                    if (i >= n - 1) break;

                    // Row partition for this column: contiguous block per group.
                    const int base_r = i + 1;
                    const int total_r = n - base_r;
                    const int chunk = (total_r + G - 1) / G;
                    const int rlo = sycl::min(n, base_r + gg * chunk);
                    const int rhi = sycl::min(n, rlo + chunk);
                    const int slo = sycl::max(rlo, i + 2);   // reflector tail start

                    for (int p = lid; p < i; p += wg) {
                        vip_local[p] = (i == p + 1) ? T(1) : Ab(i, p);
                        wip_local[p] = Wb(i, p);
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    // ---- column update: A(i, i) on group 0, this group's rows ----
                    device::detail::latrd_col_update_run(Ab, Wb, vip_local, wip_local, i, gg == 0 && lid == 0, rlo, rhi, lid, wg);

                    // Write mapping [rlo+lid, rhi), read mapping [slo+lid, rhi): for gg == 0
                    // slo == rlo + 1, so item lid reads the row item lid+1 just wrote. Masked
                    // inside one sub-group only. This is the legacy kernel's barrier, restored.
                    it.barrier(sycl::access::fence_space::global_space);

                    // ---- sumsq partial (only over rows this group just wrote) --
                    Real sumsq = Real(0);
                    for (int r = slo + lid; r < rhi; r += wg) {
                        sumsq += detail::abs2_if_complex(Ab(r, i));
                    }
                    sumsq = ex.group_reduce(sumsq);
                    if (lid == 0) red_sumsq[gg] = pack_real<T>(sumsq);

                    grid_barrier(it, bar, G);   // barrier 1

                    Real sumsq_total = Real(0);
                    for (int gi = 0; gi < G; ++gi) {
                        sumsq_total += unpack_real<T>(red_sumsq[gi]);
                    }

                    // Every work-item recomputes the reflector from identical
                    // inputs, so no broadcast is needed and all groups agree.
                    // NOTE: Ab(i+1,i) must be read here, BEFORE group 0 replaces
                    // it with 1 (which happens after barrier 2).
                    const T alpha = (i + 1 < n) ? Ab(i + 1, i) : T(0);

                    // xLARFG scalars, alg::PhaseN1 (a complex alpha reflects at the last column too).
                    const auto h = device::larfg_scalars<device::alg::PhaseN1>(alpha, sycl::sqrt(sumsq_total), n - i - 1);
                    const T tau_i = h.tau;
                    const T beta = h.beta;
                    const T scale = h.scale;

                    if (tau_i != T(0)) {
                        for (int r = slo + lid; r < rhi; r += wg) {
                            Ab(r, i) *= scale;
                        }
                    }

                    grid_barrier(it, bar, G);   // barrier 2

                    if (gg == 0 && lid == 0) {
                        E_view(i, b) = beta;
                        TAU_view(i, b) = tau_i;
                        Ab(i + 1, i) = T(1);
                    }

                    // Every group needs the whole reflector for the symv.
                    for (int c = base_r + lid; c < n; c += wg) {
                        v_local[c] = (c == base_r) ? T(1) : Ab(c, i);
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    // ---- symmetric mat-vec over this group's rows -------------
                    // ExplicitInline: the plain-inline explicit complex mac of the symv ONLY
                    // (every other complex multiply here converted cost 1.16x, register pressure).
                    // evidence: docs/perf/sytrd.md#sytrd-complex-multiply-in-the-latrd-symv
                    using SymvAr = device::arith_t<device::arith::ExplicitInline, T>;
                    for (int r = rlo + lid; r < rhi; r += wg) {
                        const int c_split = sycl::min(r, n - 1);
                        T acc = T(0);
                        #pragma unroll 4
                        for (int c = base_r; c <= c_split; ++c) {
                            SymvAr::mac(acc, Ab(r, c), v_local[c]);
                        }
                        #pragma unroll 4
                        for (int c = c_split + 1; c < n; ++c) {
                            SymvAr::mac_conj(acc, Ab(c, r), v_local[c]);
                        }
                        wcol_local[r] = acc;
                    }

                    // ---- gamma/delta partials ---------------------------------
                    for (int p = 0; p < i; ++p) {
                        T gamma_partial = T(0);
                        T delta_partial = T(0);
                        for (int c = rlo + lid; c < rhi; c += wg) {
                            const T vc = v_local[c];
                            gamma_partial += batchlas::device::conj(Wb(c, p)) * vc;
                            const T vcp = (c == p + 1) ? T(1) : ((c > p + 1) ? Ab(c, p) : T(0));
                            delta_partial += batchlas::device::conj(vcp) * vc;
                        }
                        gamma_partial = ex.group_reduce(gamma_partial);
                        delta_partial = ex.group_reduce(delta_partial);
                        if (lid == 0) {
                            red_gamma[static_cast<size_t>(gg) * ib + p] = gamma_partial;
                            red_delta[static_cast<size_t>(gg) * ib + p] = delta_partial;
                        }
                    }

                    grid_barrier(it, bar, G);   // barrier 3

                    for (int p = lid; p < i; p += wg) {
                        T gs = T(0);
                        T ds = T(0);
                        for (int gi = 0; gi < G; ++gi) {
                            gs += red_gamma[static_cast<size_t>(gi) * ib + p];
                            ds += red_delta[static_cast<size_t>(gi) * ib + p];
                        }
                        gam_local[p] = gs;
                        del_local[p] = ds;
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    for (int p = 0; p < i; ++p) {
                        const T gamma = gam_local[p];
                        const T delta = del_local[p];
                        for (int r = rlo + lid; r < rhi; r += wg) {
                            const T vrp = (r == p + 1) ? T(1) : ((r > p + 1) ? Ab(r, p) : T(0));
                            const T wrp = Wb(r, p);
                            wcol_local[r] -= vrp * gamma + wrp * delta;
                        }
                    }

                    for (int r = rlo + lid; r < rhi; r += wg) {
                        wcol_local[r] *= tau_i;
                    }

                    T dot_partial = T(0);
                    for (int r = rlo + lid; r < rhi; r += wg) {
                        dot_partial += batchlas::device::conj(v_local[r]) * wcol_local[r];
                    }
                    dot_partial = ex.group_reduce(dot_partial);
                    if (lid == 0) red_dot[gg] = dot_partial;

                    grid_barrier(it, bar, G);   // barrier 4

                    T dot = T(0);
                    for (int gi = 0; gi < G; ++gi) dot += red_dot[gi];

                    const T alpha2 = T(-0.5) * tau_i * dot;
                    for (int r = rlo + lid; r < rhi; r += wg) {
                        Wb(r, i) = wcol_local[r] + alpha2 * v_local[r];
                    }

                    grid_barrier(it, bar, G);   // barrier 5
                }

                if constexpr (FuseTrailingUpdate) {
                    const int j2 = ib;
                    const int n2 = n - j2;
                    const int tid = gg * wg + lid;
                    const int nthreads = G * wg;
                    for (int lin = tid; lin < n2 * n2; lin += nthreads) {
                        const int r = lin % n2;
                        const int c = lin / n2;
                        if (r < c) continue;

                        const int rr = j2 + r;
                        const int cc = j2 + c;
                        T acc = T(0);
                        for (int k = 0; k < ib; ++k) {
                            const T vrk = (rr == k + 1) ? T(1) : ((rr > k + 1) ? Ab(rr, k) : T(0));
                            const T vck = (cc == k + 1) ? T(1) : ((cc > k + 1) ? Ab(cc, k) : T(0));
                            const T wrk = Wb(rr, k);
                            const T wck = Wb(cc, k);
                            acc += vrk * batchlas::device::conj(wck) + wrk * batchlas::device::conj(vck);
                        }

                        T a_rc = Ab(rr, cc) - acc;
                        if constexpr (detail::is_complex<T>::value) {
                            if (rr == cc) {
                                a_rc = T(a_rc.real(), typename T::value_type(0));
                            }
                        }
                        Ab(rr, cc) = a_rc;
                    }
                }
            });
    });

    return q.get_event();
}

// ---------------------------------------------------------------------------
// Dispatcher: the legacy kernel (WG = 64/128/256) or the grid kernel.
// ---------------------------------------------------------------------------
// Chooses (G, wg) for the grid path. Co-residency is the hard constraint: the
// software grid barrier deadlocks unless all batch*G work-groups are resident, so
// never launch more than MAX_COMPUTE_UNITS work-groups in total. G*wg stays near n.
// Returns G == 1 to mean "use the legacy kernel verbatim".
struct GridLaunch { int groups; int wg; };

inline GridLaunch choose_grid_launch(Queue& q, int n, int batch) {
    GridLaunch out{1, 0};
    if (n < 4 || batch < 1) return out;

    // GPU only. The software grid barrier requires all participating
    // work-groups to be co-resident; that is argued from "one block per SM" on
    // a GPU. The CPU/host SYCL runtimes give no such guarantee (work-groups are
    // multiplexed onto a thread pool), so the barrier is unsound there and the
    // legacy single-work-group kernel must be used instead.
    if (q.device().type != DeviceType::GPU) return out;
    // Nor on Metal, not even when forced: Metal promises no co-residency of work-groups, and
    // MSL atomics are memory_order_relaxed only, so the acq_rel handshake is not what runs.
    // A hang there also stalls the display, which shares the GPU.
    if (impl::is_metal(impl::backend_of(*q))) return out;

    const int cus = static_cast<int>(q.device().get_property(DeviceProperty::MAX_COMPUTE_UNITS));
    const int resident_cap = std::max(1, cus);
    const int forced_g = batchlas::settings().geometry.latrd_grid_groups;
    int cap = resident_cap / batch;            // integer division, never rounds up

    // MEASUREMENT ONLY, DEADLOCK-CAPABLE. Without this override the cap above makes
    // BATCHLAS_LATRD_GRID_GROUPS unreachable at batch >= MAX_COMPUTE_UNITS. A forced
    // launch that is not co-resident HANGS (looks like slow JIT): run it under
    // `timeout`, and never use it to relax the default cap. Gated by the
    // BATCHLAS_ALLOW_UNSAFE_ENV build option, so ambient state cannot enable it.
    // evidence: docs/perf/sytrd.md#sytrd-the-vacuous-latrd-grid-groups-ab
    if (forced_g > 0 && batchlas::settings().unsafe.latrd_grid_force_unsafe) {
        cap = forced_g;
    }
    if (cap < 1) return out;

    const int rows = n - 1;
    int G = std::min(cap, std::max(1, (rows + 31) / 32));

    if (forced_g > 0) {
        G = std::min(forced_g, cap);           // never exceed the residency cap
    }
    if (G <= 1) return out;

    // One work-item per trailing row, rounded up to a whole sub-group.
    int wg = ((rows + G - 1) / G + 31) / 32 * 32;
    wg = std::min(256, std::max(32, wg));
    const int forced_wg = batchlas::settings().geometry.latrd_grid_wg;
    if (forced_wg == 32 || forced_wg == 64 || forced_wg == 128 || forced_wg == 256) {
        wg = forced_wg;
    }

    out.groups = G;
    out.wg = wg;
    return out;
}

template <typename T>
Event latrd_lower_panel_batched_grid_dispatch(Queue& q,
                                              const MatrixView<T, MatrixFormat::Dense>& a,
                                              const VectorView<T>& e,
                                              const VectorView<T>& tau,
                                              const MatrixView<T, MatrixFormat::Dense>& w,
                                              bool fuse_trailing_update,
                                              const GridLaunch& gl,
                                              const std::uint8_t* mask) {
    auto call = [&](auto wg_tag) {
        constexpr int WG = decltype(wg_tag)::value;
        if (fuse_trailing_update)
            return latrd_lower_panel_batched_wg_grid<T, WG, true>(q, a, e, tau, w, gl.groups, mask);
        return latrd_lower_panel_batched_wg_grid<T, WG, false>(q, a, e, tau, w, gl.groups, mask);
    };
    if (gl.wg <= 32)  return call(std::integral_constant<int, 32>{});
    if (gl.wg <= 64)  return call(std::integral_constant<int, 64>{});
    if (gl.wg <= 128) return call(std::integral_constant<int, 128>{});
    return call(std::integral_constant<int, 256>{});
}

template <typename T>
Event latrd_lower_panel_batched(Queue& q,
                                const MatrixView<T, MatrixFormat::Dense>& a,
                                const VectorView<T>& e,
                                const VectorView<T>& tau,
                                const MatrixView<T, MatrixFormat::Dense>& w,
                                int32_t wg_hint,
                                bool fuse_trailing_update) {
    const int n = a.rows();
    const std::uint8_t* mask =
        ::batchlas::detail::mask_ptr(::batchlas::detail::call_mask("latrd_lower_panel", a, e, tau, w));

    auto call_legacy = [&](auto wg_tag) {
        constexpr int WG = decltype(wg_tag)::value;
        if (fuse_trailing_update)
            return latrd_lower_panel_batched_wg_legacy<T, WG, true>(q, a, e, tau, w, mask);
        return latrd_lower_panel_batched_wg_legacy<T, WG, false>(q, a, e, tau, w, mask);
    };

    if (use_grid_latrd(n)) {
        const GridLaunch gl = choose_grid_launch(q, n, a.batch_size());
        if (gl.groups > 1) {
            return latrd_lower_panel_batched_grid_dispatch<T>(q, a, e, tau, w, fuse_trailing_update, gl, mask);
        }
        // G == 1: fall through to the legacy kernel verbatim.
    }

    if (wg_hint == 64)  return call_legacy(std::integral_constant<int, 64>{});
    if (wg_hint == 128) return call_legacy(std::integral_constant<int, 128>{});
    if (wg_hint == 256) return call_legacy(std::integral_constant<int, 256>{});
    if (n <= 64)  return call_legacy(std::integral_constant<int, 64>{});
    if (n <= 128) return call_legacy(std::integral_constant<int, 128>{});
    return call_legacy(std::integral_constant<int, 256>{});
}

template <typename T>
inline void validate_latrd_lower_panel_panel_dims(const MatrixView<T, MatrixFormat::Dense>& a,
                                                  const VectorView<T>& e,
                                                  const VectorView<T>& tau,
                                                  const MatrixView<T, MatrixFormat::Dense>& w) {
    if (a.rows() != a.cols()) {
        throw batchlas::invalid_argument("latrd_lower_panel(panel): A must be square");
    }
    if (w.rows() != a.rows()) {
        throw batchlas::invalid_argument("latrd_lower_panel(panel): W must have same number of rows as A");
    }
    const int ib = w.cols();
    if (ib < 0) {
        throw batchlas::invalid_argument("latrd_lower_panel(panel): invalid W dimensions");
    }
    if (e.size() != ib || tau.size() != ib) {
        throw batchlas::invalid_argument("latrd_lower_panel(panel): e/tau must have size equal to W.cols()");
    }
    if (a.batch_size() != e.batch_size() || a.batch_size() != tau.batch_size() || a.batch_size() != w.batch_size()) {
        throw batchlas::invalid_argument("latrd_lower_panel(panel): batch size mismatch");
    }
    if (a.batch_size() < 1) {
        throw batchlas::invalid_argument("latrd_lower_panel(panel): invalid batch size");
    }
    if (ib > std::max(0, a.rows() - 1)) {
        throw batchlas::invalid_argument("latrd_lower_panel(panel): W.cols() must be <= A.rows()-1");
    }
}

template <typename T>
inline void validate_latrd_lower_panel_dims(const MatrixView<T, MatrixFormat::Dense>& a,
                                            const VectorView<T>& e,
                                            const VectorView<T>& tau,
                                            const MatrixView<T, MatrixFormat::Dense>& w,
                                            int32_t j0,
                                            int32_t ib) {
    if (a.rows() != a.cols()) {
        throw batchlas::invalid_argument("latrd_lower_panel: A must be square");
    }
    const int n = a.rows();
    if (e.size() != std::max(0, n - 1) || tau.size() != std::max(0, n - 1)) {
        throw batchlas::invalid_argument("latrd_lower_panel: invalid e/tau sizes");
    }
    if (w.rows() != n) {
        throw batchlas::invalid_argument("latrd_lower_panel: W must have n rows");
    }
    if (w.cols() < ib) {
        throw batchlas::invalid_argument("latrd_lower_panel: W must have at least ib columns");
    }
    if (j0 < 0 || ib < 0) {
        throw batchlas::invalid_argument("latrd_lower_panel: j0/ib must be non-negative");
    }
    if (j0 > n) {
        throw batchlas::invalid_argument("latrd_lower_panel: j0 out of range");
    }
    if (a.batch_size() != e.batch_size() || a.batch_size() != tau.batch_size() || a.batch_size() != w.batch_size()) {
        throw batchlas::invalid_argument("latrd_lower_panel: batch size mismatch");
    }
    if (a.batch_size() < 1) {
        throw batchlas::invalid_argument("latrd_lower_panel: invalid batch size");
    }
}

} // namespace

template <Backend B, typename T>
Event latrd_lower_panel(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& a_panel_in,
                        const VectorView<T>& e_panel_out,
                        const VectorView<T>& tau_panel_out,
                        const MatrixView<T, MatrixFormat::Dense>& w_panel_in,
                        int32_t wg_hint,
                        bool fuse_trailing_update) {
    (void)::batchlas::detail::call_mask("latrd_lower_panel", a_panel_in, e_panel_out, tau_panel_out, w_panel_in);
    (void)B;
    validate_latrd_lower_panel_panel_dims(a_panel_in, e_panel_out, tau_panel_out, w_panel_in);

    auto& a   = const_cast<MatrixView<T, MatrixFormat::Dense>&>(a_panel_in);
    auto& e   = const_cast<VectorView<T>&>(e_panel_out);
    auto& tau = const_cast<VectorView<T>&>(tau_panel_out);
    auto& w   = const_cast<MatrixView<T, MatrixFormat::Dense>&>(w_panel_in);

    if (w.cols() == 0) {
        return ctx.get_event();
    }

    return latrd_lower_panel_batched<T>(ctx, a, e, tau, w, wg_hint, fuse_trailing_update);
}

template <Backend B, typename T>
Event latrd_lower_panel(Queue& ctx,
                        const MatrixView<T, MatrixFormat::Dense>& a_in,
                        const VectorView<T>& e_out,
                        const VectorView<T>& tau_out,
                        const MatrixView<T, MatrixFormat::Dense>& w_in,
                        int32_t j0,
                        int32_t ib,
                        int32_t wg_hint,
                        bool fuse_trailing_update) {
    (void)::batchlas::detail::call_mask("latrd_lower_panel", a_in, e_out, tau_out, w_in);
    (void)B;
    validate_latrd_lower_panel_dims(a_in, e_out, tau_out, w_in, j0, ib);

    auto& a   = const_cast<MatrixView<T, MatrixFormat::Dense>&>(a_in);
    auto& e   = const_cast<VectorView<T>&>(e_out);
    auto& tau = const_cast<VectorView<T>&>(tau_out);
    auto& w   = const_cast<MatrixView<T, MatrixFormat::Dense>&>(w_in);

    auto a_panel   = a({j0, SliceEnd()}, {j0, SliceEnd()});
    auto e_panel   = e(Slice(j0, j0 + ib));
    auto tau_panel = tau(Slice(j0, j0 + ib));
    auto w_panel   = w({j0, SliceEnd()}, {0, ib});
    return latrd_lower_panel<B, T>(ctx, a_panel, e_panel, tau_panel, w_panel, wg_hint, fuse_trailing_update);
}

#define LATRD_LOWER_PANEL_INSTANTIATE(back, fp) \
    template Event latrd_lower_panel<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        int32_t, \
        bool); \
    template Event latrd_lower_panel<back, BATCHLAS_UNPAREN fp>( \
        Queue&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const VectorView<BATCHLAS_UNPAREN fp>&, \
        const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, \
        int32_t, \
        int32_t, \
        int32_t, \
        bool);

BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(LATRD_LOWER_PANEL_INSTANTIATE)

#undef LATRD_LOWER_PANEL_INSTANTIATE

} // namespace batchlas
