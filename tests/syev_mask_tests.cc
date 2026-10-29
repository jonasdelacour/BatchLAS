// SPDX-License-Identifier: MIT
// syev honours a batch mask (W3b) in cta_fused, jacobi and the host backend's LAPACKE loop, each
// pinned and reached, on tests/masked_contract.hh: A, w and info of an inactive item untouched,
// active items the unmasked call's. The tridiagonal stages the composites call (sytrd_cta, sb2st in
// its three kernels, sb2st_hh with its back-transform, latrd's legacy, device and (CUDA) grid
// panels, and the lifted sytrd_blocked, sytrd_sy2sb and band reduction) and syev_align are checked
// without a pin. Every mask is device-only; data complex where T is, ld,
// stride and inc padded, holes inside packed sub-groups, and a 1024-item identical batch for the
// local-memory kernels. evidence: docs/design/masked-batches.md#masked-batches-tests
#include <gtest/gtest.h>

#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-vector.hh>

#include <src/ops/syev/choice.hh>
#include <src/ops/syev/sytrd/sytrd_sb2st_hh.hh>
#include <tests/masked_contract.hh>
#include <tests/test_support/storage.hh>
#include <tests/test_utils.hh>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <vector>

using namespace batchlas;

namespace {

namespace mc = masked_contract;
namespace ts = test_support;

template <typename T>
using Real = typename base_type<T>::type;

/// Item-dependent small values over 8 (identical items when `same`), complex where T is.
template <typename T>
void fill_items(T* p, std::size_t per, int batch, int salt, bool same) {
    for (std::size_t e = 0; e < per * std::size_t(batch); ++e) {
        const std::size_t item = same ? 0 : e / per, off = e % per;
        const double re = double(int((off * 7 + item * 3 + salt) % 23) - 11) / 8;
        const double im = double(int((off * 5 + item + salt) % 13) - 6) / 8;
        p[e] = MatrixView<T>::make(re, im);
    }
}

/// A batched vector at inc 2 with a gap after each item, poisoned; `item_bytes` covers the gap.
template <typename U>
struct PaddedVec {
    int n, batch, inc = 2, stride;
    UnifiedVector<U> buf;
    PaddedVec(int n_, int batch_) : n(std::max(1, n_)), batch(batch_), stride(2 * n + 3), buf(std::size_t(stride) * batch_) {}
    void poison() { std::fill_n(buf.data(), buf.size(), ts::poison<U>()); }
    VectorView<U> view(Span<const std::uint8_t> mask) {
        return VectorView<U>(buf.data(), n, batch, inc, stride).with_batch_mask(mask);
    }
    mc::Output output(std::string name) const {
        const std::size_t s = std::size_t(stride) * sizeof(U);
        return {std::move(name), reinterpret_cast<const std::byte*>(buf.data()), s, s};
    }
};

template <typename T, Backend B>
struct Cfg {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};
using Types = typename test_utils::backend_types<Cfg>::type;

template <typename Config>
class SyevMask : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    using R = Real<T>;
    static constexpr Backend B = Config::BackendVal;

    bool gpu() const { return this->ctx->device().type == DeviceType::GPU; }

    // One masked syev: A (overwritten), w and info. `wg_items` > 1: items per packed sub-group.
    struct Problem {
        Queue& q;
        int n, batch;
        JobType jobz;
        Uplo uplo;
        bool same;
        Matrix<T> A;
        UnifiedVector<R> w;  // syev's w is packed, n per item
        UnifiedVector<int32_t> info;
        Problem(Queue& q_, int n_, int batch_, JobType j, Uplo u, bool same_)
            : q(q_), n(n_), batch(batch_), jobz(j), uplo(u), same(same_), A(ts::padded<T>(n_, n_, batch_)),
              w(std::size_t(n_) * batch_), info(std::size_t(batch_)) {}

        mc::Case make_case(int wg_items) {
            mc::Case c;
            c.batch = batch;
            c.wg_items = wg_items;
            c.identical_items = same;
            c.prepare = [this] {
                fill_items(A.data_ptr(), std::size_t(A.stride()), batch, 3, same);
                std::fill_n(w.data(), w.size(), ts::poison<R>());
                std::fill_n(info.data(), info.size(), int32_t(-77));
            };
            c.call = [this](Span<const std::uint8_t> mask, Span<std::byte> ws) {
                syev<B, T>(q, A.view().with_batch_mask(mask), w.to_span(), jobz, uplo, ws,
                           info.to_span());
            };
            c.buffer_size = [this](Span<const std::uint8_t> mask) {
                return syev_buffer_size<B, T>(q, A.view().with_batch_mask(mask), w.to_span(),
                                              jobz, uplo);
            };
            c.outputs = [this] {
                return std::vector<mc::Output>{mc::Output::matrix("A", A.view()),
                                               mc::Output::span<R>("w", {w.data(), w.size()}, std::size_t(n)),
                                               mc::Output::span<int32_t>("info", {info.data(), info.size()}, 1)};
            };
            return c;
        }
    };

    void check(const std::string& spelling, int n, int batch, JobType jobz, Uplo uplo, int wg_items, bool same = false) {
        SCOPED_TRACE(spelling + " n " + std::to_string(n) + " batch " + std::to_string(batch) +
                     (jobz == JobType::EigenVectors ? " V " : " N ") + ts::uplo_name(uplo));
        Problem p(*this->ctx, n, batch, jobz, uplo, same);
        mc::check<ops::syev::SyevChoice>(*this->ctx, "syev", spelling, p.make_case(wg_items));
    }
};

TYPED_TEST_SUITE(SyevMask, Types);

// Items per 32-lane sub-group of the partition kernels: one item per P = next power of two >= n.
int items_per_subgroup(int n) {
    int p = 4;
    while (p < n) p *= 2;
    return 32 / p;
}

// cta_fused and jacobi (GPU), the LAPACKE loop (host): both jobz, both triangles, each partition size.
TYPED_TEST(SyevMask, EveryHonouringFamilyLeavesInactiveItemsUntouched) {
    const int batch = 24;
    std::vector<std::string> fams;
    if (this->gpu()) fams = {"cta_fused", "jacobi"};
    else fams = {"vendor"};
    for (const auto& f : fams)
        for (const int n : {3, 7, 13, 32})
            for (const JobType j : {JobType::EigenVectors, JobType::NoEigenVectors})
                for (const Uplo u : {Uplo::Lower, Uplo::Upper})
                    this->check(f, n, batch, j, u, items_per_subgroup(n));
    if (!this->gpu()) this->check("vendor", 45, 6, JobType::EigenVectors, Uplo::Lower, 0);
}

// The saturating batch: 1024 identical items, each active one bit-identical to the first.
TYPED_TEST(SyevMask, SaturatingIdenticalItems) {
    if (!this->gpu()) GTEST_SKIP() << "the partition kernels are GPU families";
    for (const auto* f : {"cta_fused", "jacobi"})
        for (const int n : {7, 32}) this->check(f, n, 1024, JobType::EigenVectors, Uplo::Lower, items_per_subgroup(n), true);
}

// sytrd_cta, the CTA reduction syev_cta and sytrd_blocked call: A, d, e and tau.
TYPED_TEST(SyevMask, SytrdCtaLeavesInactiveItemsUntouched) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::B;
    if (!this->gpu()) GTEST_SKIP() << "sytrd_cta is a GPU kernel";
    Queue& q = *this->ctx;
    for (const int batch : {24, 1024})
        for (const int n : {5, 13, 32})
            for (const Uplo u : {Uplo::Lower, Uplo::Upper}) {
                SCOPED_TRACE("n " + std::to_string(n) + " batch " + std::to_string(batch) + " " + ts::uplo_name(u));
                const bool same = batch >= 1024;
                Matrix<T> A = ts::padded<T>(n, n, batch);
                PaddedVec<T> d(n, batch), e(n - 1, batch), tau(n - 1, batch);
                mc::Case c;
                c.batch = batch;
                c.wg_items = items_per_subgroup(n);
                c.identical_items = same;
                c.prepare = [&] {
                    fill_items(A.data_ptr(), std::size_t(A.stride()), batch, 1, same);
                    d.poison(), e.poison(), tau.poison();
                };
                c.call = [&](Span<const std::uint8_t> m, Span<std::byte>) {
                    sytrd_cta<B, T>(q, A.view().with_batch_mask(m), d.view(m), e.view(m), tau.view(m), u, {});
                };
                c.outputs = [&] {
                    return std::vector<mc::Output>{mc::Output::matrix("A", A.view()), d.output("d"), e.output("e"),
                                                   tau.output("tau")};
                };
                mc::check(q, c);
            }
}

// sytrd_sb2st: the kd = 0 and kd = 1 copies (I3), the work-group chase (I1) and the sub-group chase (I2).
TYPED_TEST(SyevMask, Sb2stLeavesInactiveItemsUntouched) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;
    constexpr Backend B = TestFixture::B;
    if (!this->gpu()) GTEST_SKIP() << "sytrd_sb2st's chase is a GPU kernel";
    Queue& q = *this->ctx;
    const int n = 19, batch = 20;
    for (const int kd : {0, 1, 3, 6})
        for (const char* sub : {"0", "1"}) {
            if (kd <= 1 && sub[0] == '1') continue;
            SCOPED_TRACE("kd " + std::to_string(kd) + " sub-group " + sub);
            const ScopedEnvVar mode("BATCHLAS_SB2ST_SUBGROUP", sub);
            Matrix<T> AB = ts::padded<T>(kd + 1, n, batch);
            PaddedVec<R> d(n, batch), e(n - 1, batch);
            PaddedVec<T> tau(n - 1, batch);
            mc::Case c;
            c.batch = batch;
            c.wg_items = 32 / std::max(2, [&] { int p = 2; while (p < kd) p *= 2; return p; }());
            c.prepare = [&] {
                fill_items(AB.data_ptr(), std::size_t(AB.stride()), batch, 2, false);
                d.poison(), e.poison(), tau.poison();
            };
            c.call = [&](Span<const std::uint8_t> m, Span<std::byte> ws) {
                sytrd_sb2st<B, T>(q, AB.view().with_batch_mask(m), d.view(m), e.view(m), tau.view(m), Uplo::Lower, kd, ws, 0);
            };
            c.buffer_size = [&](Span<const std::uint8_t> m) {
                return sytrd_sb2st_buffer_size<B, T>(q, AB.view().with_batch_mask(m), d.view(m), e.view(m), tau.view(m),
                                                     Uplo::Lower, kd, 0);
            };
            c.outputs = [&] {
                return std::vector<mc::Output>{mc::Output::matrix("AB", AB.view()), d.output("d"), e.output("e"),
                                               tau.output("tau")};
            };
            mc::check(q, c);
        }
}

// sytrd_sb2st_hh's chase and unmqr_hb2st's three back-transform kernels (wave, tiled, plain).
TYPED_TEST(SyevMask, Sb2stHhAndItsBackTransformLeaveInactiveItemsUntouched) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;
    constexpr Backend B = TestFixture::B;
    if (!this->gpu()) GTEST_SKIP() << "sytrd_sb2st_hh is a GPU kernel";
    Queue& q = *this->ctx;
    const int n = 23, kd = 4, batch = 12;
    const auto sched = detail::build_sb2st_hh_schedule(n, kd);
    const int nrefl = std::max<int>(1, int(sched.size()));
    const auto wave_host = detail::build_sb2st_hh_wave_offsets(sched, n);
    UnifiedVector<int32_t> starts(sched.size()), lens(sched.size()), waves(wave_host.size());
    for (std::size_t k = 0; k < sched.size(); ++k) starts[k] = sched[k].start, lens[k] = sched[k].len;
    std::copy(wave_host.begin(), wave_host.end(), waves.data());
    Matrix<T> AB = ts::padded<T>(kd + 1, n, batch), tri = ts::padded<T>(2, n, batch), V = ts::padded<T>(kd, nrefl, batch);
    Matrix<T> Z = ts::padded<T>(n, n, batch);
    PaddedVec<R> d(n, batch), e(n - 1, batch);
    PaddedVec<T> tau(nrefl, batch);
    mc::Case c;
    c.batch = batch;
    c.prepare = [&] {
        fill_items(AB.data_ptr(), std::size_t(AB.stride()), batch, 4, false);
        fill_items(Z.data_ptr(), std::size_t(Z.stride()), batch, 6, false);
        std::fill_n(tri.data_ptr(), std::size_t(tri.stride()) * batch, ts::poison<T>());
        std::fill_n(V.data_ptr(), std::size_t(V.stride()) * batch, ts::poison<T>());
        d.poison(), e.poison(), tau.poison();
    };
    std::optional<std::string> back;  // BATCHLAS_SB2ST_BACK_WAVE / _TILE per run
    c.call = [&](Span<const std::uint8_t> m, Span<std::byte> ws) {
        detail::sytrd_sb2st_hh<B, T>(q, AB.view().with_batch_mask(m), tri.view().with_batch_mask(m), d.view(m), e.view(m),
                                     V.view().with_batch_mask(m), tau.view(m), Uplo::Lower, kd, ws);
        detail::unmqr_hb2st<B, T>(q, V.view().with_batch_mask(m), tau.view(m), Z.view().with_batch_mask(m), n, kd,
                                  {starts.data(), sched.size()}, {lens.data(), sched.size()},
                                  {waves.data(), wave_host.size()});
    };
    c.buffer_size = [&](Span<const std::uint8_t>) { return detail::sytrd_sb2st_hh_buffer_size<B, T>(q, n, kd, batch); };
    c.outputs = [&] {
        return std::vector<mc::Output>{mc::Output::matrix("AB", AB.view()), mc::Output::matrix("tri", tri.view()),
                                       mc::Output::matrix("V", V.view()), mc::Output::matrix("Z", Z.view()),
                                       d.output("d"), e.output("e"), tau.output("tau")};
    };
    {
        SCOPED_TRACE("wave back-transform");
        mc::check(q, c);
    }
    {
        SCOPED_TRACE("tiled back-transform");
        const ScopedEnvVar wave("BATCHLAS_SB2ST_BACK_WAVE", "0");
        mc::check(q, c);
    }
    {
        SCOPED_TRACE("plain back-transform");
        const ScopedEnvVar wave("BATCHLAS_SB2ST_BACK_WAVE", "0");
        const ScopedEnvVar tile("BATCHLAS_SB2ST_BACK_TILE", "0");
        mc::check(q, c);
    }
}

// latrd_lower_panel, sytrd_blocked's panel: the legacy and device kernels (I1), and on CUDA the grid
// kernel, whose G work-groups of an inactive item all leave before its grid barrier (it never runs
// on Metal or a host device, so a CUDA box has to cover it).
TYPED_TEST(SyevMask, LatrdPanelLeavesInactiveItemsUntouched) {
    using T = typename TestFixture::T;
    constexpr Backend B = TestFixture::B;
    if (!this->gpu()) GTEST_SKIP() << "latrd's panel is a GPU kernel";
    Queue& q = *this->ctx;
    const int ib = 4, j0 = 3, batch = 10;
    // grid at n = 100: four work-groups per item share its software grid barrier (40 resident).
    const bool cuda = B == Backend::CUDA;
    for (const char* impl : {"legacy", "device", "grid"}) {
        if (impl[0] == 'g' && !cuda) continue;  // never on Metal or a host device: a CUDA-only check
        const int n = impl[0] == 'g' ? 100 : 21;
        SCOPED_TRACE(impl);
        const ScopedEnvVar pin("BATCHLAS_LATRD_IMPL", impl);
        Matrix<T> A = ts::padded<T>(n, n, batch), W = ts::padded<T>(n, ib + 2, batch);
        PaddedVec<T> e(n - 1, batch), tau(n - 1, batch);
        mc::Case c;
        c.batch = batch;
        c.prepare = [&] {
            fill_items(A.data_ptr(), std::size_t(A.stride()), batch, 5, false);
            std::fill_n(W.data_ptr(), std::size_t(W.stride()) * batch, ts::poison<T>());
            e.poison(), tau.poison();
        };
        c.call = [&](Span<const std::uint8_t> m, Span<std::byte>) {
            latrd_lower_panel<B, T>(q, A.view().with_batch_mask(m), e.view(m), tau.view(m), W.view().with_batch_mask(m),
                                    j0, ib);
        };
        c.outputs = [&] {
            return std::vector<mc::Output>{mc::Output::matrix("A", A.view()), mc::Output::matrix("W", W.view()),
                                           e.output("e"), tau.output("tau")};
        };
        mc::check(q, c);
    }
}

// The lifted sytrd composites (lift 1), unpinned: sytrd_blocked (latrd panels, then syr2k / her2k),
// sytrd_sy2sb (geqrf panels, larft, symm / hemm, syr2k / her2k) and the band reduction (geqrf and
// ormqr per sweep). Their children route on their own Auto, which on a vendor backend takes the
// vendor unmasked, so the masked patterns compare with the all-active masked call.
TYPED_TEST(SyevMask, SytrdCompositesLeaveInactiveItemsUntouched) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;
    constexpr Backend B = TestFixture::B;
    if (!this->gpu()) GTEST_SKIP() << "on the host backend their level-3 children are its BLAS, which refuses";
    Queue& q = *this->ctx;
    const bool vendor = B == Backend::CUDA || B == Backend::ROCM;
    const int batch = 10;
    const Uplo u = Uplo::Lower;  // the only triangle the three composites implement
    // The tuned default fuses the trailing update into the latrd panel at these n; "0" runs
    // sytrd_blocked's own update kernel.
    for (const char* fuse : {"", "0"})
    for (const int n : {45, 70}) {
        {
            const ScopedEnvVar fuse_env("BATCHLAS_SYTRD_FUSE_PANEL_UPDATE", *fuse ? fuse : nullptr);
            SCOPED_TRACE("sytrd_blocked n " + std::to_string(n) + " " + ts::uplo_name(u) + " fuse " + fuse);
            Matrix<T> A = ts::padded<T>(n, n, batch);
            PaddedVec<T> d(n, batch), e(n - 1, batch), tau(n - 1, batch);
            mc::Case c;
            c.batch = batch;
            c.masked_reference = vendor;
            c.prepare = [&] {
                fill_items(A.data_ptr(), std::size_t(A.stride()), batch, 4, false);
                d.poison(), e.poison(), tau.poison();
            };
            c.call = [&](Span<const std::uint8_t> m, Span<std::byte> ws) {
                sytrd_blocked<B, T>(q, A.view().with_batch_mask(m), d.view(m), e.view(m), tau.view(m), u, ws, 16);
            };
            c.buffer_size = [&](Span<const std::uint8_t> m) {
                return sytrd_blocked_buffer_size<B, T>(q, A.view().with_batch_mask(m), d.view(m), e.view(m), tau.view(m), u,
                                                       16);
            };
            c.outputs = [&] {
                return std::vector<mc::Output>{mc::Output::matrix("A", A.view()), d.output("d"), e.output("e"),
                                               tau.output("tau")};
            };
            mc::check(q, c);
        }
    }
    for (const int kd : {4, 8}) {
        {
            const int n = 40;
            SCOPED_TRACE("sytrd_sy2sb n " + std::to_string(n) + " kd " + std::to_string(kd) + " " + ts::uplo_name(u));
            Matrix<T> A = ts::padded<T>(n, n, batch), AB = ts::padded<T>(kd + 1, n, batch);
            PaddedVec<T> tau(n - kd, batch);
            mc::Case c;
            c.batch = batch;
            c.masked_reference = vendor;
            c.prepare = [&] {
                fill_items(A.data_ptr(), std::size_t(A.stride()), batch, 6, false);
                std::fill_n(AB.data_ptr(), std::size_t(AB.stride()) * batch, ts::poison<T>());
                tau.poison();
            };
            c.call = [&](Span<const std::uint8_t> m, Span<std::byte> ws) {
                sytrd_sy2sb<B, T>(q, A.view().with_batch_mask(m), AB.view().with_batch_mask(m), tau.view(m), u, kd, ws);
            };
            c.buffer_size = [&](Span<const std::uint8_t> m) {
                return sytrd_sy2sb_buffer_size<B, T>(q, A.view().with_batch_mask(m), AB.view().with_batch_mask(m),
                                                     tau.view(m), u, kd);
            };
            c.outputs = [&] {
                return std::vector<mc::Output>{mc::Output::matrix("A", A.view()), mc::Output::matrix("AB", AB.view()),
                                               tau.output("tau")};
            };
            mc::check(q, c);
        }
    }
    for (const int kd : {3, 6}) {
        const int n = 30;
        SCOPED_TRACE("sytrd_band_reduction n " + std::to_string(n) + " kd " + std::to_string(kd));
        Matrix<T> AB = ts::padded<T>(kd + 1, n, batch);
        PaddedVec<R> d(n, batch), e(n - 1, batch);
        PaddedVec<T> tau(n - 1, batch);
        mc::Case c;
        c.batch = batch;
        c.masked_reference = vendor;
        c.prepare = [&] {
            fill_items(AB.data_ptr(), std::size_t(AB.stride()), batch, 7, false);
            d.poison(), e.poison(), tau.poison();
        };
        c.call = [&](Span<const std::uint8_t> m, Span<std::byte> ws) {
            sytrd_band_reduction<B, T>(q, AB.view().with_batch_mask(m), d.view(m), e.view(m), tau.view(m), Uplo::Lower, kd,
                                       ws, int32_t(4));
        };
        c.buffer_size = [&](Span<const std::uint8_t> m) {
            return sytrd_band_reduction_buffer_size<B, T>(q, AB.view().with_batch_mask(m), d.view(m), e.view(m), tau.view(m),
                                                          Uplo::Lower, kd, int32_t(4));
        };
        c.outputs = [&] {
            return std::vector<mc::Output>{mc::Output::matrix("AB", AB.view()), d.output("d"), e.output("e"),
                                           tau.output("tau")};
        };
        {
            // Not run-to-run deterministic on CUDA (kd 6 every run, kd 3 on item 3 now and then),
            // unmasked as masked and with the same zeroed workspace (an open defect, lift 1):
            // EXPECT_VERIFY on d and |e| against one unmasked call. Inactive items stay bit-for-bit checked.
            c.prepare();
            {
                UnifiedVector<std::byte> ws(std::max<std::size_t>(c.buffer_size({}), 1));
                c.call({}, ws.to_span());
                q.wait();
            }
            const std::vector<R> d_ref(d.buf.begin(), d.buf.end()), e_ref(e.buf.begin(), e.buf.end());
            c.verify_active = [&, d_ref, e_ref](const std::string&, std::span<const int> active) {
                R scale = 0;
                for (const R v : d_ref) scale = std::max(scale, std::abs(v));
                const R tol = R(1000) * std::numeric_limits<R>::epsilon() * std::max(scale, R(1));
                for (const int b : active) {
                    for (int i = 0; i < n; ++i) {
                        const std::size_t k = std::size_t(b) * d.stride + std::size_t(i) * d.inc;
                        EXPECT_NEAR(d.buf[k], d_ref[k], tol) << "d: active item " << b << " row " << i;
                    }
                    for (int i = 0; i < n - 1; ++i) {
                        const std::size_t k = std::size_t(b) * e.stride + std::size_t(i) * e.inc;
                        EXPECT_NEAR(std::abs(e.buf[k]), std::abs(e_ref[k]), tol) << "e: active item " << b << " row " << i;
                    }
                }
            };
        }
        mc::check(q, c);
    }
}

// syev_align: its gemm, its two kernels (I3) and the copy into its workspace. Eigenvalue clusters
// of 1, 2 and 3 columns so every Procrustes block size runs.
TYPED_TEST(SyevMask, SyevAlignLeavesInactiveItemsUntouched) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;
    constexpr Backend B = TestFixture::B;
    if (!this->gpu()) GTEST_SKIP() << "the host backend's gemm is its BLAS, which refuses a mask";
    Queue& q = *this->ctx;
    const int n = 11, m = 6, batch = 14;
    Matrix<T> V0 = ts::padded<T>(n, m, batch), V = ts::padded<T>(n, m, batch);
    UnifiedVector<R> lam(std::size_t(m) * batch);
    const R base[6] = {R(-1), R(0.5), R(0.5), R(2), R(2), R(2)};
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < m; ++j) lam[std::size_t(b) * m + j] = base[j] + R(b);
    mc::Case c;
    c.batch = batch;
    c.prepare = [&] {
        fill_items(V0.data_ptr(), std::size_t(V0.stride()), batch, 7, false);
        fill_items(V.data_ptr(), std::size_t(V.stride()), batch, 8, false);
    };
    c.call = [&](Span<const std::uint8_t> mk, Span<std::byte> ws) {
        syev_align<B, T>(q, V0.view().with_batch_mask(mk), lam.to_span(), V.view().with_batch_mask(mk), R(1e-3), ws);
    };
    c.buffer_size = [&](Span<const std::uint8_t> mk) {
        return syev_align_buffer_size<B, T>(q, V0.view().with_batch_mask(mk), lam.to_span(), V.view().with_batch_mask(mk));
    };
    c.outputs = [&] { return std::vector<mc::Output>{mc::Output::matrix("V", V.view()), mc::Output::matrix("V_old", V0.view())}; };
    mc::check(q, c);
}

}  // namespace
