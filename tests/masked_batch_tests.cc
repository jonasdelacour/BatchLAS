// Masked batches: syev_masked, sygvd_masked and sygvd_factored_masked. An inactive item's operands, eigenvalues
// and info come back bit-identical (they are poisoned, so a touched item shows); an active item's results are
// bit-identical to the unmasked call on the active items alone (same pin), satisfy A X = B X W (EXPECT_VERIFY),
// and are bit-identical to the unmasked whole batch for a per-item family and, on every family, in the
// saturating 1024 batch half masked (values bound otherwise). Patterns: none, all, alternating, single; every
// family the backend can pin; padded ld and stride; both triangles; one mask-independent workspace size.
// evidence: docs/design/masked-batches.md#masked-batches-tests
#include <gtest/gtest.h>

#include <batchlas/blas/functions/syev.hh>
#include <batchlas/blas/functions/sygvd.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/no_route.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/tolerance.hh>

#include <tests/sygvd_test_common.hh>
#include <tests/test_support/inputs.hh>
#include <tests/test_support/storage.hh>
#include <tests/test_utils.hh>

#include <src/ops/syev/choice.hh>
#include <src/ops/sygvd/choice.hh>
#include <src/select/select.hh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

using namespace batchlas;

namespace {

namespace ts = test_support;
using ts::Live;

template <typename T>
using MV = MatrixView<T, MatrixFormat::Dense>;

// A none / all / alternating / single mask over a batch, with its active and inactive items.
struct Mask {
    UnifiedVector<uint8_t> bits;
    std::vector<int> active, inactive;
    Span<const uint8_t> span() const { return {bits.data(), bits.size()}; }
};

Mask pattern(const std::string& name, int batch) {
    Mask m{UnifiedVector<uint8_t>(batch, 0), {}, {}};
    for (int b = 0; b < batch; ++b) {
        m.bits[b] = name == "all" || (name == "alternating" && b % 2) || (name == "single" && b == batch / 2);
        (m.bits[b] ? m.active : m.inactive).push_back(b);
    }
    return m;
}

// Items of len elements between a full batch and the packed batch of its active items, in order:
// `pack` gathers them out of full, else scatters packed back to their slots.
template <typename X>
void move_active(X* full, X* packed, std::size_t len, const std::vector<int>& active, bool pack) {
    for (std::size_t g = 0; g < active.size(); ++g) {
        X* f = full + static_cast<std::size_t>(active[g]) * len;
        X* p = packed + g * len;
        pack ? std::copy_n(f, len, p) : std::copy_n(p, len, f);
    }
}

// The active items of `full` as a batch of their own in the same ld and stride (none active: one poisoned item).
template <typename T>
Matrix<T> compact(Matrix<T>& full, const Mask& m) {
    Matrix<T> c = ts::padded<T>(full.rows(), full.cols(), std::max<int>(static_cast<int>(m.active.size()), 1));
    move_active(full.data_ptr(), c.data_ptr(), full.stride(), m.active, true);
    return c;
}

// The masked input: `full` with every inactive item's storage poisoned.
template <typename T>
Matrix<T> poisoned(const Matrix<T>& full, const Mask& m) {
    Matrix<T> got = full.clone();
    for (int b : m.inactive) std::fill_n(got.data_ptr() + static_cast<std::size_t>(b) * got.stride(), got.stride(), ts::poison<T>());
    return got;
}

// w as an n x 1 matrix per item, for the bit-identity checks.
template <typename R>
MV<R> items_of(const UnifiedVector<R>& w, int n) {
    return MV<R>(const_cast<R*>(w.data()), n, 1, n, n, static_cast<int>(w.size()) / n);
}

// Families whose per-item result does not depend on the batch around it; the others (blocked, two_stage,
// compose) select children and size launches by batch size, so their reference is the active items alone.
bool per_item_family(const std::string& fam) { return fam != "blocked" && fam != "two_stage" && fam != "compose"; }

template <typename Config>
class MaskedTest : public sygvd_test::SygvdFixture<Config> {
protected:
    using Base = sygvd_test::SygvdFixture<Config>;
    using T = typename Config::ScalarType;
    using R = batchlas::float_t<T>;
    static constexpr Backend BK = Config::BackendVal;
    using Base::q;
    using Base::vendor_linked;

    // n x n random Hermitian items (HPD for `spd`), packed, and their uplo triangles in padded, poisoned storage.
    std::pair<Matrix<T>, Matrix<T>> make(int n, int batch, Uplo uplo, unsigned seed, bool spd) {
        Matrix<T> h = spd ? ts::hpd<T>(q(), n, batch, 1.0, seed) : Matrix<T>(n, n, batch);
        if (!spd) (void)h.view().fill_random(q(), true, seed);
        q().wait();
        Matrix<T> s = ts::padded<T>(n, n, batch);
        ts::store<T>(s, h, Live::triangle(uplo));
        return {std::move(h), std::move(s)};
    }

    // Every syev family a pin reaches at order n on this backend.
    std::vector<std::string> syev_families(int n) {
        std::vector<std::string> f;
        if (BK != Backend::NETLIB && q().device().type == DeviceType::GPU) {
            if (n <= 32 && q().device().supports_sub_group_size(32)) f = {"cta", "cta_fused", "jacobi"};
            f.push_back("blocked");
            f.push_back("two_stage");
        }
        if (vendor_linked()) f.push_back("vendor");
        return f;
    }

    void run_syev(const MV<T>& a, UnifiedVector<R>& w, UnifiedVector<int32_t>& info, Uplo uplo) {
        const size_t need = syev_buffer_size<BK, T>(q(), a, w.to_span(), JobType::EigenVectors, uplo);
        UnifiedVector<std::byte> ws(std::max<size_t>(1, need));
        (void)syev<BK, T>(q(), a, w.to_span(), JobType::EigenVectors, uplo, ws.to_span(), info.to_span());
        q().wait();
    }

    void run_sygvd(const MV<T>& a, const MV<T>& b, UnifiedVector<R>& w, UnifiedVector<int32_t>& info, bool factored, Uplo uplo) {
        const JobType V = JobType::EigenVectors;
        const size_t need = factored ? sygvd_factored_buffer_size<BK, T>(q(), a, b, w.to_span(), 1, V, uplo)
                                     : sygvd_buffer_size<BK, T>(q(), a, b, w.to_span(), 1, V, uplo);
        UnifiedVector<std::byte> ws(std::max<size_t>(1, need));
        if (factored)
            (void)sygvd_factored<BK, T>(q(), a, b, w.to_span(), 1, V, uplo, ws.to_span(), info.to_span());
        else
            (void)sygvd<BK, T>(q(), a, b, w.to_span(), 1, V, uplo, ws.to_span(), info.to_span());
        q().wait();
    }

    // Storage (whole stride), eigenvalues and info bit-identical to the reference's on `items`.
    static void expect_same(const MV<T>& got, const MV<T>& ref, const UnifiedVector<R>& w, const UnifiedVector<R>& wr,
                            const UnifiedVector<int32_t>& info, const UnifiedVector<int32_t>& ir,
                            const std::vector<int>& items, const std::string& at) {
        ASSERT_TRUE(ts::items_bit_identical_to(got, ref, items)) << at;
        ASSERT_TRUE(ts::items_bit_identical_to(items_of(w, got.rows()), items_of(wr, got.rows()), items)) << at << ": w";
        for (int b : items) ASSERT_EQ(info[b], ir[b]) << at << " info item " << b;
    }

    // Against the unmasked call on the whole batch: bit-identical for a per-item family and for every family
    // at the saturating batch (k = 512 and 1024 select the same children: measured, pinned here), eigenvalues
    // within the values bound for the others.
    template <class Run>
    void against_full(const std::string& fam, const MV<T>& got, const Matrix<T>& full, const Mask& m,
                      const UnifiedVector<R>& w, const UnifiedVector<int32_t>& info, const std::string& at, Run run) {
        const int n = got.rows(), batch = got.batch_size();
        Matrix<T> f = full.clone();
        UnifiedVector<R> wf(w.size(), R(-4321));
        UnifiedVector<int32_t> inf(info.size(), -12345);
        run(f, wf, inf);
        if (per_item_family(fam) || batch >= 1024) return expect_same(got, f, w, wf, info, inf, m.active, at + " vs full batch");
        std::vector<std::vector<double>> ref(batch);
        double scale = 1;
        for (int b = 0; b < batch; ++b)
            for (int j = 0; j < n; ++j) {
                ref[b].push_back(double(wf[static_cast<size_t>(b) * n + j]));
                scale = std::max(scale, std::fabs(ref[b].back()));
            }
        const double ve = MV<T>::values_error(VectorView<R>(const_cast<R*>(w.data()), n, batch), ref, scale, m.active);
        EXPECT_VERIFY(T, verify::Check::values, n, ve) << at << " vs full batch";
    }
};

using MaskedTestTypes = typename test_utils::backend_types<ts::Config>::type;

}  // namespace

TYPED_TEST_SUITE(MaskedTest, MaskedTestTypes);

// syev_masked on every pinned family, every pattern, both triangles, batch 9 and a saturating 1024 half masked.
// ARMED BREAK: in src/kernels/common/batch_mask.cc move_items, index the full batch by `g` instead of `ix[g]`.
// SEEN: RED for the alternating and single patterns (and the 1024 batch) on every family; none and all GREEN.
// ARMED BREAK (2026-10-26): in both *_masked.cc buffer sizes, largest_over_counts(0, ...). SEEN: RED in both
// masked tests at the first partial pattern (workspace_error), all 8 configs; ShortMaskIsRefused GREEN.
// ARMED BREAK (2026-10-26): in syev.cc syev(), A.hermitize(q, Lower) after the launch at batch 4 or 512.
// SEEN: RED for alternating only (k = 4, 512) on every syev family and sygvd compose: blocked and two_stage
// at batch 9 by the residual alone, every family at 1024 also "vs full batch"; single, none, all GREEN.
TYPED_TEST(MaskedTest, SyevMaskedTouchesOnlyActiveItemsAndMatchesTheUnmaskedRun) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;
    constexpr Backend BK = TestFixture::BK;
    for (int n : {12, 40})
        for (const std::string& fam : this->syev_families(n))
            for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
                for (int batch : {9, 1024})
                    for (const std::string& pat : {"none", "all", "alternating", "single"}) {
                        if (batch == 1024 && pat != "alternating") continue;
                        if (batch == 1024 && uplo == Uplo::Upper) continue;
                        select::ScopedPin<ops::syev::SyevChoice> pin("syev", fam);
                        const std::string at = fam + " n=" + std::to_string(n) + " " + ts::uplo_name(uplo) + " batch=" +
                                               std::to_string(batch) + " " + pat;
                        const Mask m = pattern(pat, batch);
                        auto [A0, full] = this->make(n, batch, uplo, 11u + n + batch, false);
                        Matrix<T> got = poisoned(full, m), ref = got.clone();
                        Matrix<T> c = compact(full, m);
                        const int k = c.batch_size();
                        UnifiedVector<R> wc(static_cast<size_t>(n) * k, R(-4321)), w(static_cast<size_t>(n) * batch, R(-4321)), wr = w;
                        UnifiedVector<int32_t> ic(k, -12345), info(batch, -12345), ir = info;
                        UnifiedVector<uint8_t> ones(batch, 1);
                        size_t need = 0, need_all = 0;
                        try {
                            if (!m.active.empty()) this->run_syev(c, wc, ic, uplo);
                            need = syev_masked_buffer_size<BK, T>(this->q(), got, w.to_span(), JobType::EigenVectors, uplo, m.span());
                            need_all = syev_masked_buffer_size<BK, T>(this->q(), got, w.to_span(), JobType::EigenVectors, uplo,
                                                                      Span<const uint8_t>(ones.data(), ones.size()));
                        } catch (const NoRouteError&) {
                            GTEST_SKIP() << "no syev route";
                        }
                        // Reference: the unmasked call on the active items alone, back in their slots of the masked input.
                        move_active(ref.data_ptr(), c.data_ptr(), full.stride(), m.active, false);
                        move_active(wr.data(), wc.data(), n, m.active, false);
                        move_active(ir.data(), ic.data(), 1, m.active, false);
                        ASSERT_EQ(need, need_all) << at << ": the size depends on the mask";
                        // The all-active size is the one an SCC loop queries once and reuses as items drop out.
                        UnifiedVector<std::byte> ws(std::max<size_t>(1, need_all));
                        (void)syev_masked<BK, T>(this->q(), got, w.to_span(), JobType::EigenVectors, uplo,
                                                 Span<std::byte>(ws.data(), pat == "none" ? 0 : need_all), info.to_span(), m.span());
                        this->q().wait();
                        // Inactive items still poison and sentinels, active ones the reference's.
                        this->expect_same(got, ref, w, wr, info, ir, got.all_items(), at);
                        if (m.active.empty()) continue;
                        const double res = MV<T>::generalized_eigen_residual(this->q(), A0, MV<T>{}, got,
                                                                             VectorView<R>(w.data(), n, batch), 1, m.active);
                        EXPECT_VERIFY(T, verify::Check::eigen_residual, n, res) << at;
                        this->against_full(fam, got, full, m, w, info, at, [&](Matrix<T>& f, UnifiedVector<R>& wf, UnifiedVector<int32_t>& inf) {
                            this->run_syev(f, wf, inf, uplo);
                        });
                    }
}

// sygvd_masked and sygvd_factored_masked on every pinned family: A, B (or the factor), eigenvalues and info of
// inactive items untouched; active items bit-identical to the unmasked sygvd / sygvd_factored.
// ARMED BREAK: in src/ops/sygvd/sygvd_masked.cc run_masked, drop the scatter of B (`if (!a.factored) ...`).
// SEEN: RED at the first partial-pattern sygvd_masked case (active B), all 8 configs; syev_masked GREEN.
TYPED_TEST(MaskedTest, SygvdMaskedTouchesOnlyActiveItemsAndMatchesTheUnmaskedRun) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;
    constexpr Backend BK = TestFixture::BK;
    for (bool factored : {false, true})
        for (const std::string& fam : this->sygvd_families())
            for (int n : {6, 40})
                for (Uplo uplo : {Uplo::Lower, Uplo::Upper})
                    for (int batch : {9, 1024})
                        for (const std::string& pat : {"none", "all", "alternating", "single"}) {
                            if (factored && fam == "vendor") continue;  // no library takes a factored B
                            if (batch == 1024 && (pat != "alternating" || uplo == Uplo::Upper || n != 6)) continue;
                            select::ScopedPin<ops::sygvd::SygvdChoice> pin("sygvd", fam);
                            const std::string at = fam + (factored ? " factored" : "") + " n=" + std::to_string(n) + " " +
                                                   ts::uplo_name(uplo) + " batch=" + std::to_string(batch) + " " + pat;
                            const Mask m = pattern(pat, batch);
                            auto [A0, Afull] = this->make(n, batch, uplo, 21u + n, false);
                            auto [B0, Bfull] = this->make(n, batch, uplo, 22u + n, true);  // B0: B before any factorization
                            if (factored) {
                                UnifiedVector<int32_t> pinfo(batch, 0);
                                if (!ts::potrf_in_place<BK, T>(this->q(), Bfull, uplo, pinfo.to_span())) GTEST_SKIP() << "no potrf route";
                                ts::poison_outside<T>(Bfull, Live::triangle(uplo));  // whatever potrf left there
                            }
                            Matrix<T> A = poisoned(Afull, m), B = poisoned(Bfull, m), refA = A.clone(), refB = B.clone();
                            Matrix<T> cA = compact(Afull, m), cB = compact(Bfull, m);
                            const int k = cA.batch_size();
                            UnifiedVector<R> wc(static_cast<size_t>(n) * k, R(-4321)), w(static_cast<size_t>(n) * batch, R(-4321)), wr = w;
                            UnifiedVector<int32_t> ic(k, -12345), info(batch, -12345), ir = info;
                            const JobType V = JobType::EigenVectors;
                            UnifiedVector<uint8_t> ones(batch, 1);
                            const Span<const uint8_t> all1(ones.data(), ones.size());
                            size_t need = 0, need_all = 0;
                            try {
                                if (!m.active.empty()) this->run_sygvd(cA, cB, wc, ic, factored, uplo);
                                need = factored ? sygvd_factored_masked_buffer_size<BK, T>(this->q(), A, B, w.to_span(), 1, V, uplo, m.span())
                                                : sygvd_masked_buffer_size<BK, T>(this->q(), A, B, w.to_span(), 1, V, uplo, m.span());
                                need_all = factored ? sygvd_factored_masked_buffer_size<BK, T>(this->q(), A, B, w.to_span(), 1, V, uplo, all1)
                                                    : sygvd_masked_buffer_size<BK, T>(this->q(), A, B, w.to_span(), 1, V, uplo, all1);
                            } catch (const NoRouteError&) {
                                GTEST_SKIP() << "no sygvd route";
                            }
                            move_active(refA.data_ptr(), cA.data_ptr(), Afull.stride(), m.active, false);
                            move_active(refB.data_ptr(), cB.data_ptr(), Bfull.stride(), m.active, false);
                            move_active(wr.data(), wc.data(), n, m.active, false);
                            move_active(ir.data(), ic.data(), 1, m.active, false);
                            ASSERT_EQ(need, need_all) << at << ": the size depends on the mask";
                            UnifiedVector<std::byte> ws(std::max<size_t>(1, need_all));
                            const Span<std::byte> s(ws.data(), pat == "none" ? 0 : need_all);
                            if (factored)
                                (void)sygvd_factored_masked<BK, T>(this->q(), A, B, w.to_span(), 1, V, uplo, s, info.to_span(), m.span());
                            else
                                (void)sygvd_masked<BK, T>(this->q(), A, B, w.to_span(), 1, V, uplo, s, info.to_span(), m.span());
                            this->q().wait();
                            this->expect_same(A, refA, w, wr, info, ir, A.all_items(), at);
                            ASSERT_TRUE(ts::items_bit_identical_to<T>(B, refB)) << at << ": B";
                            if (m.active.empty()) continue;
                            const double res = MV<T>::generalized_eigen_residual(this->q(), A0, B0, A, VectorView<R>(w.data(), n, batch), 1, m.active);
                            EXPECT_VERIFY(T, verify::Check::eigen_residual, n, res) << at;
                            this->against_full(fam, A, Afull, m, w, info, at, [&](Matrix<T>& f, UnifiedVector<R>& wf, UnifiedVector<int32_t>& inf) {
                                Matrix<T> fb = Bfull.clone();
                                this->run_sygvd(f, fb, wf, inf, factored, uplo);
                            });
                        }
}

// A mask shorter than the batch is refused before anything runs.
TYPED_TEST(MaskedTest, ShortMaskIsRefused) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;
    constexpr Backend BK = TestFixture::BK;
    Matrix<T> A = ts::padded<T>(4, 4, 3);
    UnifiedVector<R> w(12);
    UnifiedVector<uint8_t> m(2, 1);
    UnifiedVector<std::byte> ws(1);
    EXPECT_THROW(((void)syev_masked<BK, T>(this->q(), A, w.to_span(), JobType::EigenVectors, Uplo::Lower,
                                         ws.to_span(), Span<int32_t>{}, Span<const uint8_t>(m.data(), m.size()))),
                 batchlas::invalid_argument);
    EXPECT_THROW(((void)sygvd_masked<BK, T>(this->q(), A, A, w.to_span(), 1, JobType::EigenVectors,
                                          Uplo::Lower, ws.to_span(), Span<int32_t>{}, Span<const uint8_t>(m.data(), m.size()))),
                 batchlas::invalid_argument);
}
