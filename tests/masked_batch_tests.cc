// Masked batches: syev_masked, sygvd_masked and sygvd_factored_masked. An inactive item's operands, eigenvalues
// and info come back bit-identical (they are poisoned, so a touched item shows); an active item's results are
// bit-identical to the unmasked call on the active items alone (same pin), satisfy A X = B X W (EXPECT_VERIFY),
// and are bit-identical to the unmasked whole batch for a per-item family and, on every family, in the
// saturating 1024 batch half masked (values bound otherwise). Patterns: none, all, alternating, single; every
// family the backend can pin; padded ld and stride; both triangles; one mask-independent workspace size.
// evidence: docs/design/masked-batches.md#masked-batches-tests
#include <gtest/gtest.h>

#include <batchlas/backend_config.h>
#include <batchlas/blas/functions/potrf.hh>
#include <batchlas/blas/functions/syev.hh>
#include <batchlas/blas/functions/sygvd.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/no_route.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/verify/tolerance.hh>

#include <tests/test_utils.hh>

#include <src/ops/potrf/choice.hh>
#include <src/ops/syev/choice.hh>
#include <src/ops/sygvd/choice.hh>
#include <src/select/select.hh>

#include <complex>
#include <cstdint>
#include <cstring>
#include <optional>
#include <random>
#include <string>
#include <vector>

using namespace batchlas;

namespace {

template <typename T>
using MV = MatrixView<T, MatrixFormat::Dense>;
template <typename T>
using RealOf = typename batchlas::base_type<T>::type;

template <typename T, Backend B>
struct MaskConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};

template <typename T>
bool same_bits(const T& a, const T& b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

template <typename T>
T poison() { return MV<T>::make(9.0e3, -7.0e3); }

// A batch of n x n items at ld = n + 3 and stride ld * n + 5; item b holds a Hermitian matrix (spd adds
// a diagonal shift that makes it positive definite) in its uplo triangle, poison elsewhere.
template <typename T>
struct Batch {
    int n = 0, batch = 0, ld = 0, stride = 0;
    UnifiedVector<T> v;
    UnifiedVector<T*> p;
    MV<T> view() { return {v.data(), n, n, ld, stride, batch, p.data()}; }
    T& at(int b, int i, int j) { return v[static_cast<size_t>(b) * stride + static_cast<size_t>(j) * ld + i]; }
};

template <typename T>
Batch<T> make_batch(int n, int batch, Uplo uplo, unsigned seed, bool spd) {
    using R = RealOf<T>;
    Batch<T> m;
    m.n = n, m.batch = batch, m.ld = n + 3, m.stride = m.ld * n + 5;
    m.v = UnifiedVector<T>(static_cast<size_t>(m.stride) * batch, poison<T>());
    m.p = UnifiedVector<T*>(batch, nullptr);
    std::mt19937 gen(seed);
    std::uniform_real_distribution<R> d(R(-1), R(1));
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                if (uplo == Uplo::Lower ? i < j : i > j) continue;
                const double re = d(gen), im = i == j ? 0.0 : double(d(gen));
                m.at(b, i, j) = i == j ? MV<T>::make(spd ? 3.0 + std::fabs(re) : re) : MV<T>::make(re / (spd ? n : 1), im / (spd ? n : 1));
            }
    return m;
}

// The masked input: the reference batch with every inactive item's storage replaced by poison.
template <typename T>
Batch<T> poisoned(const Batch<T>& ref, const std::vector<uint8_t>& mask) {
    Batch<T> m = ref;
    for (int b = 0; b < m.batch; ++b)
        if (!mask[b])
            for (int o = 0; o < m.stride; ++o) m.v[static_cast<size_t>(b) * m.stride + o] = poison<T>();
    return m;
}

// The active items of `full`, packed into a k-item batch with the same ld and stride (k = 0: one dummy item).
template <typename T>
Batch<T> compact(const Batch<T>& full, const std::vector<uint8_t>& mask) {
    Batch<T> c;
    c.n = full.n, c.ld = full.ld, c.stride = full.stride, c.batch = 0;
    for (uint8_t a : mask) c.batch += a != 0;
    c.v = UnifiedVector<T>(static_cast<size_t>(c.stride) * std::max(c.batch, 1), poison<T>());
    c.p = UnifiedVector<T*>(std::max(c.batch, 1), nullptr);
    for (int b = 0, g = 0; b < full.batch; ++b)
        if (mask[b]) {
            for (int o = 0; o < c.stride; ++o)
                c.v[static_cast<size_t>(g) * c.stride + o] = full.v[static_cast<size_t>(b) * c.stride + o];
            ++g;
        }
    return c;
}

// Item g of the packed batch back to the g-th active item of `full`; `len` entries per item.
template <typename X>
void expand(UnifiedVector<X>& full, const UnifiedVector<X>& packed, int len, const std::vector<uint8_t>& mask) {
    for (size_t b = 0, g = 0; b < mask.size(); ++b)
        if (mask[b]) {
            for (int o = 0; o < len; ++o) full[b * len + o] = packed[g * len + o];
            ++g;
        }
}

// Families whose per-item result does not depend on the batch around it; the others (blocked, two_stage,
// compose) select children and size launches by batch size, so their reference is the active items alone.
bool per_item_family(const std::string& fam) { return fam != "blocked" && fam != "two_stage" && fam != "compose"; }

std::vector<uint8_t> pattern(const std::string& name, int batch) {
    std::vector<uint8_t> m(batch, 0);
    for (int b = 0; b < batch; ++b) {
        if (name == "all") m[b] = 1;
        if (name == "alternating") m[b] = b % 2;
        if (name == "single") m[b] = b == batch / 2;
    }
    return m;
}

template <typename Config>
class MaskedTest : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    using R = RealOf<T>;
    static constexpr Backend BK = Config::BackendVal;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx) GTEST_SKIP() << "no queue";
        // cuSOLVER's Upper potrf overwrites the lower triangle and would wipe B's poison (agent guide 8.4).
        if constexpr (BK != Backend::NETLIB) potrf_pin_.emplace("potrf", "native");
    }
    Queue& q() { return *this->ctx; }
    std::optional<select::ScopedPin<ops::potrf::PotrfChoice>> potrf_pin_;

    // Item b of `m` as the whole Hermitian matrix its uplo triangle holds.
    static Matrix<T> hermitian(const Batch<T>& m, Uplo uplo) {
        Matrix<T> H(m.n, m.n, m.batch);
        auto el = [&](int b, int i, int j) { return m.v[static_cast<size_t>(b) * m.stride + static_cast<size_t>(j) * m.ld + i]; };
        for (int b = 0; b < m.batch; ++b)
            for (int j = 0; j < m.n; ++j)
                for (int i = 0; i < m.n; ++i) {
                    const bool own = uplo == Uplo::Lower ? i >= j : i <= j;
                    H(i, j, b) = own ? el(b, i, j) : MV<T>::conj(el(b, j, i));
                    if (i == j) H(i, j, b) = MV<T>::make(double(std::real(H(i, j, b))));
                }
        return H;
    }

    // ||A0 X - B0 X diag(w)|| / (||A0|| ||X||) over the active items: A0, B0 the operands before the call
    // (b null: B0 = I), X the eigenvectors in got's storage. A reference product, never the routes under test.
    double residual(const Batch<T>& a, const Batch<T>* b, Batch<T>& got, const UnifiedVector<R>& w, Uplo uplo,
                    const std::vector<uint8_t>& mask) {
        const int n = got.n, batch = got.batch;
        Matrix<T> A0 = hermitian(a, uplo), X(n, n, batch), P(n, n, batch), XW(n, n, batch);
        for (int it = 0; it < batch; ++it)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) X(i, j, it) = got.at(it, i, j);
        if (b) {
            Matrix<T> B0 = hermitian(*b, uplo);
            (void)MV<T>::product(q(), B0.view(), Transpose::NoTrans, X.view(), Transpose::NoTrans, P.view());
            q().wait();
        }
        for (int it = 0; it < batch; ++it)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) XW(i, j, it) = (b ? P(i, j, it) : X(i, j, it)) * T(w[static_cast<size_t>(it) * n + j]);
        std::vector<int> items;
        for (int it = 0; it < batch; ++it)
            if (mask[it]) items.push_back(it);
        return MV<T>::solve_residual(q(), A0, X, XW, items);
    }

    static constexpr bool vendor_linked() {
        if constexpr (BK == Backend::CUDA) return BATCHLAS_HAS_CUSOLVER;
        if constexpr (BK == Backend::NETLIB) return BATCHLAS_HAS_LAPACKE && BATCHLAS_HAS_CBLAS;
        return false;
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
    std::vector<std::string> sygvd_families() {
        std::vector<std::string> f{"compose"};
        if (vendor_linked()) f.push_back("vendor");
        return f;
    }

    // For `inactive` items: storage bit-identical to `before`, eigenvalues and info still their sentinels.
    void expect_untouched(const char* what, Batch<T>& got, const Batch<T>& before, const std::vector<uint8_t>& mask,
                          const UnifiedVector<R>& w, const UnifiedVector<int32_t>& info, const std::string& at) {
        for (int b = 0; b < got.batch; ++b) {
            if (mask[b]) continue;
            for (int o = 0; o < got.stride; ++o)
                ASSERT_TRUE(same_bits(got.v[static_cast<size_t>(b) * got.stride + o], before.v[static_cast<size_t>(b) * got.stride + o]))
                    << at << ": inactive " << what << " item " << b << " written at " << o;
            for (int j = 0; j < got.n; ++j) ASSERT_TRUE(same_bits(w[static_cast<size_t>(b) * got.n + j], R(-4321))) << at << " w item " << b;
            ASSERT_EQ(info[b], -12345) << at << " info item " << b;
        }
    }

    void run_syev(Batch<T>& b, UnifiedVector<R>& w, UnifiedVector<int32_t>& info, Uplo uplo) {
        const size_t need = syev_buffer_size<BK, T>(q(), b.view(), w.to_span(), JobType::EigenVectors, uplo);
        UnifiedVector<std::byte> ws(std::max<size_t>(1, need));
        (void)syev<BK, T>(q(), b.view(), w.to_span(), JobType::EigenVectors, uplo, ws.to_span(), info.to_span());
        q().wait();
    }

    void run_sygvd(Batch<T>& a, Batch<T>& b, UnifiedVector<R>& w, UnifiedVector<int32_t>& info, bool factored, Uplo uplo) {
        const JobType V = JobType::EigenVectors;
        const size_t need = factored ? sygvd_factored_buffer_size<BK, T>(q(), a.view(), b.view(), w.to_span(), 1, V, uplo)
                                     : sygvd_buffer_size<BK, T>(q(), a.view(), b.view(), w.to_span(), 1, V, uplo);
        UnifiedVector<std::byte> ws(std::max<size_t>(1, need));
        if (factored)
            (void)sygvd_factored<BK, T>(q(), a.view(), b.view(), w.to_span(), 1, V, uplo, ws.to_span(), info.to_span());
        else
            (void)sygvd<BK, T>(q(), a.view(), b.view(), w.to_span(), 1, V, uplo, ws.to_span(), info.to_span());
        q().wait();
    }

    // Against the unmasked call on the whole batch: bit-identical for a per-item family and for every family
    // at the saturating batch (k = 512 and 1024 select the same children: measured, pinned here), eigenvalues
    // within the values bound for the others.
    template <class Run>
    void against_full(const std::string& fam, Batch<T>& got, const Batch<T>& full, const std::vector<uint8_t>& mask,
                      const UnifiedVector<R>& w, const UnifiedVector<int32_t>& info, const std::string& at, Run run) {
        auto f = full;
        UnifiedVector<R> wf(w.size(), R(-4321));
        UnifiedVector<int32_t> inf(info.size(), -12345);
        run(f, wf, inf);
        if (per_item_family(fam) || got.batch >= 1024) {
            expect_same_active(got, f, mask, w, wf, info, inf, at + " vs full batch");
            return;
        }
        std::vector<std::vector<double>> ref(got.batch);
        std::vector<int> items;
        double scale = 1;
        for (int b = 0; b < got.batch; ++b) {
            for (int j = 0; j < got.n; ++j) {
                ref[b].push_back(double(wf[static_cast<size_t>(b) * got.n + j]));
                scale = std::max(scale, std::fabs(ref[b].back()));
            }
            if (mask[b]) items.push_back(b);
        }
        const double ve = MV<T>::values_error(VectorView<R>(const_cast<R*>(w.data()), got.n, got.batch), ref, scale, items);
        EXPECT_VERIFY(T, verify::Check::values, got.n, ve) << at << " vs full batch";
    }

    // For active items: storage, eigenvalues and info bit-identical to the unmasked reference run's.
    void expect_same_active(Batch<T>& got, Batch<T>& ref, const std::vector<uint8_t>& mask, const UnifiedVector<R>& w,
                            const UnifiedVector<R>& wr, const UnifiedVector<int32_t>& info,
                            const UnifiedVector<int32_t>& ir, const std::string& at) {
        for (int b = 0; b < got.batch; ++b) {
            if (!mask[b]) continue;
            ASSERT_EQ(info[b], ir[b]) << at << " info item " << b;
            for (int j = 0; j < got.n; ++j)
                ASSERT_TRUE(same_bits(w[static_cast<size_t>(b) * got.n + j], wr[static_cast<size_t>(b) * got.n + j]))
                    << at << " w item " << b << " j " << j;
            for (int j = 0; j < got.n; ++j)
                for (int i = 0; i < got.n; ++i)
                    ASSERT_TRUE(same_bits(got.at(b, i, j), ref.at(b, i, j))) << at << " item " << b << " (" << i << "," << j << ")";
        }
    }
};

using MaskedTestTypes = typename test_utils::backend_types<MaskConfig>::type;

const char* ul(Uplo u) { return u == Uplo::Lower ? "L" : "U"; }

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
                        const std::string at = fam + " n=" + std::to_string(n) + " " + ul(uplo) + " batch=" +
                                               std::to_string(batch) + " " + pat;
                        const auto mask = pattern(pat, batch);
                        auto full = make_batch<T>(n, batch, uplo, 11u + n + batch, false);
                        auto got = poisoned(full, mask);
                        const auto before = got;
                        // Reference: the unmasked call on the active items alone, expanded back to their slots.
                        auto c = compact(full, mask);
                        const int k = std::max(c.batch, 1);
                        UnifiedVector<R> wc(static_cast<size_t>(n) * k, R(-4321)), w(static_cast<size_t>(n) * batch, R(-4321));
                        UnifiedVector<int32_t> ic(k, -12345), info(batch, -12345);
                        UnifiedVector<uint8_t> m(batch);
                        for (int b = 0; b < batch; ++b) m[b] = mask[b];
                        const Span<const uint8_t> ms(m.data(), m.size());
                        UnifiedVector<uint8_t> ones(batch, 1);
                        size_t need = 0, need_all = 0;
                        try {
                            if (c.batch > 0) this->run_syev(c, wc, ic, uplo);
                            need = syev_masked_buffer_size<BK, T>(this->q(), got.view(), w.to_span(), JobType::EigenVectors, uplo, ms);
                            need_all = syev_masked_buffer_size<BK, T>(this->q(), got.view(), w.to_span(), JobType::EigenVectors, uplo,
                                                                      Span<const uint8_t>(ones.data(), ones.size()));
                        } catch (const NoRouteError&) {
                            GTEST_SKIP() << "no syev route";
                        }
                        auto ref = full;
                        UnifiedVector<R> wr(w.size(), R(-4321));
                        UnifiedVector<int32_t> ir(batch, -12345);
                        expand(ref.v, c.v, full.stride, mask);
                        expand(wr, wc, n, mask);
                        expand(ir, ic, 1, mask);
                        ASSERT_EQ(need, need_all) << at << ": the size depends on the mask";
                        // The all-active size is the one an SCC loop queries once and reuses as items drop out.
                        UnifiedVector<std::byte> ws(std::max<size_t>(1, need_all));
                        (void)syev_masked<BK, T>(this->q(), got.view(), w.to_span(), JobType::EigenVectors, uplo,
                                                 Span<std::byte>(ws.data(), pat == "none" ? 0 : need_all), info.to_span(), ms);
                        this->q().wait();
                        this->expect_untouched("A", got, before, mask, w, info, at);
                        this->expect_same_active(got, ref, mask, w, wr, info, ir, at);
                        if (c.batch > 0) {
                            const double res = this->residual(full, nullptr, got, w, uplo, mask);
                            EXPECT_VERIFY(T, verify::Check::eigen_residual, n, res) << at;
                        }
                        if (c.batch > 0)
                            this->against_full(fam, got, full, mask, w, info, at, [&](Batch<T>& f, UnifiedVector<R>& wf, UnifiedVector<int32_t>& inf) { this->run_syev(f, wf, inf, uplo); });
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
                            const std::string at = fam + (factored ? " factored" : "") + " n=" + std::to_string(n) +
                                                   " " + ul(uplo) + " batch=" + std::to_string(batch) + " " + pat;
                            const auto mask = pattern(pat, batch);
                            auto Ar = make_batch<T>(n, batch, uplo, 21u + n, false);
                            auto Br = make_batch<T>(n, batch, uplo, 22u + n, true);
                            const auto B0 = Br;  // B before any factorization, for the residual
                            UnifiedVector<int32_t> pinfo(batch, 0);
                            if (factored) {
                                try {
                                    UnifiedVector<std::byte> fws(std::max<size_t>(1, potrf_buffer_size<BK, T>(this->q(), Br.view(), uplo)));
                                    (void)potrf<BK, T>(this->q(), Br.view(), uplo, fws.to_span(), pinfo.to_span());
                                    this->q().wait();
                                } catch (const NoRouteError&) {
                                    GTEST_SKIP() << "no potrf route";
                                }
                                // Whatever potrf left in the other triangle, it is poison again.
                                for (int b = 0; b < batch; ++b)
                                    for (int j = 0; j < n; ++j)
                                        for (int i = 0; i < n; ++i)
                                            if (uplo == Uplo::Lower ? i < j : i > j) Br.at(b, i, j) = poison<T>();
                            }
                            const auto Afull = Ar, Bfull = Br;
                            auto A = poisoned(Afull, mask);
                            auto B = poisoned(Bfull, mask);
                            const auto Ab = A, Bb = B;
                            // Reference: the unmasked call on the active items alone, expanded back to their slots.
                            auto cA = compact(Afull, mask), cB = compact(Bfull, mask);
                            const int k = std::max(cA.batch, 1);
                            UnifiedVector<R> wc(static_cast<size_t>(n) * k, R(-4321)), w(static_cast<size_t>(n) * batch, R(-4321));
                            UnifiedVector<int32_t> ic(k, -12345), info(batch, -12345);
                            UnifiedVector<uint8_t> m(batch);
                            for (int b = 0; b < batch; ++b) m[b] = mask[b];
                            const Span<const uint8_t> ms(m.data(), m.size());
                            const JobType V = JobType::EigenVectors;
                            UnifiedVector<uint8_t> ones(batch, 1);
                            const Span<const uint8_t> all1(ones.data(), ones.size());
                            size_t need = 0, need_all = 0;
                            try {
                                if (cA.batch > 0) this->run_sygvd(cA, cB, wc, ic, factored, uplo);
                                need = factored ? sygvd_factored_masked_buffer_size<BK, T>(this->q(), A.view(), B.view(), w.to_span(), 1, V, uplo, ms)
                                                : sygvd_masked_buffer_size<BK, T>(this->q(), A.view(), B.view(), w.to_span(), 1, V, uplo, ms);
                                need_all = factored ? sygvd_factored_masked_buffer_size<BK, T>(this->q(), A.view(), B.view(), w.to_span(), 1, V, uplo, all1)
                                                    : sygvd_masked_buffer_size<BK, T>(this->q(), A.view(), B.view(), w.to_span(), 1, V, uplo, all1);
                            } catch (const NoRouteError&) {
                                GTEST_SKIP() << "no sygvd route";
                            }
                            auto refA = Afull, refB = Bfull;
                            UnifiedVector<R> wr(w.size(), R(-4321));
                            UnifiedVector<int32_t> ir(batch, -12345);
                            expand(refA.v, cA.v, Afull.stride, mask);
                            expand(refB.v, cB.v, Bfull.stride, mask);
                            expand(wr, wc, n, mask);
                            expand(ir, ic, 1, mask);
                            ASSERT_EQ(need, need_all) << at << ": the size depends on the mask";
                            UnifiedVector<std::byte> ws(std::max<size_t>(1, need_all));
                            const Span<std::byte> s(ws.data(), pat == "none" ? 0 : need_all);
                            if (factored)
                                (void)sygvd_factored_masked<BK, T>(this->q(), A.view(), B.view(), w.to_span(), 1, V, uplo, s, info.to_span(), ms);
                            else
                                (void)sygvd_masked<BK, T>(this->q(), A.view(), B.view(), w.to_span(), 1, V, uplo, s, info.to_span(), ms);
                            this->q().wait();
                            this->expect_untouched("A", A, Ab, mask, w, info, at);
                            this->expect_untouched("B", B, Bb, mask, w, info, at);
                            this->expect_same_active(A, refA, mask, w, wr, info, ir, at);
                            for (int b = 0; b < batch; ++b)
                                if (mask[b])
                                    for (int j = 0; j < n; ++j)
                                        for (int i = 0; i < n; ++i)
                                            ASSERT_TRUE(same_bits(B.at(b, i, j), refB.at(b, i, j))) << at << " B item " << b;
                            if (cA.batch > 0) {
                                const double res = this->residual(Afull, &B0, A, w, uplo, mask);
                                EXPECT_VERIFY(T, verify::Check::eigen_residual, n, res) << at;
                            }
                            if (cA.batch > 0)
                                this->against_full(fam, A, Afull, mask, w, info, at, [&](Batch<T>& f, UnifiedVector<R>& wf, UnifiedVector<int32_t>& inf) {
                                    auto fb = Bfull;
                                    this->run_sygvd(f, fb, wf, inf, factored, uplo);
                                });
                        }
}

// A mask shorter than the batch is refused before anything runs.
TYPED_TEST(MaskedTest, ShortMaskIsRefused) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;
    constexpr Backend BK = TestFixture::BK;
    auto A = make_batch<T>(4, 3, Uplo::Lower, 1u, false);
    UnifiedVector<R> w(12);
    UnifiedVector<uint8_t> m(2, 1);
    UnifiedVector<std::byte> ws(1);
    EXPECT_THROW(((void)syev_masked<BK, T>(this->q(), A.view(), w.to_span(), JobType::EigenVectors, Uplo::Lower,
                                         ws.to_span(), Span<int32_t>{}, Span<const uint8_t>(m.data(), m.size()))),
                 batchlas::invalid_argument);
    EXPECT_THROW(((void)sygvd_masked<BK, T>(this->q(), A.view(), A.view(), w.to_span(), 1, JobType::EigenVectors,
                                          Uplo::Lower, ws.to_span(), Span<int32_t>{}, Span<const uint8_t>(m.data(), m.size()))),
                 batchlas::invalid_argument);
}
