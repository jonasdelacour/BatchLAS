// The LPANEL tier's cases, in their own file only for length; they sit inside
// potrf_tests.cc's anonymous namespace so make_spd, make_planted_ldl and
// multiply_back_residual above are the same oracles the CTA cases use.
// evidence: docs/perf/potrf.md#the-lpanel-tier

template <typename Config>
class PotrfLpanelTest : public PotrfCtaTest<Config> {
protected:
    using T = typename Config::ScalarType;
    using R = RealOf<T>;
    static constexpr Backend BackendType = Config::BackendVal;

    // THE DEVICE'S ceiling, and it folds TWO caps: the local-memory slice and
    // MAX_WORK_GROUP_SIZE, because the body needs one work-item per row.
    int lp_ceiling(int nb_hint = 0) const {
        const std::size_t local_mem =
            this->ctx->device().get_property(DeviceProperty::LOCAL_MEM_SIZE);
        const std::size_t budget = (local_mem > 4096) ? (local_mem - 4096) : 0;
        const int max_wg = static_cast<int>(
            this->ctx->device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
        return sycl_potrf::potrf_lpanel_max_n_for_slm<T>(budget, max_wg,
                                                         resident::kMinBlocksPerSm, nb_hint);
    }

    struct LpGeom { int nb; int L; int G; };

    LpGeom geom(int n, int batch, int nb_hint = 0) const {
        const unsigned g = sycl_potrf::potrf_lpanel_debug_launch<T>(
            *this->ctx, n, batch, resident::kMinBlocksPerSm, nb_hint);
        return LpGeom{static_cast<int>(g >> 16),
                      static_cast<int>((g >> 4) & 0xfffu),
                      static_cast<int>(g & 0xfu)};
    }

    std::vector<int32_t> run_lpanel(const MatrixView<T, MatrixFormat::Dense>& V,
                                    Uplo uplo = Uplo::Lower, int nb_hint = 0,
                                    bool pass_info_span = true) {
        const int batch = static_cast<int>(V.batch_size());
        UnifiedVector<std::byte> ws(sycl_potrf::potrf_lpanel_buffer_size<T>(*this->ctx, V));
        UnifiedVector<int32_t> info(static_cast<size_t>(batch), int32_t(-7));
        (void)sycl_potrf::potrf_lpanel_dispatch<T>(
            *this->ctx, V, uplo, ws.to_span(),
            pass_info_span ? info.to_span() : Span<int32_t>{},
            resident::kMinBlocksPerSm, nb_hint);
        this->ctx->wait();
        return std::vector<int32_t>(info.begin(), info.end());
    }
};

TYPED_TEST_SUITE(PotrfLpanelTest, PotrfTestTypes);

// The residual sweep. Every n either straddles a panel boundary (NB = 8 or 16), a
// sub-group boundary (32) or a work-group-size step, because those are the three places
// the geometry changes. Batch shrinks as n grows: the host oracle is O(n^3) per item.
TYPED_TEST(PotrfLpanelTest, ResidualAcrossPanelBoundaries) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;

    const int cap = this->lp_ceiling();
    ASSERT_GT(cap, 0) << "no LPanel capacity for this type -- the kernel is not linked";
    ASSERT_GT(cap, 32) << "the LPanel tier exists to serve n > 32; a ceiling at or below "
                          "that means the footprint formula or the budget is wrong";

    struct Case { int n; int batch; };
    const std::vector<Case> cases = {
        {33, 3}, {40, 3}, {48, 3}, {63, 3}, {64, 3}, {65, 3}, {96, 3},
        {127, 3}, {128, 3}, {129, 3}, {192, 2},
        {255, 2}, {256, 2}, {257, 2}, {384, 1}, {512, 1},
    };

    int ran = 0;
    for (const auto& c : cases) {
        if (c.n > cap) continue;
        ++ran;
        Matrix<T, MatrixFormat::Dense> A(c.n, c.n, c.batch);
        std::vector<std::vector<T>> ref(c.batch);
        for (int b = 0; b < c.batch; ++b) {
            // Every batch item differs: identical items hide a stride bug.
            ref[b] = make_spd<T>(c.n, 5100u + 31u * b + 3u * static_cast<unsigned>(c.n));
            this->load_triangle(A, b, c.n, ref[b], Uplo::Lower,
                                make_scalar<T>(R(-999), R(777)));
        }
        const auto info = this->run_lpanel(A.view());
        for (int b = 0; b < c.batch; ++b) {
            ASSERT_EQ(info[b], 0) << "n=" << c.n << " b=" << b;
            const auto L = this->extract_L(A, b, c.n, Uplo::Lower);
            const R res = multiply_back_residual<T>(ref[b], L, c.n);
            EXPECT_LE(res, residual_tol<T>(c.n))
                << "n=" << c.n << " b=" << b << " residual/(n*eps)="
                << (res / (R(c.n) * std::numeric_limits<R>::epsilon()));
        }
    }
    ASSERT_GE(ran, 4) << "the ceiling clipped the sweep to " << ran
                      << " cases; it proves nothing about panel boundaries";
}

// THE PANEL STORE IS ONLY PUBLISHED UNDER A BARRIER, AND THE SWEEP ABOVE CANNOT SEE IT.
// The store at the end of a panel writes global columns the NEXT panel stages into sB from
// OTHER work-items; B6 orders the two. With B6 deleted the sweep above stays green at every
// n, because a handful of resident work-groups run near lockstep -- the corruption needs
// several work-groups per SM. Identical items here, deliberately against the sweep's
// "every item differs" rule: this test's oracle is cross-item agreement, and the stride bug
// that rule guards is covered by PaddedLeadingDimensionStrideAndEmptyInfoSpan.
// evidence: docs/perf/potrf.md#armed-breaks-r9--lpanel
TYPED_TEST(PotrfLpanelTest, PanelStoresArePublishedAtSaturatingBatch) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;

    const int n = std::min(128, this->lp_ceiling());
    ASSERT_GT(n, 32);
    const int batch = 1024;  // 512 was measured NOT enough on this card; 1024 is

    const auto ref = make_spd<T>(n, 90210u);
    Matrix<T, MatrixFormat::Dense> A(n, n, batch);
    for (int b = 0; b < batch; ++b) {
        this->load_triangle(A, b, n, ref, Uplo::Lower, make_scalar<T>(R(0), R(0)));
    }

    const auto info = this->run_lpanel(A.view());

    ASSERT_EQ(info[0], 0);
    const auto L0 = this->extract_L(A, 0, n, Uplo::Lower);
    EXPECT_LE((multiply_back_residual<T>(ref, L0, n)), residual_tol<T>(n))
        << "item 0, n=" << n << " batch=" << batch;

    // The kernel is deterministic for identical input and carries no atomics, so every
    // item must come back BIT-identical to item 0.
    int bad_items = 0;
    int first_bad = -1;
    for (int b = 1; b < batch; ++b) {
        bool same = (info[b] == 0);
        for (int j = 0; j < n && same; ++j) {
            for (int i = j; i < n; ++i) {
                if (!(A(i, j, b) == L0[i + static_cast<size_t>(j) * n])) { same = false; break; }
            }
        }
        if (!same) {
            ++bad_items;
            if (first_bad < 0) first_bad = b;
        }
    }
    EXPECT_EQ(bad_items, 0)
        << bad_items << " of " << (batch - 1) << " items disagree with item 0 (first: "
        << first_bad << "); n=" << n;
}

// The other triangle is neither READ nor WRITTEN, and THAT TAKES TWO POISONS. A quiet NaN
// alone cannot see the WRITE: the value the kernel would store into the upper triangle is
// itself derived from the NaN it read, so `isnan` passes on the corrupted output too --
// poison the guard swallows. The second pass therefore uses a FINITE sentinel and demands
// bit-equality, which is what actually arms the "not written" half.
// evidence: docs/perf/potrf.md#armed-breaks-r9--lpanel
TYPED_TEST(PotrfLpanelTest, OtherTriangleIsNeitherReadNorWritten) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;

    const int n = std::min(96, this->lp_ceiling());
    ASSERT_GT(n, 32);
    const int batch = 2;
    const R qnan = std::numeric_limits<R>::quiet_NaN();

    for (int pass = 0; pass < 2; ++pass) {
        const bool nan_pass = (pass == 0);
        const T poison = nan_pass ? make_scalar<T>(qnan, qnan) : make_scalar<T>(R(-999), R(777));

        Matrix<T, MatrixFormat::Dense> A(n, n, batch);
        std::vector<std::vector<T>> ref(batch);
        for (int b = 0; b < batch; ++b) {
            ref[b] = make_spd<T>(n, 2200u + 5u * b);
            this->load_triangle(A, b, n, ref[b], Uplo::Lower, poison);
        }
        const auto info = this->run_lpanel(A.view());

        for (int b = 0; b < batch; ++b) {
            ASSERT_EQ(info[b], 0) << "b=" << b << " pass=" << pass;
            for (int i = 0; i < n; ++i) {
                for (int j = i + 1; j < n; ++j) {
                    const bool intact = nan_pass ? std::isnan(host_real(A(i, j, b)))
                                                 : (A(i, j, b) == poison);
                    ASSERT_TRUE(intact)
                        << "the kernel WROTE the upper triangle at (" << i << "," << j
                        << ") b=" << b << " pass=" << pass;
                }
            }
            const auto L = this->extract_L(A, b, n, Uplo::Lower);
            const R res = multiply_back_residual<T>(ref[b], L, n);
            ASSERT_FALSE(std::isnan(res))
                << "the kernel READ the poisoned upper triangle, b=" << b;
            EXPECT_LE(res, residual_tol<T>(n)) << "b=" << b << " pass=" << pass;
        }
    }
}

// THE TRANSITION ORACLE. Where both tiers serve the shape they must agree -- elementwise
// within tolerance, not bit-exact: the two kernels sum the same inner products in
// different orders. evidence: docs/perf/potrf.md#the-cta-comparison-is-elementwise-not-bit-exact
TYPED_TEST(PotrfLpanelTest, AgreesWithTheCtaKernel) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;

    const int cap = std::min(this->ceiling(), this->lp_ceiling());
    ASSERT_GT(cap, 32) << "the two tiers do not overlap above n = 32 on this device; "
                          "the oracle would be vacuous";

    for (int n : {33, 48, cap}) {
        if (n > cap) continue;
        const int batch = 3;
        Matrix<T, MatrixFormat::Dense> A(n, n, batch);
        Matrix<T, MatrixFormat::Dense> Bm(n, n, batch);
        for (int b = 0; b < batch; ++b) {
            const auto ref = make_spd<T>(n, 3300u + 7u * b + static_cast<unsigned>(n));
            this->load_triangle(A, b, n, ref, Uplo::Lower, make_scalar<T>(R(0), R(0)));
            this->load_triangle(Bm, b, n, ref, Uplo::Lower, make_scalar<T>(R(0), R(0)));
        }
        const auto info_lp = this->run_lpanel(A.view());
        const auto info_cta = this->run_cta(Bm, Uplo::Lower);

        R scale = R(0);
        for (int b = 0; b < batch; ++b) {
            for (int i = 0; i < n; ++i)
                for (int j = 0; j <= i; ++j)
                    scale = std::max(scale, std::abs(host_real(Bm(i, j, b))));
        }
        const R tol = R(32) * R(n) * std::numeric_limits<R>::epsilon() * std::max(scale, R(1));

        for (int b = 0; b < batch; ++b) {
            ASSERT_EQ(info_lp[b], 0);
            ASSERT_EQ(info_cta[b], info_lp[b]) << "the tiers disagree about info, b=" << b;
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j <= i; ++j) {
                    EXPECT_NEAR(host_real(A(i, j, b)), host_real(Bm(i, j, b)), tol)
                        << "n=" << n << " (" << i << "," << j << ") b=" << b;
                    EXPECT_NEAR(host_imag(A(i, j, b)), host_imag(Bm(i, j, b)), tol)
                        << "n=" << n << " (" << i << "," << j << ") b=" << b;
                }
            }
        }
    }
}

// `info` names the 1-based GLOBAL column of the FIRST non-positive pivot. The sweep is
// built around the panel width the launcher actually chose, so the columns that straddle
// a panel boundary are the ones tested rather than the ones that happened to be typed.
TYPED_TEST(PotrfLpanelTest, InfoIndexIsExact) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;

    const int n = std::min(96, this->lp_ceiling());
    ASSERT_GT(n, 66) << "the sweep needs room for several panels";
    const int nb = this->geom(n, 1).nb;
    ASSERT_GT(nb, 0) << "the launcher reports this shape does not fit";

    std::vector<int> cols = {0, 1, nb - 1, nb, nb + 1, 2 * nb, 31, 32, 33, 64, 65, n - 1};
    std::sort(cols.begin(), cols.end());
    cols.erase(std::unique(cols.begin(), cols.end()), cols.end());

    for (int c : cols) {
        if (c < 0 || c >= n) continue;
        Matrix<T, MatrixFormat::Dense> A(n, n, 1);
        const auto ref = make_planted_ldl<T>(n, {c}, 4242u + static_cast<unsigned>(c));
        // THE TEST ASSERTS ITS OWN SENSITIVITY: for c >= 1 the ORIGINAL diagonal at the
        // failure column is positive, so only a kernel testing the UPDATED pivot names it.
        if (c >= 1) {
            ASSERT_GT(host_real(ref[c + static_cast<size_t>(c) * n]), R(0))
                << "the planted matrix is not discriminating at column " << c;
        }
        this->load_triangle(A, 0, n, ref, Uplo::Lower, make_scalar<T>(R(0), R(0)));
        const auto info = this->run_lpanel(A.view());
        EXPECT_EQ(info[0], c + 1)
            << "planted a non-positive pivot at global column " << c << " of " << n
            << " (nb=" << nb << ")";
    }
}

// FIRST FAILURE WINS, including across a panel boundary, and a failed item stays finite.
TYPED_TEST(PotrfLpanelTest, InfoFirstFailureWinsAndFailedItemsStayFinite) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;

    const int n = std::min(64, this->lp_ceiling());
    ASSERT_GT(n, 32);
    const int nb = this->geom(n, 1).nb;
    ASSERT_GT(nb, 0);

    // The second failure is deliberately in a LATER panel than the first.
    for (int c : {1, 3, 17}) {
        const int c2 = c + 2 * nb;
        if (c2 >= n) continue;
        ASSERT_NE(c / nb, c2 / nb) << "both failures landed in the same panel";

        Matrix<T, MatrixFormat::Dense> A(n, n, 1);
        const auto ref = make_planted_ldl<T>(n, {c, c2}, 777u + static_cast<unsigned>(c));
        this->load_triangle(A, 0, n, ref, Uplo::Lower, make_scalar<T>(R(0), R(0)));
        const auto info = this->run_lpanel(A.view());
        EXPECT_EQ(info[0], c + 1) << "failures planted at " << c << " and " << c2;

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                ASSERT_TRUE(std::isfinite(host_real(A(i, j, 0))))
                    << "a failed item went non-finite at (" << i << "," << j << ")";
                ASSERT_TRUE(std::isfinite(host_imag(A(i, j, 0))));
            }
        }
    }
}

// PACKED LAUNCHES ARE INDEPENDENT. G > 1 is legal here only because every barrier is a
// WORK-GROUP barrier over a schedule the packed matrices share; a failure in one matrix
// must not disturb its neighbours, and the tail work-group's dead slots must write nothing.
TYPED_TEST(PotrfLpanelTest, PackedBatchIsIndependentAndTheTailSlotsAreInert) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;

    const int n = std::min(48, this->lp_ceiling());
    ASSERT_GT(n, 32);
    const int G = this->geom(n, 64).G;
    if (G <= 1) {
        GTEST_SKIP() << "this device packs one matrix per work-group at n = " << n
                     << "; the packing path is not exercised";
    }

    // 4G + 3 leaves the LAST work-group partly dead, which is the case the `store` flag
    // in potrf_lpanel_body exists for.
    const int batch = 4 * G + 3;
    ASSERT_NE(batch % G, 0) << "batch is a multiple of G: no dead slots, nothing proved";

    const int bad_item = G;            // not item 0, so a slot offset bug is visible
    const int bad_col = n / 2;

    Matrix<T, MatrixFormat::Dense> A(n, n, batch);
    std::vector<std::vector<T>> ref(batch);
    for (int b = 0; b < batch; ++b) {
        ref[b] = (b == bad_item)
                     ? make_planted_ldl<T>(n, {bad_col}, 61u)
                     : make_spd<T>(n, 400u + 11u * b);
        this->load_triangle(A, b, n, ref[b], Uplo::Lower, make_scalar<T>(R(0), R(0)));
    }
    const auto info = this->run_lpanel(A.view());

    EXPECT_EQ(info[bad_item], bad_col + 1);
    for (int b = 0; b < batch; ++b) {
        if (b == bad_item) continue;
        ASSERT_EQ(info[b], 0) << "healthy item " << b << " reported a failure";
        const auto L = this->extract_L(A, b, n, Uplo::Lower);
        EXPECT_LE((multiply_back_residual<T>(ref[b], L, n)), residual_tol<T>(n))
            << "item " << b << " (packing G=" << G << ", batch=" << batch << ")";
    }
}

// A PADDED LEADING DIMENSION AND A STRIDE THAT IS NOT ld * cols; and the same call with
// an EMPTY info span, which must draw pool scratch rather than write through a null.
TYPED_TEST(PotrfLpanelTest, PaddedLeadingDimensionStrideAndEmptyInfoSpan) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;

    const int n = std::min(65, this->lp_ceiling());
    ASSERT_GT(n, 32);
    const int batch = 5;
    const int ld = n + 7;                       // != rows
    const int stride = ld * n + 13;             // != ld * cols

    for (bool pass_info : {true, false}) {
        UnifiedVector<T> buf(static_cast<size_t>(stride) * batch,
                             make_scalar<T>(R(-11), R(5)));   // non-PD poison
        MatrixView<T, MatrixFormat::Dense> V(buf.data(), n, n, ld, stride, batch);

        std::vector<std::vector<T>> ref(batch);
        for (int b = 0; b < batch; ++b) {
            ref[b] = make_spd<T>(n, 4711u + 13u * b);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    buf[static_cast<size_t>(b) * stride + i + static_cast<size_t>(j) * ld] =
                        (i >= j) ? ref[b][i + static_cast<size_t>(j) * n]
                                 : make_scalar<T>(R(0), R(0));
                }
            }
        }

        const auto info = this->run_lpanel(V, Uplo::Lower, 0, pass_info);
        for (int b = 0; b < batch; ++b) {
            if (pass_info) ASSERT_EQ(info[b], 0) << "b=" << b;
            std::vector<T> L(static_cast<size_t>(n) * n, T{});
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j <= i; ++j) {
                    L[i + static_cast<size_t>(j) * n] =
                        buf[static_cast<size_t>(b) * stride + i + static_cast<size_t>(j) * ld];
                }
            }
            EXPECT_LE((multiply_back_residual<T>(ref[b], L, n)), residual_tol<T>(n))
                << "b=" << b << " ld=" << ld << " stride=" << stride
                << " info_span=" << pass_info;
        }
    }
}

// The direct entry point re-applies every gate supports() applies, because it is
// reachable WITHOUT the table; Uplo::Upper is one of them here and is not on the CTA arm.
TYPED_TEST(PotrfLpanelTest, DirectEntryPointRefusesWhatSupportsRefuses) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;

    const int cap = this->lp_ceiling();
    ASSERT_GT(cap, 32);
    UnifiedVector<int32_t> info(8, int32_t(0));

    // (a) not square.
    {
        Matrix<T, MatrixFormat::Dense> A(40, 33, 1);
        A.fill(make_scalar<T>(R(1), R(0)));
        UnifiedVector<std::byte> ws(64);
        EXPECT_THROW(sycl_potrf::potrf_lpanel_dispatch<T>(*this->ctx, A.view(), Uplo::Lower,
                                                          ws.to_span(), info.to_span()),
                     std::invalid_argument);
    }

    // (b) Uplo::Upper -- the left-looking update reads the LOWER triangle.
    {
        const int n = std::min(40, cap);
        Matrix<T, MatrixFormat::Dense> A(n, n, 1);
        A.fill(T{});
        for (int i = 0; i < n; ++i) A(i, i, 0) = make_scalar<T>(R(2), R(0));
        UnifiedVector<std::byte> ws(
            sycl_potrf::potrf_lpanel_buffer_size<T>(*this->ctx, A.view()));
        EXPECT_THROW(sycl_potrf::potrf_lpanel_dispatch<T>(*this->ctx, A.view(), Uplo::Upper,
                                                          ws.to_span(), info.to_span()),
                     std::invalid_argument);
    }

    // (c) one order past the ceiling: a HARD capacity, not a slow path.
    {
        const int n = cap + 1;
        Matrix<T, MatrixFormat::Dense> A(n, n, 1);
        A.fill(T{});
        for (int i = 0; i < n; ++i) A(i, i, 0) = make_scalar<T>(R(2), R(0));
        EXPECT_EQ(this->geom(n, 1).nb, 0)
            << "debug_launch says order " << n << " fits, one past the advertised ceiling";
        UnifiedVector<std::byte> ws(
            sycl_potrf::potrf_lpanel_buffer_size<T>(*this->ctx, A.view()));
        EXPECT_THROW(sycl_potrf::potrf_lpanel_dispatch<T>(*this->ctx, A.view(), Uplo::Lower,
                                                          ws.to_span(), info.to_span()),
                     std::invalid_argument);
    }
}

// THE WINDOW, AND THE R8b INVARIANT THAT GUARDS IT. `preferred()` now answers true inside
// one measured window; outside it, and for every type and `uplo` that has no grid, `Auto`
// must still take the vendor. The arm COUNTS below are the real guard: automatic() returns
// on the first supports && preferred hit, so two arms answering is a silent mis-route.
// evidence: docs/perf/potrf.md#the-measured-lpanel-window
TYPED_TEST(PotrfLpanelTest, RouteWindowIsExactlyTheMeasuredOne) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;
    static constexpr Backend B = TestFixture::BackendType;
    using Tbl = dispatch::RouteTable<dispatch::Op::potrf, T>;
    const dispatch::Route lp{dispatch::Origin::Native, dispatch::Algorithm::LPanel};

    // This case asserts what the TABLE does with no route pinned, so it has to say so:
    // ctest's potrf_tests_native variant runs this same binary with
    // BATCHLAS_POTRF_ROUTE=native in the environment, which forces every resolve below
    // native and turns the vendor-present assertion red. Empty reads as unset.
    const ScopedEnvVar clear_potrf("BATCHLAS_POTRF_ROUTE", "");

    // The vocabulary: a missing to_string case prints "?" in every coverage row, and a
    // missing parse case is SILENT -- the pin becomes nullopt and the arm measures Auto.
    EXPECT_EQ(dispatch::to_string(dispatch::Algorithm::LPanel), "lpanel");
    ASSERT_TRUE(dispatch::parse_algorithm_word("lpanel").has_value());
    EXPECT_EQ(*dispatch::parse_algorithm_word("lpanel"), dispatch::Algorithm::LPanel);
    // APPENDED, never inserted: route.hh is installed behind a SOVERSION.
    EXPECT_EQ(static_cast<int>(dispatch::Algorithm::LPanel), 14);

    // float and complex<float> are the types the grid covers; double and complex<double>
    // keep the pre-P3 walk, so this same body asserts BOTH outcomes.
    constexpr bool measured =
        std::is_same_v<T, float> || std::is_same_v<T, std::complex<float>>;
    const int cta_last = std::is_same_v<T, float> ? 35 : 32;

    for (int n : {32, 33, 64, 256, 257, 384}) {
        if (n > this->lp_ceiling() && n > this->ceiling()) continue;

        Matrix<T, MatrixFormat::Dense> A(n, n, 4);
        A.fill(T{});
        for (int b = 0; b < 4; ++b)
            for (int i = 0; i < n; ++i) A(i, i, b) = make_scalar<T>(R(1), R(0));

        const auto lower = backend::potrf_op_shape<B, T>(*this->ctx, A.view(), Uplo::Lower);
        const auto upper = backend::potrf_op_shape<B, T>(*this->ctx, A.view(), Uplo::Upper);
        ASSERT_TRUE(lower.has_value());
        ASSERT_TRUE(upper.has_value());

        EXPECT_FALSE(Tbl::supports(lp, *upper)) << "LPanel must refuse Upper, n=" << n;
        EXPECT_EQ(Tbl::supports(lp, *lower), n <= this->lp_ceiling()) << "n=" << n;

        const bool in_window = measured && n > 32 && n <= 256;

        int pref_hits = 0;
        int pref_hits_upper = 0;
        int tier_hits = 0;
        for (const dispatch::Route* r = Tbl::order_begin(); r != Tbl::order_end(); ++r) {
            if (!dispatch::is_native(*r)) continue;
            if (Tbl::supports(*r, *lower) && Tbl::preferred(*r, *lower)) ++pref_hits;
            if (Tbl::supports(*r, *upper) && Tbl::preferred(*r, *upper)) ++pref_hits_upper;
            if (Tbl::supports(*r, *lower) && Tbl::native_tier_preferred(*r, *lower)) ++tier_hits;
        }
        EXPECT_EQ(pref_hits, in_window ? 1 : 0) << "n=" << n;
        EXPECT_EQ(pref_hits_upper, 0) << "Upper has no grid behind it, n=" << n;
        EXPECT_EQ(tier_hits, 1) << "n=" << n << ": the vendor-free walk is an accident of the "
                                   "order array rather than a stated decision";

        const auto route = backend::potrf_route<B, T>(*this->ctx, A.view(), Uplo::Lower,
                                                      /*vendor_available=*/true);
        if (in_window) {
            ASSERT_TRUE(dispatch::is_native(route)) << "n=" << n;
            EXPECT_EQ(route.algo, (n <= cta_last) ? dispatch::Algorithm::CTA
                                                  : dispatch::Algorithm::LPanel) << "n=" << n;
        } else {
            EXPECT_TRUE(dispatch::is_vendor(route))
                << "a shape OUTSIDE the measured window left the vendor, n=" << n;
        }
        EXPECT_TRUE(dispatch::is_vendor(
            backend::potrf_route<B, T>(*this->ctx, A.view(), Uplo::Upper,
                                       /*vendor_available=*/true)))
            << "Upper moved off the vendor at n=" << n;

        const auto free_route = backend::potrf_route<B, T>(*this->ctx, A.view(), Uplo::Lower,
                                                           /*vendor_available=*/false);
        const bool lpanel_owns = measured && n > cta_last && n <= this->lp_ceiling();
        const dispatch::Algorithm want = lpanel_owns ? dispatch::Algorithm::LPanel
                                       : (n <= this->ceiling()) ? dispatch::Algorithm::CTA
                                                                : dispatch::Algorithm::Blocked;
        EXPECT_EQ(free_route.algo, want) << "vendor-free walk, n=" << n;
    }
}

// The facade reaches the kernel under an explicit pin, and potrf_buffer_size covers the
// tier. Bit-exactness against the direct call observes EXECUTION; a residual would not.
TYPED_TEST(PotrfLpanelTest, FacadeReachesTheLpanelKernel) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;
    static constexpr Backend B = TestFixture::BackendType;

    const int n = std::min(48, this->lp_ceiling());
    ASSERT_GT(n, 32);
    const int batch = 3;

    const ScopedEnvVar route_pin("BATCHLAS_POTRF_ROUTE", "lpanel");

    Matrix<T, MatrixFormat::Dense> A(n, n, batch);
    Matrix<T, MatrixFormat::Dense> direct(n, n, batch);
    std::vector<std::vector<T>> ref(batch);
    for (int b = 0; b < batch; ++b) {
        ref[b] = make_spd<T>(n, 8080u + b);
        this->load_triangle(A, b, n, ref[b], Uplo::Lower, make_scalar<T>(R(0), R(0)));
        this->load_triangle(direct, b, n, ref[b], Uplo::Lower, make_scalar<T>(R(0), R(0)));
    }

    // The route assertion LOCALISES a failure -- which link broke -- but is NOT the guard.
    const auto route = backend::potrf_route<B, T>(*this->ctx, A.view(), Uplo::Lower,
                                                  /*vendor_available=*/true);
    ASSERT_TRUE(dispatch::is_native(route));
    ASSERT_EQ(route.algo, dispatch::Algorithm::LPanel);

    const std::size_t bytes = potrf_buffer_size<B, T>(*this->ctx, A.view(), Uplo::Lower);
    ASSERT_GE(bytes, sycl_potrf::potrf_lpanel_buffer_size<T>(*this->ctx, A.view()))
        << "potrf_buffer_size does not cover the LPanel tier it just resolved to";
    UnifiedVector<std::byte> ws(bytes);
    UnifiedVector<int32_t> info(batch, int32_t(-7));
    potrf<B, T>(*this->ctx, A.view(), Uplo::Lower, ws.to_span(), info.to_span());
    this->ctx->wait();

    const auto info_direct = this->run_lpanel(direct.view());
    for (int b = 0; b < batch; ++b) {
        ASSERT_EQ(info[b], 0);
        ASSERT_EQ(info_direct[b], 0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                ASSERT_EQ(host_real(A(i, j, b)), host_real(direct(i, j, b)))
                    << "the facade did not run the LPanel kernel at (" << i << "," << j
                    << ") b=" << b;
                ASSERT_EQ(host_imag(A(i, j, b)), host_imag(direct(i, j, b)));
            }
        }
    }
}

// NB is an A/B knob, and a hint the build does not instantiate is an ERROR rather than a
// silent downgrade -- otherwise the NB sweep measures NB = 8 twice and reports a tie.
TYPED_TEST(PotrfLpanelTest, NbSixteenIsInstantiatedForFloatOnly) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;

    const int cap16 = this->lp_ceiling(16);
    UnifiedVector<int32_t> probe(4, int32_t(0));

    if constexpr (std::is_same_v<T, float>) {
        ASSERT_GT(cap16, 32) << "NB = 16 advertises no usable order for float";
        // Doubling NB doubles the panel footprint, so the ceiling must FALL.
        EXPECT_LT(cap16, this->lp_ceiling(8));

        const int n = std::min(96, cap16);
        const int batch = 3;
        Matrix<T, MatrixFormat::Dense> A(n, n, batch);
        std::vector<std::vector<T>> ref(batch);
        for (int b = 0; b < batch; ++b) {
            ref[b] = make_spd<T>(n, 1717u + b);
            this->load_triangle(A, b, n, ref[b], Uplo::Lower, make_scalar<T>(R(0), R(0)));
        }
        EXPECT_EQ(this->geom(n, batch, 16).nb, 16);
        const auto info = this->run_lpanel(A.view(), Uplo::Lower, /*nb_hint=*/16);
        for (int b = 0; b < batch; ++b) {
            ASSERT_EQ(info[b], 0) << "b=" << b;
            const auto L = this->extract_L(A, b, n, Uplo::Lower);
            EXPECT_LE((multiply_back_residual<T>(ref[b], L, n)), residual_tol<T>(n));
        }
    } else {
        EXPECT_EQ(cap16, 0) << "NB = 16 advertises capacity for a type it is not built for";
        const int n = std::min(40, this->lp_ceiling());
        Matrix<T, MatrixFormat::Dense> A(n, n, 1);
        A.fill(T{});
        for (int i = 0; i < n; ++i) A(i, i, 0) = make_scalar<T>(R(2), R(0));
        UnifiedVector<std::byte> ws(
            sycl_potrf::potrf_lpanel_buffer_size<T>(*this->ctx, A.view()));
        EXPECT_THROW(sycl_potrf::potrf_lpanel_dispatch<T>(
                         *this->ctx, A.view(), Uplo::Lower, ws.to_span(), probe.to_span(),
                         resident::kMinBlocksPerSm, /*nb_hint=*/16),
                     std::exception);
    }
}

// The ceiling is TWO caps, and the work-group one is not slack: the body needs one
// work-item per row, so an order above MAX_WORK_GROUP_SIZE would race silently.
TYPED_TEST(PotrfLpanelTest, CeilingIsCappedByBothMemoryAndWorkGroupSize) {
    using T = typename TestFixture::T;

    const int max_wg = static_cast<int>(
        this->ctx->device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));
    const int cap = this->lp_ceiling();
    ASSERT_GT(cap, 0);
    EXPECT_LE(cap, (max_wg / 32) * 32);

    // A budget so large that only the work-group cap can bind.
    EXPECT_EQ((sycl_potrf::potrf_lpanel_max_n_for_slm<T>(std::size_t(1) << 40, max_wg)),
              (max_wg / 32) * 32);

    // ... and a work-group cap so large that only memory can bind; that answer must be
    // strictly larger than the CTA tier's at the same budget, which is the whole point.
    const std::size_t local_mem =
        this->ctx->device().get_property(DeviceProperty::LOCAL_MEM_SIZE);
    const std::size_t budget = (local_mem > 4096) ? (local_mem - 4096) : 0;
    EXPECT_GT((sycl_potrf::potrf_lpanel_max_n_for_slm<T>(budget, 1 << 20)),
              (sycl_potrf::potrf_cta_max_n_for_slm<T>(budget)))
        << "the LPanel tier holds one n x NB panel and the CTA tier the whole matrix; a "
           "ceiling that is not larger means the footprint formula is wrong";

    // The shipped ceilings, pinned against the BUDGET-parameterised query rather than a
    // device, so a footprint change is loud rather than a silently narrower tier.
    // evidence: docs/perf/potrf.md#the-lpanel-tier
    constexpr std::size_t kRef = 97280;   // the 4090's LOCAL_MEM_SIZE - 4096
    int adv = 0, res = 0;
    if constexpr (std::is_same_v<T, float>)                      { adv = 744; res = 1024; }
    else if constexpr (std::is_same_v<T, double>)                { adv = 368; res = 1024; }
    else if constexpr (std::is_same_v<T, std::complex<float>>)   { adv = 368; res = 1024; }
    else                                                         { adv = 180; res = 750; }
    EXPECT_EQ((sycl_potrf::potrf_lpanel_max_n_for_slm<T>(kRef, 1024)), adv);
    EXPECT_EQ((sycl_potrf::potrf_lpanel_max_n_for_slm<T>(kRef, 1024, 1)), res);

    // L is a whole number of sub-groups and never below n.
    for (int n : {33, 64, 65, std::min(192, cap)}) {
        if (n > cap) continue;
        const auto g = this->geom(n, 8);
        ASSERT_GT(g.nb, 0) << "n=" << n;
        EXPECT_GE(g.L, n) << "n=" << n << ": fewer lanes than rows is a silent race";
        EXPECT_EQ(g.L % 32, 0) << "n=" << n;
        EXPECT_GE(g.G, 1);
    }
}
