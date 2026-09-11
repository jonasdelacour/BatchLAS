// The TINY tier: one matrix per sub-group partition, held in registers. Included by
// potrf_tests.cc inside its anonymous namespace, so make_spd, make_planted_ldl,
// multiply_back_residual and residual_tol are in scope.
//
// Every case here drives sycl_potrf::potrf_tiny_dispatch DIRECTLY: a forced route that
// supports() rejects silently becomes the vendor, which would make a green run mean
// nothing. evidence: docs/perf/potrf.md#the-tiny-tier

// One batch of n x n matrices at an arbitrary (ld, stride), everything outside the `uplo`
// triangle set to `pad`. Matrix<T>(n,n,batch) cannot express a padded ld or a stride that
// is not ld*cols, and those are what make an off-by-one store falsifiable.
template <typename T>
struct TinyProb {
    int n = 0, batch = 0, ld = 0, stride = 0;
    UnifiedVector<T> buf;
    std::vector<std::vector<T>> ref;

    MatrixView<T, MatrixFormat::Dense> view() {
        return MatrixView<T, MatrixFormat::Dense>(buf.data(), n, n, ld, stride, batch);
    }
    T at(int i, int j, int b) const {
        return buf[static_cast<size_t>(b) * stride + i + static_cast<size_t>(j) * ld];
    }
};

// `seed_shift` scales item b's matrix so a cross-matrix write changes a value rather
// than reproducing the neighbour's.
template <typename T>
TinyProb<T> make_tiny_prob(int n, int batch, int ld, int stride, Uplo uplo,
                           unsigned seed, T pad) {
    using R = RealOf<T>;
    TinyProb<T> p;
    p.n = n; p.batch = batch; p.ld = ld; p.stride = stride;
    p.buf = UnifiedVector<T>(static_cast<size_t>(stride) * batch, pad);
    p.ref.resize(batch);
    for (int b = 0; b < batch; ++b) {
        p.ref[b] = make_spd<T>(n, seed + 31u * static_cast<unsigned>(b),
                               R(1) + R(b % 5));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                const bool in_tri = (uplo == Uplo::Lower) ? (i >= j) : (i <= j);
                if (!in_tri) continue;
                p.buf[static_cast<size_t>(b) * stride + i + static_cast<size_t>(j) * ld] =
                    p.ref[b][i + static_cast<size_t>(j) * n];
            }
        }
    }
    return p;
}

template <typename Config>
class PotrfTinyTest : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    using R = RealOf<T>;
    static constexpr Backend BackendType = Config::BackendVal;

    void SetUp() override {
        test_utils::BatchLASTest<Config>::SetUp();
        if (this->HasFatalFailure() || ::testing::Test::IsSkipped()) return;
        if (!this->ctx) GTEST_SKIP() << "no queue";
        // The suite's own gate must mirror supports(): enumerated sub-group support,
        // never MAX_SUB_GROUP_SIZE >= 32, which reports entry [0].
        if (this->ctx->device().type != DeviceType::GPU) {
            GTEST_SKIP() << "potrf_tiny is a GPU kernel";
        }
        if (!this->ctx->device().supports_sub_group_size(32)) {
            GTEST_SKIP() << "device does not offer sub-group size 32";
        }
        if (sycl_potrf::potrf_tiny_max_n<T>() < 1) {
            GTEST_SKIP() << "the tiny kernel is not linked for this type";
        }
    }

    int cap() const { return sycl_potrf::potrf_tiny_max_n<T>(); }

    // Matrices per work-group, read from the kernel's own geometry hook rather than
    // recomputed here: a test that re-derives the launch cannot see the two disagree.
    int per_wg(int n) const {
        const unsigned g = sycl_potrf::potrf_tiny_debug_launch<T>(*this->ctx, n);
        if (g == 0u) return 0;
        return static_cast<int>(g >> 16) * static_cast<int>(g & 0xffffu);
    }

    std::vector<int32_t> run_tiny(MatrixView<T, MatrixFormat::Dense> V, Uplo uplo,
                                  bool pass_info_span = true) {
        const int batch = V.batch_size();
        UnifiedVector<std::byte> ws(sycl_potrf::potrf_tiny_buffer_size<T>(*this->ctx, V));
        UnifiedVector<int32_t> info(batch, int32_t(-7));
        if (pass_info_span) {
            sycl_potrf::potrf_tiny_dispatch<T>(*this->ctx, V, uplo, ws.to_span(),
                                               info.to_span());
        } else {
            sycl_potrf::potrf_tiny_dispatch<T>(*this->ctx, V, uplo, ws.to_span(),
                                               Span<int32_t>{});
        }
        this->ctx->wait();
        return std::vector<int32_t>(info.begin(), info.end());
    }

    // L for item b: for Upper the stored object is U with A = U^H U, so L = U^H.
    std::vector<T> extract_L(const TinyProb<T>& p, int b, Uplo uplo) {
        std::vector<T> L(static_cast<size_t>(p.n) * p.n, T{});
        for (int i = 0; i < p.n; ++i) {
            for (int j = 0; j <= i; ++j) {
                L[i + static_cast<size_t>(j) * p.n] =
                    (uplo == Uplo::Lower) ? p.at(i, j, b) : host_conj(p.at(j, i, b));
            }
        }
        return L;
    }
};

TYPED_TEST_SUITE(PotrfTinyTest, PotrfTestTypes);

// EVERY order the tier claims, both triangles, ld = n and ld = n + 5, batch 1 and a
// batch that leaves the last work-group partially filled. n = 2 and 3 are mandatory:
// the stale-pivot defect first shows at n = 2.
TYPED_TEST(PotrfTinyTest, TinyResidualBothTriangles) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;

    const int cap = this->cap();
    ASSERT_GT(cap, 0);

    for (int n = 1; n <= cap; ++n) {
        const int pw = this->per_wg(n);
        ASSERT_GT(pw, 0) << "n=" << n << " is inside the cap but the geometry hook says unfit";
        // A partial last work-group: `pw + 3` is never a multiple of pw for pw >= 2.
        for (int batch : {1, pw + 3}) {
            for (int ld_pad : {0, 5}) {
                const int ld = n + ld_pad;
                const int stride = ld * n + (ld_pad ? 13 : 0);
                for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
                    auto p = make_tiny_prob<T>(n, batch, ld, stride, uplo,
                                               1000u + 7u * static_cast<unsigned>(n),
                                               make_scalar<T>(R(-11), R(5)));
                    const auto info = this->run_tiny(p.view(), uplo);
                    for (int b = 0; b < batch; ++b) {
                        ASSERT_EQ(info[b], 0)
                            << "n=" << n << " b=" << b << " ld=" << ld
                            << " uplo=" << static_cast<int>(uplo);
                        const auto L = this->extract_L(p, b, uplo);
                        EXPECT_LE((multiply_back_residual<T>(p.ref[b], L, n)),
                                  residual_tol<T>(n))
                            << "n=" << n << " b=" << b << " ld=" << ld
                            << " uplo=" << static_cast<int>(uplo);
                    }
                }
            }
        }
    }
}

// AGREEMENT WITH THE CTA TIER, ELEMENTWISE AND NOT BIT-EXACT. At n <= 8 the CTA kernel runs
// exactly ONE NB=8 panel, so the recurrence is the same sum in the same order and the bound
// can be far tighter than a residual. It must NOT be tightened to bit-identity: FMA
// contraction differs between a local-memory load and a shuffle. It stops at 8 by
// construction -- above NB the CTA kernel is a blocked recurrence with a different
// summation order. evidence: docs/perf/potrf.md#the-cta-comparison-is-elementwise-not-bit-exact
TYPED_TEST(PotrfTinyTest, TinyAgreesWithCtaWithinOnePanel) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;

    const int cta_cap = sycl_potrf::potrf_cta_max_n<T>();
    ASSERT_GE(cta_cap, 8);

    for (int n = 1; n <= 8; ++n) {
        for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
            const int batch = 5;
            auto tiny = make_tiny_prob<T>(n, batch, n, n * n, uplo, 5150u,
                                          make_scalar<T>(R(0), R(0)));
            auto cta = make_tiny_prob<T>(n, batch, n, n * n, uplo, 5150u,
                                         make_scalar<T>(R(0), R(0)));

            const auto info_t = this->run_tiny(tiny.view(), uplo);

            UnifiedVector<std::byte> ws(
                sycl_potrf::potrf_cta_buffer_size<T>(*this->ctx, cta.view()));
            UnifiedVector<int32_t> info_c(batch, int32_t(-7));
            sycl_potrf::potrf_cta_dispatch<T>(*this->ctx, cta.view(), uplo, ws.to_span(),
                                              info_c.to_span());
            this->ctx->wait();

            for (int b = 0; b < batch; ++b) {
                ASSERT_EQ(info_t[b], 0);
                ASSERT_EQ(info_c[b], 0);
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        const bool in_tri = (uplo == Uplo::Lower) ? (i >= j) : (i <= j);
                        if (!in_tri) continue;
                        const T a = tiny.at(i, j, b);
                        const T c = cta.at(i, j, b);
                        const R scale =
                            std::max(R(1), std::max(std::abs(host_real(c)),
                                                    std::abs(host_imag(c))));
                        const R tol = R(8) * std::numeric_limits<R>::epsilon() * scale;
                        ASSERT_NEAR(host_real(a), host_real(c), tol)
                            << "tiny != cta at n=" << n << " (" << i << "," << j
                            << ") b=" << b << " uplo=" << static_cast<int>(uplo);
                        ASSERT_NEAR(host_imag(a), host_imag(c), tol);
                    }
                }
            }
        }
    }
}

// PADDING IS INERT. Rows n..N-1 of a live partition and every row of a DEAD partition carry
// a register identity, so nothing outside the leading n x n triangle can reach the answer.
// The guard is bit-identity between a quiet-NaN pad and a zero pad: a residual test would
// accept a kernel that reads the pad and happens to get a finite number back.
TYPED_TEST(PotrfTinyTest, TinyPaddingIsInert) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;

    const int cap = this->cap();
    const R nan = std::numeric_limits<R>::quiet_NaN();

    // n = 5 in the N = 8 bucket, 9 in N = 16, 17 and 31 in N = 32 -- one interior order
    // per bucket, plus the exact bucket edges where the pad is empty.
    for (int n : {2, 5, 8, 9, 17, 31, 32}) {
        if (n > cap) continue;
        const int pw = this->per_wg(n);
        // batch = 1 leaves pw - 1 DEAD partitions in the only work-group.
        for (int batch : {1, pw + 1}) {
            for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
                const int ld = n + 5;
                const int stride = ld * n + 13;
                auto clean = make_tiny_prob<T>(n, batch, ld, stride, uplo, 31337u,
                                               make_scalar<T>(R(0), R(0)));
                auto poisoned = make_tiny_prob<T>(n, batch, ld, stride, uplo, 31337u,
                                                  make_scalar<T>(nan, nan));

                const auto info_c = this->run_tiny(clean.view(), uplo);
                const auto info_p = this->run_tiny(poisoned.view(), uplo);

                for (int b = 0; b < batch; ++b) {
                    ASSERT_EQ(info_c[b], 0) << "n=" << n << " b=" << b;
                    ASSERT_EQ(info_p[b], info_c[b])
                        << "the NaN pad changed info at n=" << n << " b=" << b;
                    for (int i = 0; i < n; ++i) {
                        for (int j = 0; j < n; ++j) {
                            const bool in_tri =
                                (uplo == Uplo::Lower) ? (i >= j) : (i <= j);
                            if (!in_tri) continue;
                            ASSERT_EQ(host_real(clean.at(i, j, b)),
                                      host_real(poisoned.at(i, j, b)))
                                << "the pad reached the factor at n=" << n << " ("
                                << i << "," << j << ") b=" << b;
                            ASSERT_EQ(host_imag(clean.at(i, j, b)),
                                      host_imag(poisoned.at(i, j, b)));
                        }
                    }
                }
            }
        }
    }
}

// The other triangle is neither read nor written. A finite sentinel read back proves
// NOT WRITTEN; a quiet NaN with the factor asserted NaN-free proves NOT READ.
TYPED_TEST(PotrfTinyTest, TinyOtherTriangleIsNeitherReadNorWritten) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;

    const int cap = this->cap();
    for (int n : {7, 8, 16, 17, 32}) {
        if (n > cap) continue;
        for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
            const int batch = 6;
            const T sentinel = make_scalar<T>(R(-3.5), R(11.25));
            auto p = make_tiny_prob<T>(n, batch, n + 5, (n + 5) * n + 13, uplo, 55u,
                                       sentinel);
            const auto info = this->run_tiny(p.view(), uplo);
            for (int b = 0; b < batch; ++b) {
                ASSERT_EQ(info[b], 0);
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        const bool in_tri = (uplo == Uplo::Lower) ? (i >= j) : (i <= j);
                        if (in_tri) continue;
                        ASSERT_EQ(host_real(p.at(i, j, b)), host_real(sentinel))
                            << "wrote outside the triangle at (" << i << "," << j << ")";
                        ASSERT_EQ(host_imag(p.at(i, j, b)), host_imag(sentinel));
                    }
                }
            }

            const R nan = std::numeric_limits<R>::quiet_NaN();
            auto q = make_tiny_prob<T>(n, batch, n + 5, (n + 5) * n + 13, uplo, 55u,
                                       make_scalar<T>(nan, nan));
            const auto info2 = this->run_tiny(q.view(), uplo);
            for (int b = 0; b < batch; ++b) {
                ASSERT_EQ(info2[b], 0);
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        const bool in_tri = (uplo == Uplo::Lower) ? (i >= j) : (i <= j);
                        if (!in_tri) continue;
                        ASSERT_FALSE(std::isnan(host_real(q.at(i, j, b))))
                            << "NaN leaked from the untouched triangle into ("
                            << i << "," << j << ")";
                        ASSERT_FALSE(std::isnan(host_imag(q.at(i, j, b))));
                    }
                }
            }
        }
    }
}

// PACKED LAUNCHES ARE INDEPENDENT. Several matrices share a sub-group at N < 32 and two
// sub-groups share a work-group at every N, so a partition index that is not unique in
// the work-group, or a store that runs past the triangle, shows up here and nowhere
// else. The neighbours' unused rows carry quiet NaN so a cross-matrix READ is visible
// too, not only a cross-matrix write.
TYPED_TEST(PotrfTinyTest, TinyPackedBatchMatchesSolo) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;

    const int cap = this->cap();
    const R nan = std::numeric_limits<R>::quiet_NaN();
    int packed_ns = 0;

    for (int n : {5, 8, 12, 16, 20, 32}) {
        if (n > cap) continue;
        const int pw = this->per_wg(n);
        if (pw <= 1) continue;
        ++packed_ns;

        const int batch = 3 * pw + 1;
        const int ld = n + 5;
        const int stride = ld * n + 13;
        auto packed = make_tiny_prob<T>(n, batch, ld, stride, Uplo::Lower, 2000u,
                                        make_scalar<T>(nan, nan));
        const auto info_packed = this->run_tiny(packed.view(), Uplo::Lower);

        for (int b = 0; b < batch; ++b) {
            ASSERT_EQ(info_packed[b], 0) << "n=" << n << " b=" << b;

            // The same matrix, alone in its launch. Item b of `packed` is rebuilt here
            // with the identical seed, so any difference is the packing.
            TinyProb<T> solo;
            solo.n = n; solo.batch = 1; solo.ld = ld; solo.stride = stride;
            solo.buf = UnifiedVector<T>(static_cast<size_t>(stride),
                                        make_scalar<T>(nan, nan));
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j <= i; ++j) {
                    solo.buf[i + static_cast<size_t>(j) * ld] =
                        packed.ref[b][i + static_cast<size_t>(j) * n];
                }
            }
            const auto info_solo = this->run_tiny(solo.view(), Uplo::Lower);
            ASSERT_EQ(info_solo[0], 0);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j <= i; ++j) {
                    ASSERT_EQ(host_real(packed.at(i, j, b)), host_real(solo.at(i, j, 0)))
                        << "packed vs solo differ at n=" << n << " b=" << b
                        << " (" << i << "," << j << ")";
                    ASSERT_EQ(host_imag(packed.at(i, j, b)), host_imag(solo.at(i, j, 0)));
                }
            }
        }
    }
    // Anti-vacuity: without a packing launch every assertion above was vacuous.
    ASSERT_GT(packed_ns, 0) << "no order packed more than one matrix per work-group";
}

// `info` is the EXACT 1-based column whose UPDATED Schur diagonal was not > 0, the
// first failure wins, and a failed item stays FINITE. The planted matrix's original
// diagonal at the failure column is positive, so a kernel reading the stale pivot
// cannot name it.
TYPED_TEST(PotrfTinyTest, TinyInfoIsExactAndFirstFailureWins) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;

    const int n = std::min(32, this->cap());
    ASSERT_GE(n, 16);

    for (Uplo uplo : {Uplo::Lower, Uplo::Upper}) {
        for (int c : {0, 1, 5, 8, 9, 15, n - 1}) {
            if (c < 0 || c >= n) continue;
            auto p = make_tiny_prob<T>(n, 1, n + 5, (n + 5) * n + 13, uplo, 1u,
                                       make_scalar<T>(R(0), R(0)));
            const auto ref = make_planted_ldl<T>(n, {c}, 4242u + static_cast<unsigned>(c));
            if (c >= 1) {
                ASSERT_GT(host_real(ref[c + static_cast<size_t>(c) * n]), R(0))
                    << "the planted matrix is not discriminating at column " << c;
            }
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    const bool in_tri = (uplo == Uplo::Lower) ? (i >= j) : (i <= j);
                    if (in_tri) {
                        p.buf[i + static_cast<size_t>(j) * p.ld] =
                            ref[i + static_cast<size_t>(j) * n];
                    }
                }
            }
            const auto info = this->run_tiny(p.view(), uplo);
            EXPECT_EQ(info[0], c + 1)
                << "planted a non-positive pivot at column " << c << " of " << n
                << " uplo=" << static_cast<int>(uplo);
        }
    }

    // Two failures, one launch: the earlier column must win.
    for (int c : {2, 7, 11}) {
        const int c2 = c + 5;
        if (c2 >= n) continue;
        auto p = make_tiny_prob<T>(n, 1, n, n * n, Uplo::Lower, 2u,
                                   make_scalar<T>(R(0), R(0)));
        const auto ref = make_planted_ldl<T>(n, {c, c2}, 777u + static_cast<unsigned>(c));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                p.buf[i + static_cast<size_t>(j) * p.ld] = ref[i + static_cast<size_t>(j) * n];
            }
        }
        EXPECT_EQ(this->run_tiny(p.view(), Uplo::Lower)[0], c + 1)
            << "failures planted at " << c << " and " << c2;
    }

    // At batch scale, with the failures at different columns so a shared flag shows,
    // and with the healthy neighbours still correct.
    {
        const int batch = 4 * this->per_wg(n) + 3;
        const std::vector<int> bad_items = {0, 5, batch - 1};
        const std::vector<int> bad_cols = {0, 6, n - 1};
        auto p = make_tiny_prob<T>(n, batch, n + 5, (n + 5) * n + 13, Uplo::Lower, 300u,
                                   make_scalar<T>(R(0), R(0)));
        for (size_t k = 0; k < bad_items.size(); ++k) {
            const int b = bad_items[k];
            p.ref[b] = make_planted_ldl<T>(n, {bad_cols[k]},
                                           90u + static_cast<unsigned>(b));
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j <= i; ++j) {
                    p.buf[static_cast<size_t>(b) * p.stride + i +
                          static_cast<size_t>(j) * p.ld] =
                        p.ref[b][i + static_cast<size_t>(j) * n];
                }
            }
        }
        const auto info = this->run_tiny(p.view(), Uplo::Lower);
        for (size_t k = 0; k < bad_items.size(); ++k) {
            EXPECT_EQ(info[bad_items[k]], bad_cols[k] + 1) << "item " << bad_items[k];
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j <= i; ++j) {
                    const T v = p.at(i, j, bad_items[k]);
                    ASSERT_TRUE(std::isfinite(host_real(v)))
                        << "failed item went non-finite at (" << i << "," << j << ")";
                    ASSERT_TRUE(std::isfinite(host_imag(v)));
                }
            }
        }
        for (int b = 0; b < batch; ++b) {
            if (std::find(bad_items.begin(), bad_items.end(), b) != bad_items.end()) continue;
            ASSERT_EQ(info[b], 0) << "healthy item " << b << " reported a failure";
            const auto L = this->extract_L(p, b, Uplo::Lower);
            EXPECT_LE((multiply_back_residual<T>(p.ref[b], L, n)), residual_tol<T>(n))
                << "healthy item " << b << " next to a failed one";
        }
    }
}

// An empty `info` span means "not requested" and must not change the answer: the
// fallback is UNINITIALISED pool scratch, and a workspace of zero bytes would make the
// allocator hand back nothing at all.
TYPED_TEST(PotrfTinyTest, TinyEmptyInfoSpanStillFactorises) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;

    const int n = std::min(29, this->cap());
    const int batch = 2 * this->per_wg(n) + 1;
    auto p = make_tiny_prob<T>(n, batch, n + 5, (n + 5) * n + 13, Uplo::Lower, 606u,
                               make_scalar<T>(R(0), R(0)));

    EXPECT_GT(sycl_potrf::potrf_tiny_buffer_size<T>(*this->ctx, p.view()), 0u)
        << "a zero-byte tier workspace makes potrf_buffer_size's native_need == 0 "
           "unimplemented check fire on a Tiny-only shape";

    this->run_tiny(p.view(), Uplo::Lower, /*pass_info_span=*/false);
    for (int b = 0; b < batch; ++b) {
        const auto L = this->extract_L(p, b, Uplo::Lower);
        EXPECT_LE((multiply_back_residual<T>(p.ref[b], L, n)), residual_tol<T>(n)) << "b=" << b;
    }
}

// The launch geometry, asserted against INVARIANTS and never against a pinned S: S is a
// tuning constant the design calls for A/B-ing, and pinning it turns the constant into a
// change-detector. Do not add assertions re-derived from the hook's own answer -- they are
// identities no break can move. evidence:
// docs/perf/potrf.md#what-the-geometry-test-can-and-cannot-check
TYPED_TEST(PotrfTinyTest, TinyLaunchGeometryIsReported) {
    using T = typename TestFixture::T;

    const int max_wg = static_cast<int>(
        this->ctx->device().get_property(DeviceProperty::MAX_WORK_GROUP_SIZE));

    int prev_bucket = 0;
    for (int n = 1; n <= this->cap(); ++n) {
        const unsigned g = sycl_potrf::potrf_tiny_debug_launch<T>(*this->ctx, n);
        ASSERT_NE(g, 0u) << "n=" << n << " is inside the cap but reports unfit";
        const int S = static_cast<int>(g >> 16);
        const int G = static_cast<int>(g & 0xffffu);

        ASSERT_GE(G, 1) << "n=" << n << ": no partitions per sub-group";
        ASSERT_EQ(32 % G, 0) << "n=" << n << ": " << G << " partitions do not divide the "
                                "sub-group, so the hook's S and G are not a partition of "
                                "one work-group";
        const int N = 32 / G;                 // the bucket the hook actually chose

        // The ladder, read OUT of the hook's answer: the bucket must hold n, be one of the
        // three instantiated widths, and be the TIGHTEST such.
        EXPECT_TRUE(N == 8 || N == 16 || N == 32)
            << "n=" << n << ": bucket " << N << " is not one of the instantiated widths";
        EXPECT_GE(N, n) << "n=" << n << ": bucket " << N << " is narrower than the order";
        EXPECT_TRUE(N == 8 || n > N / 2)
            << "n=" << n << ": bucket " << N << " is wider than the order needs -- the "
               "next bucket down, " << N / 2 << ", holds it";
        EXPECT_GE(N, prev_bucket) << "n=" << n << ": the bucket ladder is not monotone";
        prev_bucket = N;

        EXPECT_GE(S, 2) << "R3 wants at least two sub-groups per work-group";
        const int wg = S * 32;
        EXPECT_LE(wg, max_wg) << "n=" << n << ": the work-group is wider than the device's "
                                 "maximum, which aborts the enqueue";
    }

    EXPECT_EQ(sycl_potrf::potrf_tiny_debug_launch<T>(*this->ctx, this->cap() + 1), 0u)
        << "one order past the cap must report unfit";
}

// The route table sees the tier, and the tier sits FIRST in the order array.
TYPED_TEST(PotrfTinyTest, TinyRouteTableAndVocabulary) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::BackendType;
    using Tbl = dispatch::RouteTable<dispatch::Op::potrf, T>;
    const dispatch::Route tiny{dispatch::Origin::Native, dispatch::Algorithm::Tiny};

    // The vocabulary round trip. A missing parse case is SILENT: the pin becomes
    // nullopt, is dropped, and a benchmark arm measures whatever automatic() picked.
    EXPECT_EQ(dispatch::to_string(dispatch::Algorithm::Tiny), "tiny");
    ASSERT_TRUE(dispatch::parse_algorithm_word("tiny").has_value());
    EXPECT_EQ(*dispatch::parse_algorithm_word("tiny"), dispatch::Algorithm::Tiny);

    EXPECT_EQ(dispatch::kPotrfOrder[0].algo, dispatch::Algorithm::Tiny);
    EXPECT_TRUE(dispatch::is_native(dispatch::kPotrfOrder[0]));

    const int cap = this->cap();
    Matrix<T, MatrixFormat::Dense> A(cap, cap, 2);
    A.fill(T{});
    const auto shape = backend::potrf_op_shape<B, T>(*this->ctx, A.view(), Uplo::Lower);
    ASSERT_TRUE(shape.has_value());
    EXPECT_EQ(shape->tiny_max_n, cap);

    auto at_cap = *shape;
    EXPECT_TRUE(Tbl::supports(tiny, at_cap));
    // FALSE on purpose, exactly as getrf and geqrf answer it. Tiny is routable by an
    // explicit pin and nothing else: a true answer here hands the tier the whole vendor-free
    // walk and every bare `native` pin at order <= tiny_max_n. The flip belongs in the
    // commit that MEASURES the window, together with preferred().
    EXPECT_FALSE(Tbl::native_tier_preferred(tiny, at_cap));
    // Exactly one native tier answers true, or the vendor-free walk is an accident of
    // the order array (R8b, from the other direction). At the tiny cap that is CTA.
    EXPECT_TRUE(Tbl::native_tier_preferred(
        {dispatch::Origin::Native, dispatch::Algorithm::CTA}, at_cap));
    EXPECT_FALSE(Tbl::native_tier_preferred(
        {dispatch::Origin::Native, dispatch::Algorithm::Blocked}, at_cap));

    auto past = *shape;
    past.m = past.n = past.k = cap + 1;
    EXPECT_FALSE(Tbl::supports(tiny, past));
    EXPECT_FALSE(Tbl::native_tier_preferred(tiny, past));

    // preferred() is still empty for every tier: the window is set from measurement.
    EXPECT_FALSE(Tbl::preferred(tiny, at_cap));
}

// The facade actually reaches the TINY kernel; the guard is bit-exactness against the
// direct call, because the table answering {Native, Tiny} says nothing about what ran.
TYPED_TEST(PotrfTinyTest, TinyFacadeReachesTheTinyKernel) {
    using T = typename TestFixture::T;
    using R = typename TestFixture::R;
    static constexpr Backend B = TestFixture::BackendType;

    const int n = std::min(17, this->cap());
    const int batch = 5;

    // A bare ::setenv is invisible: batchlas::settings() snapshots the environment
    // before main() and parse_route_env reads only that snapshot.
    const ScopedEnvVar route_pin("BATCHLAS_POTRF_ROUTE", "tiny");

    auto through = make_tiny_prob<T>(n, batch, n + 5, (n + 5) * n + 13, Uplo::Lower,
                                     8080u, make_scalar<T>(R(0), R(0)));
    auto direct = make_tiny_prob<T>(n, batch, n + 5, (n + 5) * n + 13, Uplo::Lower,
                                    8080u, make_scalar<T>(R(0), R(0)));

    const auto route = backend::potrf_route<B, T>(*this->ctx, through.view(), Uplo::Lower,
                                                  /*vendor_available=*/true);
    ASSERT_TRUE(dispatch::is_native(route))
        << "BATCHLAS_POTRF_ROUTE=tiny did not resolve to a native route";
    ASSERT_EQ(route.algo, dispatch::Algorithm::Tiny);

    UnifiedVector<std::byte> ws(
        potrf_buffer_size<B, T>(*this->ctx, through.view(), Uplo::Lower));
    UnifiedVector<int32_t> info(batch, int32_t(-7));
    potrf<B, T>(*this->ctx, through.view(), Uplo::Lower, ws.to_span(), info.to_span());
    this->ctx->wait();

    const auto info_direct = this->run_tiny(direct.view(), Uplo::Lower);
    for (int b = 0; b < batch; ++b) {
        ASSERT_EQ(info[b], 0);
        ASSERT_EQ(info_direct[b], 0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                ASSERT_EQ(host_real(through.at(i, j, b)), host_real(direct.at(i, j, b)))
                    << "the facade did not run the tiny kernel at (" << i << "," << j
                    << ") b=" << b;
                ASSERT_EQ(host_imag(through.at(i, j, b)), host_imag(direct.at(i, j, b)));
            }
        }
    }
}

// potrf_buffer_size must cover the tier it now routes to, and must not shrink below
// what the tier's own query asks for.
TYPED_TEST(PotrfTinyTest, TinyBufferSizeCoversTheTier) {
    using T = typename TestFixture::T;
    static constexpr Backend B = TestFixture::BackendType;

    for (int n : {1, 8, 16, 32}) {
        if (n > this->cap()) continue;
        Matrix<T, MatrixFormat::Dense> A(n, n, 64);
        A.fill(T{});
        const std::size_t tier = sycl_potrf::potrf_tiny_buffer_size<T>(*this->ctx, A.view());
        std::size_t facade = 0;
        ASSERT_NO_THROW((facade = potrf_buffer_size<B, T>(*this->ctx, A.view(), Uplo::Lower)));
        EXPECT_GE(facade, tier) << "n=" << n;
    }
}

// The direct entry point re-applies every supports() gate and THROWS. A forced route
// that supports() rejects falls through to automatic() and silently runs the vendor,
// so a caller reaching this function past a gate must be told, not served.
TYPED_TEST(PotrfTinyTest, TinyDirectEntryPointRefusesWhatSupportsRefuses) {
    using T = typename TestFixture::T;

    UnifiedVector<int32_t> info(4, int32_t(0));
    UnifiedVector<std::byte> ws(1024);

    {   // above the register ceiling
        const int n = this->cap() + 1;
        Matrix<T, MatrixFormat::Dense> A(n, n, 1);
        A.fill(T{});
        EXPECT_THROW(sycl_potrf::potrf_tiny_dispatch<T>(*this->ctx, A.view(), Uplo::Lower,
                                                        ws.to_span(), info.to_span()),
                     std::invalid_argument);
    }
    {   // not square
        Matrix<T, MatrixFormat::Dense> A(8, 6, 1);
        A.fill(T{});
        EXPECT_THROW(sycl_potrf::potrf_tiny_dispatch<T>(*this->ctx, A.view(), Uplo::Lower,
                                                        ws.to_span(), info.to_span()),
                     std::invalid_argument);
    }
}


// ZERO LOCAL MEMORY, ZERO WORK-GROUP BARRIERS -- a SOURCE check, because the property is a
// design invariant with no observable behaviour on this box: a work-group barrier in a
// partition kernel is a race and a local_accessor is an occupancy cost, and both stay green
// in a functional test. The 48 KB launch hole is structurally unreachable only while both
// stay absent.
#ifdef BATCHLAS_POTRF_TINY_CC_PATH

// The first line of `path` whose CODE (comments stripped) contains `token`, or "".
inline std::string FirstTinySourceLineWith(const std::string& path, const char* token) {
    std::FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return std::string("<could not open ") + path + ">";
    char line[8192];
    std::string found;
    while (std::fgets(line, sizeof(line), f)) {
        const std::string text(line);
        const size_t comment = text.find("//");
        const std::string code =
            comment == std::string::npos ? text : text.substr(0, comment);
        if (code.find(token) != std::string::npos) { found = text; break; }
    }
    std::fclose(f);
    return found;
}

TEST(PotrfTinySource, IsBarrierAndLocalMemoryFree) {
    const std::string path = BATCHLAS_POTRF_TINY_CC_PATH;
    for (const char* token : {"local_accessor", "group_barrier"}) {
        const std::string offending = FirstTinySourceLineWith(path, token);
        EXPECT_TRUE(offending.empty())
            << "potrf_tiny.cc names `" << token
            << "` in code: the tier's whole argument is that it has no memory to order "
               "and nothing to synchronise. Offending line: " << offending;
    }
    // Anti-vacuity: the scan must be able to find something that IS there.
    EXPECT_FALSE(FirstTinySourceLineWith(path, "select_from_group").empty())
        << "the source scan found nothing at all -- check BATCHLAS_POTRF_TINY_CC_PATH";
}
#endif  // BATCHLAS_POTRF_TINY_CC_PATH
