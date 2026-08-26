// ===========================================================================
// L8b. GETRS, THE TWO PERMUTATION SPELLINGS.
//
// WP8-I2 gave the composed getrs driver a SECOND spelling of the row
// permutation: instead of walking the interchange list down every column of B
// (lu_laswp.hh's lu_laswp_launch), it applies the list ONCE to an identity index
// array in local memory and then gathers, dst[i] = src[idxs[i]], contiguously.
// Which one runs is a SPEED decision -- BATCHLAS_GETRS_LASWP, or B.cols()
// against kGetrsPermGatherMinNrhs -- and never a correctness one.
//
// So the two arms must agree BIT FOR BIT: the permutation is a pure data move
// and the two trsm calls that follow it are identical, so any difference at all
// is a defect and not rounding. That is a strictly stronger assertion than the
// residual, which both arms would pass with the SAME wrong permutation.
//
// THREE ANTI-VACUITY GUARDS, and none is decorative:
//
//  (1) THE SPELLING IS READ BACK, per arm, from getrs_perm_spelling_debug --
//      the driver's OWN resolution, not a re-derivation. Two things it catches:
//      an environment read that did not take, and the gather's SILENT FALLBACK
//      to the walk when the tile does not fit local memory. Without it, a test
//      that believes it is exercising the gather can be running the walk twice
//      and asserting that the walk equals itself.
//  (2) THE PERMUTATION MUST NOT BE AN INVOLUTION. If it were, the forward and
//      reversed index walks would coincide and the Trans/ConjTrans rows -- the
//      only place the reversed direction is exercised -- would prove nothing.
//  (3) nrhs = 70 EXCEEDS THE TILE WIDTH for every scalar type on this device, so
//      the multi-chunk loop and its partial last chunk both run. At nrhs = 5 the
//      first chunk is already partial. A single mid-size nrhs would test neither.
//
// n = 96 and n = 257 are both below and above the 256-wide work-group, which
// selects different branches of the (column, row) flattening; 257 is odd, so the
// odd-ld padding is a no-op there and a real pad at 96.
// ===========================================================================
TYPED_TEST(LuTest, GetrsPermutationSpellingsAgreeBitForBit) {
    using T = typename TestFixture::T;
    const int batch = 3;

    for (int n : {96, 257}) {
        for (int nrhs : {5, 70}) {
            auto p = make_dominant_permuted<T>(n, batch, 4242u + unsigned(n + nrhs));
            this->run_blocked(p);
            ASSERT_GE(non_diagonal_pivots(p, 0), n / 4);
            ASSERT_FALSE(interchange_is_involution(p.expect_piv))
                << "this matrix's permutation is SELF-INVERSE, so the gather's REVERSED "
                   "index walk is indistinguishable from its forward one and the Trans and "
                   "ConjTrans rows below prove nothing";
            check_factor(p, "getrs/spellings/factor");
            if (this->HasFailure()) return;

            auto rhs = make_rhs<T>(n, nrhs, batch, 1313u + unsigned(n + nrhs));

            for (Transpose op : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
                std::vector<std::vector<T>> answer;
                for (const char* spelling : {"walk", "gather"}) {
                    setenv("BATCHLAS_GETRS_LASWP", spelling, 1);

                    // GUARD (1). The driver's own resolution, for THIS shape, on
                    // THIS queue -- so a fallback the caller cannot see is visible
                    // here.
                    const int got =
                        sycl_getrs::getrs_perm_spelling_debug<T>(*this->ctx, n, nrhs);
                    const int want = (std::strcmp(spelling, "gather") == 0) ? 1 : 0;
                    ASSERT_EQ(got, want)
                        << "BATCHLAS_GETRS_LASWP=" << spelling << " at n=" << n
                        << " nrhs=" << nrhs << " resolved spelling " << got
                        << ". A gather that fell back to the walk would make the "
                           "bit-identity assertion below compare the walk with itself.";

                    reset_rhs(rhs);
                    auto A = view_of(p);
                    auto Bv = view_of(rhs);
                    // The query must stay 0 under BOTH spellings: the gather is in
                    // place. See GetrsPermGatherBuysNoWorkspace below.
                    UnifiedVector<std::byte> ws(std::max<std::size_t>(
                        1, sycl_getrs::getrs_blocked_buffer_size<T>(*this->ctx, A, Bv, op)));
                    ASSERT_NO_THROW(sycl_getrs::getrs_blocked_dispatch<T>(
                        *this->ctx, A, Bv, op, p.piv.to_span(), ws.to_span(),
                        this->getrs_seam()));
                    this->ctx->wait();

                    for (int b = 0; b < batch; ++b) {
                        const double res = solve_residual<T>(
                            p.a0.data() + size_t(b) * p.stride,
                            rhs.buf.data() + size_t(b) * rhs.stride,
                            rhs.b0.data() + size_t(b) * rhs.stride,
                            n, nrhs, p.ld, rhs.ld, op);
                        EXPECT_LE(res, solve_tol<T>(n))
                            << "getrs spelling=" << spelling << " transA=" << int(op)
                            << " n=" << n << " nrhs=" << nrhs << " b=" << b;
                    }
                    answer.emplace_back(rhs.buf.begin(), rhs.buf.end());
                    if (this->HasFailure()) { unsetenv("BATCHLAS_GETRS_LASWP"); return; }
                }
                unsetenv("BATCHLAS_GETRS_LASWP");

                // THE STRONG ASSERTION. Same permutation, same two solves, so the
                // answers must be identical to the last bit.
                size_t diff = 0;
                for (size_t i = 0; i < answer[0].size(); ++i)
                    if (std::memcmp(&answer[0][i], &answer[1][i], sizeof(T)) != 0) ++diff;
                EXPECT_EQ(diff, size_t(0))
                    << "the walk and the collapsed gather disagree in " << diff
                    << " of " << answer[0].size() << " elements at transA=" << int(op)
                    << " n=" << n << " nrhs=" << nrhs
                    << ". They apply the SAME permutation to the SAME buffer and then run "
                       "the SAME two trsm calls, so any difference is a defect.";
                if (this->HasFailure()) return;
            }
        }
    }
}

// ===========================================================================
// L8c. THE SPELLING DECISION SURFACE, WITHOUT RUNNING A KERNEL.
//
// getrs_perm_spelling_debug resolves through the driver's own perm_spelling()
// and its own capacity arithmetic, so this test pins the two boundaries the
// driver actually has:
//
//   THE nrhs BOUNDARY, kGetrsPermGatherMinNrhs, which is the default policy.
//   The constant is TRANSCRIBED from the header rather than written out, so a
//   later retune moves the test with the code -- but the CELLS ON EITHER SIDE
//   are asserted, which is what a wrongly-inverted comparison would break.
//
//   THE CAPACITY REFUSAL. The gather needs 2*n ints plus one column of B in
//   local memory; above that it enqueues NOTHING and the driver re-schedules the
//   identical composition with the walk. That fallback is silent by design --
//   RouteTable<Op::getrs,T> has no field to advertise a laswp capacity -- and it
//   is therefore invisible to every other test in this suite. Asserting it here
//   costs no kernel launch, because the query takes n as an integer.
// ===========================================================================
TYPED_TEST(LuTest, GetrsPermSpellingDecisionSurface) {
    using T = typename TestFixture::T;
    if (this->ctx->device().type != DeviceType::GPU) GTEST_SKIP() << "the gather is GPU-only";

    unsetenv("BATCHLAS_GETRS_LASWP");
    constexpr int kMin = sycl_getrs::kGetrsPermGatherMinNrhs;
    ASSERT_GE(kMin, 1) << "a boundary below 1 would make the walk unreachable by default";

    // THE DEFAULT nrhs BOUNDARY, both sides.
    if (kMin > 1) {
        EXPECT_EQ(sycl_getrs::getrs_perm_spelling_debug<T>(*this->ctx, 128, kMin - 1), 0)
            << "nrhs just below kGetrsPermGatherMinNrhs must take the WALK by default";
    }
    EXPECT_EQ(sycl_getrs::getrs_perm_spelling_debug<T>(*this->ctx, 128, kMin), 1)
        << "nrhs at kGetrsPermGatherMinNrhs must take the GATHER by default";
    EXPECT_EQ(sycl_getrs::getrs_perm_spelling_debug<T>(*this->ctx, 128, 4 * kMin), 1);

    // linalg::solve issues getrs at nrhs = 1 and is the only caller in the tree.
    // It must keep the walk, which is what makes "the boundary buys nothing at
    // the narrow end" a property of the shipped library and not of a comment.
    EXPECT_EQ(sycl_getrs::getrs_perm_spelling_debug<T>(*this->ctx, 512, 1), kMin <= 1 ? 1 : 0);

    // THE OVERRIDES beat the boundary in both directions.
    setenv("BATCHLAS_GETRS_LASWP", "walk", 1);
    EXPECT_EQ(sycl_getrs::getrs_perm_spelling_debug<T>(*this->ctx, 128, 4 * kMin), 0)
        << "BATCHLAS_GETRS_LASWP=walk must force the walk above the boundary";
    setenv("BATCHLAS_GETRS_LASWP", "gather", 1);
    EXPECT_EQ(sycl_getrs::getrs_perm_spelling_debug<T>(*this->ctx, 128, 1), 1)
        << "BATCHLAS_GETRS_LASWP=gather must force the gather below the boundary";

    // THE CAPACITY REFUSAL, forced on, at an order no tile can hold. This is the
    // only assertion in the suite that the fallback branch is reachable at all.
    EXPECT_EQ(sycl_getrs::getrs_perm_spelling_debug<T>(*this->ctx, 1 << 20, 4 * kMin), 0)
        << "the gather must REFUSE (and fall back to the walk) at an order whose column "
           "cannot fit local memory, rather than launching a kernel that cannot run";

    // ...and it must NOT refuse at an order the suite and the benchmarks reach.
    // A capacity that fires early is a lever that never runs -- 'linked is not
    // reachable', with the sign flipped.
    EXPECT_EQ(sycl_getrs::getrs_perm_spelling_debug<T>(*this->ctx, 1024, 4 * kMin), 1)
        << "the gather must still fit at n = 1024, the largest order this pass measured";
    unsetenv("BATCHLAS_GETRS_LASWP");
}

// ===========================================================================
// L8d. THE GATHER BUYS NO WORKSPACE, AT ANY WIDTH.
//
// THE HAZARD THIS GUARDS, stated exactly, because nothing else in the suite can
// see it. src/dispatch/entry_points/factorization.cc:846-866 takes the workspace
// maximum over EVERY NATIVE TIER THAT supports() THE SHAPE, not over the tier
// the route named -- and at nrhs <= kGetrsFusedMaxRhs BOTH tiers supports(). So
// a gather implemented the way the WP6 plan budgets for it -- an out-of-place
// RHS plus an int32[n] per item, bought in getrs_blocked_buffer_size -- would
// bill every nrhs = 1 call that routes to the FUSED tier and needs nothing:
// 1,310,720 B at cdouble n=512 batch=128, on linalg::solve's hot path.
//
// The shipped gather permutes IN LOCAL MEMORY, in place, so there is nothing to
// bill. This test is what keeps that true: it asserts ZERO at a wide nrhs under
// BOTH spellings, so a later out-of-place strategy cannot arrive silently.
// ===========================================================================
TYPED_TEST(LuTest, GetrsPermGatherBuysNoWorkspace) {
    using T = typename TestFixture::T;
    const int n = 96, batch = 3;
    auto p = make_dominant_permuted<T>(n, batch, 606u);

    for (int nrhs : {1, 8, 128}) {
        auto rhs = make_rhs<T>(n, nrhs, batch, 707u + unsigned(nrhs));
        auto A = view_of(p);
        auto Bv = view_of(rhs);
        for (const char* spelling : {"walk", "gather"}) {
            setenv("BATCHLAS_GETRS_LASWP", spelling, 1);
            for (Transpose op : {Transpose::NoTrans, Transpose::Trans, Transpose::ConjTrans}) {
                EXPECT_EQ(sycl_getrs::getrs_blocked_buffer_size<T>(*this->ctx, A, Bv, op),
                          std::size_t(0))
                    << "the composed getrs must stay workspace-free at nrhs=" << nrhs
                    << " spelling=" << spelling << " transA=" << int(op)
                    << ". A buffer bought here is charged to every narrow call that "
                       "routes to the FUSED tier, because the facade maxes over every "
                       "SUPPORTED native tier and not over the routed one.";
            }
        }
    }
    unsetenv("BATCHLAS_GETRS_LASWP");
}
