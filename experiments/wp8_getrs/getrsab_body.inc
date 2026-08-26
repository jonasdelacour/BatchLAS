// ---- the getrs A/B body; spliced onto lubench6.cpp's scaffolding by build.py.
//
// WP8-I2 A/B: the composed getrs driver's ROW PERMUTATION, two spellings,
// INTERLEAVED INSIDE ONE PROCESS.
//
// WHY A NEW HARNESS RATHER THAN TWO RUNS OF lubench6. GATE-B asks for the two
// arms interleaved within one session. Two DRIVER SPELLINGS are neither two
// routes nor two builds, so no pin and no library swap can separate them -- but
// BATCHLAS_GETRS_LASWP is re-read on every call once it is PRESENT in the
// environment (src/extensions/getrs_native.cc's perm_spelling(); presence
// latches, the value does not), so setenv() between reps is a real per-rep
// interleave. Everything above this line is lubench6.cpp's, copied by script
// rather than re-derived: the diagonally dominant then row-permuted matrix, the
// host LAPACKE factorisation, the NaN-propagating solve probe, Tol<T>, the
// resolved-route column.
//
// THE EXTRA ORACLE THIS HARNESS HAS AND lubench6 DOES NOT: the two arms compute
// the SAME permutation of the same buffer followed by the SAME two trsm calls,
// so their SOLUTIONS must be BIT-IDENTICAL. That is a stronger statement than
// either residual, and it is exactly what the collapse claims.
//
// ANTI-VACUITY, and it is not optional here for TWO independent reasons:
//   * the env presence flag latches, so a mis-ordered harness would run one arm
//     twice and report a flat 1.00x;
//   * the gather FALLS BACK to the walk when the tile does not fit, silently.
// Both are read back from sycl_getrs::getrs_perm_spelling_debug<T>, which
// resolves through the driver's own functions, and a row whose two arms report
// the SAME spelling is marked BAD.

static double warm_s2() { const char* e = std::getenv("WARM_S"); return e ? std::atof(e) : 1.0; }
static int nprobe2()    { const char* e = std::getenv("NPROBE"); return e ? std::atoi(e) : 1; }

template <typename T>
static int run_ab(const char* tn, int n, int nrhs, int batch, int reps,
                  const std::string& armA, const std::string& armB, Transpose tr) {
    // The knob must be PRESENT before the first getrs call.
    setenv("BATCHLAS_GETRS_LASWP", armA.c_str(), 1);

    auto q = std::make_shared<Queue>(Device("gpu"), BE);
    const size_t sa = size_t(n) * n, sb = size_t(n) * nrhs;

    UnifiedVector<T> A0(sa * batch), A(sa * batch);
    UnifiedVector<int64_t> piv(size_t(n) * batch);
    UnifiedVector<int32_t> info(batch);
    fill_A0<T>(A0, n, batch, 12345);

    UnifiedVector<T*> pA0(batch), pA(batch);
    MatrixView<T, MatrixFormat::Dense> A0v(A0.data(), n, n, n, int(sa), batch, pA0.data());
    MatrixView<T, MatrixFormat::Dense> Av(A.data(), n, n, n, int(sa), batch, pA.data());
    MatrixView<T, MatrixFormat::Dense>::copy(*q, Av, A0v);
    q->wait();

    const size_t f_ws = getrf_buffer_size<BE, T>(*q, Av);
    UnifiedVector<std::byte> fws(f_ws ? f_ws : 1);
    getrf<BE, T>(*q, Av, piv.to_span(), fws.to_span(), info.to_span());
    q->wait();
    const int* pivi = reinterpret_cast<const int*>(piv.data());
    const int ntp = nontrivial_pivots(pivi, n);
    const double fres = getrf_probe<T>(A, A0, pivi, n, n, batch, nprobe2());
    const std::string rf = rstr(backend::getrf_route<BE, T>(*q, Av, kVendorF));

    // B0 is the right-hand side; X is the buffer solved in place. XA keeps arm
    // A's answer for the bit-identity check.
    UnifiedVector<T> B0(sb * batch), X(sb * batch), XA(sb * batch);
    { Rng rg(777); for (size_t i = 0; i < B0.size(); ++i) B0[i] = mk<T>(rg.next(), rg.next()); }
    UnifiedVector<T*> pB0(batch), pX(batch);
    MatrixView<T, MatrixFormat::Dense> B0v(B0.data(), n, nrhs, n, int(sb), batch, pB0.data());
    MatrixView<T, MatrixFormat::Dense> Xv(X.data(), n, nrhs, n, int(sb), batch, pX.data());

    const size_t s_ws = getrs_buffer_size<BE, T>(*q, Av, Xv, tr);
    UnifiedVector<std::byte> sws(s_ws ? s_ws : 1);
    const std::string rs = rstr(backend::getrs_route<BE, T>(*q, Av, Xv, tr, kVendorF));

    auto arm = [&](const std::string& a) { setenv("BATCHLAS_GETRS_LASWP", a.c_str(), 1); };
    auto spell = [&](const std::string& a) {
        arm(a);
        return sycl_getrs::getrs_perm_spelling_debug<T>(*q, n, nrhs);
    };
    const int sA = spell(armA), sB = spell(armB);

    // The RHS copy is OUTSIDE the timed region -- the same discipline lubench6's
    // getrs mode uses, because the copy is not part of the op.
    auto once = [&]() {
        MatrixView<T, MatrixFormat::Dense>::copy(*q, Xv, B0v);
        q->wait();
        const auto t0 = std::chrono::steady_clock::now();
        getrs<BE, T>(*q, Av, Xv, tr, piv.to_span(), sws.to_span());
        q->wait();
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    };

    // WARM BOTH ARMS: JIT, clocks, first touch. Untimed and discarded.
    const auto w0 = std::chrono::steady_clock::now();
    do { arm(armA); once(); arm(armB); once(); }
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() < warm_s2());

    arm(armA); once();
    std::memcpy(XA.data(), X.data(), sb * batch * sizeof(T));
    const double resA = solve_probe<T>(X, B0, A0, n, nrhs, batch, nprobe2(), tr);

    arm(armB); once();
    size_t diff = 0;
    if (std::memcmp(XA.data(), X.data(), sb * batch * sizeof(T)) != 0) {
        const unsigned char* p = reinterpret_cast<const unsigned char*>(XA.data());
        const unsigned char* r = reinterpret_cast<const unsigned char*>(X.data());
        for (size_t i = 0; i < sb * batch; ++i)
            if (std::memcmp(p + i * sizeof(T), r + i * sizeof(T), sizeof(T)) != 0) ++diff;
    }
    const double resB = solve_probe<T>(X, B0, A0, n, nrhs, batch, nprobe2(), tr);

    // INTERLEAVED reps, arm by arm, inside the one process.
    std::vector<double> msA, msB;
    for (int r = 0; r < reps; ++r) {
        arm(armA); msA.push_back(once());
        arm(armB); msB.push_back(once());
    }
    const Stat stA = stat_of(msA), stB = stat_of(msB);

    const bool okc = std::isfinite(resA) && resA <= Tol<T>::v &&
                     std::isfinite(resB) && resB <= Tol<T>::v &&
                     std::isfinite(fres) && fres <= Tol<T>::v &&
                     ntp > 0 && diff == 0 && sA != sB;

    // ratio > 1 means arm B (the second name, by convention the NEW arm) is faster.
    std::printf("%s,%d,%d,%d,%s,%s,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.3e,%.3e,%zu,%zu,%s,%d,%s\n",
                tn, n, nrhs, batch, armA.c_str(), armB.c_str(), sA, sB,
                stA.med, stB.med, stA.med / stB.med, stA.relsd, stB.relsd,
                resA, resB, diff, s_ws, (rf + "|" + rs).c_str(), ntp,
                okc ? "ok" : "BAD");
    return okc ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr,
            "usage: getrsab <type> <n> <nrhs> <batch> <reps> [armA=walk] [armB=gather] [N|T|C]\n"
            "cols  : type,n,nrhs,batch,armA,armB,spellA,spellB,medA,medB,ratio,"
            "relsdA,relsdB,resA,resB,bitdiff,ws,route,ntpiv,flag\n");
        return 2;
    }
    const std::string t = argv[1];
    const int n = std::atoi(argv[2]), nrhs = std::atoi(argv[3]);
    const int batch = std::atoi(argv[4]), reps = std::atoi(argv[5]);
    const std::string a = (argc > 6) ? argv[6] : "walk";
    const std::string b = (argc > 7) ? argv[7] : "gather";
    const char tc = (argc > 8) ? argv[8][0] : 'N';
    const Transpose tr = (tc == 'T') ? Transpose::Trans
                       : (tc == 'C') ? Transpose::ConjTrans
                                     : Transpose::NoTrans;
    if (t == "float")   return run_ab<float>(t.c_str(), n, nrhs, batch, reps, a, b, tr);
    if (t == "double")  return run_ab<double>(t.c_str(), n, nrhs, batch, reps, a, b, tr);
    if (t == "cfloat")  return run_ab<std::complex<float>>(t.c_str(), n, nrhs, batch, reps, a, b, tr);
    if (t == "cdouble") return run_ab<std::complex<double>>(t.c_str(), n, nrhs, batch, reps, a, b, tr);
    std::fprintf(stderr, "unknown type %s\n", t.c_str());
    return 2;
}
