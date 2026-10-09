// P0 of docs/design/small-n-factorization-plan.md: the in-tree, correct,
// saturation-aware A/B harness for the batched factorizations
// (potrf / getrf / getrs / geqrf / orgqr / gesv / posv) at n = 4..512.
// It ports the proven pieces of wp4_potrf/phase2_ab/realpotrf.cpp, wp6_lu/bench/lubench6.cpp and
// wp5_qr/bench/qrbench.cpp (tag perf-evidence/vendor-independence) into the tree. Inputs, residuals
// and bounds come from batchlas::verify (docs/design/verification.md).
// ONE CELL PER PROCESS. The binary takes exactly one (op, type, shape, batch)
// and never loops over shapes internally. That is not a style choice: the SLM
// carve-out attribute is sticky per CUfunction, so an earlier, larger launch in
// the same process can make a later launch that should have failed succeed, and
// the result then depends on iteration order. run_factor_grid.sh forks per cell.
//
// WHAT IT REFUSES TO DO. It never reports a timing it has not checked in the
// same process. Every timed arm is followed by an untimed run of the same route
// whose output is verified on the host, in double (or complex<double>)
// promotion, on items 0 AND batch-1 -- item 0 alone is blind to a wrong batch
// stride. A row that fails any gate carries bad=1 and a reason, and the caller
// is expected to drop it rather than quote it.
//
//   usage: factor_bench <op> <type> <m> <n> <nrhs> <batch> <reps>
//                       [--route=<pin>] [--csv=<path>] [--arms=a,b,...]
//   env:   WARM_S (seconds, default 1.5), LD_PAD (ld = m + LD_PAD)
//
#include <batchlas/blas/functions/geqrf.hh>
#include <batchlas/blas/functions/gesv.hh>
#include <batchlas/blas/functions/getrf.hh>
#include <batchlas/blas/functions/getrs.hh>
#include <batchlas/blas/functions/orgqr.hh>
#include <batchlas/blas/functions/posv.hh>
#include <batchlas/blas/functions/potrf.hh>
#include <batchlas/backend_config.h>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>

#include <batchlas/settings.hh>

#include "../src/extensions/getrf_native.hh"
#include "../src/extensions/potrf_native.hh"
#include "../src/sycl/trsm_native.hh"
#include "../src/ops/geqrf/choice.hh"
#include "../src/ops/orgqr/choice.hh"
#include "../src/ops/getrf/choice.hh"
#include "../src/ops/getrs/choice.hh"
#include "../src/ops/gesv/choice.hh"
#include "../src/ops/posv/choice.hh"
#include "../src/ops/potrf/choice.hh"

#include <batchlas/verify/inputs.hh>
#include <batchlas/verify/reference.hh>
#include <batchlas/verify/residuals.hh>
#include <batchlas/verify/tolerance.hh>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace batchlas;
// The vendor arm is whichever vendor library this build links: cuSOLVER/cuBLAS on
// CUDA, rocSOLVER/rocBLAS on ROCm.
#if BATCHLAS_HAS_CUDA_BACKEND
static constexpr Backend BE = Backend::CUDA;
#else
static constexpr Backend BE = Backend::ROCM;
#endif

static double warm_s() { const char* e = std::getenv("WARM_S"); return e ? std::atof(e) : 1.5; }
static int ld_pad()    { const char* e = std::getenv("LD_PAD"); return e ? std::atoi(e) : 0; }

struct Stat { double med = 0, mean = 0, relsd = 0; };
static Stat stat_of(std::vector<double> v) {
    if (v.empty()) return {};
    std::sort(v.begin(), v.end());
    Stat s;
    s.med = v[v.size() / 2];
    for (double x : v) s.mean += x;
    s.mean /= double(v.size());
    double sd = 0;
    for (double x : v) sd += (x - s.mean) * (x - s.mean);
    sd = std::sqrt(sd / double(v.size()));
    s.relsd = s.mean > 0 ? sd / s.mean : 0.0;
    return s;
}

// ------------------------------------------------------------- route pins
enum class OpKind { potrf, getrf, getrs, geqrf, orgqr, gesv, posv };

// P2: the FUSED ops have no vendor arm of their own -- see composed_pins below.
static bool is_solve_op(OpKind k) { return k == OpKind::gesv || k == OpKind::posv; }

static const char* pin_variable(OpKind k) {
    switch (k) {
        case OpKind::potrf: return "BATCHLAS_POTRF_ROUTE";
        case OpKind::getrf: return "BATCHLAS_GETRF_ROUTE";
        case OpKind::getrs: return "BATCHLAS_GETRS_ROUTE";
        case OpKind::geqrf: return "BATCHLAS_GEQRF_ROUTE";
        case OpKind::orgqr: return "BATCHLAS_ORGQR_ROUTE";
        case OpKind::gesv:  return "BATCHLAS_GESV_ROUTE";
        case OpKind::posv:  return "BATCHLAS_POSV_ROUTE";
    }
    return "";
}

// The leaf A/B variable per op; nullptr for an op with no leaf seam, which makes an
// `/leaf=` arm on that op a loud no-op rather than a quiet one.
static const char* leaf_variable(OpKind op) {
    switch (op) {
        case OpKind::getrf: return "BATCHLAS_GETRF_LEAF";
        case OpKind::geqrf: return "BATCHLAS_GEQRF_LEAF";
        default: return nullptr;
    }
}
static const char* op_text(OpKind k) {
    switch (k) {
        case OpKind::potrf: return "potrf";
        case OpKind::getrf: return "getrf";
        case OpKind::getrs: return "getrs";
        case OpKind::geqrf: return "geqrf";
        case OpKind::orgqr: return "orgqr";
        case OpKind::gesv:  return "gesv";
        case OpKind::posv:  return "posv";
    }
    return "?";
}

// PIN_PARSED SAYS THE VALUE WAS UNDERSTOOD, NOT THAT THE ROUTE TOOK: the resolved
// route is a separate readback (run_factor_grid.sh re-runs each cell once, untimed, with
// BATCHLAS_COVERAGE_OUT set and greps the `reached,` row). Every op here is flat-selected
// (src/ops/<op>/): a pin is auto, native, vendor or a choice spelling (`lpanel:panel=8`), and
// one that cannot run THROWS. The coverage readback carries the spelling in chosen_algo,
// e.g. `native:lpanel:panel=8`.
//
// It is queried INSIDE the ScopedEnvVar scope on purpose: settings() is a pre-main
// snapshot, and only ScopedEnvVar's reload_settings() makes the pin readable at all.
template <class Choice>
static bool select_pin_parsed(OpKind k) {
    const char* raw = settings().routing.route(op_text(k)).get();
    if (raw == nullptr) return false;
    std::string text = raw;
    for (char& ch : text) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    if (text == "auto" || text == "native" || text == "vendor") return true;
    return select::parse<Choice>(text).has_value();
}

static bool pin_parsed_now(OpKind k) {
    switch (k) {
        case OpKind::potrf: return select_pin_parsed<ops::potrf::PotrfChoice>(k);
        case OpKind::getrf: return select_pin_parsed<ops::getrf::GetrfChoice>(k);
        case OpKind::getrs: return select_pin_parsed<ops::getrs::GetrsChoice>(k);
        case OpKind::geqrf: return select_pin_parsed<ops::geqrf::GeqrfChoice>(k);
        case OpKind::orgqr: return select_pin_parsed<ops::orgqr::OrgqrChoice>(k);
        case OpKind::gesv:  return select_pin_parsed<ops::gesv::GesvChoice>(k);
        case OpKind::posv:  return select_pin_parsed<ops::posv::PosvChoice>(k);
    }
    return false;
}

// The explicit Q (m x k) of item b from the packed reflectors, Q = H_0 ... H_{k-1} applied to the
// first k columns of I_m. batchlas::verify has no equivalent (qr_residual applies the reflectors
// in place), and orthogonality() needs Q itself.
template <typename T>
static void form_Q(const UnifiedVector<T>& F, const UnifiedVector<T>& tau,
                   int b, int m, int k, int ld, size_t stride, size_t taustride,
                   std::vector<verify::promoted_t<T>>& Q) {
    using D = verify::promoted_t<T>;
    const size_t o = size_t(b) * stride;
    Q.assign(size_t(m) * size_t(k), D(0));
    for (int j = 0; j < k; ++j) Q[size_t(j) * size_t(m) + size_t(j)] = D(1);
    for (int step = 0; step < k; ++step) {
        const int i = k - 1 - step;
        const D t = verify::up(tau[taustride * size_t(b) + size_t(i)]);
        for (int c = 0; c < k; ++c) {
            D s = Q[size_t(c) * size_t(m) + size_t(i)];      // v(i) == 1 implicitly
            for (int r = i + 1; r < m; ++r)
                s += verify::conj(verify::up(F[o + size_t(i) * size_t(ld) + size_t(r)])) *
                     Q[size_t(c) * size_t(m) + size_t(r)];
            Q[size_t(c) * size_t(m) + size_t(i)] -= t * s;
            for (int r = i + 1; r < m; ++r)
                Q[size_t(c) * size_t(m) + size_t(r)] -=
                    t * verify::up(F[o + size_t(i) * size_t(ld) + size_t(r)]) * s;
        }
    }
}

// ------------------------------------------------------------- one arm
struct Arm {
    std::string name;      // "vendor" | "native" | "<pin>/leaf=<x>"
    std::string pin;       // the value actually written to BATCHLAS_<OP>_ROUTE
    std::string leaf;      // P4: BATCHLAS_GETRF_LEAF for this arm's reps only, or empty
    // P2: extra (variable, value) pins held for this arm's reps only; see composed_pins.
    std::vector<std::pair<std::string, std::string>> sub_pins;
    bool pin_parsed = false;
    Stat st;
    double residual = std::numeric_limits<double>::quiet_NaN();
    double extra = 0;
    int info_nonzero = 0;
    int bad = 0;
    std::string reason;
    bool refused = false;  // a pin this shape cannot run: the arm is reported, never timed
};

// P4 needs an A/B the route pin cannot express: both arms are the SAME route
// (native blocked getrf) and differ only in which panel leaf the driver calls, which
// is BATCHLAS_GETRF_LEAF. An arm named "blocked/leaf=reg" pins both, for that arm's
// reps only, so the leaf A/B is interleaved in one process like every other arm here
// instead of ratioed across two. evidence: docs/perf/lu.md#lu-the-register-leaf-ab
// P5 needs the same A/B for geqrf's panel leaf, whose variable is BATCHLAS_GEQRF_LEAF, so the
// leaf variable is chosen by OP rather than hardcoded: setting getrf's variable on a geqrf cell
// is a silent no-op and the two arms would then be the same arm measured twice.
// P2: gesv and posv have NO vendor arm of their own. Their `Blocked` route is a
// COMPOSITION -- `getrf; getrs` and `potrf; trsm; trsm` -- and each composed op routes
// independently, so "the vendor arm" and "the two-launch native arm" are the same
// gesv/posv route under two different sub-op pinnings. Naming the arm alone would
// measure whatever the composed ops happened to route to, which is the "diff the
// ROUTE, not the timing" defect; these pins make the composition explicit and
// `pin_parsed_now` still proves the OUTER pin landed.
// A potrf or getrf pin the shape cannot run throws, so `composed` pins tiny only inside its
// ceiling (16 for cdouble) and the best runnable native tier above it; trsm `cta` likewise.
static std::vector<std::pair<std::string, std::string>>
composed_pins(OpKind op, const std::string& arm_name, bool potrf_tiny_fits, bool trsm_cta_fits,
              bool getrf_tiny_fits) {
    if (op == OpKind::gesv) {
        if (arm_name == "vendor")
            return {{"BATCHLAS_GETRF_ROUTE", "vendor"}, {"BATCHLAS_GETRS_ROUTE", "vendor"}};
        if (arm_name == "native")
            return {{"BATCHLAS_GETRF_ROUTE", "native"}, {"BATCHLAS_GETRS_ROUTE", "native"}};
        if (arm_name == "composed")
            return {{"BATCHLAS_GETRF_ROUTE", getrf_tiny_fits ? "tiny" : "native"},
                    {"BATCHLAS_GETRS_ROUTE", "cta"}};
    }
    if (op == OpKind::posv) {
        if (arm_name == "vendor")
            return {{"BATCHLAS_POTRF_ROUTE", "vendor"}, {"BATCHLAS_TRSM_ROUTE", "vendor"}};
        if (arm_name == "native")
            return {{"BATCHLAS_POTRF_ROUTE", "native"}, {"BATCHLAS_TRSM_ROUTE", "native"}};
        if (arm_name == "composed")
            return {{"BATCHLAS_POTRF_ROUTE", potrf_tiny_fits ? "tiny" : "native"},
                    {"BATCHLAS_TRSM_ROUTE", trsm_cta_fits ? "cta" : "native"}};
    }
    return {};
}

// For a solve op the outer route is what separates the arms: "vendor" and "composed"
// pin the two-launch `blocked` composition, "native" takes the SHIPPED native walk --
// the fused tiny kernel inside its window, the composition above it. Pinning "native"
// to `blocked` measured a composition the library never selects at n <= 32 and put
// posv at 0.25-0.62x where the shipped fused kernel wins 1.1-6.9x.
static std::string solve_outer_pin(const std::string& arm_name) {
    if (arm_name == "vendor" || arm_name == "composed") return "blocked";
    return arm_name;
}

struct ArmEnv {
    ScopedEnvVar route;
    std::unique_ptr<ScopedEnvVar> leaf;
    std::vector<std::unique_ptr<ScopedEnvVar>> extra;
    ArmEnv(const char* var, const Arm& a, const char* leaf_var) : route(var, a.pin.c_str()) {
        if (!a.leaf.empty() && leaf_var != nullptr) {
            leaf = std::make_unique<ScopedEnvVar>(leaf_var, a.leaf.c_str());
        }
        for (const auto& kv : a.sub_pins)
            extra.push_back(std::make_unique<ScopedEnvVar>(kv.first.c_str(), kv.second.c_str()));
    }
};

static void flag(Arm& a, const char* why) {
    a.bad = 1;
    if (!a.reason.empty()) a.reason += "+";
    a.reason += why;
}

// A refused pin (std::invalid_argument from choose()) costs only that arm, never the cell.
// The message goes in a CSV field, so its commas and quotes are replaced.
static void refuse(Arm& a, const std::invalid_argument& e) {
    std::string why = std::string("pin refused: ") + e.what();
    for (char& ch : why)
        if (ch == ',') ch = ';';
        else if (ch == '"') ch = '\'';
    a.refused = true;
    flag(a, why.c_str());
}

static void gate(Arm& a, bool residual_within_bound, int reps) {
    if (!std::isfinite(a.residual)) flag(a, "residual_nonfinite");
    else if (!residual_within_bound) flag(a, "residual");
    if (!std::isfinite(a.extra)) flag(a, "extra_nonfinite");
    if (a.info_nonzero != 0) flag(a, "info");
    if (reps > 1 && a.st.relsd > 0.10) flag(a, "relsd");
    if (a.bad == 0) a.reason = "ok";
}

struct Cfg {
    OpKind op = OpKind::potrf;
    std::string type;
    int m = 0, n = 0, nrhs = 0, batch = 0, reps = 0;
    std::string route_pin;      // --route=, applied to the NATIVE arm only
    std::string csv;
    // The arms to interleave, in order. Each name is also its own route pin unless it
    // is "native" and --route= overrides it. Defaults to the two-arm vendor/native A/B.
    // NOT a set of bools: comparing three tiers (vendor, cta, tiny) in ONE process is
    // what keeps clock drift out of the ratio, and two processes ratioing through a
    // shared vendor arm puts it back. evidence: docs/perf/small-n-baseline.md#the-arms-list
    std::vector<std::string> arms{"vendor", "native"};
    // potrf/posv only. The grids on docs/perf were Lower-only because this harness could
    // not express anything else, which is why the Upper windows read "untested".
    bool upper = false;
};

static Uplo up_of(const Cfg& c) { return c.upper ? Uplo::Upper : Uplo::Lower; }

static void emit(const Cfg& c, const Arm& a, std::FILE* csv) {
    char buf[2048];
    std::snprintf(buf, sizeof(buf),
                  "%s,%s,%d,%d,%d,%d,%s,%s,%d,%.6f,%.6f,%.4f,%d,%.3e,%.3e,%d,%d,%s\n",
                  op_text(c.op), c.type.c_str(), c.m, c.n, c.nrhs, c.batch,
                  a.name.c_str(), a.pin.c_str(), a.pin_parsed ? 1 : 0,
                  a.st.med, a.st.mean, a.st.relsd, c.reps,
                  a.residual, a.extra, a.info_nonzero, a.bad, a.reason.c_str());
    std::fputs(buf, stdout);
    if (csv) std::fputs(buf, csv);
}

// ------------------------------------------------------------- driver
template <typename T>
static int run(const Cfg& c) {
    auto q = std::make_shared<Queue>(Device("gpu"), BE);
    using MV = MatrixView<T, MatrixFormat::Dense>;

    const int m = c.m, n = c.n, nrhs = c.nrhs, batch = c.batch;
    const int pad = ld_pad();
    const int lda = m + pad;                       // the strided-ld trap, honoured
    const size_t sa = size_t(lda) * size_t(n);
    const int ldb = n + pad;
    const size_t sb = size_t(ldb) * size_t(nrhs > 0 ? nrhs : 1);
    const int kmin = std::min(m, n);
    const bool needs_factor = (c.op == OpKind::getrs || c.op == OpKind::orgqr);

    const size_t nbatch = size_t(batch);
    UnifiedVector<T> A0(sa * nbatch), A(sa * nbatch);
    // THESE ARE VIEWS OVER THE CALLER'S BUFFERS, never Matrix: Matrix's
    // (const T*, ...) constructor COPIES into its own storage, so factorising a
    // Matrix built that way leaves the array this program checks untouched --
    // which reads as "info == 0 with a huge residual", a wrong-answer report for
    // a correct kernel.
    // And EVERY VIEW GETS ITS OWN POINTER ARRAY: the vendor batched paths call
    // data_ptrs(ctx) and throw "data_ptrs target is null" on a view built
    // without one, and sharing a single array between two views is a recorded
    // aliasing trap.
    UnifiedVector<T*> pA0(nbatch), pA(nbatch);
    MV A0v(A0.data(), m, n, lda, int(sa), batch, pA0.data());
    MV Av(A.data(), m, n, lda, int(sa), batch, pA.data());

    UnifiedVector<int32_t> info(nbatch, 0);
    UnifiedVector<int64_t> piv(size_t(n) * nbatch, 0);
    // getrf packs 1-based int32 pivots in the FIRST HALF of the int64 span
    // (src/extensions/getrf_native.hh), n per item.
    const int* pivi = reinterpret_cast<const int*>(piv.data());
    const int pstride = n;
    UnifiedVector<T> tau(size_t(kmin) * nbatch);

    const bool has_rhs = (c.op == OpKind::getrs) || is_solve_op(c.op);
    const size_t bsz = has_rhs ? sb * nbatch : size_t(1);
    UnifiedVector<T> B0(bsz), X(bsz);
    UnifiedVector<T*> pB0(nbatch), pX(nbatch);
    MV B0v, Xv;
    if (has_rhs) {
        B0v = MV(B0.data(), n, nrhs, ldb, int(sb), batch, pB0.data());
        Xv  = MV(X.data(),  n, nrhs, ldb, int(sb), batch, pX.data());
        verify::fill_random(B0v, 777);
    }

    switch (c.op) {
        case OpKind::potrf:
        case OpKind::posv: verify::fill_spd(A0v); break;
        case OpKind::getrf:
        case OpKind::gesv:
        case OpKind::getrs: verify::fill_lu(A0v, 12345); break;
        case OpKind::geqrf:
        case OpKind::orgqr: verify::fill_gauss(A0v, 12345); break;
    }

    auto reset_A = [&] { (void)MV::copy(*q, Av, A0v); q->wait(); };

    // getrs and orgqr both consume a factorisation. Produce it ONCE, untimed,
    // through the public entry point under the AUTOMATIC route, and keep a host
    // copy, so both arms are handed byte-identical input every rep.
    const size_t fsz = needs_factor ? sa * nbatch : size_t(1);
    UnifiedVector<T> F(fsz);
    if (needs_factor) {
        reset_A();
        if (c.op == OpKind::getrs) {
            const size_t fw = getrf_buffer_size<BE, T>(*q, Av);
            const size_t fwsz = fw ? fw : size_t(1);
            UnifiedVector<std::byte> fws(fwsz);
            (void)getrf<BE, T>(*q, Av, piv.to_span(), fws.to_span(), info.to_span());
        } else {
            const size_t gw = geqrf_buffer_size<BE, T>(*q, Av, tau.to_span());
            const size_t gwsz = gw ? gw : size_t(1);
            UnifiedVector<std::byte> gws(gwsz);
            (void)geqrf<BE, T>(*q, Av, tau.to_span(), gws.to_span());
        }
        q->wait();
        std::memcpy(F.data(), A.data(), sa * nbatch * sizeof(T));
    }

    // FACTORIZATION IS DESTRUCTIVE, so the input is restored before EVERY rep.
    // For getrs the factored A must survive instead, so the reset restores the
    // right-hand side; for orgqr it restores the packed reflectors.
    auto reset = [&] {
        if (c.op == OpKind::getrs) { (void)MV::copy(*q, Xv, B0v); q->wait(); }
        else if (c.op == OpKind::orgqr) { std::memcpy(A.data(), F.data(), sa * nbatch * sizeof(T)); }
        // A fused solve overwrites A with the factor AND B with X, so both are restored.
        else if (is_solve_op(c.op)) { reset_A(); (void)MV::copy(*q, Xv, B0v); q->wait(); }
        else reset_A();
    };

    // An arm's NAME is its pin, so --arms=vendor,cta,tiny needs no new flag per tier;
    // --route= still overrides the pin of the arm literally named "native", which is
    // the two-arm form every existing script uses.
    std::vector<Arm> arms;
    for (const std::string& nm : c.arms) {
        Arm a;
        a.name = nm;
        std::string pin = nm;
        const auto cut = nm.find("/leaf=");
        if (cut != std::string::npos) {
            pin = nm.substr(0, cut);
            a.leaf = nm.substr(cut + 6);
        }
        a.pin = (pin == "native" && !c.route_pin.empty()) ? c.route_pin : pin;
        if (is_solve_op(c.op)) {
            a.sub_pins = composed_pins(c.op, pin, n <= sycl_potrf::potrf_tiny_max_n<T>(), n <= sycl_trsm::trsm_cta_max_n<T>(),
                                       n <= sycl_getrf::getrf_tiny_max_n<T>());
            if (a.pin == pin) a.pin = solve_outer_pin(pin);
        }
        arms.push_back(a);
    }
    if (arms.empty()) { std::fprintf(stderr, "factor_bench: no arms selected\n"); return 2; }

    const char* var = pin_variable(c.op);

    // The workspace is sized per arm INSIDE that arm's pin and the MAXIMUM is
    // allocated: the two routes do not agree on how much they need, and handing
    // a native tier a vendor-sized workspace is the recorded "108x too small"
    // defect. Over-allocating is safe; under-allocating is not.
    size_t wneed = 0;
    for (size_t i = 0; i < arms.size(); ++i) {
        ArmEnv pinned(var, arms[i], leaf_variable(c.op));
        arms[i].pin_parsed = pin_parsed_now(c.op);
        size_t need = 0;
        try {
            switch (c.op) {
                case OpKind::potrf: need = potrf_buffer_size<BE, T>(*q, Av, up_of(c)); break;
                case OpKind::getrf: need = getrf_buffer_size<BE, T>(*q, Av); break;
                case OpKind::getrs: need = getrs_buffer_size<BE, T>(*q, Av, Xv, Transpose::NoTrans); break;
                case OpKind::geqrf: need = geqrf_buffer_size<BE, T>(*q, Av, tau.to_span()); break;
                case OpKind::orgqr: need = orgqr_buffer_size<BE, T>(*q, Av, tau.to_span()); break;
                case OpKind::gesv: need = gesv_buffer_size<BE, T>(*q, Av, Xv); break;
                case OpKind::posv: need = posv_buffer_size<BE, T>(*q, Av, Xv, up_of(c)); break;
            }
        } catch (const std::invalid_argument& e) {
            refuse(arms[i], e);
        }
        wneed = std::max(wneed, need);
    }
    const size_t wsz = wneed ? wneed : size_t(1);
    UnifiedVector<std::byte> ws(wsz);

    auto call = [&] {
        switch (c.op) {
            case OpKind::potrf: (void)potrf<BE, T>(*q, Av, up_of(c), ws.to_span(), info.to_span()); break;
            case OpKind::getrf: (void)getrf<BE, T>(*q, Av, piv.to_span(), ws.to_span(), info.to_span()); break;
            case OpKind::getrs: (void)getrs<BE, T>(*q, Av, Xv, Transpose::NoTrans, piv.to_span(), ws.to_span()); break;
            case OpKind::geqrf: (void)geqrf<BE, T>(*q, Av, tau.to_span(), ws.to_span()); break;
            case OpKind::orgqr: (void)orgqr<BE, T>(*q, Av, tau.to_span(), ws.to_span()); break;
            case OpKind::gesv:
                (void)gesv<BE, T>(*q, Av, Xv, piv.to_span(), ws.to_span(), info.to_span());
                break;
            case OpKind::posv:
                (void)posv<BE, T>(*q, Av, Xv, up_of(c), ws.to_span(), info.to_span());
                break;
        }
        q->wait();
    };
    // One arm's run under its pin; false (and the arm refused) if the pin throws.
    auto run_arm = [&](Arm& a) {
        if (a.refused) return false;
        try {
            call();
            return true;
        } catch (const std::invalid_argument& e) {
            refuse(a, e);
            return false;
        }
    };

    // TIME-BASED WARM-UP, DISCARDED, and INTERLEAVED in the timed loop's arm order -- measured,
    // not stylistic: a per-arm warm-up made arm 0's first timed rep 2.2x slow every time and
    // tripped the rel_sd gate on a stable cell. evidence: docs/perf/small-n-baseline.md#warm-up-order-and-the-variance-gate
    {
        const double budget = warm_s() * double(arms.size());
        const auto w0 = std::chrono::steady_clock::now();
        do {
            for (size_t i = 0; i < arms.size(); ++i) {
                if (arms[i].refused) continue;
                ArmEnv pinned(var, arms[i], leaf_variable(c.op));
                reset();
                (void)run_arm(arms[i]);
            }
        } while (std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count() < budget);
    }

    // INTERLEAVED A/B IN ONE PROCESS: every rep runs every arm back to back, so
    // clock drift hits both arms equally. Medians are taken per arm afterwards.
    std::vector<std::vector<double>> ms(arms.size());
    for (int r = 0; r < c.reps; ++r) {
        for (size_t i = 0; i < arms.size(); ++i) {
            if (arms[i].refused) continue;
            ArmEnv pinned(var, arms[i], leaf_variable(c.op));
            reset();
            const auto t0 = std::chrono::steady_clock::now();
            if (!run_arm(arms[i])) continue;
            const auto t1 = std::chrono::steady_clock::now();
            ms[i].push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
    }

    // DUMP_MS prints the raw per-rep times. rel_sd is a gate, so it has to be
    // auditable: a cell rejected for rel_sd is either a noisy neighbour, a
    // single cold outlier, or a genuinely bimodal route, and the median alone
    // cannot tell those apart.
    if (std::getenv("DUMP_MS")) {
        for (size_t i = 0; i < arms.size(); ++i) {
            std::fprintf(stderr, "  raw %s:", arms[i].name.c_str());
            for (double v : ms[i]) std::fprintf(stderr, " %.4f", v);
            std::fprintf(stderr, "\n");
        }
    }

    // CORRECTNESS IN THE SAME PROCESS, per arm, from one extra UNTIMED run of
    // that arm's route -- so a fast wrong answer cannot be reported as a win.
    for (size_t i = 0; i < arms.size(); ++i) {
        Arm& a = arms[i];
        if (a.refused) continue;
        a.st = stat_of(ms[i]);
        {
            ArmEnv pinned(var, a, leaf_variable(c.op));
            for (int b = 0; b < batch; ++b) info[b] = 0;
            reset();
            if (!run_arm(a)) continue;
        }
        // Items 0 AND batch-1: item 0 alone is blind to a wrong batch stride.
        const std::vector<int> items{0, batch - 1};
        using Check = verify::Check;
        bool within = false;
        switch (c.op) {
            case OpKind::potrf:
                a.residual = verify::potrf_residual(A0v, Av, up_of(c), items);
                within = verify::pass<T>(Check::factorization, n, a.residual);
                break;
            case OpKind::getrf: {
                // Packed 1-based int32 pivots, n per item, in the first half of the int64 span.
                VectorView<int32_t> pivv(reinterpret_cast<int32_t*>(piv.data()), n, batch, 1, pstride);
                a.residual = verify::getrf_residual(A0v, Av, pivv, items);
                within = verify::pass<T>(Check::factorization, n, a.residual);
                // The DISCRIMINATING oracle: the pivot sequence, elementwise, against an
                // independent host xGETRF on the same input.
                int mism = 0;
                std::vector<int32_t> hp;
                for (int b : items) {
                    auto h = verify::copy_item(A0v, b);
                    if (!verify::getrf_pivots(n, n, h, hp)) { flag(a, "pivot_oracle_unavailable"); break; }
                    for (int k = 0; k < n; ++k)
                        if (hp[size_t(k)] != pivi[size_t(b) * size_t(pstride) + size_t(k)]) ++mism;
                }
                a.extra = double(mism);
                if (mism != 0) flag(a, "pivot_mismatch");
                int nontrivial = 0;
                for (int k = 0; k < n; ++k) nontrivial += (pivi[k] != k + 1);
                if (nontrivial == 0) flag(a, "vacuous_pivots");
                break;
            }
            case OpKind::gesv:
            case OpKind::posv:
            case OpKind::getrs:
                a.residual = verify::solve_residual(A0v, Xv, B0v, items);
                within = verify::pass<T>(Check::solve, n, a.residual);
                break;
            case OpKind::geqrf: {
                using D = verify::promoted_t<T>;
                using QV = MatrixView<D, MatrixFormat::Dense>;
                VectorView<T> tauv(tau.data(), kmin, batch, 1, kmin);
                a.residual = verify::qr_residual(A0v, Av, tauv, items);
                std::vector<D> Q;
                double orth = 0;
                for (int b : items) {
                    form_Q<T>(A, tau, b, m, kmin, lda, sa, size_t(kmin), Q);
                    D* qp = Q.data();
                    QV Qv(Q.data(), m, kmin, m, m * kmin, 1, &qp);
                    orth = verify::nanmax(orth, verify::orthogonality(Qv));
                }
                a.extra = orth;
                within = verify::pass<T>(Check::factorization, n, a.residual);
                if (!verify::pass<T>(Check::orthogonality, m, orth)) flag(a, "orthogonality");
                break;
            }
            case OpKind::orgqr: {
                // orgqr has no factorisation residual of its own.
                a.residual = verify::orthogonality(Av, items);
                a.extra = a.residual;
                within = verify::pass<T>(Check::orthogonality, m, a.residual);
                break;
            }
        }
        if (c.op == OpKind::potrf || c.op == OpKind::getrf)
            for (int b = 0; b < batch; ++b) if (info[b] != 0) ++a.info_nonzero;
        gate(a, within, c.reps);
    }

    std::FILE* csv = nullptr;
    if (!c.csv.empty()) {
        std::FILE* probe = std::fopen(c.csv.c_str(), "r");
        const bool fresh = (probe == nullptr);
        if (probe) std::fclose(probe);
        csv = std::fopen(c.csv.c_str(), "a");
        if (csv && fresh)
            std::fputs("op,type,m,n,nrhs,batch,arm,pin,pin_parsed,median_ms,mean_ms,"
                       "rel_sd,reps,residual,extra_check,info_nonzero,bad,reason\n", csv);
    }
    int rc = 0;
    for (const Arm& a : arms) { emit(c, a, csv); rc |= a.bad; }
    if (csv) std::fclose(csv);
    return rc;
}

// ------------------------------------------------------------- main
int main(int argc, char** argv) {
    if (argc < 8) {
        std::fprintf(stderr,
            "usage: factor_bench <op> <type> <m> <n> <nrhs> <batch> <reps>\n"
            "                    [--route=<pin>] [--csv=<path>] [--arms=a,b,...]\n"
            "  op   : potrf getrf getrs geqrf orgqr gesv posv\n"
            "  type : float double cfloat cdouble\n"
            "  --uplo=upper|lower : potrf and posv only, default lower\n"
            "  env  : WARM_S (seconds, default 1.5), LD_PAD (ld = m + LD_PAD)\n"
            "prints: op,type,m,n,nrhs,batch,arm,pin,pin_parsed,median_ms,mean_ms,"
            "rel_sd,reps,residual,extra_check,info_nonzero,bad,reason\n");
        return 2;
    }
    Cfg c;
    const std::string opn = argv[1];
    if      (opn == "potrf") c.op = OpKind::potrf;
    else if (opn == "getrf") c.op = OpKind::getrf;
    else if (opn == "getrs") c.op = OpKind::getrs;
    else if (opn == "geqrf") c.op = OpKind::geqrf;
    else if (opn == "orgqr") c.op = OpKind::orgqr;
    else if (opn == "gesv")  c.op = OpKind::gesv;
    else if (opn == "posv")  c.op = OpKind::posv;
    else { std::fprintf(stderr, "factor_bench: unknown op %s\n", opn.c_str()); return 2; }

    c.type  = argv[2];
    c.m     = std::atoi(argv[3]);
    c.n     = std::atoi(argv[4]);
    c.nrhs  = std::atoi(argv[5]);
    c.batch = std::atoi(argv[6]);
    c.reps  = std::atoi(argv[7]);
    for (int i = 8; i < argc; ++i) {
        const std::string a = argv[i];
        if (a.rfind("--route=", 0) == 0) c.route_pin = a.substr(8);
        else if (a.rfind("--csv=", 0) == 0) c.csv = a.substr(6);
        else if (a == "--uplo=upper") c.upper = true;
        else if (a == "--uplo=lower") c.upper = false;
        else if (a.rfind("--arms=", 0) == 0) {
            // SPLIT ON COMMAS EXACTLY, never a substring test: `--arms=native` under a
            // `find("vendor")` test dropped the vendor arm silently, and `--arms=vendor`
            // kept it while dropping native -- the same class of defect as the recorded
            // substring `--name` trap in the minibench harness.
            const std::string v = a.substr(7);
            c.arms.clear();
            size_t pos = 0;
            while (pos <= v.size()) {
                const size_t comma = v.find(',', pos);
                const size_t end = (comma == std::string::npos) ? v.size() : comma;
                const std::string tok = v.substr(pos, end - pos);
                if (!tok.empty()) c.arms.push_back(tok);
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
            if (c.arms.empty()) {
                std::fprintf(stderr, "factor_bench: --arms= listed no arms\n");
                return 2;
            }
        } else { std::fprintf(stderr, "factor_bench: unknown flag %s\n", a.c_str()); return 2; }
    }
    if (c.m <= 0 || c.n <= 0 || c.batch <= 0 || c.reps <= 0) {
        std::fprintf(stderr, "factor_bench: need m>0 n>0 batch>0 reps>0\n");
        return 2;
    }
    if ((c.op == OpKind::getrs || is_solve_op(c.op)) && c.nrhs <= 0) {
        std::fprintf(stderr, "factor_bench: %s needs nrhs > 0\n", opn.c_str());
        return 2;
    }
    if ((c.op == OpKind::potrf || c.op == OpKind::getrf || c.op == OpKind::getrs ||
         is_solve_op(c.op)) && c.m != c.n) {
        std::fprintf(stderr, "factor_bench: %s is square; m must equal n\n", opn.c_str());
        return 2;
    }
    if (c.n > c.m) {
        std::fprintf(stderr, "factor_bench: this harness assumes n <= m\n");
        return 2;
    }

    if (c.type == "float")   return run<float>(c);
    if (c.type == "double")  return run<double>(c);
    if (c.type == "cfloat")  return run<std::complex<float>>(c);
    if (c.type == "cdouble") return run<std::complex<double>>(c);
    std::fprintf(stderr, "factor_bench: unknown type %s\n", c.type.c_str());
    return 2;
}
