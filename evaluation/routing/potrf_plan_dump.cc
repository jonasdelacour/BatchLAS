// potrf_plan_dump: the launch plan and cost terms of every potrf route at given shapes, for
// evaluation/routing/fit.py. Needs no GPU: the device is described by flags.
//
//   stdin : one shape per line, "<dtype> <n> <batch> [L|U]", dtype float|double|cfloat|cdouble
//   stdout: one JSON object per shape
//   flags : --local-mem B --max-wg N --cus N --max-threads-per-cu N --max-groups-per-cu N
//
// "auto" is today's RouteTable choice with a vendor present, "auto_vendor_free" without one.
// The vendor row is potrf_plan::vendor_pseudo_plan: it has no launch plan of its own.

#include "../../src/backends/potrf_route.hh"
#include "../../src/extensions/potrf_launch_plan.hh"
#include "../../src/sycl/trsm_native.hh"

#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>

using namespace batchlas;
using launch_plan::CostTerms;
using launch_plan::DeviceFacts;
using launch_plan::LaunchPlan;

namespace {

void emit_plan(std::ostream& o, const char* name, bool supported, const LaunchPlan& p,
               const DeviceFacts& d) {
    const CostTerms t = launch_plan::cost_terms(p, d);
    o << "\"" << name << "\":{\"supported\":" << (supported ? "true" : "false")
      << ",\"fits\":" << (p.fits ? "true" : "false") << ",\"launches\":" << p.launches
      << ",\"wave_launches\":" << p.wave_launches << ",\"groups\":" << p.groups
      << ",\"wg_size\":" << p.wg_size << ",\"slm_per_group\":" << p.slm_per_group
      << ",\"regs_per_item\":" << p.regs_per_item
      << ",\"resident_groups_per_cu\":" << p.resident_groups_per_cu << ",\"flops\":" << p.flops
      << ",\"useful_flops\":" << p.useful_flops << ",\"bytes\":" << p.bytes
      << ",\"serial_steps\":" << p.serial_steps << ",\"terms\":[" << t.launch << "," << t.flop
      << "," << t.byte << "," << t.step << "]}";
}

std::string route_name(dispatch::Route r) {
    if (dispatch::is_vendor(r)) return "vendor";
    switch (r.algo) {
        case dispatch::Algorithm::Tiny: return "native:tiny";
        case dispatch::Algorithm::CTA: return "native:cta";
        case dispatch::Algorithm::LPanel: return "native:lpanel";
        case dispatch::Algorithm::Blocked: return "native:blocked";
        default: return "unknown";
    }
}

template <typename T>
void dump(std::ostream& o, const char* dtype, int n, std::int64_t batch, Uplo uplo,
          const DeviceFacts& d) {
    const auto s = backend::potrf_op_shape_from_facts<Backend::CUDA, T>(
        d, n, n, batch, uplo, /*is_gpu=*/true, /*has_sg32=*/true, /*heterogeneous=*/false,
        sycl_potrf::potrf_blocked_available<T>());
    using Table = dispatch::RouteTable<dispatch::Op::potrf, T>;
    using dispatch::Algorithm;
    using dispatch::Origin;
    auto sup = [&](Algorithm a) { return Table::supports({Origin::Native, a}, s); };

    const dispatch::Route none{};
    const auto pick = dispatch::resolve_route_uninstrumented<dispatch::Op::potrf, T>(none, s, true);
    const auto pick_vf = dispatch::resolve_route_uninstrumented<dispatch::Op::potrf, T>(none, s, false);

    o << "{\"dtype\":\"" << dtype << "\",\"n\":" << n << ",\"batch\":" << batch
      << ",\"uplo\":\"" << (uplo == Uplo::Upper ? "U" : "L") << "\",\"auto\":\""
      << route_name(pick) << "\",\"auto_vendor_free\":\"" << route_name(pick_vf)
      << "\",\"routes\":{";
    emit_plan(o, "native:tiny", sup(Algorithm::Tiny), potrf_plan::tiny_plan<T>(n, batch, d), d);
    o << ",";
    emit_plan(o, "native:cta", sup(Algorithm::CTA), potrf_plan::cta_plan<T>(n, batch, d), d);
    o << ",";
    emit_plan(o, "native:lpanel", sup(Algorithm::LPanel), potrf_plan::lpanel_plan<T>(n, batch, d), d);
    o << ",";
    emit_plan(o, "native:blocked", sup(Algorithm::Blocked),
              potrf_plan::blocked_plan<T>(n, batch, d, sycl_trsm::trsm_cta_max_n<T>()), d);
    o << ",";
    emit_plan(o, "vendor", true, potrf_plan::vendor_pseudo_plan<T>(n, batch), d);
    o << "}}\n";
}

long long flag(int argc, char** argv, const char* name, long long dflt) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) return std::atoll(argv[i + 1]);
    }
    return dflt;
}

}  // namespace

int main(int argc, char** argv) {
    DeviceFacts d;
    d.local_mem_bytes = static_cast<std::size_t>(flag(argc, argv, "--local-mem", 0));
    d.max_wg_size = static_cast<int>(flag(argc, argv, "--max-wg", 1024));
    d.compute_units = static_cast<int>(flag(argc, argv, "--cus", 0));
    d.max_threads_per_cu = static_cast<int>(flag(argc, argv, "--max-threads-per-cu", 0));
    d.max_groups_per_cu = static_cast<int>(flag(argc, argv, "--max-groups-per-cu", 0));
    if (d.local_mem_bytes == 0 || d.compute_units == 0) {
        std::fprintf(stderr, "potrf_plan_dump: --local-mem and --cus are required\n");
        return 2;
    }
    std::cout.precision(17);
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        std::string dtype, ul = "L";
        long long n = 0, batch = 0;
        if (!(in >> dtype >> n >> batch)) continue;
        in >> ul;
        const Uplo uplo = (ul == "U" || ul == "Upper") ? Uplo::Upper : Uplo::Lower;
        const int ni = static_cast<int>(n);
        if (dtype == "float") dump<float>(std::cout, "float", ni, batch, uplo, d);
        else if (dtype == "double") dump<double>(std::cout, "double", ni, batch, uplo, d);
        else if (dtype == "cfloat") dump<std::complex<float>>(std::cout, "cfloat", ni, batch, uplo, d);
        else if (dtype == "cdouble") dump<std::complex<double>>(std::cout, "cdouble", ni, batch, uplo, d);
        else std::fprintf(stderr, "potrf_plan_dump: unknown dtype %s\n", dtype.c_str());
    }
    return 0;
}
