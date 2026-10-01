// potrf_plan_dump: the launch plan and cost terms of every potrf route at given shapes, for
// evaluation/routing/fit.py. Needs no GPU: the device is described by flags.
//
//   stdin : one shape per line, "<dtype> <n> <batch> [L|U]", dtype float|double|cfloat|cdouble
//   stdout: one JSON object per shape
//   flags : --local-mem B --max-wg N --cus N --max-threads-per-cu N --max-groups-per-cu N
//           --regs <dtype>=t8:R,t16:R,t32:R,cta_sg:R,cta_wg:R,lp:R,lp16:R  (repeatable; from
//           evaluation/routing/profiles/registers.json via fit.py)
//           --profile sm_89|sm_120: also price with that profile's generated constants and
//             print "auto_model" (the RouteTable choice with the cost model, if its gate passed)
//           --device: also resolve each shape through the LIBRARY's potrf_route on GPU 0's
//             queue ("device_route", "device_route_vendor_free"); honours
//             BATCHLAS_ROUTING_PROFILE and BATCHLAS_POTRF_ROUTE like a real call
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
#include <map>
#include <memory>
#include <optional>
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
      << "," << t.byte << "," << t.step << "," << t.slot << "," << t.item << ","
      << (t.additive ? 1 : 0) << "]}";
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

std::map<std::string, potrf_plan::KernelRegs> g_regs;
arch::RoutingProfile g_profile = arch::RoutingProfile::Unset;
std::unique_ptr<Queue> g_queue;

// "t8:54,t16:64,cta_sg:64" -> KernelRegs; unknown keys are an error, not a silent 0.
bool parse_regs(const std::string& spec, potrf_plan::KernelRegs& r) {
    std::istringstream in(spec);
    std::string kv;
    while (std::getline(in, kv, ',')) {
        const auto c = kv.find(':');
        if (c == std::string::npos) return false;
        const std::string k = kv.substr(0, c);
        const int v = std::atoi(kv.c_str() + c + 1);
        if (k == "t8") r.tiny[0] = v;
        else if (k == "t16") r.tiny[1] = v;
        else if (k == "t32") r.tiny[2] = v;
        else if (k == "cta_sg") r.cta_sg = v;
        else if (k == "cta_wg") r.cta_wg = v;
        else if (k == "lp") r.lpanel = v;
        else if (k == "lp16") r.lpanel_nb16 = v;
        else return false;
    }
    return true;
}

template <typename T>
void dump(std::ostream& o, const char* dtype, int n, std::int64_t batch, Uplo uplo,
          const DeviceFacts& d) {
    const potrf_plan::KernelRegs r = g_regs[dtype];
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
      << route_name(pick) << "\",\"auto_vendor_free\":\"" << route_name(pick_vf) << "\"";
    if (g_profile != arch::RoutingProfile::Unset) {
        auto sm = s;
        sm.profile = g_profile;
        int nb_env = 0, w_env = 0;
        sycl_potrf::potrf_blocked_overrides(nb_env, w_env);
        backend::potrf_price_routes<T>(sm, d, sycl_trsm::trsm_cta_max_n<T>(), nb_env, w_env);
        const auto pm = dispatch::resolve_route_uninstrumented<dispatch::Op::potrf, T>(none, sm, true);
        o << ",\"model_enabled\":" << (sm.model_enabled ? "true" : "false")
          << ",\"auto_model\":\"" << route_name(pm) << "\"";
    }
    if (g_queue) {
        const MatrixView<T, MatrixFormat::Dense> A(nullptr, n, n, n, n * n, static_cast<int>(batch));
        const auto dr = backend::potrf_route<Backend::CUDA, T>(*g_queue, A, uplo, true);
        const auto dv = backend::potrf_route<Backend::CUDA, T>(*g_queue, A, uplo, false);
        o << ",\"device_route\":\"" << route_name(dr) << "\",\"device_route_vendor_free\":\""
          << route_name(dv) << "\"";
    }
    o << ",\"routes\":{";
    emit_plan(o, "native:tiny", sup(Algorithm::Tiny), potrf_plan::tiny_plan<T>(n, batch, d, r), d);
    o << ",";
    emit_plan(o, "native:cta", sup(Algorithm::CTA), potrf_plan::cta_plan<T>(n, batch, d, r), d);
    o << ",";
    emit_plan(o, "native:lpanel", sup(Algorithm::LPanel), potrf_plan::lpanel_plan<T>(n, batch, d, r), d);
    o << ",";
    emit_plan(o, "native:blocked", sup(Algorithm::Blocked),
              potrf_plan::blocked_plan<T>(n, batch, d, sycl_trsm::trsm_cta_max_n<T>(), r), d);
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
    // --query-device: what this box's GPU reports, to check evaluation/routing/profiles/.
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--query-device") == 0) {
            Device dev("gpu");
            std::cout << "{\"local_mem\":" << dev.get_property(DeviceProperty::LOCAL_MEM_SIZE)
                      << ",\"max_wg\":" << dev.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE)
                      << ",\"cus\":" << dev.get_property(DeviceProperty::MAX_COMPUTE_UNITS)
                      << ",\"cuda_cc\":" << dev.cuda_compute_capability() << "}\n";
            return 0;
        }
    }
    DeviceFacts d;
    d.local_mem_bytes = static_cast<std::size_t>(flag(argc, argv, "--local-mem", 0));
    d.max_wg_size = static_cast<int>(flag(argc, argv, "--max-wg", 1024));
    d.compute_units = static_cast<int>(flag(argc, argv, "--cus", 0));
    d.max_threads_per_cu = static_cast<int>(flag(argc, argv, "--max-threads-per-cu", 0));
    d.max_groups_per_cu = static_cast<int>(flag(argc, argv, "--max-groups-per-cu", 0));
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--regs") != 0) continue;
        const std::string spec = argv[i + 1];
        const auto eq = spec.find('=');
        if (eq == std::string::npos || !parse_regs(spec.substr(eq + 1), g_regs[spec.substr(0, eq)])) {
            std::fprintf(stderr, "potrf_plan_dump: bad --regs %s\n", spec.c_str());
            return 2;
        }
    }
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--device") == 0) {
            g_queue = std::make_unique<Queue>(Device("gpu"), Backend::CUDA, true);
        }
        if (std::strcmp(argv[i], "--profile") == 0 && i + 1 < argc) {
            const auto p = arch::parse_routing_profile(argv[i + 1]);
            if (!p) {
                std::fprintf(stderr, "potrf_plan_dump: unknown profile %s\n", argv[i + 1]);
                return 2;
            }
            g_profile = *p;
        }
    }
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
