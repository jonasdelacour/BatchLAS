#include "spec.hh"

#include <algorithm>
#include <stdexcept>

namespace batchlas::tune {

namespace {
std::vector<const OpSpec*>& registry() {
    static std::vector<const OpSpec*> r;
    return r;
}
}  // namespace

void register_spec(const OpSpec* spec) { registry().push_back(spec); }

const OpSpec* find_spec(std::string_view op) {
    for (const OpSpec* s : registry())
        if (s->op() == op) return s;
    return nullptr;
}

std::vector<const OpSpec*> all_specs() { return registry(); }

std::vector<std::string> ld_audit_arms(const std::vector<ArmOutcome>& out, const std::vector<std::string>& wanted) {
    std::vector<std::string> arms;
    for (const ArmOutcome& o : out)
        if ((o.status == "ok" || o.status == "eliminated") &&
            std::find(wanted.begin(), wanted.end(), o.arm) != wanted.end())
            arms.push_back(o.arm);
    return arms;
}

void apply_ld_audit(std::vector<ArmOutcome>& out, const std::vector<ArmOutcome>& audit) {
    for (const ArmOutcome& a : audit)
        for (ArmOutcome& o : out) {
            if (o.arm != a.arm) continue;
            if (a.status == "ok") {
                o.ld_audit = "pass";
            } else if (a.status == "skipped") {
                o.ld_audit = "skipped";
            } else {
                o.ld_audit = "fail";
                o.status = "bad";
                o.reason = "ld audit: ld_pad +" + std::to_string(kLdAuditPad) + ": " + a.status +
                           (a.reason.empty() ? "" : " " + a.reason);
                o.residual = a.residual;
            }
        }
}

std::vector<ArmOutcome> run_cell_audited(const OpSpec& spec, const CellRequest& req) {
    std::vector<ArmOutcome> out = spec.run_cell(req);
    if (req.ld_audit.empty() || (req.mode != "race" && req.mode != "time")) return out;
    CellRequest a = req;
    a.mode = "verify";
    a.ld_pad = req.ld_pad + kLdAuditPad;
    a.arms = ld_audit_arms(out, req.ld_audit);
    a.seed_order.clear();
    a.ld_audit.clear();
    if (a.arms.empty()) return out;
    std::vector<ArmOutcome> got;
    try {
        got = spec.run_cell(a);
    } catch (const std::exception& e) {
        for (const std::string& arm : a.arms) got.push_back({arm, "error", e.what(), {}, {}, 0, 0, ""});
    }
    apply_ld_audit(out, got);
    return out;
}

std::vector<std::int64_t> OpSpec::dims(const std::string& dtype, const CellKey& key) const {
    static_cast<void>(dtype);
    std::vector<std::int64_t> out;
    for (const auto& [name, value] : key) {
        if (name == "batch") continue;
        std::size_t used = 0;
        std::int64_t v = 0;
        try {
            v = std::stoll(value, &used);
        } catch (const std::exception&) {
            continue;  // exact keys (uplo, form, trans) are words
        }
        if (used == value.size()) out.push_back(v);
    }
    return out;
}

KernelBlock OpSpec::kernel_block(const std::string& repo) const { return kernel_block_from_file(repo, spec_file()); }

std::map<std::string, std::vector<std::string>> OpSpec::family_sources(const std::string& repo) const {
    const KernelBlock b = kernel_block(repo);
    std::map<std::string, std::vector<std::string>> out = b.family;
    for (const auto& [fam, files] : b.deps) out[fam].insert(out[fam].end(), files.begin(), files.end());
    return out;
}

std::vector<CellKey> OpSpec::grid(const std::string& dtype,
                                  const std::map<std::string, std::vector<std::string>>& overrides) const {
    static_cast<void>(dtype);
    auto ax = axes();
    for (const auto& [name, values] : overrides) {
        auto it = std::find_if(ax.begin(), ax.end(), [&](const auto& a) { return a.first == name; });
        if (it == ax.end()) throw std::invalid_argument(op() + " has no grid axis '" + name + "'");
        it->second = values;
    }
    std::vector<CellKey> cells{CellKey{}};
    for (const auto& [name, values] : ax) {
        std::vector<CellKey> next;
        for (const CellKey& c : cells)
            for (const std::string& v : values) {
                CellKey k = c;
                k.push_back({name, v});
                next.push_back(std::move(k));
            }
        cells = std::move(next);
    }
    return cells;
}

}  // namespace batchlas::tune
