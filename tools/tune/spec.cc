#include "spec.hh"

#ifndef BATCHLAS_TUNE_SOURCE_DIR
#define BATCHLAS_TUNE_SOURCE_DIR "."
#endif

#include <algorithm>
#include <fstream>
#include <sstream>
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

KernelBlock OpSpec::kernel_block() const {
    std::ifstream f(std::string(BATCHLAS_TUNE_SOURCE_DIR) + "/" + spec_file());
    if (!f) throw std::runtime_error("cannot read " + spec_file());
    std::ostringstream ss;
    ss << f.rdbuf();
    return parse_kernel_block(ss.str());
}

std::map<std::string, std::vector<std::string>> OpSpec::family_sources() const {
    const KernelBlock b = kernel_block();
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
