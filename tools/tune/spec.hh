#pragma once

// The per-op tuner spec (docs/design/flat-kernel-selection.md §6.1): what a cell is, how to
// build and verify its problem, which choices to time and which sources the table depends on.
// SYCL-free: the driver sees only this interface; each tools/tune/<op>_spec.cc implements it
// with the op's choice.hh and registers itself with BATCHLAS_TUNE_REGISTER.

#include "tune_core.hh"

#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace batchlas::tune {

// What one --cell child does. `arms` are pin words handed to select::ScopedPin: choice
// spellings when tuning, "<old spelling>" and "auto" in --gate.
struct CellRequest {
    std::string dtype;
    CellKey key;
    std::vector<std::string> arms;
    std::string mode;  // "time" | "jit" | "probe"
    int reps = 16;
    double warm_s = 1.5;
    bool reverse = false;  // pass 2: the base arm order reversed (§6.3 step 4)
    int ld_pad = 0;        // non-natural ld = n + ld_pad
};

struct ArmOutcome {
    std::string arm;
    std::string status;  // ok | skipped (pin refused: can_run false) | bad (verification) | error
    std::string reason;
    std::vector<double> ms;  // timed reps, in rep order
    std::vector<int> slot;   // the arm's position within each rep's rotated order
    double residual = 0;
    int info_nonzero = 0;
};

class OpSpec {
public:
    virtual ~OpSpec() = default;
    virtual std::string op() const = 0;
    // choice.hh key_names, e.g. {"uplo:exact", "n:log:3", "batch:log"}.
    virtual std::vector<std::string> key_names() const = 0;
    // to_string of every candidates<T>() entry, in list (tie-break) order.
    virtual std::vector<std::string> candidates(const std::string& dtype) const = 0;
    // The coarse grid's axes in key order, from choice.hh; `overrides` replaces whole axes.
    virtual std::vector<std::pair<std::string, std::vector<std::string>>> axes() const = 0;
    // Bytes of the op's inputs at a cell (the 4 GiB cap) and the key §6.2 bisects.
    virtual double bytes(const std::string& dtype, const CellKey& key) const = 0;
    virtual std::string refine_key() const { return "n"; }
    // §6.4: repo-relative kernel sources (the kernel-sources block CI and CMake read too).
    virtual std::vector<std::string> kernel_sources() const = 0;
    virtual std::string spec_file() const = 0;
    // A coverage `reached` route -> this op's pin spelling; then the child side: run one cell.
    virtual std::string normalize_route(const std::string& origin, const std::string& algo) const = 0;
    virtual std::vector<ArmOutcome> run_cell(const CellRequest& req) const = 0;

    std::vector<CellKey> grid(const std::map<std::string, std::vector<std::string>>& overrides) const;
};

inline double dtype_bytes(const std::string& dtype) {
    return dtype == "float" ? 4 : dtype == "cdouble" ? 16 : 8;
}

void register_spec(const OpSpec* spec);
const OpSpec* find_spec(std::string_view op);
std::vector<const OpSpec*> all_specs();

struct SpecRegistrar {
    explicit SpecRegistrar(const OpSpec* s) { register_spec(s); }
};

}  // namespace batchlas::tune

#define BATCHLAS_TUNE_REGISTER(Type)                                       \
    namespace {                                                            \
    const Type kBatchlasTuneSpecInstance{};                                \
    const ::batchlas::tune::SpecRegistrar kBatchlasTuneRegistrar{&kBatchlasTuneSpecInstance}; \
    }
