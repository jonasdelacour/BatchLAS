#pragma once

// Auto's expected pick for one shape, read from the shipped tables, so a retune changes the
// expectation and not the test. Not via select::choose: the nearest row of each table in borrow
// order, first entry a strict ScopedPin accepts while `probe` runs the op. Keys are spelled by
// the test; AutoReadsEveryKeyField (synthetic tables) proves key_of's wiring independently.
// evidence: docs/design/flat-kernel-selection.md#phase-5-the-eleven-transcribed-ops

#include "../src/select/select.hh"

#include <stdexcept>
#include <string>
#include <string_view>

namespace test_utils {

inline constexpr const char* kNoTableEntryRuns = "<no table entry runs: last resort>";

// `native_only`: what the class word "native" resolves to (the vendor family is skipped).
template <class Choice, class Probe>
std::string shipped_table_pick(std::string_view op, std::string_view dtype, const batchlas::select::Device& d,
                               const batchlas::select::Key& key, Probe&& probe, bool native_only = false) {
    for (const batchlas::select::Table* t : batchlas::select::tables_in_borrow_order(op, dtype, d)) {
        const batchlas::select::TableRow* row = t->nearest(key);
        if (!row) continue;
        for (const batchlas::select::TableEntry& e : row->ranked) {
            if (native_only && e.spelling.rfind("vendor", 0) == 0) continue;
            try {
                const batchlas::select::ScopedPin<Choice> pin(op, e.spelling, batchlas::select::StrictPin{});
                probe();
            } catch (const std::invalid_argument& err) {  // resolve_pin's three refusals; else taken
                const std::string w = err.what();
                if (w.find("cannot run") != std::string::npos || w.find("(strict): no ") != std::string::npos ||
                    w.find("is not a compiled") != std::string::npos)
                    continue;
            } catch (...) {  // the pin was taken; the launch failing is the family's business
            }
            return e.spelling;
        }
    }
    return kNoTableEntryRuns;
}

}  // namespace test_utils
