// SPDX-License-Identifier: MIT
#pragma once

// Auto's expected pick for one shape, read from the shipped tables, so a retune changes the
// expectation and not the test. Not via select::choose: the nearest row of each table in borrow
// order, first entry a strict ScopedPin accepts while `probe` runs the op. Keys are spelled by
// the test; AutoReadsEveryKeyField (synthetic tables) proves key_of's wiring independently.
// evidence: docs/design/flat-kernel-selection.md#flat-selection-tables-and-provenance

#include <src/select/select.hh>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace test_utils {

inline constexpr const char* kNoTableEntryRuns = "<no table entry runs: last resort>";

// Tests spell a synthetic table for one dtype ("<op>.<dtype>.<device>.txt", keys without dtype);
// set_builtin_tables wants tuned/<op>.<device>.txt files keyed by dtype first
// (docs/design/tuning-data.md). Other names pass through unchanged.
inline std::vector<std::pair<std::string, std::string>> dtype_keyed(std::vector<std::pair<std::string, std::string>> files) {
    for (auto& [name, text] : files) {
        std::vector<std::string> part;
        for (std::size_t a = 0, e; a <= name.size(); a = e + 1) {
            e = std::min(name.find('.', a), name.size());
            part.push_back(name.substr(a, e - a));
        }
        if (part.size() != 4 || part[3] != "txt") continue;
        const std::string dt = part[1];
        std::string out;
        for (std::size_t a = 0, e; a < text.size(); a = e + 1) {
            e = std::min(text.find('\n', a), text.size());
            std::string line = text.substr(a, e - a);
            if (line.rfind("# keys:", 0) == 0) line = "# keys: dtype:exact" + line.substr(7);
            else if (!line.empty() && line[0] == '#') {
                if (const auto p = line.find(" dtype=" + dt); p != std::string::npos) line.erase(p, 7 + dt.size());
            } else if (!line.empty()) line = "dtype=" + dt + " " + line;
            out += line + "\n";
        }
        name = part[0] + "." + part[2] + ".txt";
        text = out;
    }
    return files;
}

// `native_only`: what the class word "native" resolves to (the vendor family is skipped).
template <typename Choice, typename Probe>
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
