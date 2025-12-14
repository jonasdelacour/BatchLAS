#pragma once

#include <sycl/sycl.hpp>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace batchlas_kernel_trace {

struct Record {
    sycl::event event;
    std::string name;
    std::string device_name;
    std::uint64_t submit_id = 0;
    std::uint32_t tid = 0;
};

inline std::atomic<bool> g_initialized{false};
inline std::atomic<bool> g_enabled{false};
inline std::mutex g_mu;
inline std::vector<Record> g_records;
inline std::atomic<std::uint64_t> g_submit_counter{0};

inline thread_local const char* tl_scope_name = nullptr;

inline bool _env_truthy(const char* v) {
    if (!v) return false;
    return (std::string(v) == "1" || std::string(v) == "true" || std::string(v) == "TRUE" ||
            std::string(v) == "on" || std::string(v) == "ON");
}

inline const char* trace_path() {
    if (const char* p = std::getenv("BATCHLAS_KERNEL_TRACE_PATH")) {
        if (*p) return p;
    }
    if (const char* p = std::getenv("BATCHLAS_TRACE_PATH")) {
        if (*p) return p;
    }
    return "batchlas_kernels.trace.json";
}

inline void flush();

inline void _init_once() {
    bool expected = false;
    if (!g_initialized.compare_exchange_strong(expected, true)) return;

    const bool enabled = _env_truthy(std::getenv("BATCHLAS_KERNEL_TRACE")) ||
                         _env_truthy(std::getenv("BATCHLAS_TRACE_KERNELS"));
    g_enabled.store(enabled);

    if (enabled) {
        std::atexit([]() {
            // Best-effort flush at process exit.
            try {
                batchlas_kernel_trace::flush();
            } catch (...) {
                // Swallow all exceptions during atexit.
            }
        });
    }
}

inline bool enabled() {
    _init_once();
    return g_enabled.load();
}

inline const char* current_scope_name() {
    return tl_scope_name;
}

struct Scope {
    const char* prev_ = nullptr;
    explicit Scope(const char* name) {
        prev_ = tl_scope_name;
        tl_scope_name = name;
    }
    ~Scope() { tl_scope_name = prev_; }
};

inline std::string _safe_device_name(const sycl::queue& q) {
    try {
        return q.get_device().get_info<sycl::info::device::name>();
    } catch (...) {
        return "unknown";
    }
}

inline void record_event(const sycl::queue& q, const sycl::event& e, std::string name, std::uint32_t tid) {
    if (!enabled()) return;

    Record r;
    r.event = e;
    r.name = std::move(name);
    r.device_name = _safe_device_name(q);
    r.submit_id = ++g_submit_counter;
    r.tid = tid;

    std::lock_guard<std::mutex> lock(g_mu);
    g_records.push_back(std::move(r));
}

inline void flush() {
    if (!enabled()) return;

    std::vector<Record> records;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        if (g_records.empty()) return;
        records.swap(g_records);
    }

    struct Ev {
        std::string name;
        std::string device;
        std::uint64_t start_ns;
        std::uint64_t end_ns;
        std::uint64_t submit_id;
        std::uint32_t tid;
    };

    std::vector<Ev> evs;
    evs.reserve(records.size());

    std::optional<std::uint64_t> min_start;

    for (auto& r : records) {
        // Ensure completion so profiling info is available.
        try {
            r.event.wait();
        } catch (...) {
            continue;
        }

        try {
            auto start = r.event.get_profiling_info<sycl::info::event_profiling::command_start>();
            auto end = r.event.get_profiling_info<sycl::info::event_profiling::command_end>();
            if (end <= start) continue;

            if (!min_start || start < *min_start) min_start = start;

            evs.push_back(Ev{
                r.name,
                r.device_name,
                static_cast<std::uint64_t>(start),
                static_cast<std::uint64_t>(end),
                r.submit_id,
                r.tid,
            });
        } catch (...) {
            // Some backends/commands may not expose profiling info.
            continue;
        }
    }

    if (evs.empty() || !min_start) return;

    const auto base = *min_start;

    std::ofstream f(trace_path());
    if (!f.is_open()) return;

    // Minimal Chrome trace writer (no external deps)
    f << "{\n";
    f << "  \"displayTimeUnit\": \"ms\",\n";
    f << "  \"traceEvents\": [\n";

    for (size_t i = 0; i < evs.size(); ++i) {
        const auto& e = evs[i];
        const double ts_us = double(e.start_ns - base) / 1000.0;
        const double dur_us = double(e.end_ns - e.start_ns) / 1000.0;

        auto json_escape = [](const std::string& s) {
            std::string out;
            out.reserve(s.size());
            for (char c : s) {
                switch (c) {
                    case '\\': out += "\\\\"; break;
                    case '"': out += "\\\""; break;
                    case '\n': out += "\\n"; break;
                    case '\r': out += "\\r"; break;
                    case '\t': out += "\\t"; break;
                    default: out += c; break;
                }
            }
            return out;
        };

        f << "    {";
        f << "\"name\":\"" << json_escape(e.name) << "\",";
        f << "\"cat\":\"sycl\",";
        f << "\"ph\":\"X\",";
        f << "\"ts\":" << ts_us << ',';
        f << "\"dur\":" << dur_us << ',';
        f << "\"pid\":1,";
        f << "\"tid\":" << e.tid << ',';
        f << "\"args\":{";
        f << "\"device\":\"" << json_escape(e.device) << "\",";
        f << "\"submit_id\":" << e.submit_id;
        f << "}}";
        f << (i + 1 < evs.size() ? "," : "") << "\n";
    }

    f << "  ]\n";
    f << "}\n";
}

} // namespace batchlas_kernel_trace

// Convenience macro: annotate a region; all queue submissions within may inherit the name.
// Note: this is best-effort labeling; submissions without a scope will be labeled "sycl_submit".
#define BATCHLAS_KERNEL_TRACE_SCOPE(name_literal) \
    ::batchlas_kernel_trace::Scope BATCHLAS_KERNEL_TRACE_SCOPE_##__COUNTER__ { name_literal }
