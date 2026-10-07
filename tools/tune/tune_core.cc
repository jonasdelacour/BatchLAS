#include "tune_core.hh"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace batchlas::tune {

std::vector<std::string> split(std::string_view s, char sep) {
    std::vector<std::string> out;
    std::size_t start = 0;
    for (std::size_t i = 0; i <= s.size(); ++i)
        if (i == s.size() || s[i] == sep) {
            if (i > start) out.emplace_back(s.substr(start, i - start));
            start = i + 1;
        }
    return out;
}

std::vector<std::string> split_fields(std::string_view s, char sep) {
    std::vector<std::string> out(1);
    for (char ch : s) {
        if (ch == sep) out.emplace_back();
        else out.back() += ch;
    }
    return out;
}

std::vector<std::string> parse_csv_line(std::string_view line) {
    if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
    std::vector<std::string> out(1);
    bool quoted = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];
        if (quoted && ch == '"' && i + 1 < line.size() && line[i + 1] == '"') {
            out.back() += '"';
            ++i;
        } else if (ch == '"') {
            quoted = !quoted;
        } else if (ch == ',' && !quoted) {
            out.emplace_back();
        } else {
            out.back() += ch;
        }
    }
    return out;
}

std::string csv_field(std::string_view s) {
    if (s.find_first_of(",\"\n\r") == std::string_view::npos) return std::string(s);
    std::string out = "\"";
    for (char ch : s) {
        if (ch == '\n' || ch == '\r') ch = ' ';
        if (ch == '"') out += '"';
        out += ch;
    }
    return out + "\"";
}

std::string key_text(const CellKey& k) {
    std::string out;
    for (const KV& f : k) out += (out.empty() ? "" : " ") + f.name + "=" + f.value;
    return out;
}

std::string key_arg(const CellKey& k) {
    std::string out;
    for (const KV& f : k) out += (out.empty() ? "" : ",") + f.name + "=" + f.value;
    return out;
}

std::optional<CellKey> parse_key_arg(std::string_view s) {
    CellKey k;
    for (const std::string& tok : split(s, ',')) {
        const auto eq = tok.find('=');
        if (eq == std::string::npos || eq == 0 || eq + 1 == tok.size()) return std::nullopt;
        k.push_back({tok.substr(0, eq), tok.substr(eq + 1)});
    }
    if (k.empty()) return std::nullopt;
    return k;
}

const std::string* key_get(const CellKey& k, std::string_view name) {
    for (const KV& f : k)
        if (f.name == name) return &f.value;
    return nullptr;
}

std::int64_t key_int(const CellKey& k, std::string_view name) {
    const std::string* v = key_get(k, name);
    std::int64_t out = 0;
    if (!v || std::from_chars(v->data(), v->data() + v->size(), out).ptr != v->data() + v->size())
        throw std::runtime_error("cell " + key_text(k) + " has no integer '" + std::string(name) + "'");
    return out;
}

CellKey key_with(CellKey k, std::string_view name, std::string value) {
    for (KV& f : k)
        if (f.name == name) f.value = std::move(value);
    return k;
}

double median(std::vector<double> v) {
    if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(v.begin(), v.end());
    const std::size_t h = v.size() / 2;
    return v.size() % 2 ? v[h] : 0.5 * (v[h - 1] + v[h]);
}

std::vector<std::size_t> rep_order(std::size_t k, int rep, bool reverse) {
    std::vector<std::size_t> out(k);
    for (std::size_t i = 0; i < k; ++i) {
        const std::size_t j = (i + static_cast<std::size_t>(rep)) % k;
        out[i] = reverse ? k - 1 - j : j;
    }
    return out;
}

std::vector<std::string> rank(const std::map<std::string, double>& times, const std::vector<std::string>& order,
                              double tie) {
    auto pos = [&](const std::string& c) {
        return static_cast<std::size_t>(std::find(order.begin(), order.end(), c) - order.begin());
    };
    double best = std::numeric_limits<double>::infinity();
    for (const auto& [c, t] : times) best = std::min(best, t);
    std::vector<std::string> tied, rest;
    for (const auto& [c, t] : times) (t <= best * (1.0 + tie) ? tied : rest).push_back(c);
    std::sort(tied.begin(), tied.end(), [&](const auto& a, const auto& b) { return pos(a) < pos(b); });
    std::sort(rest.begin(), rest.end(), [&](const auto& a, const auto& b) {
        return std::pair(times.at(a), pos(a)) < std::pair(times.at(b), pos(b));
    });
    tied.insert(tied.end(), rest.begin(), rest.end());
    return tied;
}

bool needs_remeasure(const std::map<std::string, std::vector<double>>& pass_medians, double limit) {
    for (const auto& [c, m] : pass_medians) {
        if (m.size() < 2) continue;
        const auto [lo, hi] = std::minmax_element(m.begin(), m.end());
        if (*lo > 0 && *hi / *lo - 1.0 > limit) return true;
    }
    return false;
}

std::vector<std::int64_t> refine_midpoints(std::vector<LinePoint> line, double ratio) {
    std::sort(line.begin(), line.end(), [](const LinePoint& a, const LinePoint& b) { return a.n < b.n; });
    std::vector<std::int64_t> out;
    for (std::size_t i = 0; i + 1 < line.size(); ++i) {
        const std::int64_t lo = line[i].n, hi = line[i + 1].n;
        if (line[i].winner == line[i + 1].winner || lo <= 0) continue;
        if (static_cast<double>(hi) / static_cast<double>(lo) < ratio || hi - lo < 2) continue;
        auto mid = static_cast<std::int64_t>(std::llround(std::sqrt(static_cast<double>(lo) * static_cast<double>(hi))));
        mid = std::clamp(mid, lo + 1, hi - 1);
        out.push_back(mid);
    }
    return out;
}

std::map<std::string, double> attempt_times(const Attempt& a, const std::vector<std::string>& cands) {
    std::map<std::string, double> out;
    if (a.empty()) return out;
    for (const auto& cand : cands) {
        double sum = 0;
        bool all = true;
        for (const PassData& p : a) {
            const auto it = p.arms.find(cand);
            all = all && it != p.arms.end() && it->second.status == "ok";
            if (!all) break;
            sum += it->second.median;
        }
        if (all) out[cand] = sum / double(a.size());
    }
    return out;
}

std::map<std::string, double> final_times(const std::vector<Attempt>& attempts, const std::vector<std::string>& cands,
                                          int* used) {
    for (std::size_t i = attempts.size(); i-- > 0;) {
        auto t = attempt_times(attempts[i], cands);
        if (t.empty()) continue;
        if (used) *used = int(i);
        return t;
    }
    if (used) *used = -1;
    return {};
}

bool attempt_needs_remeasure(const Attempt& a, const std::vector<std::string>& cands, double limit) {
    std::map<std::string, std::vector<double>> med;
    for (const auto& [cand, t] : attempt_times(a, cands))
        for (const PassData& p : a) med[cand].push_back(p.arms.at(cand).median);
    return needs_remeasure(med, limit);
}

RefineRound refine_round(const std::map<CellKey, std::map<std::string, double>>& measured, const std::string& refine_key,
                         const std::vector<std::string>& order, double tie, double ratio) {
    std::map<CellKey, std::vector<LinePoint>> lines;
    std::map<CellKey, CellKey> member;
    for (const auto& [k, t] : measured) {
        if (t.empty()) continue;
        const CellKey line = key_with(k, refine_key, "*");
        lines[line].push_back({key_int(k, refine_key), rank(t, order, tie).front()});
        member[line] = k;
    }
    RefineRound out;
    for (auto& [line, pts] : lines) {
        std::sort(pts.begin(), pts.end(), [](const LinePoint& a, const LinePoint& b) { return a.n < b.n; });
        for (std::int64_t mid : refine_midpoints(pts, ratio)) {
            CellKey k = key_with(member[line], refine_key, std::to_string(mid));
            if (!measured.count(k)) {
                out.next.push_back(std::move(k));
                continue;
            }
            const auto hi = std::find_if(pts.begin(), pts.end(), [&](const LinePoint& p) { return p.n > mid; });
            out.stalled.push_back(key_text(line) + ": edge between " + refine_key + "=" + std::to_string((hi - 1)->n) +
                                  " (" + (hi - 1)->winner + ") and " + std::to_string(hi->n) + " (" + hi->winner +
                                  ") stays wide: " + refine_key + "=" + std::to_string(mid) + " has no winner");
        }
    }
    return out;
}

std::optional<std::pair<std::string, std::string>> reached_route(std::string_view coverage, std::string_view op) {
    std::optional<std::pair<std::string, std::string>> found;
    std::size_t c_kind = 0, c_op = 0, c_origin = 0, c_algo = 0, width = 0;
    for (const std::string& line : split(coverage, '\n')) {
        const auto v = split_fields(line, ',');
        if (!v.empty() && v[0] == "kind") {
            auto col = [&](const char* n) { return std::size_t(std::find(v.begin(), v.end(), n) - v.begin()); };
            c_kind = col("kind"), c_op = col("op"), c_origin = col("chosen_origin"), c_algo = col("chosen_algo");
            width = std::max({c_kind, c_op, c_origin, c_algo}) < v.size() ? v.size() : 0;
            continue;
        }
        if (width && v.size() >= width && v[c_kind] == "reached" && v[c_op] == op) found = {{v[c_origin], v[c_algo]}};
    }
    return found;
}

// ---- JSON -----------------------------------------------------------------------------------

namespace {

std::string quote(std::string_view s) {
    std::string out = "\"";
    for (char ch : s) {
        if (ch == '"' || ch == '\\') {
            out += '\\';
            out += ch;
        } else if (static_cast<unsigned char>(ch) < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(static_cast<unsigned char>(ch)));
            out += buf;
        } else {
            out += ch;
        }
    }
    return out + "\"";
}

bool all_digits(std::string_view s) {
    return !s.empty() && s.size() < 19 && std::all_of(s.begin(), s.end(), [](char c) { return c >= '0' && c <= '9'; });
}

}  // namespace

void Json::sep(std::string_view k) {
    body_ += body_.empty() ? "{" : ", ";
    body_ += quote(k) + ": ";
}

Json& Json::str(std::string_view k, std::string_view v) {
    sep(k);
    body_ += quote(v);
    return *this;
}

Json& Json::num(std::string_view k, double v) {
    sep(k);
    if (!std::isfinite(v)) {
        body_ += "null";
        return *this;
    }
    char buf[64];
    const auto r = std::to_chars(buf, buf + sizeof(buf), v);
    body_.append(buf, r.ptr);
    return *this;
}

Json& Json::integer(std::string_view k, std::int64_t v) {
    sep(k);
    body_ += std::to_string(v);
    return *this;
}

Json& Json::boolean(std::string_view k, bool v) {
    sep(k);
    body_ += v ? "true" : "false";
    return *this;
}

Json& Json::key(const CellKey& key) {
    for (const KV& f : key) {
        if (all_digits(f.value)) integer(f.name, std::stoll(f.value));
        else str(f.name, f.value);
    }
    return *this;
}

std::string Json::line() const { return (body_.empty() ? std::string("{") : body_) + "}\n"; }

std::string Record::get(const std::string& k, const std::string& dflt) const {
    const auto it = s.find(k);
    return it == s.end() ? dflt : it->second;
}

double Record::number(const std::string& k) const {
    const auto it = s.find(k);
    double v = std::numeric_limits<double>::quiet_NaN();
    if (it == s.end() || it->second == "null") return v;
    const auto& t = it->second;
    if (std::from_chars(t.data(), t.data() + t.size(), v).ptr != t.data() + t.size())
        return std::numeric_limits<double>::quiet_NaN();
    return v;
}

std::optional<Record> parse_record(std::string_view line, std::string* err) {
    Record r;
    std::size_t i = 0;
    auto fail = [&](const char* why) -> std::optional<Record> {
        if (err) *err = std::string(why) + " at column " + std::to_string(i);
        return std::nullopt;
    };
    auto ws = [&] {
        while (i < line.size() && (line[i] == ' ' || line[i] == '\t' || line[i] == '\r' || line[i] == '\n')) ++i;
    };
    auto string_at = [&](std::string& out) {
        if (i >= line.size() || line[i] != '"') return false;
        for (++i; i < line.size(); ++i) {
            if (line[i] == '"') {
                ++i;
                return true;
            }
            if (line[i] != '\\') {
                out += line[i];
                continue;
            }
            if (++i >= line.size()) return false;
            const char e = line[i];
            if (e == 'u' && i + 4 < line.size()) {
                out += static_cast<char>(std::stoi(std::string(line.substr(i + 1, 4)), nullptr, 16));
                i += 4;
            } else {
                out += e == 'n' ? '\n' : e == 't' ? '\t' : e;
            }
        }
        return false;
    };
    ws();
    if (i >= line.size() || line[i++] != '{') return fail("expected '{'");
    ws();
    if (i < line.size() && line[i] == '}') return r;
    while (true) {
        ws();
        std::string k, v;
        if (!string_at(k)) return fail("expected a key string");
        ws();
        if (i >= line.size() || line[i++] != ':') return fail("expected ':'");
        ws();
        if (i < line.size() && line[i] == '"') {
            if (!string_at(v)) return fail("unterminated string");
        } else {
            const std::size_t start = i;
            while (i < line.size() && line[i] != ',' && line[i] != '}' && line[i] != ' ') ++i;
            v = std::string(line.substr(start, i - start));
            if (v.empty() || v[0] == '{' || v[0] == '[') return fail("nested values are not supported");
        }
        r.s[k] = v;
        ws();
        if (i >= line.size()) return fail("unterminated object");
        if (line[i] == ',') {
            ++i;
            continue;
        }
        if (line[i] == '}') return r;
        return fail("expected ',' or '}'");
    }
}

std::vector<Record> read_records(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::vector<Record> out;
    std::string line, err;
    for (int no = 1; std::getline(f, line); ++no) {
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        auto r = parse_record(line, &err);
        if (!r) throw std::runtime_error(path + ":" + std::to_string(no) + ": " + err);
        out.push_back(std::move(*r));
    }
    return out;
}

// ---- SHA-256 (FIPS 180-4) -------------------------------------------------------------------

namespace {

constexpr std::array<std::uint32_t, 64> K256{
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

constexpr std::uint32_t rotr(std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

}  // namespace

std::string sha256_hex(std::string_view data) {
    std::array<std::uint32_t, 8> h{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::string msg(data);
    const std::uint64_t bits = static_cast<std::uint64_t>(data.size()) * 8;
    msg += static_cast<char>(0x80);
    while (msg.size() % 64 != 56) msg += '\0';
    for (int s = 56; s >= 0; s -= 8) msg += static_cast<char>((bits >> s) & 0xff);
    std::array<std::uint32_t, 64> w{};
    for (std::size_t off = 0; off < msg.size(); off += 64) {
        for (int t = 0; t < 16; ++t) {
            w[t] = 0;
            for (int b = 0; b < 4; ++b)
                w[t] = (w[t] << 8) | static_cast<unsigned char>(msg[off + static_cast<std::size_t>(4 * t + b)]);
        }
        for (int t = 16; t < 64; ++t) {
            const std::uint32_t s0 = rotr(w[t - 15], 7) ^ rotr(w[t - 15], 18) ^ (w[t - 15] >> 3);
            const std::uint32_t s1 = rotr(w[t - 2], 17) ^ rotr(w[t - 2], 19) ^ (w[t - 2] >> 10);
            w[t] = w[t - 16] + s0 + w[t - 7] + s1;
        }
        auto v = h;
        for (int t = 0; t < 64; ++t) {
            const std::uint32_t S1 = rotr(v[4], 6) ^ rotr(v[4], 11) ^ rotr(v[4], 25);
            const std::uint32_t ch = (v[4] & v[5]) ^ (~v[4] & v[6]);
            const std::uint32_t t1 = v[7] + S1 + ch + K256[t] + w[t];
            const std::uint32_t S0 = rotr(v[0], 2) ^ rotr(v[0], 13) ^ rotr(v[0], 22);
            const std::uint32_t maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            for (int j = 7; j > 0; --j) v[j] = v[j - 1];
            v[4] += t1;
            v[0] = t1 + S0 + maj;
        }
        for (int j = 0; j < 8; ++j) h[j] += v[j];
    }
    char out[65];
    for (int j = 0; j < 8; ++j) std::snprintf(out + 8 * j, 9, "%08x", h[j]);
    return std::string(out, 64);
}

namespace {
std::vector<std::string> quoted_paths(const std::string& line) {
    std::vector<std::string> out;
    for (std::size_t a = line.find('"'); a != std::string::npos; a = line.find('"', a + 1)) {
        const std::size_t b = line.find('"', a + 1);
        if (b == std::string::npos) break;
        out.push_back(line.substr(a + 1, b - a - 1));
        a = b;
    }
    return out;
}

// "// family: name" -> name; empty when the line is not a family marker.
std::string family_marker(const std::string& line) {
    std::size_t c = line.find_first_not_of(" \t");
    if (c == std::string::npos || line.compare(c, 2, "//") != 0) return {};
    c = line.find_first_not_of(" \t", c + 2);
    if (c == std::string::npos || line.compare(c, 7, "family:") != 0) return {};
    std::size_t a = c + 7;
    while (a < line.size() && line[a] == ' ') ++a;
    std::size_t e = a;
    while (e < line.size() && line[e] != ' ' && line[e] != '"') ++e;
    return line.substr(a, e - a);
}

// A line that is exactly "// common" (trailing words or a path on the line do not count).
bool is_common_marker(const std::string& line) {
    const std::size_t a = line.find_first_not_of(" \t");
    const std::size_t e = line.find_last_not_of(" \t\r");
    return a != std::string::npos && line.substr(a, e - a + 1) == "// common";
}
}  // namespace

KernelBlock parse_kernel_block(std::string_view spec_source) {
    KernelBlock out;
    enum class In { none, sources, deps } in_block = In::none;
    std::string section;  // empty = common
    std::istringstream in{std::string(spec_source)};
    for (std::string line; std::getline(in, line);) {
        if (line.find("kernel-sources-begin") != std::string::npos) {
            in_block = In::sources;
            section.clear();
            continue;
        }
        if (line.find("kernel-sources-end") != std::string::npos) {
            in_block = In::none;
            continue;
        }
        if (line.find("kernel-deps-begin") != std::string::npos) {
            in_block = In::deps;
            continue;
        }
        if (line.find("kernel-deps-end") != std::string::npos) break;
        if (in_block == In::none) continue;
        const std::string fam = family_marker(line);
        if (in_block == In::deps) {
            if (fam.empty()) continue;
            auto& v = out.deps[fam];
            for (std::string& p : quoted_paths(line)) v.push_back(std::move(p));
            continue;
        }
        if (!fam.empty())
            section = fam;
        else if (is_common_marker(line))
            section.clear();
        for (std::string& p : quoted_paths(line)) {
            out.all.push_back(p);
            (section.empty() ? out.common : out.family[section]).push_back(std::move(p));
        }
    }
    return out;
}

KernelBlock kernel_block_from_file(const std::string& repo, const std::string& spec_file) {
    const std::string path = repo + "/" + spec_file;
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return parse_kernel_block(ss.str());
}

std::vector<std::string> parse_kernel_list(std::string_view spec_source) {
    return parse_kernel_block(spec_source).all;
}

std::map<std::string, std::string> family_hashes(const std::string& repo, const KernelBlock& b,
                                                 const std::vector<std::string>& families) {
    std::map<std::string, std::string> out;
    for (const std::string& f : families) {
        std::vector<std::string> paths = b.common;
        for (const auto* m : {&b.family, &b.deps})
            if (const auto it = m->find(f); it != m->end()) paths.insert(paths.end(), it->second.begin(), it->second.end());
        std::string missing;
        const auto h = kernel_hash(repo, paths, &missing);
        if (!h) throw std::runtime_error("kernel source missing for family '" + f + "': " + missing);
        out[f] = *h;
    }
    return out;
}

std::optional<std::string> kernel_hash(const std::string& repo, const std::vector<std::string>& paths,
                                       std::string* missing) {
    std::string manifest;
    for (const std::string& p : paths) {
        std::ifstream f(repo + "/" + p, std::ios::binary);
        if (!f) {
            if (missing) *missing = p;
            return std::nullopt;
        }
        std::ostringstream ss;
        ss << f.rdbuf();
        manifest += sha256_hex(ss.str()) + "  " + p + "\n";
    }
    return sha256_hex(manifest).substr(0, 8);
}

std::string gate_verdict(double ratio_p1, double ratio_p2, bool bad_row, double limit) {
    if (ratio_p1 > limit && ratio_p2 > limit) return "FAIL";
    return bad_row ? "BAD_ROW" : "pass";
}

std::optional<std::vector<OldChoice>> parse_old_csv(std::string_view text, const std::vector<std::string>& key_names,
                                                    const std::vector<std::string>& dtypes, std::string* err) {
    auto fail = [&](const std::string& why) -> std::optional<std::vector<OldChoice>> {
        if (err) *err = why;
        return std::nullopt;
    };
    const auto lines = split(text, '\n');
    if (lines.empty()) return fail("empty file");
    const auto head = parse_csv_line(lines[0]);
    std::vector<std::size_t> cols;
    for (const std::string& n : [&] {
             std::vector<std::string> want{"dtype", "old"};
             want.insert(want.end(), key_names.begin(), key_names.end());
             return want;
         }()) {
        const auto it = std::find(head.begin(), head.end(), n);
        if (it == head.end()) return fail("no column '" + n + "'");
        cols.push_back(std::size_t(it - head.begin()));
    }
    std::vector<OldChoice> out;
    for (std::size_t i = 1; i < lines.size(); ++i) {
        if (lines[i].find_first_not_of(" \t\r") == std::string::npos) continue;
        const auto v = parse_csv_line(lines[i]);
        const std::string where = "row " + std::to_string(i + 1);
        if (v.size() != head.size())
            return fail(where + " has " + std::to_string(v.size()) + " fields, the header " + std::to_string(head.size()));
        OldChoice c{v[cols[0]], {}, v[cols[1]]};
        for (std::size_t k = 0; k < key_names.size(); ++k) c.key.push_back({key_names[k], v[cols[2 + k]]});
        if (c.old.empty()) return fail(where + " has an empty 'old'");
        if (!dtypes.empty() && std::find(dtypes.begin(), dtypes.end(), c.dtype) == dtypes.end())
            return fail(where + " is dtype " + c.dtype + ", not in --dtype");
        out.push_back(std::move(c));
    }
    if (out.empty()) return fail("no data rows");
    return out;
}

std::pair<std::string, std::string> split_origin(const std::string& spelling) {
    for (const char* o : {"native:", "vendor:"})
        if (spelling.rfind(o, 0) == 0) return {std::string(o, std::strlen(o) - 1), spelling.substr(std::strlen(o))};
    return {"native", spelling};
}

void gate_count(GateCounts& c, const std::string& v) {
    (v == "same" ? c.same : v == "pass" ? c.pass : v == "FAIL" ? c.fail : v == "BAD_ROW" ? c.bad : v == "cap" ? c.cap
                                                                                                      : c.error)++;
}

int gate_exit_code(const GateCounts& c) {
    if (c.fail) return 1;
    if (c.error || c.bad || c.same + c.pass == 0) return 3;
    return 0;
}

std::string gate_summary(const GateCounts& c) {
    return std::to_string(c.same) + " same, " + std::to_string(c.pass) + " timed pass, " + std::to_string(c.fail) +
           " FAIL, " + std::to_string(c.bad) + " BAD_ROW, " + std::to_string(c.error) + " ERROR, " +
           std::to_string(c.cap) + " over the cap";
}

std::optional<std::string> devices_problem(const std::vector<int>& devices, bool given,
                                           const std::optional<std::string>& parent_visible) {
    if (!given && !parent_visible) return "--devices is required";
    if (!given) return "CUDA_VISIBLE_DEVICES=" + *parent_visible + " is set: give --devices (nvidia-smi indices inside it)";
    if (devices.empty()) return "--devices is empty";
    if (!parent_visible) return std::nullopt;
    const std::string& cvd = *parent_visible;
    std::vector<int> allowed;
    for (const std::string& f : split_fields(cvd, ',')) {
        int v = -1;
        if (f.empty() || std::from_chars(f.data(), f.data() + f.size(), v).ptr != f.data() + f.size())
            return "CUDA_VISIBLE_DEVICES=" + cvd + " is not a list of GPU indices; unset it and use --devices";
        allowed.push_back(v);
    }
    for (int d : devices)
        if (std::find(allowed.begin(), allowed.end(), d) == allowed.end())
            return "--devices " + std::to_string(d) + " is outside CUDA_VISIBLE_DEVICES=" + cvd;
    return std::nullopt;
}

AppScan scan_compute_apps(std::string_view out, long self_pid) {
    AppScan s;
    for (std::string line : split(out, '\n')) {
        line.erase(0, line.find_first_not_of(" \t\r"));
        line.erase(line.find_last_not_of(" \t\r") + 1);
        if (line.empty()) continue;
        long pid = -1;
        const bool num = std::from_chars(line.data(), line.data() + line.size(), pid).ptr == line.data() + line.size();
        if (num && pid == self_pid) s.self = true;
        else s.foreign.push_back(line);
    }
    return s;
}

namespace {
bool is_pid(const std::string& s) {
    return !s.empty() && s.find_first_not_of("0123456789") == std::string::npos;
}
std::string bracket(const std::vector<std::string>& v) {
    std::string out;
    for (const auto& s : v) out += (out.empty() ? "" : ",") + s;
    return "[" + out + "]";
}
}  // namespace

GuardCheck guard_before(const AppScan& scan, double util, double ceiling, bool allow_idle_foreign) {
    GuardCheck g;
    const bool all_pids = std::all_of(scan.foreign.begin(), scan.foreign.end(), is_pid);
    if (!scan.foreign.empty() && (!allow_idle_foreign || !all_pids)) {
        g.refuse = "compute processes " + bracket(scan.foreign);
        return g;
    }
    char u[32];
    std::snprintf(u, sizeof(u), "%g", util);
    if (util > ceiling) {
        g.refuse = "utilization " + std::string(u) + "%";
        if (!scan.foreign.empty()) g.refuse += " with compute processes " + bracket(scan.foreign);
        return g;
    }
    g.tolerated = scan.foreign;
    return g;
}

std::vector<std::string> guard_new_foreign(const AppScan& after, const std::vector<std::string>& tolerated) {
    std::vector<std::string> fresh;
    for (const auto& f : after.foreign)
        if (!is_pid(f) || std::find(tolerated.begin(), tolerated.end(), f) == tolerated.end()) fresh.push_back(f);
    return fresh;
}

}  // namespace batchlas::tune
