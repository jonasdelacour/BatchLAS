#include "race.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

#include "tune_core.hh"

namespace batchlas::tune {

std::size_t lower_order_stat(int n, double confidence) {
    if (n < 1) return 0;
    const double alpha = 1.0 - confidence + 1e-12;
    double c = std::pow(0.5, n), cdf = 0;  // c = C(n, i) / 2^n
    std::size_t k = 0;
    for (int i = 0; i < n; ++i) {
        cdf += c;  // P(X <= i) = P(X < i + 1)
        if (cdf > alpha) break;
        k = static_cast<std::size_t>(i) + 1;
        c = c * (n - i) / (i + 1);
    }
    return k;
}

namespace {

double run_median(const std::vector<double>& v) {
    std::vector<double> ok;
    for (double x : v)
        if (!std::isnan(x)) ok.push_back(x);
    return median(std::move(ok));
}

}  // namespace

RaceVerdict race_step(RaceState& s, const TierParams& p, double tie) {
    const std::size_t rounds = s.ms.empty() ? 0 : s.ms[0].size();
    int leader = -1;
    double best = std::numeric_limits<double>::infinity();
    for (std::size_t c = 0; c < s.cands.size(); ++c) {
        if (!s.alive[c]) continue;
        const double m = run_median(s.ms[c]);
        if (!std::isnan(m) && m < best) best = m, leader = static_cast<int>(c);
    }
    bool all_tied = leader >= 0;
    if (leader >= 0) {
        std::vector<std::size_t> dying;
        for (std::size_t c = 0; c < s.cands.size(); ++c) {
            if (!s.alive[c] || static_cast<int>(c) == leader) continue;
            std::vector<double> q;
            for (std::size_t r = 0; r < s.ms[c].size() && r < s.ms[leader].size(); ++r) {
                const double a = s.ms[c][r], b = s.ms[leader][r];
                if (!std::isnan(a) && !std::isnan(b) && b > 0) q.push_back(a / b);
            }
            const int n = static_cast<int>(q.size());
            const std::size_t k = lower_order_stat(n, p.confidence);
            std::sort(q.begin(), q.end());
            if (n >= 1 && median(q) > kGrossLoserRatio) {
                dying.push_back(c);  // evidence: docs/design/tiered-tuning.md#engine-readiness-for-a-deep-run-over-all-19-ops
            } else if (n >= p.min_reps && k >= 1) {
                if (q[k - 1] > 1.0 + tie) dying.push_back(c);
                else if (!(q[n - k] < 1.0 + tie)) all_tied = false;
            } else {
                all_tied = false;
            }
        }
        for (std::size_t c : dying) s.alive[c] = false;
    }
    if (std::count(s.alive.begin(), s.alive.end(), true) <= 1) return RaceVerdict::winner;
    if (all_tied) return RaceVerdict::tie;
    return static_cast<int>(rounds) >= p.max_reps ? RaceVerdict::cap : RaceVerdict::more;
}

bool race_over(RaceVerdict v, int rounds, const TierParams& p) {
    return v != RaceVerdict::more && !(v == RaceVerdict::winner && rounds < p.min_reps);
}

std::vector<std::string> race_ranking(const RaceState& s, const std::vector<std::string>& order, double tie) {
    std::map<std::string, double> alive;
    std::vector<std::pair<double, std::size_t>> dead;
    auto pos = [&](const std::string& c) {
        return static_cast<std::size_t>(std::find(order.begin(), order.end(), c) - order.begin());
    };
    const double inf = std::numeric_limits<double>::infinity();
    for (std::size_t c = 0; c < s.cands.size(); ++c) {
        double m = run_median(s.ms[c]);
        if (std::isnan(m)) m = inf;
        if (s.alive[c]) alive[s.cands[c]] = m;
        else dead.emplace_back(m, c);
    }
    auto out = rank(alive, order, tie);
    std::sort(dead.begin(), dead.end(), [&](const auto& a, const auto& b) {
        return std::pair(a.first, pos(s.cands[a.second])) < std::pair(b.first, pos(s.cands[b.second]));
    });
    for (const auto& d : dead) out.push_back(s.cands[d.second]);
    return out;
}

}  // namespace batchlas::tune
