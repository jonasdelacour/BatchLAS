// SPDX-License-Identifier: MIT
// Every public entry point that takes a view and does not honour a batch mask refuses one before it
// launches or writes (W0): vendor, per-family, sub-op, util and verification entries, and the
// routed ops. Each runs on a small problem with a device-only mask {0,1,0,1} (a host read faults on
// a GPU) and must throw batchlas::unsupported (NoRouteError is one) with every operand's bytes
// unchanged. A direct entry must say "batch mask"; a routed op refuses through can_run, so its
// unmasked twin must run first, or the throw could be a missing route.
#include <gtest/gtest.h>

#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/extra.hh>
#include <batchlas/blas/functions.hh>
#include <batchlas/blas/linalg-ops.hh>
#include <batchlas/error.hh>
#include <batchlas/internal/ormqr_blocked.hh>
#include <batchlas/util/sycl-vector.hh>

#include <src/ops/ormqr/wy_spelling.hh>
#include <src/select/vendor.hh>
#include <tests/masked_contract.hh>
#include <tests/test_utils.hh>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <span>
#include <string>
#include <tuple>
#include <vector>

using namespace batchlas;

namespace {

template <typename T>
using MV = MatrixView<T, MatrixFormat::Dense>;
template <typename T>
using Real = typename base_type<T>::type;

constexpr int kN = 8, kLd = 10, kStride = kLd * kN + 3, kBatch = 4, kKd = 2;
constexpr int kVecStride = kN + 2;
const std::vector<std::uint8_t> kAlternating{0, 1, 0, 1};

// Every operand of every entry, primed with well-posed data: Hermitian, diagonally dominant items,
// a diagonal CSR pattern, a positive tridiagonal (d, e). With `only` < 0 every view carries `mask`;
// otherwise only view number `only` does (see masks()).
template <typename T>
struct Pool {
    Queue& q;
    Span<const std::uint8_t> mask;
    Span<std::byte> ws;
    int only = -1, views = 0;
    std::vector<std::pair<int, std::string>>* used = nullptr;
    UnifiedVector<T> a, b, c, z, v, tq, tp, tau, vals;
    UnifiedVector<Real<T>> d, e, w, s;
    UnifiedVector<T*> pa, pb, pc;
    UnifiedVector<std::int64_t> piv;
    UnifiedVector<std::int32_t> piv32, info, m, counts;
    UnifiedVector<int> ro, ci, full, first_empty, second_k0;

    static std::size_t mat_len() { return std::size_t(kStride) * kBatch; }
    static std::size_t vec_len() { return std::size_t(kVecStride) * kBatch; }
    Pool(Queue& queue, Span<const std::uint8_t> m_, Span<std::byte> work, int only_ = -1)
        : q(queue), mask(m_), ws(work), only(only_), a(mat_len()), b(mat_len()), c(mat_len()), z(mat_len()),
          v(mat_len()), tq(vec_len()), tp(vec_len()), tau(vec_len()), vals(vec_len()), d(vec_len()), e(vec_len()),
          w(vec_len()), s(vec_len()), pa(kBatch), pb(kBatch), pc(kBatch), piv(vec_len()), piv32(vec_len()),
          info(kBatch), m(kBatch), counts(kBatch), ro(std::size_t(kN + 1) * kBatch), ci(vec_len()), full(kBatch, kN),
          first_empty(kBatch, kN), second_k0(kBatch, kN) {
        for (auto* buf : {&a, &b, &c, &z, &v})
            for (std::size_t k = 0; k < buf->size(); ++k) {
                const std::size_t it = k / kStride, j = (k % kStride) / kLd, i = (k % kStride) % kLd;
                const Real<T> re = i == j ? Real<T>(2 * kN + it) : Real<T>(0.25) * Real<T>(i + j + 1);
                (*buf)[k] = T(re);
            }
        for (std::size_t k = 0; k < vec_len(); ++k) {
            const auto i = Real<T>(k % kVecStride);
            d[k] = Real<T>(4) + i;
            e[k] = Real<T>(0.5);
            w[k] = i;
            s[k] = Real<T>(kN) - i;
            tq[k] = tp[k] = tau[k] = T(0.5);
            vals[k] = T(Real<T>(3));
            piv[k] = std::int64_t(k % kVecStride % kN) + 1;
            piv32[k] = std::int32_t(k % kVecStride % kN) + 1;
        }
        for (int i = 0; i < kBatch; ++i) info[i] = m[i] = counts[i] = -7;
        for (std::size_t k = 0; k < ro.size(); ++k) ro[k] = int(k % (kN + 1));
        for (std::size_t k = 0; k < vec_len(); ++k) ci[k] = int(k % kVecStride % kN);  // per-item copies
        first_empty[0] = 0;  // item 0 (inactive) has m == 0, item 1 (active) k == 0
        second_k0[1] = 0;
    }
    bool masked() const { return !mask.empty(); }
    // Every byte an entry could write, named; the workspace is scratch and excluded.
    std::vector<std::tuple<const char*, const std::byte*, std::size_t>> buffers() const {
        std::vector<std::tuple<const char*, const std::byte*, std::size_t>> out;
        const auto add = [&](const char* name, const auto& buf) {
            out.emplace_back(name, reinterpret_cast<const std::byte*>(buf.data()), buf.size() * sizeof(*buf.data()));
        };
        add("A", a), add("B", b), add("C", c), add("Z", z), add("V", v), add("tauq", tq), add("taup", tp);
        add("tau", tau), add("csr", vals), add("d", d), add("e", e), add("w", w), add("s", s), add("piv", piv);
        add("piv32", piv32), add("info", info), add("m", m), add("counts", counts);
        return out;
    }
    // Keyed on the view, not the buffer: the k-th view the call builds, so two views of one buffer
    // are masked one at a time. `used` gets each view's ordinal and buffer name.
    bool masks(const void* data) {
        const int ordinal = views++;
        if (used) {
            std::string name = "?";
            for (const auto& [n, ptr, bytes] : buffers())
                if (ptr == data) name = n;
            used->emplace_back(ordinal, name + " (view " + std::to_string(ordinal) + ")");
        }
        return masked() && (only < 0 || only == ordinal);
    }
    MV<T> mat(UnifiedVector<T>& buf, int rows = kN, int cols = kN) {
        T** ptrs = &buf == &a ? pa.data() : &buf == &b ? pb.data() : &buf == &c ? pc.data() : nullptr;
        const MV<T> view(buf.data(), rows, cols, kLd, kStride, kBatch, ptrs);
        return masks(buf.data()) ? view.with_batch_mask(mask) : view;
    }
    template <typename U>
    VectorView<U> vec(UnifiedVector<U>& buf, int size = kN) {
        const VectorView<U> view(buf.data(), size, kBatch, 1, kVecStride);
        return masks(buf.data()) ? view.with_batch_mask(mask) : view;
    }
    MatrixView<T, MatrixFormat::CSR> csr() {
        const MatrixView<T, MatrixFormat::CSR> view(vals.data(), ro.data(), ci.data(), kN, kN, NonZeros{kN}, kVecStride,
                                                    kN + 1, kBatch);
        return masks(vals.data()) ? view.with_batch_mask(mask) : view;
    }
    template <typename U>
    static Span<U> flat(UnifiedVector<U>& buf) { return {buf.data(), buf.size()}; }
    static Span<int> ints(UnifiedVector<int>& buf) { return {buf.data(), buf.size()}; }
    Span<Real<T>> wspan() { return flat(w); }
    Span<T> tspan() { return flat(tau); }
    Span<std::int64_t> pspan() { return flat(piv); }
    Span<std::int32_t> ispan() { return flat(info); }

    std::vector<std::vector<std::byte>> snapshot() const {
        std::vector<std::vector<std::byte>> out;
        for (const auto& [name, data, bytes] : buffers()) out.emplace_back(data, data + bytes);
        return out;
    }
};

template <typename T>
struct Entry {
    std::string name;
    bool routed = false;  ///< refuses through can_run (NoRouteError text varies): an unmasked twin must run
    std::function<void(Pool<T>&)> call;
    bool pinned = false;  ///< runs under a strict pin, so "cannot run this shape" is can_run's refusal
};

// One masked run (`only` as for Pool): a refusal, and every operand byte unchanged.
template <typename T>
void expect_refused(Queue& q, const Entry<T>& entry, Span<const std::uint8_t> mask, Span<std::byte> ws, int only,
                    const std::string& view, std::vector<std::pair<int, std::string>>* used) {
    Pool<T> p(q, mask, ws, only);
    p.used = used;
    SCOPED_TRACE(only < 0 ? std::string("every operand masked") : "only " + view + " masked");
    const auto before = p.snapshot();
    std::string what;
    bool refused = false;
    try {
        entry.call(p);
        q.wait();
        what = "ran a masked call";
    } catch (const batchlas::unsupported& e) {
        refused = true;
        what = e.what();
    } catch (const batchlas::invalid_argument& e) {
        refused = entry.pinned && std::string(e.what()).find("cannot run this shape") != std::string::npos;
        what = std::string(refused ? "" : "threw something other than a refusal: ") + e.what();
    } catch (const std::exception& e) {
        what = std::string("threw something other than a refusal: ") + e.what();
    }
    EXPECT_TRUE(refused) << what;
    if (refused && !entry.routed) EXPECT_NE(what.find("batch mask"), std::string::npos) << what;
    const auto after = p.snapshot();
    for (std::size_t i = 0; i < before.size(); ++i)
        EXPECT_TRUE(after[i] == before[i]) << "wrote operand " << std::get<0>(p.buffers()[i]);
}

// Every operand masked, then each operand view alone: an entry that refuses on some operands
// only ignores the mask on the others. False when a routed op has no route in this build at all
// (`unroutable` allows only that): nothing a mask could be ignored by, nothing to check.
template <typename T>
bool check_refuses(Queue& q, const Entry<T>& entry, Span<const std::uint8_t> mask, Span<std::byte> ws,
                   bool unroutable = false) {
    SCOPED_TRACE(entry.name);
    if (entry.routed) {
        Pool<T> plain(q, {}, ws);
        try {
            entry.call(plain);
            q.wait();
        } catch (const NoRouteError& e) {
            if (unroutable) return false;
            ADD_FAILURE() << "the unmasked call threw, so a masked throw proves nothing: " << e.what();
            return false;
        } catch (const std::exception& e) {
            ADD_FAILURE() << "the unmasked call threw, so a masked throw proves nothing: " << e.what();
            return false;
        }
    }
    std::vector<std::pair<int, std::string>> used;  // the views the call builds, recorded by the every-operand run
    expect_refused(q, entry, mask, ws, -1, "", &used);
    EXPECT_FALSE(used.empty()) << "the entry views no operand";
    for (const auto& [ordinal, view] : used) expect_refused(q, entry, mask, ws, ordinal, view, nullptr);
    return true;
}

// gemm, whose native families honour the mask (W1a): plain, and heterogeneous with a k == 0 item.
template <typename T, Backend B>
std::vector<Entry<T>> gemm_entries() {
    using P = Pool<T>;
    const T one(1), half(0.5);
    const auto N = Transpose::NoTrans;
    return {
        {"gemm", true, [=](P& p) { gemm<B, T>(p.q, p.mat(p.a), p.mat(p.b), p.mat(p.c), one, half, N, N); }},
        {"gemm(heterogeneous, a k == 0 item)", true,
         [=](P& p) {
             // Item 0 (inactive) is empty and item 1 (active) has k == 0: the loop writes C there itself.
             const auto A = p.mat(p.a).with_active_dims(p.first_empty.to_span(), p.second_k0.to_span());
             const auto Bm = p.mat(p.b).with_active_dims(p.second_k0.to_span(), p.full.to_span());
             const auto Cm = p.mat(p.c).with_active_dims(p.first_empty.to_span(), p.full.to_span());
             gemm<B, T>(p.q, A, Bm, Cm, one, half, N, N);
         }},
    };
}

// The routed ops: the public op entries, each refused by every family's can_run term.
// gemv honours the mask (W1a): gemv_candidates_tests' Masked* cases hold its contract.
template <typename T, Backend B>
std::vector<Entry<T>> routed_entries() {
    using P = Pool<T>;
    const T one(1), half(0.5);
    const auto L = Uplo::Lower;
    const auto N = Transpose::NoTrans;
    const auto all = SvdVectors::All;
    std::vector<Entry<T>> out = {
        {"trsm", true, [=](P& p) { trsm<B, T>(p.q, p.mat(p.a), p.mat(p.b), one, Side::Left, L, N, Diag::NonUnit); }},
        {"trmm", true,
         [=](P& p) { trmm<B, T>(p.q, p.mat(p.a), p.mat(p.b), p.mat(p.c), one, Side::Left, L, N, Diag::NonUnit); }},
        {"potrf", true, [=](P& p) { potrf<B, T>(p.q, p.mat(p.a), L, p.ws, p.ispan()); }},
        {"getrf", true, [=](P& p) { getrf<B, T>(p.q, p.mat(p.a), p.pspan(), p.ws, p.ispan()); }},
        {"getrs", true,
         [=](P& p) {
             if (!p.masked()) getrf<B, T>(p.q, p.mat(p.a), p.pspan(), p.ws, p.ispan());
             getrs<B, T>(p.q, p.mat(p.a), p.mat(p.b), N, p.pspan(), p.ws);
         }},
        {"getri", true,
         [=](P& p) {
             if (!p.masked()) getrf<B, T>(p.q, p.mat(p.a), p.pspan(), p.ws, p.ispan());
             getri<B, T>(p.q, p.mat(p.a), p.mat(p.c), p.pspan(), p.ws, p.ispan());
         }},
        {"syev", true, [=](P& p) { syev<B, T>(p.q, p.mat(p.a), p.wspan(), JobType::EigenVectors, L, p.ws, p.ispan()); }},
        {"gesvd", true,
         [=](P& p) { gesvd<B, T>(p.q, p.mat(p.a), p.wspan(), p.mat(p.b), p.mat(p.c), all, all, p.ws, p.ispan()); }},
        {"gesvd(hermitian)", true,
         [=](P& p) { gesvd<B, T>(p.q, p.mat(p.a), p.wspan(), p.mat(p.b), p.mat(p.c), all, all, L, p.ws, p.ispan()); }},
        {"gesv", true, [=](P& p) { gesv<B, T>(p.q, p.mat(p.a), p.mat(p.b), p.pspan(), p.ws, p.ispan()); }},
        {"posv", true, [=](P& p) { posv<B, T>(p.q, p.mat(p.a), p.mat(p.b), L, p.ws, p.ispan()); }},
        {"potri", true,
         [=](P& p) {
             if (!p.masked()) potrf<B, T>(p.q, p.mat(p.a), L, p.ws, p.ispan());
             potri<B, T>(p.q, p.mat(p.a), L, true, p.ws, p.ispan());
         }},
        {"potrs", true,
         [=](P& p) {
             if (!p.masked()) potrf<B, T>(p.q, p.mat(p.a), L, p.ws, p.ispan());
             potrs<B, T>(p.q, p.mat(p.a), p.mat(p.b), L);
         }},
        {"spmm", true,
         [=](P& p) { spmm<B, T, MatrixFormat::CSR>(p.q, p.csr(), p.mat(p.b), p.mat(p.c), one, half, N, N, p.ws); }},
        {"sygst", true,
         [=](P& p) {
             if (!p.masked()) potrf<B, T>(p.q, p.mat(p.b), L, p.ws, p.ispan());
             sygst<B, T>(p.q, p.mat(p.a), p.mat(p.b), 1, L, p.ws);
         }},
        {"sygvd", true,
         [=](P& p) {
             sygvd<B, T>(p.q, p.mat(p.a), p.mat(p.b), p.wspan(), 1, JobType::EigenVectors, L, p.ws, p.ispan());
         }},
        {"sygvd_factored", true,
         [=](P& p) {
             if (!p.masked()) potrf<B, T>(p.q, p.mat(p.b), L, p.ws, p.ispan());
             sygvd_factored<B, T>(p.q, p.mat(p.a), p.mat(p.b), p.wspan(), 1, JobType::EigenVectors, L, p.ws,
                                  p.ispan());
         }},
        {"sytrf", true, [=](P& p) { sytrf<B, T>(p.q, p.mat(p.a), L, p.pspan(), p.ws, p.ispan()); }},
        {"sytrs", true,
         [=](P& p) {
             if (!p.masked()) sytrf<B, T>(p.q, p.mat(p.a), L, p.pspan(), p.ws, p.ispan());
             sytrs<B, T>(p.q, p.mat(p.a), p.mat(p.b), L, p.pspan(), p.ws);
         }},
        {"sysv", true, [=](P& p) { sysv<B, T>(p.q, p.mat(p.a), p.mat(p.b), L, p.pspan(), p.ws, p.ispan()); }},
    };
    // The host backend's natives yield to its BLAS, which cannot skip an item: gemm refuses there.
    if constexpr (B == Backend::NETLIB)
        for (auto& e : gemm_entries<T, B>()) out.push_back(std::move(e));
    // geqrf honours the mask (W2c): tests/geqrf_mask_tests.cc holds its contract. orgqr and
    // ormqr (Left) honour it except where blocked's Left apply reaches trmm (W1b) on a device.
    if (B != Backend::NETLIB && ops::ormqr::wy_reaches_trmm<B, T>(Side::Left)) {
        out.push_back({"orgqr", true, [=](P& p) {
                           if (!p.masked()) geqrf<B, T>(p.q, p.mat(p.a), p.tspan(), p.ws);
                           orgqr<B, T>(p.q, p.mat(p.a), p.tspan(), p.ws);
                       }});
        out.push_back({"ormqr", true, [=](P& p) {
                           if (!p.masked()) geqrf<B, T>(p.q, p.mat(p.a), p.tspan(), p.ws);
                           ormqr<B, T>(p.q, p.mat(p.a), p.mat(p.c), Side::Left, N, p.tspan(), p.ws);
                       }});
    }
    if constexpr (B != Backend::NETLIB) {  // blocked is a GPU family
        out.push_back({"getri(pinned blocked)", true,
                       [=](P& p) {
                           // blocked writes P into C before its trsm: only getri's own C check stops a masked C.
                           const select::ScopedPin<std::monostate> pin("getri", "blocked", select::StrictPin{});
                           if (!p.masked()) getrf<B, T>(p.q, p.mat(p.a), p.pspan(), p.ws, p.ispan());
                           getri<B, T>(p.q, p.mat(p.a), p.mat(p.c), p.pspan(), p.ws, p.ispan());
                       },
                       true});
    }
    if constexpr (is_std_complex_v<T>) {
        out.push_back({"hemm", true,
                       [=](P& p) { hemm<B, T>(p.q, p.mat(p.a), p.mat(p.b), p.mat(p.c), one, half, Side::Left, L); }});
        out.push_back({"herk", true, [=](P& p) { herk<B, T>(p.q, p.mat(p.a), p.mat(p.c), Real<T>(1), Real<T>(0.5), L, N); }});
        out.push_back({"her2k", true,
                       [=](P& p) { her2k<B, T>(p.q, p.mat(p.a), p.mat(p.b), p.mat(p.c), one, Real<T>(0.5), L, N); }});
    } else {
        out.push_back({"symm", true,
                       [=](P& p) { symm<B, T>(p.q, p.mat(p.a), p.mat(p.b), p.mat(p.c), one, half, Side::Left, L); }});
        out.push_back({"syrk", true, [=](P& p) { syrk<B, T>(p.q, p.mat(p.a), p.mat(p.c), one, half, L, N); }});
        out.push_back({"syr2k", true,
                       [=](P& p) { syr2k<B, T>(p.q, p.mat(p.a), p.mat(p.b), p.mat(p.c), one, half, L, N); }});
    }
    return out;
}

// The vendor entries this build links (backend::*_vendor), each the library call of one routed op.
template <typename T, Backend B>
std::vector<Entry<T>> vendor_entries() {
    using P = Pool<T>;
    namespace sv = batchlas::select;
    const T one(1), half(0.5);
    const auto L = Uplo::Lower;
    const auto N = Transpose::NoTrans;
    std::vector<Entry<T>> out;
    const auto add = [&](const char* name, std::function<void(P&)> f) { out.push_back({name, false, std::move(f)}); };
    if constexpr (sv::level3_vendor_available<B>) {
        add("gemm_vendor", [=](P& p) {
            backend::gemm_vendor<B, T>(p.q, p.mat(p.a), p.mat(p.b), p.mat(p.c), one, half, N, N, ComputePrecision::Default);
        });
        add("gemv_vendor", [=](P& p) { backend::gemv_vendor<B, T>(p.q, p.mat(p.a), p.vec(p.tq), p.vec(p.tp), one, half, N); });
        add("trsm_vendor", [=](P& p) {
            backend::trsm_vendor<B, T>(p.q, p.mat(p.a), p.mat(p.b), Side::Left, L, N, Diag::NonUnit, one);
        });
        add("trmm_vendor", [=](P& p) {
            backend::trmm_vendor<B, T>(p.q, p.mat(p.a), p.mat(p.b), p.mat(p.c), one, Side::Left, L, N, Diag::NonUnit);
        });
        if constexpr (is_std_complex_v<T>) {
            add("hemm_vendor", [=](P& p) {
                backend::hemm_vendor<B, T>(p.q, p.mat(p.a), p.mat(p.b), p.mat(p.c), one, half, Side::Left, L);
            });
            add("herk_vendor", [=](P& p) {
                backend::herk_vendor<B, T>(p.q, p.mat(p.a), p.mat(p.c), Real<T>(1), Real<T>(0.5), L, N);
            });
            add("her2k_vendor", [=](P& p) {
                backend::her2k_vendor<B, T>(p.q, p.mat(p.a), p.mat(p.b), p.mat(p.c), one, Real<T>(0.5), L, N);
            });
        } else {
            add("symm_vendor", [=](P& p) {
                backend::symm_vendor<B, T>(p.q, p.mat(p.a), p.mat(p.b), p.mat(p.c), one, half, Side::Left, L);
            });
            add("syrk_vendor", [=](P& p) { backend::syrk_vendor<B, T>(p.q, p.mat(p.a), p.mat(p.c), one, half, L, N); });
            add("syr2k_vendor", [=](P& p) {
                backend::syr2k_vendor<B, T>(p.q, p.mat(p.a), p.mat(p.b), p.mat(p.c), one, half, L, N);
            });
        }
    }
    if constexpr (sv::factorization_vendor_available<B>) {
        // NETLIB's QR host loops honour the mask (W2c, idiom I4); a strided vendor call cannot.
        if constexpr (B != Backend::NETLIB) {
            add("geqrf_vendor", [=](P& p) { backend::geqrf_vendor<B, T>(p.q, p.mat(p.a), p.tspan(), p.ws); });
            add("orgqr_vendor", [=](P& p) { backend::orgqr_vendor<B, T>(p.q, p.mat(p.a), p.tspan(), p.ws); });
            add("ormqr_vendor", [=](P& p) {
                backend::ormqr_vendor<B, T>(p.q, p.mat(p.a), p.mat(p.c), Side::Left, N, p.tspan(), p.ws);
            });
        }
        add("getrf_vendor", [=](P& p) { backend::getrf_vendor<B, T>(p.q, p.mat(p.a), p.pspan(), p.ws, p.ispan()); });
        add("getrs_vendor", [=](P& p) { backend::getrs_vendor<B, T>(p.q, p.mat(p.a), p.mat(p.b), N, p.pspan(), p.ws); });
        add("getri_vendor", [=](P& p) {
            backend::getri_vendor<B, T>(p.q, p.mat(p.a), p.mat(p.c), p.pspan(), p.ws, p.ispan());
        });
    }
    if constexpr (sv::solver_vendor_available<B>) {
        add("potrf_vendor", [=](P& p) { backend::potrf_vendor<B, T>(p.q, p.mat(p.a), L, p.ws, p.ispan()); });
        add("syev_vendor", [=](P& p) {
            backend::syev_vendor<B, T>(p.q, p.mat(p.a), p.wspan(), JobType::EigenVectors, L, p.ws, p.ispan());
        });
        add("gesvd_vendor", [=](P& p) {
            backend::gesvd_vendor<B, T>(p.q, p.mat(p.a), p.wspan(), p.mat(p.b), p.mat(p.c), SvdVectors::All,
                                        SvdVectors::All, p.ws, p.ispan());
        });
    }
    if constexpr (sv::sparse_vendor_available<B>) {
        add("spmm_vendor", [=](P& p) {
            backend::spmm_vendor<B, T, MatrixFormat::CSR>(p.q, p.csr(), p.mat(p.b), p.mat(p.c), one, half, N, N, p.ws);
        });
    }
    if constexpr ((B == Backend::CUDA && sv::kHasCudaVendor) || (B == Backend::NETLIB && sv::kHasNetlib)) {
        add("sygvd_vendor", [=](P& p) {
            backend::sygvd_vendor<B, T>(p.q, p.mat(p.a), p.mat(p.b), p.wspan(), 1, JobType::EigenVectors, L, p.ws,
                                        p.ispan());
        });
        add("sytrf_vendor", [=](P& p) { backend::sytrf_vendor<B, T>(p.q, p.mat(p.a), L, p.pspan(), p.ws, p.ispan()); });
    }
    if constexpr (B == Backend::NETLIB && sv::kHasNetlib) {
        add("sytrs_vendor", [=](P& p) {
            backend::sytrs_vendor<B, T>(p.q, p.mat(p.a), p.mat(p.b), L, p.pspan(), nullptr);
        });
        add("sygst_vendor", [=](P& p) { backend::sygst_vendor<B, T>(p.q, p.mat(p.a), p.mat(p.b), 1, L, p.ws); });
    }
    return out;
}

// The per-family, sub-op and util entries of blas/extensions.hh, extra.hh and internal/ (real T).
template <typename T, Backend B>
std::vector<Entry<T>> direct_entries() {
    using P = Pool<T>;
    using R = Real<T>;
    const auto L = Uplo::Lower;
    const auto N = Transpose::NoTrans;
    const auto all = SvdVectors::All;
    const auto ev = JobType::EigenVectors;
    std::vector<Entry<T>> out;
    const auto add = [&](const char* name, std::function<void(P&)> f) { out.push_back({name, false, std::move(f)}); };
    add("syev_cta", [=](P& p) { syev_cta<B, T>(p.q, p.mat(p.a), p.wspan(), ev, L, p.ws); });
    add("syev_cta_fused", [=](P& p) { syev_cta_fused<B, T>(p.q, p.mat(p.a), p.wspan(), ev, L, p.ws); });
    add("syev_jacobi_cta", [=](P& p) { syev_jacobi_cta<B, T>(p.q, p.mat(p.a), p.wspan(), ev, L, p.ws); });
    add("syev_blocked", [=](P& p) { syev_blocked<B, T>(p.q, p.mat(p.a), p.wspan(), ev, L, p.ws); });
    add("syev_two_stage", [=](P& p) { syev_two_stage<B, T>(p.q, p.mat(p.a), p.wspan(), ev, L, p.ws); });
    add("gesvd_blocked", [=](P& p) {
        gesvd_blocked<B, T>(p.q, p.mat(p.a), p.wspan(), p.mat(p.b), p.mat(p.c), all, all, p.ws, p.ispan());
    });
    add("gesvd_blocked(hermitian)", [=](P& p) {
        gesvd_blocked<B, T>(p.q, p.mat(p.a), p.wspan(), p.mat(p.b), p.mat(p.c), all, all, L, p.ws, p.ispan());
    });
    add("gesvd_cta", [=](P& p) {
        gesvd_cta<B, T>(p.q, p.mat(p.a), p.wspan(), p.mat(p.b), p.mat(p.c), all, all, p.ws, p.ispan());
    });
    add("gesvd_cta(hermitian)", [=](P& p) {
        gesvd_cta<B, T>(p.q, p.mat(p.a), p.wspan(), p.mat(p.b), p.mat(p.c), all, all, L, p.ws, p.ispan());
    });
    add("gesvdj_cta", [=](P& p) { gesvdj_cta<B, T>(p.q, p.mat(p.a), p.wspan(), p.mat(p.b), p.mat(p.c), all, all, p.ws); });
    // ormqx_cta honours the mask (W2c). ormqr_blocked's Left apply refuses where it reaches trmm;
    // on NETLIB its gemm children refuse.
    if (B == Backend::NETLIB || ops::ormqr::wy_reaches_trmm<B, T>(Side::Left))
        add("ormqr_blocked",
            [=](P& p) { ormqr_blocked<B, T>(p.q, p.mat(p.a), p.mat(p.c), Side::Left, N, p.tspan(), p.ws); });
    add("sytrd_cta", [=](P& p) { sytrd_cta<B, T>(p.q, p.mat(p.a), p.vec(p.tq), p.vec(p.tp, kN - 1), p.vec(p.tau), L, p.ws); });
    add("sytrd_blocked", [=](P& p) {
        sytrd_blocked<B, T>(p.q, p.mat(p.a), p.vec(p.tq), p.vec(p.tp, kN - 1), p.vec(p.tau), L, p.ws);
    });
    add("sytrd_sy2sb", [=](P& p) { sytrd_sy2sb<B, T>(p.q, p.mat(p.a), p.mat(p.z, kKd + 1), p.vec(p.tau), L, kKd, p.ws); });
    add("sytrd_sb2st", [=](P& p) {
        sytrd_sb2st<B, T>(p.q, p.mat(p.z, kKd + 1), p.vec(p.d), p.vec(p.e, kN - 1), p.vec(p.tau), L, kKd, p.ws, 0);
    });
    add("sytrd_band_reduction", [=](P& p) {
        sytrd_band_reduction<B, T>(p.q, p.mat(p.z, kKd + 1), p.vec(p.d), p.vec(p.e, kN - 1), p.vec(p.tau), L, kKd, p.ws,
                                   int32_t(4));
    });
    add("sytrd_band_reduction(params)", [=](P& p) {
        sytrd_band_reduction<B, T>(p.q, p.mat(p.z, kKd + 1), p.vec(p.d), p.vec(p.e, kN - 1), p.vec(p.tau), L, kKd, p.ws,
                                   SytrdBandReductionParams{});
    });
    add("sytrd_band_reduction_single_step", [=](P& p) {
        sytrd_band_reduction_single_step<B, T>(p.q, p.mat(p.z, kKd + 1), p.mat(p.v, kN - 1), L, kKd, p.ws,
                                               SytrdBandReductionParams{});
    });
    add("latrd_lower_panel", [=](P& p) {
        latrd_lower_panel<B, T>(p.q, p.mat(p.a), p.vec(p.tp, kN - 1), p.vec(p.tau), p.mat(p.z, kN, 2), 0, 2);
    });
    add("latrd_lower_panel(panel)", [=](P& p) {
        latrd_lower_panel<B, T>(p.q, p.mat(p.a), p.vec(p.tp, kN - 1), p.vec(p.tau), p.mat(p.z, kN, 2));
    });
    add("gebrd_unblocked", [=](P& p) {
        gebrd_unblocked<B, T>(p.q, p.mat(p.a), p.vec(p.d), p.vec(p.e, kN - 1), p.vec(p.tq), p.vec(p.tp));
    });
    add("gebrd_cta",
        [=](P& p) { gebrd_cta<B, T>(p.q, p.mat(p.a), p.vec(p.d), p.vec(p.e, kN - 1), p.vec(p.tq), p.vec(p.tp)); });
    add("gebrd_blocked", [=](P& p) {
        gebrd_blocked<B, T>(p.q, p.mat(p.a), p.vec(p.d), p.vec(p.e, kN - 1), p.vec(p.tq), p.vec(p.tp), p.ws);
    });
    add("ormbr", [=](P& p) { ormbr<B, T>(p.q, p.mat(p.a), p.vec(p.tau), p.mat(p.c), 'Q', Side::Left, N, p.ws); });
    add("syevx", [=](P& p) { syevx<B, T, MatrixFormat::Dense>(p.q, p.mat(p.a), p.wspan(), 2, p.ws, ev, p.mat(p.z)); });
    add("syevx(m)", [=](P& p) {
        syevx<B, T, MatrixFormat::Dense>(p.q, p.mat(p.a), p.wspan(), P::flat(p.m), 2, p.ws, ev, p.mat(p.z));
    });
    add("syevx_direct", [=](P& p) {
        syevx_direct<B, T, MatrixFormat::Dense>(p.q, p.mat(p.a), p.wspan(), P::flat(p.m), 2, p.ws, ev, p.mat(p.z),
                                                SyevxParams<T>());
    });
    add("syevx_direct_subset", [=](P& p) {
        syevx_direct_subset<B, T, MatrixFormat::Dense>(p.q, p.mat(p.a), p.wspan(), P::flat(p.m), 2, p.ws, ev,
                                                       p.mat(p.z), SyevxParams<T>());
    });
    add("syevx_lobpcg", [=](P& p) {
        syevx_lobpcg<B, T, MatrixFormat::Dense>(p.q, p.mat(p.a), p.wspan(), 2, p.ws, ev, p.mat(p.z), SyevxParams<T>());
    });
    add("syevx_filtered", [=](P& p) {
        syevx_filtered<B, T, MatrixFormat::Dense>(p.q, p.mat(p.a), p.wspan(), 2, p.ws, ev, p.mat(p.z), SyevxParams<T>());
    });
    add("ritz_values", [=](P& p) {
        ritz_values<B, T, MatrixFormat::Dense>(p.q, p.mat(p.a), p.mat(p.z), p.vec(p.w), p.ws);
    });
    add("ortho", [=](P& p) { ortho<B, T>(p.q, p.mat(p.a, kN, 2), N, p.ws); });
    add("ortho(M)", [=](P& p) { ortho<B, T>(p.q, p.mat(p.a, kN, 2), p.mat(p.z, kN, 2), N, N, p.ws); });
    add("inv", [=](P& p) { inv<B, T>(p.q, p.mat(p.a), p.mat(p.c), p.ws); });
    add("inv(owning)", [=](P& p) { (void)inv<B, T>(p.q, p.mat(p.a)); });
    add("norm", [=](P& p) { norm<T, MatrixFormat::Dense>(p.q, p.mat(p.a), NormType::Frobenius, P::flat(p.w)); });
    add("norm(owning)", [=](P& p) { (void)norm<T, MatrixFormat::Dense>(p.q, p.mat(p.a)); });
    add("syfn", [=](P& p) { syfn<B, T>(p.q, p.mat(p.a), p.mat(p.c), SyfnFunction<R>::exp(), L, p.ws, p.ispan()); });
    add("syfn_from_eigen", [=](P& p) {
        syfn_from_eigen<B, T>(p.q, p.mat(p.z), p.wspan(), p.mat(p.c), SyfnFunction<R>::exp(), p.ws, p.ispan());
    });
    add("syev_align", [=](P& p) { syev_align<B, T>(p.q, p.mat(p.z), p.wspan(), p.mat(p.v), R(1e-3), p.ws); });
    add("iluk_factorize", [=](P& p) { (void)iluk_factorize<B, T>(p.q, p.csr()); });
    add("iluk_factorize(workspace)",
        [=](P& p) { (void)iluk_factorize<B, T>(p.q, p.csr(), p.ws, ILUKParams<T>(), nullptr); });
    add("iluk_apply", [=](P& p) {
        // The factor comes from the unmasked matrix (a workspace of its own): only rhs and out are masked.
        UnifiedVector<std::byte> fws(std::size_t(1) << 20);
        const MatrixView<T, MatrixFormat::CSR> plain(p.vals.data(), p.ro.data(), p.ci.data(), kN, kN, NonZeros{kN},
                                                     kVecStride, kN + 1, kBatch);
        const auto M = iluk_factorize<B, T>(p.q, plain, Span<std::byte>(fws.data(), fws.size()), ILUKParams<T>(), nullptr);
        p.q.wait();
        iluk_apply<B, T>(p.q, M, p.mat(p.b), p.mat(p.c), p.ws);
    });
    if constexpr (!is_std_complex_v<T>) {  // real-only: tridiagonal and bidiagonal stages, lanczos, lascl, cond
        add("steqr", [=](P& p) {
            steqr<B, T>(p.q, p.vec(p.d), p.vec(p.e, kN - 1), p.vec(p.w), p.ws, ev, SteqrParams<T>(), p.mat(p.z), p.ispan());
        });
        add("steqr_cta", [=](P& p) {
            steqr_cta<B, T>(p.q, p.vec(p.d), p.vec(p.e, kN - 1), p.vec(p.w), p.ws, ev, SteqrParams<T>(), p.mat(p.z),
                            p.ispan());
        });
        add("stedc", [=](P& p) {
            stedc<B, T>(p.q, p.vec(p.d), p.vec(p.e, kN - 1), p.vec(p.w), p.ws, ev, StedcParams<T>(), p.mat(p.z), p.ispan());
        });
        add("stebz", [=](P& p) { stebz<B, T>(p.q, p.vec(p.d), p.vec(p.e, kN - 1), p.vec(p.w), P::flat(p.m), p.ws); });
        add("stein", [=](P& p) { stein<B, T>(p.q, p.vec(p.d), p.vec(p.e, kN - 1), p.vec(p.w), kN, p.mat(p.z), p.ws); });
        add("stein(counts)", [=](P& p) {
            stein<B, T>(p.q, p.vec(p.d), p.vec(p.e, kN - 1), p.vec(p.w), kN, Span<const std::int32_t>(p.counts.data(), kBatch),
                        p.mat(p.z), p.ws);
        });
        add("bdsqr", [=](P& p) { bdsqr<B, R>(p.q, p.vec(p.d), p.vec(p.e, kN - 1), P::flat(p.s), p.ws, true, p.ispan()); });
        add("bdsqr(vectors)", [=](P& p) {
            bdsqr<B, R>(p.q, p.vec(p.d), p.vec(p.e, kN - 1), P::flat(p.s), p.ws, p.mat(p.b), p.mat(p.c), true, p.ispan());
        });
        add("bdsdc", [=](P& p) { bdsdc<B, R>(p.q, p.vec(p.d), p.vec(p.e, kN - 1), P::flat(p.s), p.ws, true, p.ispan()); });
        add("bdsdc(vectors)", [=](P& p) {
            bdsdc<B, R>(p.q, p.vec(p.d), p.vec(p.e, kN - 1), P::flat(p.s), p.ws, p.mat(p.b), p.mat(p.c), true, p.ispan());
        });
        add("cond", [=](P& p) { cond<B, T, MatrixFormat::Dense>(p.q, p.mat(p.a), NormType::Frobenius, P::flat(p.tq), p.ws); });
        add("cond(owning)", [=](P& p) { (void)cond<B, T, MatrixFormat::Dense>(p.q, p.mat(p.a), NormType::Frobenius); });
        add("tridiagonal_solver", [=](P& p) {
            tridiagonal_solver<B, T>(p.q, P::flat(p.tq), P::flat(p.tp), p.wspan(), p.ws, ev, p.mat(p.z), kN, kBatch);
        });
        add("lanczos", [=](P& p) { lanczos<B, T, MatrixFormat::Dense>(p.q, p.mat(p.a), p.wspan(), p.ws, ev, p.mat(p.z)); });
        add("lascl", [=](P& p) { lascl<MatrixFormat::Dense, T>(p.q, p.mat(p.a), T(1), T(2)); });
    }
    return out;
}

// The verification statics of MatrixView, and the Matrix and view conversions that copy data.
template <typename T>
std::vector<Entry<T>> check_entries() {
    using P = Pool<T>;
    using V = MV<T>;
    const auto N = Transpose::NoTrans;
    const auto L = Uplo::Lower;
    std::vector<Entry<T>> out;
    const auto add = [&](const char* name, std::function<void(P&)> f) { out.push_back({name, false, std::move(f)}); };
    add("MatrixView::product", [=](P& p) { V::product(p.q, p.mat(p.a), N, p.mat(p.b), N, p.mat(p.c)); });
    add("potrf_residual", [=](P& p) { (void)V::potrf_residual(p.q, p.mat(p.a), p.mat(p.b), L); });
    add("similarity_residual", [=](P& p) { (void)V::similarity_residual(p.q, p.mat(p.a), p.mat(p.b), p.mat(p.c)); });
    add("getrf_residual", [=](P& p) { (void)V::getrf_residual(p.q, p.mat(p.a), p.mat(p.b), p.vec(p.piv32)); });
    add("qr_residual", [=](P& p) { (void)V::qr_residual(p.q, p.mat(p.a), p.mat(p.b), p.vec(p.tau)); });
    add("form_q(item)", [=](P& p) { (void)V::form_q(p.q, p.mat(p.a), p.vec(p.tau), 1, kN); });
    add("form_q(items)", [=](P& p) { (void)V::form_q(p.q, p.mat(p.a), p.vec(p.tau), std::span<const int>{}, kN); });
    add("svd_residual",
        [=](P& p) { (void)V::svd_residual(p.q, p.mat(p.a), p.mat(p.b), p.vec(p.s), p.mat(p.c)); });
    add("qr_reconstruction", [=](P& p) { (void)V::qr_reconstruction(p.q, p.mat(p.a), p.mat(p.b), p.mat(p.c)); });
    add("solve_residual(trans)", [=](P& p) { (void)V::solve_residual(p.q, p.mat(p.a), N, p.mat(p.b), p.mat(p.c)); });
    add("solve_residual", [=](P& p) { (void)V::solve_residual(p.q, p.mat(p.a), p.mat(p.b), p.mat(p.c)); });
    add("lu_solve_residual", [=](P& p) {
        (void)V::lu_solve_residual(p.q, p.mat(p.a), p.vec(p.piv32), N, p.mat(p.b), p.mat(p.c));
    });
    add("gemm_backward_error", [=](P& p) {
        (void)V::gemm_backward_error(p.q, p.mat(p.a), N, p.mat(p.b), N, p.mat(p.c), p.mat(p.z), 1.0, 0.5);
    });
    add("gemm_max_denominator",
        [=](P& p) { (void)V::gemm_max_denominator(p.q, p.mat(p.a), N, p.mat(p.b), N, p.mat(p.c), 1.0, 0.5); });
    add("gemv_backward_error", [=](P& p) {
        (void)V::gemv_backward_error(p.q, p.mat(p.a), N, p.vec(p.tq), p.vec(p.tp), p.vec(p.tau), 1.0, 0.5);
    });
    add("trsm_residual", [=](P& p) {
        (void)V::trsm_residual(p.q, p.mat(p.a), Side::Left, L, N, Diag::NonUnit, p.mat(p.b), p.mat(p.c), 1.0);
    });
    add("spmm_backward_error", [=](P& p) {
        (void)V::spmm_backward_error(p.q, p.csr(), N, p.mat(p.b), N, p.mat(p.c), p.mat(p.z), 1.0, 0.5);
    });
    add("rank2k_backward_error", [=](P& p) {
        (void)V::rank2k_backward_error(p.q, p.mat(p.a), p.mat(p.b), N, p.mat(p.c), p.mat(p.z), L, 1.0, 0.5, false);
    });
    add("orthogonality", [=](P& p) { (void)V::orthogonality(p.q, p.mat(p.a)); });
    add("eigen_residual", [=](P& p) { (void)V::eigen_residual(p.q, p.mat(p.a), p.mat(p.z), p.vec(p.w)); });
    add("generalized_eigen_residual", [=](P& p) {
        (void)V::generalized_eigen_residual(p.q, p.mat(p.a), p.mat(p.b), p.mat(p.z), p.vec(p.w));
    });
    add("values_error", [=](P& p) {
        (void)V::values_error(p.vec(p.w), std::vector<std::vector<double>>(kBatch, std::vector<double>(kN, 0.0)), 1.0);
    });
    add("pivot_ratio", [=](P& p) { (void)V::pivot_ratio(p.mat(p.a)); });
    add("pivots_valid", [=](P& p) { (void)V::pivots_valid(p.vec(p.piv32), kN); });
    add("Matrix::copy_from", [=](P& p) {
        Matrix<T> dst(kN, kN, kBatch);
        dst.copy_from(p.mat(p.a));
    });
    add("MatrixView::astype(device-only mask)", [=](P& p) { (void)p.mat(p.a).template astype<T>(); });
    add("MatrixView::astype(items)", [=](P& p) { (void)p.mat(p.a).template astype<T>(std::span<const int>{}); });
    add("VectorView::astype", [=](P& p) { (void)p.vec(p.tq).template astype<T>(); });
    // A Matrix carries its own (shared) store, so these copy the device-only bytes into it.
    const auto owned = [](P& p) {
        Matrix<T> M(p.a.data(), kN, kN, kLd, kStride, kBatch);
        if (p.masks(p.a.data())) {
            auto store = M.enable_batch_mask();
            p.q->memcpy(store.data(), p.mask.data(), store.size());
            p.q.wait();
        }
        return M;
    };
    add("Matrix::to_column_major", [=](P& p) { (void)owned(p).to_column_major(p.q); });
    add("Matrix::to_column_major(own queue)", [=](P& p) { (void)owned(p).to_column_major(); });
    add("Matrix::to_row_major", [=](P& p) { (void)owned(p).to_row_major(p.q); });
    add("Matrix::to_row_major(own queue)", [=](P& p) { (void)owned(p).to_row_major(); });
    add("Matrix::convert_to<CSR>", [=](P& p) { (void)owned(p).template convert_to<MatrixFormat::CSR>(); });
    return out;
}

// The header-only value-returning wrappers of blas/linalg-ops.hh: their work copies are unmasked.
template <typename T>
std::vector<Entry<T>> linalg_entries() {
    using P = Pool<T>;
    std::vector<Entry<T>> out;
    const auto add = [&](const char* name, std::function<void(P&)> f) { out.push_back({name, false, std::move(f)}); };
    SyevxParams<T> subset;
    subset.select = SyevxSelect::Index;
    subset.il = 0;
    subset.iu = 1;
    add("linalg::add", [=](P& p) { (void)linalg::add<T>(p.q, p.mat(p.a), p.mat(p.b)); });
    add("linalg::subtract", [=](P& p) { (void)linalg::subtract<T>(p.q, p.mat(p.a), p.mat(p.b)); });
    add("linalg::multiply", [=](P& p) { (void)linalg::multiply<T>(p.q, p.mat(p.a), p.mat(p.b)); });
    add("linalg::divide", [=](P& p) { (void)linalg::divide<T>(p.q, p.mat(p.a), p.mat(p.b)); });
    add("linalg::scaled", [=](P& p) { (void)linalg::scaled<T>(p.q, p.mat(p.a), T(2)); });
    add("linalg::matmul", [=](P& p) { (void)linalg::matmul<T>(p.q, p.mat(p.a), p.mat(p.b)); });
    add("linalg::matvec", [=](P& p) { (void)linalg::matvec<T>(p.q, p.mat(p.a), p.vec(p.tq)); });
    add("linalg::gram", [=](P& p) { (void)linalg::gram<T>(p.q, p.mat(p.a)); });
    add("linalg::solve_triangular", [=](P& p) { (void)linalg::solve_triangular<T>(p.q, p.mat(p.a), p.mat(p.b)); });
    add("linalg::cholesky", [=](P& p) { (void)linalg::cholesky<T>(p.q, p.mat(p.a)); });
    add("linalg::eigvalsh", [=](P& p) { (void)linalg::eigvalsh<T>(p.q, p.mat(p.a)); });
    add("linalg::eigh", [=](P& p) { (void)linalg::eigh<T>(p.q, p.mat(p.a)); });
    add("linalg::eigh(subset)", [=](P& p) { (void)linalg::eigh<T>(p.q, p.mat(p.a), subset); });
    add("linalg::eigvalsh(subset)", [=](P& p) { (void)linalg::eigvalsh<T>(p.q, p.mat(p.a), subset); });
    add("linalg::solve", [=](P& p) { (void)linalg::solve<T>(p.q, p.mat(p.a), p.mat(p.b)); });
    add("linalg::solve_spd", [=](P& p) { (void)linalg::solve_spd<T>(p.q, p.mat(p.a), p.mat(p.b)); });
    add("linalg::triu", [=](P& p) { (void)linalg::triu<T>(p.q, p.mat(p.a)); });
    add("linalg::tril", [=](P& p) { (void)linalg::tril<T>(p.q, p.mat(p.a)); });
    add("linalg::inv", [=](P& p) { (void)linalg::inv<T>(p.q, p.mat(p.a)); });
    add("linalg::svd", [=](P& p) { (void)linalg::svd<T>(p.q, p.mat(p.a)); });
    add("linalg::svd(options)", [=](P& p) {
        (void)linalg::svd<T>(p.q, p.mat(p.a), GesvdOptions{.jobu = SvdVectors::All, .jobvh = SvdVectors::None});
    });
    add("linalg::lu", [=](P& p) { (void)linalg::lu<T>(p.q, p.mat(p.a)); });
    add("linalg::norm", [=](P& p) { (void)linalg::norm<T>(p.q, p.mat(p.a)); });
    if constexpr (!is_std_complex_v<T>) add("linalg::cond", [=](P& p) { (void)linalg::cond<T>(p.q, p.mat(p.a)); });
    return out;
}

template <typename T, Backend B>
struct EntryConfig {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};
using C = std::complex<float>;
using EntryTuple = decltype(std::tuple_cat(
#if BATCHLAS_HAS_HOST_BACKEND && BATCHLAS_HAS_CPU_TARGET
    std::tuple<EntryConfig<float, Backend::NETLIB>, EntryConfig<C, Backend::NETLIB>>{},
#endif
#if BATCHLAS_HAS_CUDA_BACKEND
    std::tuple<EntryConfig<float, Backend::CUDA>, EntryConfig<C, Backend::CUDA>>{},
#endif
#if BATCHLAS_HAS_ROCM_BACKEND
    std::tuple<EntryConfig<float, Backend::ROCM>, EntryConfig<C, Backend::ROCM>>{},
#endif
    std::tuple<>{}));
using EntryTypes = test_utils::tuple_to_types<EntryTuple>::type;

template <typename Config>
class BatchMaskEntries : public test_utils::BatchLASTest<Config> {
protected:
    UnifiedVector<std::byte> work{std::size_t(1) << 26};
    Span<std::byte> ws() { return {work.data(), work.size()}; }
};
TYPED_TEST_SUITE(BatchMaskEntries, EntryTypes);

TYPED_TEST(BatchMaskEntries, RoutedOpsRefuseADeviceOnlyMaskBeforeWriting) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    Queue& q = *this->ctx;
    const masked_contract::DeviceMask mask(q, kAlternating);
    // Only a build without the backend's library may lack a route (herk has no native kernel).
    const bool unroutable =
        (B == Backend::NETLIB && !select::kHasNetlib) || (B == Backend::CUDA && !select::kHasCudaVendor);
    int checked = 0;
    for (const auto& e : routed_entries<T, B>()) checked += check_refuses<T>(q, e, mask.span(), this->ws(), unroutable);
    EXPECT_GT(checked, 0) << "no routed op ran unmasked, so no refusal was checked";
}

// gemm runs a masked batch natively on a device backend, whichever operand carries the mask: C's
// inactive items (0, 2) bit-identical, its active ones the unmasked call's, A and B unwritten.
TYPED_TEST(BatchMaskEntries, GemmRunsADeviceOnlyMaskNatively) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    if constexpr (B == Backend::NETLIB) GTEST_SKIP() << "the host backend's gemm is its BLAS, which refuses";
    Queue& q = *this->ctx;
    const masked_contract::DeviceMask mask(q, kAlternating);
    for (const auto& e : gemm_entries<T, B>()) {
        SCOPED_TRACE(e.name);
        Pool<T> plain(q, {}, this->ws());
        e.call(plain);
        q.wait();
        const auto want = plain.snapshot();
        for (const int only : {-1, 0, 1, 2}) {
            SCOPED_TRACE("operand " + std::to_string(only) + " masked (-1: every one)");
            Pool<T> p(q, mask.span(), this->ws(), only);
            const auto before = p.snapshot();
            ASSERT_NO_THROW({
                e.call(p);
                q.wait();
            });
            const auto after = p.snapshot();
            for (std::size_t i = 0; i < before.size(); ++i) {
                const std::string name = std::get<0>(p.buffers()[i]);
                if (name != "C") {
                    EXPECT_TRUE(after[i] == before[i]) << "wrote operand " << name;
                    continue;
                }
                for (int b = 0; b < kBatch; ++b) {
                    const std::size_t off = std::size_t(b) * kStride * sizeof(T), len = kStride * sizeof(T);
                    const auto& ref = kAlternating[std::size_t(b)] ? want[i] : before[i];
                    EXPECT_EQ(std::memcmp(after[i].data() + off, ref.data() + off, len), 0)
                        << "C item " << b << (kAlternating[std::size_t(b)] ? " differs from the unmasked call" : " was written");
                }
            }
        }
    }
}

TYPED_TEST(BatchMaskEntries, VendorEntriesRefuseADeviceOnlyMask) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    Queue& q = *this->ctx;
    const masked_contract::DeviceMask mask(q, kAlternating);
    for (const auto& e : vendor_entries<T, B>()) check_refuses<T>(q, e, mask.span(), this->ws());
}

TYPED_TEST(BatchMaskEntries, FamilyStageAndUtilEntriesRefuseADeviceOnlyMask) {
    using T = typename TestFixture::ScalarType;
    constexpr Backend B = TestFixture::BackendType;
    Queue& q = *this->ctx;
    const masked_contract::DeviceMask mask(q, kAlternating);
    for (const auto& e : direct_entries<T, B>()) check_refuses<T>(q, e, mask.span(), this->ws());
}

TYPED_TEST(BatchMaskEntries, VerificationEntriesRefuseADeviceOnlyMask) {
    using T = typename TestFixture::ScalarType;
    Queue& q = *this->ctx;
    const masked_contract::DeviceMask mask(q, kAlternating);
    for (const auto& e : check_entries<T>()) check_refuses<T>(q, e, mask.span(), this->ws());
}

TYPED_TEST(BatchMaskEntries, LinalgWrappersRefuseADeviceOnlyMask) {
    using T = typename TestFixture::ScalarType;
    Queue& q = *this->ctx;
    const masked_contract::DeviceMask mask(q, kAlternating);
    for (const auto& e : linalg_entries<T>()) check_refuses<T>(q, e, mask.span(), this->ws());
}

// The syev capability queries answer from can_run, so a masked view is one no family supports.
TYPED_TEST(BatchMaskEntries, SyevSupportQueriesAnswerNoForAMaskedView) {
    using T = typename TestFixture::ScalarType;
    namespace sd = ops::syev::detail;
    constexpr Backend B = TestFixture::BackendType;
    if (B == Backend::NETLIB) GTEST_SKIP() << "no syev family serves the host backend";
    Queue& q = *this->ctx;
    const masked_contract::DeviceMask mask(q, kAlternating);
    Pool<T> plain(q, {}, this->ws());
    const auto A = plain.mat(plain.a);
    ASSERT_TRUE(sd::syev_supports_cta<T>(q, A) || sd::syev_supports_blocked<T>(q, A, Uplo::Lower) ||
                sd::syev_supports_two_stage<T>(q, A, Uplo::Lower))
        << "no family supports the unmasked view, so the masked answers prove nothing";
    Pool<T> p(q, mask.span(), this->ws());
    EXPECT_FALSE(sd::syev_supports_cta<T>(q, p.mat(p.a)));
    EXPECT_FALSE(sd::syev_supports_blocked<T>(q, p.mat(p.a), Uplo::Lower));
    EXPECT_FALSE(sd::syev_supports_two_stage<T>(q, p.mat(p.a), Uplo::Lower));
}

// astype copies the mask on the host with no Queue: shared USM copies, a device-only or a foreign
// (non-USM) mask is refused before any data is converted.
TYPED_TEST(BatchMaskEntries, AstypeCopiesOnlyAHostReadableUsmMask) {
    using T = typename TestFixture::ScalarType;
    Queue& q = *this->ctx;
    Pool<T> p(q, {}, this->ws());
    UnifiedVector<std::uint8_t> shared(kBatch);
    for (int b = 0; b < kBatch; ++b) shared[b] = kAlternating[std::size_t(b)];
    const auto copied = p.mat(p.a).with_batch_mask(shared.to_span()).template astype<T>();
    ASSERT_TRUE(copied.has_batch_mask());
    for (int b = 0; b < kBatch; ++b) EXPECT_EQ(copied.batch_mask()[b], kAlternating[std::size_t(b)]);
    const masked_contract::DeviceMask device(q, kAlternating);
    std::vector<std::uint8_t> foreign(kAlternating);
    for (const auto m : {device.span(), Span<const std::uint8_t>(foreign.data(), foreign.size())})
        EXPECT_THROW((void)p.mat(p.a).with_batch_mask(m).template astype<T>(), batchlas::unsupported);
}

}  // namespace
