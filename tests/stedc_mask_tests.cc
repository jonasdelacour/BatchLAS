// SPDX-License-Identifier: MIT
// stedc honours a batch mask (W3a) on tests/masked_contract.hh: the NETLIB arm (steqr_legacy), the
// recursive driver and the level driver (its per-level expanded mask, the root solved in the
// caller's packed matrix, padded trees), jobz V and N, every GPU merge variant on a GPU queue and
// the CPU merge path on a CPU SYCL queue. d (edited in place), e, w, Z and info are all outputs;
// d and e have non-unit increments, w and Z padded strides. The n = 512 cases cross the host
// deflation-maximum decision (narrow vs full GEMM) with items whose deflation differs; the heavy
// tier adds the saturating identical-item batch. evidence: docs/design/masked-batches.md#masked-batches-tests
#include <gtest/gtest.h>

#include <batchlas/blas/extensions.hh>
#include <batchlas/blas/matrix.hh>
#include <batchlas/util/sycl-vector.hh>

#include <tests/masked_contract.hh>
#include <tests/test_support/storage.hh>
#include <tests/test_utils.hh>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <span>
#include <string>
#include <vector>

using namespace batchlas;

namespace {

namespace mc = masked_contract;

struct Spec {
    std::string name;
    int n = 0, batch = 0, threshold = 16;
    StedcAlgorithm algorithm = StedcAlgorithm::Recursive;
    JobType jobz = JobType::EigenVectors;
    bool packed_z = false;      ///< natural ld and stride: the level driver's root runs in Z itself
    bool identical = false;     ///< every item the same input (the saturating case)
    bool deflation_mix = false; ///< odd items deflate most of every merge, even items almost none
    StedcMergeVariant merge = StedcMergeVariant::Auto;
    StedcSecularSolver secular = StedcSecularSolver::Rocm;
    bool cpu_queue = true;      ///< false: n >= 256, where the CPU queue's float stedc is NaN on base too
};

template <typename T, Backend B>
struct Problem {
    Queue& q;
    Spec s;
    int dinc = 2, dstride, estride, wstride;
    UnifiedVector<T> d, e, w;
    Matrix<T> Z;
    UnifiedVector<std::int32_t> info;

    Problem(Queue& queue, const Spec& spec)
        : q(queue), s(spec), dstride(2 * s.n + 3), estride(s.n + 4), wstride(s.n + 3),
          d(std::size_t(dstride) * s.batch), e(std::size_t(estride) * s.batch), w(std::size_t(wstride) * s.batch),
          Z(s.packed_z ? Matrix<T>(s.n, s.n, s.batch) : test_support::padded<T>(s.n, s.n, s.batch)),
          info(std::size_t(s.batch)) {}

    StedcParams<T> params() const {
        StedcParams<T> p{.recursion_threshold = s.threshold, .algorithm = s.algorithm, .secular_solver = s.secular};
        p.merge_variant = s.merge;
        return p;
    }
    VectorView<T> dv() const { return VectorView<T>(d.data(), s.n, s.batch, dinc, dstride); }
    VectorView<T> ev() const { return VectorView<T>(e.data(), s.n - 1, s.batch, 1, estride); }
    VectorView<T> wv() const { return VectorView<T>(w.data(), s.n, s.batch, 1, wstride); }

    // Graded diagonal, item-dependent unless identical. deflation_mix decouples an odd item's left
    // half (tiny e), so its merges keep few poles, and gives an even item a ramp that keeps most.
    void fill() {
        std::fill_n(d.data(), d.size(), test_support::poison<T>());
        std::fill_n(e.data(), e.size(), test_support::poison<T>());
        for (int b = 0; b < s.batch; ++b) {
            for (int i = 0; i < s.n; ++i) d[std::size_t(b) * dstride + std::size_t(i) * dinc] = T(input(b, i, true));
            for (int i = 0; i + 1 < s.n; ++i) e[std::size_t(b) * estride + i] = T(input(b, i, false));
        }
    }

    double input(int b, int i, bool diag) const {
        const double item = s.identical ? 0.0 : double(b);
        if (s.deflation_mix && b % 2 == 0) {
            // A slow ramp under a unit off-diagonal: delocalised eigenvectors and distinct poles,
            // so the top merge keeps more than 3/4 of them.
            return double(T(diag ? (0.25 + 0.05 * b) * double(i) / s.n : 1.0));
        }
        if (diag) return double(T((1.0 + double(i) / s.n) * std::cos(0.37 * i + 1.3 * item)));
        const bool tiny = s.deflation_mix && b % 2 == 1 && i + 1 < s.n / 2;
        return tiny ? double(T(1e-30)) : double(T(0.5 + 0.25 * std::sin(0.91 * i + 0.7 * item)));
    }

    mc::Case make_case() {
        mc::Case c;
        c.batch = s.batch;
        c.wg_items = 4;
        c.identical_items = s.identical;
        c.prepare = [this] {
            fill();
            std::fill_n(w.data(), w.size(), test_support::poison<T>());
            std::fill_n(Z.data_ptr(), std::size_t(Z.stride()) * s.batch, test_support::poison<T>());
            std::fill_n(info.data(), info.size(), std::int32_t(-7));
        };
        c.call = [this](Span<const std::uint8_t> mask, Span<std::byte> ws) {
            (void)stedc<B, T>(q, dv().with_batch_mask(mask), ev().with_batch_mask(mask), wv().with_batch_mask(mask), ws, s.jobz,
                        params(), Z.view().with_batch_mask(mask), Span<std::int32_t>(info.data(), info.size()));
        };
        c.buffer_size = [this](Span<const std::uint8_t>) {
            return stedc_buffer_size<B, T>(q, std::size_t(s.n), std::size_t(s.batch), s.jobz, params());
        };
        c.outputs = [this] {
            return std::vector<mc::Output>{
                mc::Output::matrix("Z", Z.view()),
                mc::Output::span("w", std::span<const T>(w.data(), w.size()), std::size_t(wstride)),
                mc::Output::span("info", std::span<const std::int32_t>(info.data(), info.size()), 1),
                mc::Output::span("d", std::span<const T>(d.data(), d.size()), std::size_t(dstride)),
                mc::Output::span("e", std::span<const T>(e.data(), e.size()), std::size_t(estride))};
        };
        return c;
    }
};

template <typename T, Backend B>
struct Cfg {
    using ScalarType = T;
    static constexpr Backend BackendVal = B;
};
using Types = typename test_utils::backend_types_filtered<Cfg, false>::type;

template <typename Config>
class StedcMask : public test_utils::BatchLASTest<Config> {
protected:
    using T = typename Config::ScalarType;
    static constexpr Backend B = Config::BackendVal;

    // The fixture's queue, and on the SYCL backend also a CPU queue: there stedc takes the CPU
    // merge path (the legacy secular solver), which the GPU queue never runs.
    std::vector<std::pair<std::string, Queue*>> queues() {
        std::vector<std::pair<std::string, Queue*>> out{{"default", this->ctx.get()}};
        if constexpr (B == Backend::SYCL) {
            if (!cpu_) cpu_ = std::make_unique<Queue>(Device("cpu"), B, true);
            out.emplace_back("cpu", cpu_.get());
        }
        return out;
    }
    void check(const Spec& s) {
        for (auto& [qname, q] : queues()) {
            if (qname == "cpu" && !s.cpu_queue) continue;
            SCOPED_TRACE(s.name + " on the " + qname + " queue");
            Problem<T, B> p(*q, s);
            mc::check(*q, p.make_case());
        }
    }
    bool gpu() const { return this->ctx->device().type == DeviceType::GPU; }

private:
    std::unique_ptr<Queue> cpu_;
};

TYPED_TEST_SUITE(StedcMask, Types);

constexpr auto kRec = StedcAlgorithm::Recursive;
constexpr auto kLvl = StedcAlgorithm::Levels;
constexpr auto kNoVec = JobType::NoEigenVectors;

TYPED_TEST(StedcMask, RecursiveDriver) {
    this->check({"recursive n=48", 48, 6, 8, kRec});
    this->check({"recursive n=37 odd split", 37, 5, 8, kRec});
    this->check({"recursive jobz N", 48, 6, 8, kRec, kNoVec});
}

// n = 37 pads to 40 (own top buffer, Z padded); n = 64 and 128 are unpadded and packed, so the
// root level is solved in Z itself and the deeper levels (L = 2 and 3) must stay out of it.
TYPED_TEST(StedcMask, LevelDriver) {
    this->check({"levels n=37 padded tree", 37, 6, 16, kLvl});
    this->check({"levels n=64 packed Z", 64, 6, 16, kLvl, JobType::EigenVectors, true});
    this->check({"levels n=128 packed Z", 128, 5, 16, kLvl, JobType::EigenVectors, true});
    this->check({"levels jobz N", 37, 6, 16, kLvl, kNoVec});
}

// Every merge kernel family the GPU tuning can pick, pinned through StedcParams.
TYPED_TEST(StedcMask, GpuMergeVariants) {
    if (TestFixture::B == Backend::NETLIB || !this->gpu()) GTEST_SKIP() << "the merge variants are GPU paths";
    using V = StedcMergeVariant;
    for (const auto alg : {kRec, kLvl}) {
        const std::string a = alg == kRec ? "recursive " : "levels ";
        this->check({a + "baseline rocm", 64, 6, 16, alg, JobType::EigenVectors, true, false, false, V::Baseline});
        this->check({a + "baseline legacy", 64, 6, 16, alg, JobType::EigenVectors, true, false, false, V::Baseline,
                     StedcSecularSolver::Legacy});
        this->check({a + "fused", 64, 6, 16, alg, JobType::EigenVectors, true, false, false, V::Fused});
        this->check({a + "fused cta", 64, 6, 16, alg, JobType::EigenVectors, true, false, false, V::FusedCta});
    }
}

// n >= 512: the back-transform width is the batch-wide maximum of the kept poles, read on the host.
// An inactive item reports 0, so a pattern with only odd items active takes the narrow GEMM (16
// columns on Metal float) where the unmasked call takes the full one (512): the masked-batches
// exception. It decided bit-identical here, so Z is held to the unmasked bits, not to bounds. The
// CPU queue is left out: its float stedc returns NaN at n >= 256 on base 02801019 as well.
TYPED_TEST(StedcMask, DeflationMaximumAt512) {
    using V = StedcMergeVariant;
    constexpr auto kRocm = StedcSecularSolver::Rocm;
    this->check({"recursive n=512 deflation mix", 512, 4, 32, kRec, JobType::EigenVectors, false, false, true, V::Auto, kRocm, false});
    this->check({"levels n=512 deflation mix", 512, 4, 32, kLvl, JobType::EigenVectors, true, false, true, V::Auto, kRocm, false});
}

TYPED_TEST(StedcMask, SaturatingIdenticalItems) {
    this->check({"recursive saturating", 64, 1024, 16, kRec, JobType::EigenVectors, false, true});
    this->check({"levels saturating", 64, 1024, 16, kLvl, JobType::EigenVectors, true, true});
}

}  // namespace
