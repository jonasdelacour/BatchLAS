#include <gtest/gtest.h>
#include <util/minibench.hh>
#include <blas/enums.hh>
#include <batchlas/backend_config.h>

static int last_arg = 0;
static void dummy_bench(minibench::State& state) { last_arg = state.range(0); }
MINI_BENCHMARK_REGISTER_SIZES((dummy_bench), [](auto* b){ b->Args({1}); });

// Benchmarks used for backend/type filtering
static int called_fc = 0;
static int called_dn = 0;

template <typename T, batchlas::Backend B>
static void typed_bench(minibench::State&) {}

#ifdef BATCHLAS_HAS_CUDA_BACKEND
template <>
void typed_bench<float, batchlas::Backend::CUDA>(minibench::State&) { ++called_fc; }
MINI_BENCHMARK_REGISTER_SIZES((typed_bench<float, batchlas::Backend::CUDA>), [](auto* b){ b->Args({1}); });
#endif

#ifdef BATCHLAS_HAS_ROCM_BACKEND
template <>
void typed_bench<float, batchlas::Backend::ROCM>(minibench::State&) { ++called_fc; }
MINI_BENCHMARK_REGISTER_SIZES((typed_bench<float, batchlas::Backend::ROCM>), [](auto* b){ b->Args({1}); });
#endif

#ifdef BATCHLAS_HAS_MKL_BACKEND
template <>
void typed_bench<float, batchlas::Backend::MKL>(minibench::State&) { ++called_fc; }
MINI_BENCHMARK_REGISTER_SIZES((typed_bench<float, batchlas::Backend::MKL>), [](auto* b){ b->Args({1}); });
#endif

#ifdef BATCHLAS_HAS_HOST_BACKEND
template <>
void typed_bench<double, batchlas::Backend::NETLIB>(minibench::State&) { ++called_dn; }
MINI_BENCHMARK_REGISTER_SIZES((typed_bench<double, batchlas::Backend::NETLIB>), [](auto* b){ b->Args({1}); });
#endif

TEST(MiniBenchCliTest, OverridesRegisteredArgs) {
    const char* argv[] = {"prog", "5"};
    int argc = 2;
    auto opts = minibench::ParseCommandLine(argc, const_cast<char**>(argv));

    for (auto& b : minibench::registry()) {
        if (b.name.find("dummy_bench") != std::string::npos && !opts.args_list.empty())
            b.args_list = opts.args_list;
    }

    auto it = std::find_if(minibench::registry().begin(), minibench::registry().end(),
                           [](const auto& b){ return b.name.find("dummy_bench") != std::string::npos; });
    ASSERT_NE(it, minibench::registry().end());
    auto& bench = *it;
    ASSERT_EQ(bench.args_list.size(), 1u);
    EXPECT_EQ(bench.args_list[0][0], 5);

    last_arg = 0;
    minibench::run_benchmark(bench, bench.args_list[0], opts.cfg);
    EXPECT_EQ(last_arg, 5);
}

TEST(MiniBenchCliTest, BackendTypeFiltering) {
#if BATCHLAS_HAS_HOST_BACKEND
    const char* argv[] = {"prog", "--backend=NETLIB", "--type=double"};
    int argc = 3;
    auto opts = minibench::ParseCommandLine(argc, const_cast<char**>(argv));
    minibench::RunRegisteredBenchmarks(opts.cfg, "", opts.backends, opts.types);
    EXPECT_EQ(called_fc, 0);
    EXPECT_GT(called_dn, 0);
#elif BATCHLAS_HAS_CUDA_BACKEND
    const char* argv[] = {"prog", "--backend=CUDA", "--type=float"};
    int argc = 3;
    auto opts = minibench::ParseCommandLine(argc, const_cast<char**>(argv));
    minibench::RunRegisteredBenchmarks(opts.cfg, "", opts.backends, opts.types);
    EXPECT_EQ(called_dn, 0);
    EXPECT_GT(called_fc, 0);
#elif BATCHLAS_HAS_ROCM_BACKEND
    const char* argv[] = {"prog", "--backend=ROCM", "--type=float"};
    int argc = 3;
    auto opts = minibench::ParseCommandLine(argc, const_cast<char**>(argv));
    minibench::RunRegisteredBenchmarks(opts.cfg, "", opts.backends, opts.types);
    EXPECT_EQ(called_dn, 0);
    EXPECT_GT(called_fc, 0);
#elif BATCHLAS_HAS_MKL_BACKEND
    const char* argv[] = {"prog", "--backend=MKL", "--type=float"};
    int argc = 3;
    auto opts = minibench::ParseCommandLine(argc, const_cast<char**>(argv));
    minibench::RunRegisteredBenchmarks(opts.cfg, "", opts.backends, opts.types);
    EXPECT_EQ(called_dn, 0);
    EXPECT_GT(called_fc, 0);
#else
    GTEST_SKIP() << "No supported backend compiled";
#endif
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
