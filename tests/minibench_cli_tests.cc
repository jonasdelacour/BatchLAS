#include <gtest/gtest.h>
#include <util/minibench.hh>

static int last_arg = 0;

static void dummy_bench(minibench::State& state) {
    last_arg = state.range(0);
}

// Register default size which will be overridden
MINI_BENCHMARK_REGISTER_SIZES((dummy_bench), [](auto* b){ b->Args({1}); });

TEST(MiniBenchCliTest, OverridesRegisteredArgs) {
    const char* argv[] = {"prog", "5"};
    int argc = 2;
    auto opts = minibench::ParseCommandLine(argc, const_cast<char**>(argv));

    // Apply CLI sizes to all benchmarks
    for (auto& b : minibench::registry()) {
        if (!opts.args_list.empty()) {
            b.args_list = opts.args_list;
        }
    }

    ASSERT_EQ(minibench::registry().size(), 1u);
    auto& bench = minibench::registry().front();
    ASSERT_EQ(bench.args_list.size(), 1u);
    EXPECT_EQ(bench.args_list[0][0], 5);

    last_arg = 0;
    minibench::run_benchmark(bench, bench.args_list[0], opts.cfg);
    EXPECT_EQ(last_arg, 5);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
