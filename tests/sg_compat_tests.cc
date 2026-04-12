// Tests for SubGroupPartition<P> (sg_compat.hh).
//
// Each test kernel exercises one collective, writes results to a device buffer,
// and validates on the host.  The tests are parameterised over partition size P
// and the full sub-group size (32 lanes on AMD gfx1200 / Intel).

#include <gtest/gtest.h>

#include <sycl/sycl.hpp>
#include <util/sycl-device-queue.hh>

// Include the compatibility layer directly.
#include "../src/extensions/sg_compat.hh"

#include <cmath>
#include <numeric>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// SYCL queue helpers
// ---------------------------------------------------------------------------
static sycl::queue make_gpu_queue() {
    try {
        return sycl::queue{sycl::gpu_selector_v};
    } catch (...) {
        return sycl::queue{sycl::default_selector_v};
    }
}

// ---------------------------------------------------------------------------
// Kernel: test SubGroupPartition identity functions
//   get_local_linear_id      == lane within chunk  [0, P)
//   get_local_linear_range   == P
//   get_group_linear_id      == chunk index within sub-group
//   get_group_linear_range   == sg_size / P
//   leader()                 == (lane_within_chunk == 0)
// ---------------------------------------------------------------------------
template <size_t P, size_t SG>
void test_partition_identity(sycl::queue& q) {
    constexpr int N = static_cast<int>(SG);

    std::vector<uint32_t> local_id_out(N, 99),
                           local_range_out(N, 99),
                           group_id_out(N, 99),
                           group_range_out(N, 99),
                           leader_out(N, 99);

    {
        sycl::buffer<uint32_t, 1> b_lid(local_id_out.data(), N);
        sycl::buffer<uint32_t, 1> b_lr(local_range_out.data(), N);
        sycl::buffer<uint32_t, 1> b_gid(group_id_out.data(), N);
        sycl::buffer<uint32_t, 1> b_gr(group_range_out.data(), N);
        sycl::buffer<uint32_t, 1> b_ldr(leader_out.data(), N);

        q.submit([&](sycl::handler& h) {
            auto a_lid = b_lid.get_access<sycl::access::mode::write>(h);
            auto a_lr  = b_lr .get_access<sycl::access::mode::write>(h);
            auto a_gid = b_gid.get_access<sycl::access::mode::write>(h);
            auto a_gr  = b_gr .get_access<sycl::access::mode::write>(h);
            auto a_ldr = b_ldr.get_access<sycl::access::mode::write>(h);

            h.parallel_for(
                sycl::nd_range<1>{N, N},
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                    const auto sg   = it.get_sub_group();
                    const auto part = batchlas::make_partition<P>(sg);
                    const uint32_t wi = static_cast<uint32_t>(it.get_global_linear_id());
                    a_lid[wi]  = part.get_local_linear_id();
                    a_lr [wi]  = part.get_local_linear_range();
                    a_gid[wi]  = part.get_group_linear_id();
                    a_gr [wi]  = part.get_group_linear_range();
                    a_ldr[wi]  = part.leader() ? 1u : 0u;
                });
        }).wait();
    }

    for (int wi = 0; wi < N; ++wi) {
        const uint32_t expected_lid  = static_cast<uint32_t>(wi) % static_cast<uint32_t>(P);
        const uint32_t expected_gid  = static_cast<uint32_t>(wi) / static_cast<uint32_t>(P);
        EXPECT_EQ(local_id_out[wi],    expected_lid)    << "P=" << P << " wi=" << wi << " local_id";
        EXPECT_EQ(local_range_out[wi], static_cast<uint32_t>(P)) << "P=" << P << " wi=" << wi << " local_range";
        EXPECT_EQ(group_id_out[wi],    expected_gid)    << "P=" << P << " wi=" << wi << " group_id";
        EXPECT_EQ(group_range_out[wi], static_cast<uint32_t>(SG / P)) << "P=" << P << " wi=" << wi << " group_range";
        EXPECT_EQ(leader_out[wi],      (expected_lid == 0u) ? 1u : 0u) << "P=" << P << " wi=" << wi << " leader";
    }
}

// ---------------------------------------------------------------------------
// Kernel: test permute_group_by_xor (XOR shuffle within chunk)
//   Each lane i XORs with mask 1: result should be value from lane (i ^ 1).
//   Since i ^ 1 stays within the same chunk (mask < P), this is intra-chunk.
// ---------------------------------------------------------------------------
template <size_t P, size_t SG>
void test_permute_xor(sycl::queue& q) {
    constexpr int N = static_cast<int>(SG);
    std::vector<uint32_t> out(N, 0);

    {
        sycl::buffer<uint32_t, 1> buf(out.data(), N);
        q.submit([&](sycl::handler& h) {
            auto acc = buf.get_access<sycl::access::mode::write>(h);
            h.parallel_for(
                sycl::nd_range<1>{N, N},
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                    const auto sg   = it.get_sub_group();
                    const auto part = batchlas::make_partition<P>(sg);
                    const uint32_t wi = static_cast<uint32_t>(it.get_global_linear_id());
                    // Value = absolute lane id
                    const uint32_t result = permute_group_by_xor(part, wi, 1u);
                    acc[wi] = result;
                });
        }).wait();
    }

    for (int wi = 0; wi < N; ++wi) {
        // mask=1 XOR within subgroup — result is value at lane (wi ^ 1)
        // BUT the absolute lane is what was stored as value, so the swap
        // partner within the chunk is (wi ^ 1) as long as 1 < P.
        if constexpr (P >= 2) {
            const uint32_t expected = static_cast<uint32_t>(wi ^ 1);
            EXPECT_EQ(out[wi], expected) << "P=" << P << " wi=" << wi << " xor_shuffle";
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel: test select_from_group (broadcast a specific lane's value)
//   Every lane reads lane 0's value (the leader).  Expected = base of chunk.
// ---------------------------------------------------------------------------
template <size_t P, size_t SG>
void test_select_from_group(sycl::queue& q) {
    constexpr int N = static_cast<int>(SG);
    std::vector<uint32_t> out(N, 0);

    {
        sycl::buffer<uint32_t, 1> buf(out.data(), N);
        q.submit([&](sycl::handler& h) {
            auto acc = buf.get_access<sycl::access::mode::write>(h);
            h.parallel_for(
                sycl::nd_range<1>{N, N},
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                    const auto sg   = it.get_sub_group();
                    const auto part = batchlas::make_partition<P>(sg);
                    const uint32_t wi = static_cast<uint32_t>(it.get_global_linear_id());
                    // Broadcast the absolute lane id of the chunk's lane 0
                    const uint32_t result = select_from_group(part, wi, 0u);
                    acc[wi] = result;
                });
        }).wait();
    }

    for (int wi = 0; wi < N; ++wi) {
        // Leader's absolute lane id = (wi / P) * P
        const uint32_t expected = static_cast<uint32_t>((wi / static_cast<int>(P)) * static_cast<int>(P));
        EXPECT_EQ(out[wi], expected) << "P=" << P << " wi=" << wi << " select";
    }
}

// ---------------------------------------------------------------------------
// Kernel: test shift_group_left (shift within sub-group)
//   shift_group_left(part, v, 1): lane i reads lane (i+1)'s value.
//   Last lane of each chunk reads from across the boundary; result undefined
//   by SYCL spec — we check only non-boundary lanes.
// ---------------------------------------------------------------------------
template <size_t P, size_t SG>
void test_shift_left(sycl::queue& q) {
    constexpr int N = static_cast<int>(SG);
    std::vector<uint32_t> out(N, 0);

    {
        sycl::buffer<uint32_t, 1> buf(out.data(), N);
        q.submit([&](sycl::handler& h) {
            auto acc = buf.get_access<sycl::access::mode::write>(h);
            h.parallel_for(
                sycl::nd_range<1>{N, N},
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                    const auto sg   = it.get_sub_group();
                    const auto part = batchlas::make_partition<P>(sg);
                    const uint32_t wi = static_cast<uint32_t>(it.get_global_linear_id());
                    const uint32_t result = shift_group_left(part, wi, 1u);
                    acc[wi] = result;
                });
        }).wait();
    }

    for (int wi = 0; wi < N; ++wi) {
        const bool is_last_in_chunk = ((wi + 1) % static_cast<int>(P) == 0);
        if (!is_last_in_chunk) {
            EXPECT_EQ(out[wi], static_cast<uint32_t>(wi + 1)) << "P=" << P << " wi=" << wi << " shift_left";
        }
    }
}

// ---------------------------------------------------------------------------
// Kernel: test in-partition butterfly reduction using permute_group_by_xor
//   Each chunk reduces its lanes' values (0..P-1 relative ids) to their sum.
//   Expected sum for chunk k = sum(k*P .. k*P + P - 1) = k*P*P + P*(P-1)/2.
// ---------------------------------------------------------------------------
template <size_t P, size_t SG>
void test_butterfly_reduce(sycl::queue& q) {
    constexpr int N = static_cast<int>(SG);
    std::vector<uint32_t> out(N, 0);

    {
        sycl::buffer<uint32_t, 1> buf(out.data(), N);
        q.submit([&](sycl::handler& h) {
            auto acc = buf.get_access<sycl::access::mode::write>(h);
            h.parallel_for(
                sycl::nd_range<1>{N, N},
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                    const auto sg   = it.get_sub_group();
                    const auto part = batchlas::make_partition<P>(sg);
                    const uint32_t wi = static_cast<uint32_t>(it.get_global_linear_id());
                    uint32_t v = wi;  // each lane starts with its absolute id
                    for (uint32_t offset = static_cast<uint32_t>(P) / 2u; offset > 0u; offset >>= 1u) {
                        v += permute_group_by_xor(part, v, offset);
                    }
                    acc[wi] = v;
                });
        }).wait();
    }

    for (int wi = 0; wi < N; ++wi) {
        // Sum of the chunk's absolute lane ids
        const int base = (wi / static_cast<int>(P)) * static_cast<int>(P);
        uint32_t expected = 0u;
        for (int j = 0; j < static_cast<int>(P); ++j) expected += static_cast<uint32_t>(base + j);
        EXPECT_EQ(out[wi], expected) << "P=" << P << " wi=" << wi << " butterfly";
    }
}

// ---------------------------------------------------------------------------
// Kernel: test group_barrier is a no-op (doesn't deadlock, pass-through)
//   Write a value, call group_barrier, read it back unchanged.
// ---------------------------------------------------------------------------
template <size_t P, size_t SG>
void test_group_barrier(sycl::queue& q) {
    constexpr int N = static_cast<int>(SG);
    std::vector<uint32_t> out(N, 0);

    {
        sycl::buffer<uint32_t, 1> buf(out.data(), N);
        q.submit([&](sycl::handler& h) {
            auto acc = buf.get_access<sycl::access::mode::write>(h);
            h.parallel_for(
                sycl::nd_range<1>{N, N},
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                    const auto sg   = it.get_sub_group();
                    const auto part = batchlas::make_partition<P>(sg);
                    const uint32_t wi = static_cast<uint32_t>(it.get_global_linear_id());
                    const uint32_t v = wi * 2u;
                    group_barrier(part);   // must not hang
                    group_barrier(part);
                    acc[wi] = v;
                });
        }).wait();
    }

    for (int wi = 0; wi < N; ++wi) {
        EXPECT_EQ(out[wi], static_cast<uint32_t>(wi * 2)) << "P=" << P << " wi=" << wi << " barrier";
    }
}

// ---------------------------------------------------------------------------
// Kernel: full end-to-end — replicate what the CTA kernels do:
//   per-partition sum-reduce of floating-point values using butterfly.
// ---------------------------------------------------------------------------
template <size_t P, size_t SG>
void test_float_butterfly_reduce(sycl::queue& q) {
    constexpr int N = static_cast<int>(SG);
    std::vector<float> input(N), out(N, 0.f);
    std::iota(input.begin(), input.end(), 1.f);  // 1, 2, ..., N

    {
        sycl::buffer<float, 1> b_in(input.data(), N);
        sycl::buffer<float, 1> b_out(out.data(), N);

        q.submit([&](sycl::handler& h) {
            auto a_in  = b_in.get_access<sycl::access::mode::read>(h);
            auto a_out = b_out.get_access<sycl::access::mode::write>(h);
            h.parallel_for(
                sycl::nd_range<1>{N, N},
                [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                    const auto sg   = it.get_sub_group();
                    const auto part = batchlas::make_partition<P>(sg);
                    const uint32_t wi = static_cast<uint32_t>(it.get_global_linear_id());
                    float v = a_in[wi];
                    for (uint32_t offset = static_cast<uint32_t>(P) / 2u; offset > 0u; offset >>= 1u) {
                        v += permute_group_by_xor(part, v, offset);
                    }
                    a_out[wi] = v;
                });
        }).wait();
    }

    for (int wi = 0; wi < N; ++wi) {
        // Input values for this chunk are consecutive integers
        const int base = (wi / static_cast<int>(P)) * static_cast<int>(P);
        float expected = 0.f;
        for (int j = 0; j < static_cast<int>(P); ++j) expected += static_cast<float>(base + j + 1);
        EXPECT_NEAR(out[wi], expected, 1e-4f) << "P=" << P << " wi=" << wi << " float_butterfly";
    }
}

// ---------------------------------------------------------------------------
// Fixture + typed test macros  
// ---------------------------------------------------------------------------
class SgCompatTest : public ::testing::Test {
protected:
    sycl::queue q = make_gpu_queue();
};

// Macro to reduce boilerplate — runs one sub-test for every P we care about
#define RUN_FOR_ALL_P(test_fn, SG)   \
    do {                              \
        test_fn<1,  SG>(q);           \
        test_fn<2,  SG>(q);           \
        test_fn<4,  SG>(q);           \
        test_fn<8,  SG>(q);           \
        test_fn<16, SG>(q);           \
        test_fn<32, SG>(q);           \
    } while (false)

// SG=32 is the smallest possible sub-group on both AMD gfx1200 and Intel GPU
TEST_F(SgCompatTest, PartitionIdentitySG32)  { RUN_FOR_ALL_P(test_partition_identity, 32); }
TEST_F(SgCompatTest, PermuteXorSG32)         { RUN_FOR_ALL_P(test_permute_xor,         32); }
TEST_F(SgCompatTest, SelectFromGroupSG32)    { RUN_FOR_ALL_P(test_select_from_group,   32); }
TEST_F(SgCompatTest, ShiftLeftSG32)          { RUN_FOR_ALL_P(test_shift_left,           32); }
TEST_F(SgCompatTest, ButterflyReduceSG32)    { RUN_FOR_ALL_P(test_butterfly_reduce,     32); }
TEST_F(SgCompatTest, GroupBarrierSG32)       { RUN_FOR_ALL_P(test_group_barrier,        32); }
TEST_F(SgCompatTest, FloatButterflyReduceSG32) { RUN_FOR_ALL_P(test_float_butterfly_reduce, 32); }

} // namespace
