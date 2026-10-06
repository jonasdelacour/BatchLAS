#include <gtest/gtest.h>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/backend_config.h>

#include <cstdlib>
#include <set>
#include <string>
#include <vector>
#ifdef BATCHLAS_UTIL_TESTS_HAVE_CUDART
#include <cuda_runtime_api.h>
#endif

TEST(DeviceTest, DefaultConstruction) {
    Device device;
    EXPECT_EQ(device.idx, 0);
    EXPECT_EQ(device.type, DeviceType::HOST);
}

TEST(DeviceTest, IndexAndTypeConstruction) {
    Device device(1, DeviceType::CPU);
    EXPECT_EQ(device.idx, 1);
    EXPECT_EQ(device.type, DeviceType::CPU);
}

TEST(DeviceTest, GetDevices) {
    auto cpus = Device::get_devices(DeviceType::CPU);
    auto gpus = Device::get_devices(DeviceType::GPU);
    auto accelerators = Device::get_devices(DeviceType::ACCELERATOR);
    // Every entry must name its own device: get_devices() once returned idx 0 for
    // all of them, so gpus.at(1) silently was GPU 0.
    for (const auto* list : {&cpus, &gpus, &accelerators}) {
        for (size_t i = 0; i < list->size(); ++i) EXPECT_EQ((*list)[i].idx, i);
    }

    // We can't guarantee specific hardware is available on the test system
    // But we can at least check that the API returns something reasonable
    EXPECT_NO_THROW({
        auto default_device = Device::default_device();
    });
}

TEST(DeviceTest, StringConstruction) {
    // Test might fail if specific hardware isn't available, so we'll wrap in try/catch
    try {
        Device cpu_device("cpu");
        EXPECT_EQ(cpu_device.type, DeviceType::CPU);
    } catch (const std::runtime_error&) {
        // No CPU device available, that's ok for the test
    }
}

TEST(DeviceTest, DeviceProperties) {
    try {
        // Get default device, which should always be available
        Device device = Device::default_device();
        
        // Test getting various properties
        EXPECT_NO_THROW({
            size_t wg_size = device.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE);
            size_t compute_units = device.get_property(DeviceProperty::MAX_COMPUTE_UNITS);
        });
        
        // Device name and vendor should return something
        EXPECT_FALSE(device.get_name().empty());
        EXPECT_FALSE(device.get_vendor() == Vendor::OTHER);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Skipping device property tests due to no devices available";
    }
}

TEST(EventTest, Basic) {
    Event event;
    
    // Basic construction and move operations should work
    EXPECT_NO_THROW({
        Event event2;
        Event event3 = std::move(event2);
    });
}

TEST(QueueTest, DefaultConstruction) {
    EXPECT_NO_THROW({
        Queue queue;
    });
}

TEST(QueueTest, DeviceConstruction) {
    try {
        // Get default device and create queue
        Device device = Device::default_device();
        
        EXPECT_NO_THROW({
            Queue queue(device);
            EXPECT_EQ(queue.device().idx, device.idx);
            EXPECT_EQ(queue.device().type, device.type);
            EXPECT_TRUE(queue.in_order());
        });
        
        EXPECT_NO_THROW({
            Queue queue(device, false);
            EXPECT_FALSE(queue.in_order());
        });
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Skipping queue tests due to no devices available";
    }
}

TEST(QueueTest, MoveOperations) {
    try {
        Device device = Device::default_device();
        
        Queue queue1(device);
        
        // Test move construction
        EXPECT_NO_THROW({
            Queue queue2 = std::move(queue1);
            EXPECT_EQ(queue2.device().idx, device.idx);
            EXPECT_EQ(queue2.device().type, device.type);
        });
        
        // Test move assignment
        EXPECT_NO_THROW({
            Queue queue3;
            queue3 = Queue(device);
            EXPECT_EQ(queue3.device().idx, device.idx);
            EXPECT_EQ(queue3.device().type, device.type);
        });
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Skipping queue tests due to no devices available";
    }
}

TEST(QueueTest, GetEvent) {
    try {
        Queue queue(Device::default_device());
        
        EXPECT_NO_THROW({
            Event event = queue.get_event();
        });
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Skipping event tests due to no devices available";
    }
}

TEST(QueueTest, EnqueueEvent) {
    try {
        Queue queue(Device::default_device());
        Event event = queue.get_event();
        
        EXPECT_NO_THROW({
            queue.enqueue(event);
        });
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Skipping enqueue tests due to no devices available";
    }
}

TEST(QueueTest, EnqueueMultipleEvents) {
    try {
        Queue queue(Device::default_device());
        std::vector<Event> events;
        
        for (int i = 0; i < 3; i++) {
            events.push_back(queue.get_event());
        }
        
        EXPECT_NO_THROW({
            queue.enqueue(events);
        });
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Skipping enqueue tests due to no devices available";
    }
}

TEST(QueueTest, WaitAndThrow) {
    try {
        Queue queue(Device::default_device());

        EXPECT_NO_THROW({
            queue.wait();
            queue.wait_and_throw();
        });
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Skipping wait tests due to no devices available";
    }
}

// Pins the contract that every kernel-geometry knob in src/extensions now
// depends on. These call sites each used to carry their own atoi-plus-`> 0`
// parser; the shared helper only preserves them if it agrees on the whole input
// space, and the interesting half of that space (unset, empty, non-positive,
// unparseable, trailing junk) is exactly what no existing test reaches -- the
// knobs the suite already sets are all positive integers, where every candidate
// parser agrees trivially.
TEST(EnvHelpers, PositiveIntOrClampsAndFallsBack) {
    const char* kKey = "BATCHLAS_TEST_ENV_POSITIVE_INT";

    {   // unset
        batchlas::ScopedEnvVar v(kKey, nullptr);
        EXPECT_EQ(batchlas::env_positive_int_or(kKey, 7), 7);
    }
    {   // empty string: stoi throws, so fallback
        batchlas::ScopedEnvVar v(kKey, "");
        EXPECT_EQ(batchlas::env_positive_int_or(kKey, 7), 7);
    }
    {   // zero and negatives are "meaningless geometry", i.e. unset
        batchlas::ScopedEnvVar v(kKey, "0");
        EXPECT_EQ(batchlas::env_positive_int_or(kKey, 7), 7);
    }
    {
        batchlas::ScopedEnvVar v(kKey, "-3");
        EXPECT_EQ(batchlas::env_positive_int_or(kKey, 7), 7);
    }
    {   // unparseable falls back rather than silently reading as 0
        batchlas::ScopedEnvVar v(kKey, "abc");
        EXPECT_EQ(batchlas::env_positive_int_or(kKey, 7), 7);
    }
    {   // stoi takes the leading integer and ignores the tail -- documented here
        // because atoi did the same, so routing the old call sites through this
        // did not change them.
        batchlas::ScopedEnvVar v(kKey, "8junk");
        EXPECT_EQ(batchlas::env_positive_int_or(kKey, 7), 8);
    }
    {
        batchlas::ScopedEnvVar v(kKey, "16");
        EXPECT_EQ(batchlas::env_positive_int_or(kKey, 7), 16);
    }
}

// env_truthy/env_falsy take the VALUE, not the name, and an unset variable is
// neither -- that is what lets a caller tell "forced off" from "not specified".
TEST(EnvHelpers, TruthyAndFalsyAreNotComplements) {
    EXPECT_FALSE(batchlas::env_truthy(nullptr));
    EXPECT_FALSE(batchlas::env_falsy(nullptr));

    EXPECT_TRUE(batchlas::env_truthy("1"));
    EXPECT_TRUE(batchlas::env_truthy("true"));
    EXPECT_TRUE(batchlas::env_truthy("ON"));
    EXPECT_FALSE(batchlas::env_truthy("True"));  // exact spellings only

    EXPECT_TRUE(batchlas::env_falsy("0"));
    EXPECT_TRUE(batchlas::env_falsy("off"));
    EXPECT_FALSE(batchlas::env_falsy("False"));  // exact spellings only
}

// Per-architecture routing keys on this value, and 0 must mean "not CUDA" so every
// other device keeps the as-measured (sm_89) windows.
// evidence: docs/perf/blackwell.md#compute-capability-key
TEST(DeviceTest, CudaComputeCapabilityIsZeroOffCuda) {
    for (const Device& d : Device::get_devices(DeviceType::CPU)) {
        EXPECT_EQ(d.cuda_compute_capability(), 0) << d.get_name();
    }
    for (const Device& d : Device::get_devices(DeviceType::GPU)) {
        if (d.get_vendor() != Vendor::NVIDIA) EXPECT_EQ(d.cuda_compute_capability(), 0) << d.get_name();
    }
}

TEST(DeviceTest, CudaComputeCapabilityOnNvidiaGpu) {
    std::vector<Device> nv;
    for (const Device& d : Device::get_devices(DeviceType::GPU)) {
        if (d.get_vendor() == Vendor::NVIDIA) nv.push_back(d);
    }
    if (nv.empty()) GTEST_SKIP() << "no NVIDIA GPU visible";
#ifdef BATCHLAS_UTIL_TESTS_HAVE_CUDART
    // Independent oracle: the CUDA runtime, not the SYCL version string we parse.
    // Paired per device (the CUDA adapter enumerates in CUDA ordinal order, and the
    // name check proves the pairing), so on a mixed-GPU box one device reporting
    // another's cc still fails.
    int count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
    ASSERT_EQ(static_cast<size_t>(count), nv.size()) << "SYCL and CUDA see different NVIDIA GPUs";
#endif
    for (size_t i = 0; i < nv.size(); ++i) {
        const Device& d = nv[i];
        const int cc = d.cuda_compute_capability();
#if BATCHLAS_HAS_CUDA_BACKEND
        EXPECT_GE(cc, 50) << d.get_name();
        EXPECT_LT(cc, 1000) << d.get_name();
#endif
        EXPECT_EQ(d.cuda_compute_capability(), cc) << "memoized value changed";
#ifdef BATCHLAS_UTIL_TESTS_HAVE_CUDART
        cudaDeviceProp prop{};
        ASSERT_EQ(cudaGetDeviceProperties(&prop, static_cast<int>(i)), cudaSuccess);
        ASSERT_EQ(d.get_name(), std::string(prop.name)) << "SYCL/CUDA ordinal pairing broke at " << i;
        EXPECT_EQ(cc, prop.major * 10 + prop.minor) << d.get_name() << " (CUDA ordinal " << i << ")";
#endif
    }
}

// Straddle both edges of the family window: sm_120 and sm_121 are in; sm_89,
// sm_100 (datacenter Blackwell) and a future sm_130 are out.
TEST(DeviceTest, Sm120FamilyWindow) {
    using batchlas::is_sm120_family;
    static_assert(!is_sm120_family(0) && !is_sm120_family(89) && !is_sm120_family(100));
    EXPECT_FALSE(is_sm120_family(119));
    EXPECT_TRUE(is_sm120_family(120));
    EXPECT_TRUE(is_sm120_family(121));
    EXPECT_TRUE(is_sm120_family(129));
    EXPECT_FALSE(is_sm120_family(130));
}