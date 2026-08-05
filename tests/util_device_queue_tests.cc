#include <gtest/gtest.h>
#include <util/env.hh>
#include <util/sycl-device-queue.hh>

#include <cstdlib>
#include <string>

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

namespace {
// setenv/unsetenv rather than putenv: putenv keeps the caller's buffer alive in
// the environment, which is a lifetime trap in a test that restores on scope
// exit.
struct ScopedEnvVar {
    std::string key;
    std::string prev;
    bool had_prev;

    ScopedEnvVar(const char* k, const char* v) : key(k) {
        const char* old = std::getenv(k);
        had_prev = old != nullptr;
        if (had_prev) prev = old;
        if (v) setenv(k, v, 1); else unsetenv(k);
    }
    ~ScopedEnvVar() {
        if (had_prev) setenv(key.c_str(), prev.c_str(), 1); else unsetenv(key.c_str());
    }
};
}  // namespace

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
        ScopedEnvVar v(kKey, nullptr);
        EXPECT_EQ(batchlas::env_positive_int_or(kKey, 7), 7);
    }
    {   // empty string: stoi throws, so fallback
        ScopedEnvVar v(kKey, "");
        EXPECT_EQ(batchlas::env_positive_int_or(kKey, 7), 7);
    }
    {   // zero and negatives are "meaningless geometry", i.e. unset
        ScopedEnvVar v(kKey, "0");
        EXPECT_EQ(batchlas::env_positive_int_or(kKey, 7), 7);
    }
    {
        ScopedEnvVar v(kKey, "-3");
        EXPECT_EQ(batchlas::env_positive_int_or(kKey, 7), 7);
    }
    {   // unparseable falls back rather than silently reading as 0
        ScopedEnvVar v(kKey, "abc");
        EXPECT_EQ(batchlas::env_positive_int_or(kKey, 7), 7);
    }
    {   // stoi takes the leading integer and ignores the tail -- documented here
        // because atoi did the same, so routing the old call sites through this
        // did not change them.
        ScopedEnvVar v(kKey, "8junk");
        EXPECT_EQ(batchlas::env_positive_int_or(kKey, 7), 8);
    }
    {
        ScopedEnvVar v(kKey, "16");
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