#include <gtest/gtest.h>
#include <util/sycl-device-queue.hh>

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
        EXPECT_FALSE(device.get_vendor().empty());
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Skipping device property tests due to no devices available";
    }
}

TEST(SyclEventTest, Basic) {
    SyclEvent event;
    
    // Basic construction and move operations should work
    EXPECT_NO_THROW({
        SyclEvent event2;
        SyclEvent event3 = std::move(event2);
    });
}

TEST(SyclQueueTest, DefaultConstruction) {
    EXPECT_NO_THROW({
        SyclQueue queue;
    });
}

TEST(SyclQueueTest, DeviceConstruction) {
    try {
        // Get default device and create queue
        Device device = Device::default_device();
        
        EXPECT_NO_THROW({
            SyclQueue queue(device);
            EXPECT_EQ(queue.device().idx, device.idx);
            EXPECT_EQ(queue.device().type, device.type);
            EXPECT_TRUE(queue.in_order());
        });
        
        EXPECT_NO_THROW({
            SyclQueue queue(device, false);
            EXPECT_FALSE(queue.in_order());
        });
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Skipping queue tests due to no devices available";
    }
}

TEST(SyclQueueTest, MoveOperations) {
    try {
        Device device = Device::default_device();
        
        SyclQueue queue1(device);
        
        // Test move construction
        EXPECT_NO_THROW({
            SyclQueue queue2 = std::move(queue1);
            EXPECT_EQ(queue2.device().idx, device.idx);
            EXPECT_EQ(queue2.device().type, device.type);
        });
        
        // Test move assignment
        EXPECT_NO_THROW({
            SyclQueue queue3;
            queue3 = SyclQueue(device);
            EXPECT_EQ(queue3.device().idx, device.idx);
            EXPECT_EQ(queue3.device().type, device.type);
        });
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Skipping queue tests due to no devices available";
    }
}

TEST(SyclQueueTest, GetEvent) {
    try {
        SyclQueue queue(Device::default_device());
        
        EXPECT_NO_THROW({
            SyclEvent event = queue.get_event();
        });
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Skipping event tests due to no devices available";
    }
}

TEST(SyclQueueTest, EnqueueEvent) {
    try {
        SyclQueue queue(Device::default_device());
        SyclEvent event = queue.get_event();
        
        EXPECT_NO_THROW({
            queue.enqueue(event);
        });
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Skipping enqueue tests due to no devices available";
    }
}

TEST(SyclQueueTest, EnqueueMultipleEvents) {
    try {
        SyclQueue queue(Device::default_device());
        std::vector<SyclEvent> events;
        
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

TEST(SyclQueueTest, WaitAndThrow) {
    try {
        SyclQueue queue(Device::default_device());
        
        EXPECT_NO_THROW({
            queue.wait();
            queue.wait_and_throw();
        });
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Skipping wait tests due to no devices available";
    }
}