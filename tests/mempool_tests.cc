#include <gtest/gtest.h>
#include <batchlas/util/mempool.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/util/sycl-span.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <vector>
#include <complex>
#include <memory>
#include <cstring>
#include <algorithm>
#include <thread>
#include <stdexcept>

class BumpAllocatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        queue = std::make_unique<Queue>();
        device = queue->device();
        // Create a large buffer for testing
        buffer_size = 1024 * 1024; // 1MB
        buffer = std::make_unique<std::byte[]>(buffer_size);
        std::memset(buffer.get(), 0, buffer_size);
    }

    std::unique_ptr<Queue> queue;
    Device device;
    std::unique_ptr<std::byte[]> buffer;
    size_t buffer_size;
};

// Test basic construction and simple allocations
TEST_F(BumpAllocatorTest, BasicConstruction) {
    BumpAllocator pool(buffer.get(), buffer_size);
    
    // Test small allocation
    auto span1 = pool.allocate<int>(device, 10);
    EXPECT_EQ(span1.size(), 10);
    EXPECT_NE(span1.data(), nullptr);
    
    // Test allocation after first one
    auto span2 = pool.allocate<float>(device, 5);
    EXPECT_EQ(span2.size(), 5);
    EXPECT_NE(span2.data(), nullptr);
    
    // Ensure spans don't overlap
    EXPECT_TRUE(reinterpret_cast<char*>(span2.data()) >= 
                reinterpret_cast<char*>(span1.data()) + span1.size() * sizeof(int));
}

// Test construction from Span
TEST_F(BumpAllocatorTest, ConstructionFromSpan) {
    Span<std::byte> span(buffer.get(), buffer_size);
    BumpAllocator pool(span);
    
    auto allocated = pool.allocate<double>(device, 100);
    EXPECT_EQ(allocated.size(), 100);
    EXPECT_NE(allocated.data(), nullptr);
}

// Test zero-size allocations
TEST_F(BumpAllocatorTest, ZeroSizeAllocations) {
    BumpAllocator pool(buffer.get(), buffer_size);
    
    auto span = pool.allocate<int>(device, 0);
    EXPECT_EQ(span.size(), 0);
    EXPECT_EQ(span.data(), nullptr);
    
    // Ensure we can still allocate after zero-size allocation
    auto span2 = pool.allocate<float>(device, 10);
    EXPECT_EQ(span2.size(), 10);
    EXPECT_NE(span2.data(), nullptr);
}

// Test alignment requirements
TEST_F(BumpAllocatorTest, AlignmentRequirements) {
    BumpAllocator pool(buffer.get(), buffer_size);
    
    // Test different types with different alignment requirements
    auto char_span = pool.allocate<char>(device, 1);
    auto int_span = pool.allocate<int>(device, 1);
    auto double_span = pool.allocate<double>(device, 1);
    auto complex_span = pool.allocate<std::complex<double>>(device, 1);
    
    // Check alignment based on actual BumpAllocator implementation
    auto device_align = std::max((size_t)16, device.get_property(DeviceProperty::MEM_BASE_ADDR_ALIGN) / 8);
    auto char_align = std::max(device_align, static_cast<std::uintptr_t>(alignof(char)));
    auto int_align = std::max(device_align, static_cast<std::uintptr_t>(alignof(int)));
    auto double_align = std::max(device_align, static_cast<std::uintptr_t>(alignof(double)));
    auto complex_align = std::max(device_align, static_cast<std::uintptr_t>(alignof(std::complex<double>)));
    
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(char_span.data()) % char_align, 0);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(int_span.data()) % int_align, 0);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(double_span.data()) % double_align, 0);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(complex_span.data()) % complex_align, 0);
}

// Test allocation_size static methods
TEST_F(BumpAllocatorTest, AllocationSizeCalculation) {
    // First let's understand the device alignment requirements
    auto device_align = std::max((size_t)16, device.get_property(DeviceProperty::MEM_BASE_ADDR_ALIGN) / 8);
    std::cout << "Device alignment: " << device_align << " bytes" << std::endl;
    
    // Test allocation_size with device
    auto size1 = BumpAllocator::allocation_size<int>(device, 100);
    auto size2 = BumpAllocator::allocation_size<double>(device, 100);
    auto size3 = BumpAllocator::allocation_size<std::complex<float>>(device, 100);
    
    std::cout << "Size for 100 ints: " << size1 << " bytes" << std::endl;
    std::cout << "Size for 100 doubles: " << size2 << " bytes" << std::endl;
    std::cout << "Size for 100 complex<float>: " << size3 << " bytes" << std::endl;
    
    EXPECT_GE(size1, 100 * sizeof(int));
    EXPECT_GE(size2, 100 * sizeof(double));
    EXPECT_GE(size3, 100 * sizeof(std::complex<float>));
    
    // Verify sizes are aligned to proper boundaries
    auto int_align = std::max(device_align, static_cast<std::uintptr_t>(alignof(int)));
    auto double_align = std::max(device_align, static_cast<std::uintptr_t>(alignof(double)));
    auto complex_align = std::max(device_align, static_cast<std::uintptr_t>(alignof(std::complex<float>)));
    
    EXPECT_EQ(size1 % int_align, 0);
    EXPECT_EQ(size2 % double_align, 0);
    EXPECT_EQ(size3 % complex_align, 0);
    
    // Test allocation_size with queue
    auto size1_q = BumpAllocator::allocation_size<int>(*queue, 100);
    auto size2_q = BumpAllocator::allocation_size<double>(*queue, 100);
    auto size3_q = BumpAllocator::allocation_size<std::complex<float>>(*queue, 100);
    
    EXPECT_EQ(size1, size1_q);
    EXPECT_EQ(size2, size2_q);
    EXPECT_EQ(size3, size3_q);
    
    // Test zero size
    EXPECT_EQ(BumpAllocator::allocation_size<int>(device, 0), 0);
}

// Test buffer alignment and exhaustion
TEST_F(BumpAllocatorTest, BufferExhaustion) {
    // First, let's understand the allocation size behavior
    auto device_align = std::max((size_t)16, device.get_property(DeviceProperty::MEM_BASE_ADDR_ALIGN) / 8);
    std::cout << "Device alignment: " << device_align << " bytes" << std::endl;
    
    // Check how much space a single int actually needs
    auto single_int_size = BumpAllocator::allocation_size<int>(device, 1);
    std::cout << "Size for 1 int: " << single_int_size << " bytes" << std::endl;
    
    // The size depends on alignment requirements, not a fixed cache line
    auto int_align = std::max(device_align, static_cast<std::uintptr_t>(alignof(int)));
    EXPECT_EQ(single_int_size, int_align);
    
    // Test with a properly aligned buffer that's exactly one allocation unit
    void* aligned_ptr = std::aligned_alloc(int_align, int_align);
    ASSERT_NE(aligned_ptr, nullptr);
    
    BumpAllocator aligned_pool(reinterpret_cast<std::byte*>(aligned_ptr), int_align);
    
    // First allocation should succeed since buffer is already aligned
    auto span1 = aligned_pool.allocate<int>(device, 1);
    EXPECT_EQ(span1.size(), 1);
    EXPECT_NE(span1.data(), nullptr);
    
    // Second allocation will fail because no space left after first allocation
    EXPECT_THROW({
        aligned_pool.allocate<int>(device, 1);
    }, std::runtime_error);
    
    std::free(aligned_ptr);
    
    // Test with buffer smaller than alignment requirement
    const size_t small_size = 4;
    auto small_buffer = std::make_unique<std::byte[]>(small_size);
    BumpAllocator small_pool(small_buffer.get(), small_size);

    // This should fail because we need int_align bytes but only have 4 bytes
    EXPECT_THROW({
        small_pool.allocate<int>(device, 1);
    }, std::runtime_error);
    
    // Test with larger buffer to show multiple allocations can fit
    const size_t large_size = 8 * int_align; // Should fit multiple allocations
    void* large_aligned_ptr = std::aligned_alloc(int_align, large_size);
    ASSERT_NE(large_aligned_ptr, nullptr);
    
    BumpAllocator large_pool(reinterpret_cast<std::byte*>(large_aligned_ptr), large_size);
    
    // Multiple allocations should fit
    std::vector<Span<int>> spans;
    for (int i = 0; i < 1000; ++i) { // Try many allocations
        try {
            auto span = large_pool.allocate<int>(device, 1);
            spans.push_back(span);
        } catch (const std::runtime_error& e) {
            std::cout << "Ran out of space after " << spans.size() << " int allocations" << std::endl;
            break;
        }
    }
    
    // Should have been able to fit multiple ints (not the 4KB assumption)
    EXPECT_GT(spans.size(), 1); // At least 2 ints should fit in 8 * alignment
    
    std::free(large_aligned_ptr);
    
    // Test with allocation larger than buffer
    void* small_aligned_ptr = std::aligned_alloc(int_align, int_align);
    ASSERT_NE(small_aligned_ptr, nullptr);
    
    BumpAllocator large_alloc_pool(reinterpret_cast<std::byte*>(small_aligned_ptr), int_align);
    
    // Try to allocate more than the buffer size can hold
    size_t max_ints = int_align / sizeof(int);
    EXPECT_THROW({
        large_alloc_pool.allocate<int>(device, max_ints + 1); // One more than can fit
    }, std::runtime_error);
    
    std::free(small_aligned_ptr);
}

// Test device allocation behavior
TEST_F(BumpAllocatorTest, DeviceAllocationBehavior) {
    auto device_align = std::max((size_t)16, device.get_property(DeviceProperty::MEM_BASE_ADDR_ALIGN) / 8);
    
    // allocation_size returns the space needed including alignment
    auto char_align = std::max(device_align, static_cast<std::uintptr_t>(alignof(char)));
    auto int_align = std::max(device_align, static_cast<std::uintptr_t>(alignof(int)));
    auto double_align = std::max(device_align, static_cast<std::uintptr_t>(alignof(double)));
    
    EXPECT_EQ(BumpAllocator::allocation_size<char>(device, 1), char_align);
    EXPECT_EQ(BumpAllocator::allocation_size<int>(device, 1), int_align);
    EXPECT_EQ(BumpAllocator::allocation_size<double>(device, 1), double_align);
    
    // For larger allocations, size is rounded up to alignment boundary
    size_t large_int_count = (2 * int_align) / sizeof(int); // Should require 2 * alignment
    EXPECT_EQ(BumpAllocator::allocation_size<int>(device, large_int_count), 2 * int_align);
    
    // The actual allocator allocates with proper alignment
    // Use a larger buffer to accommodate multiple allocations with alignment
    const size_t buffer_size = 16 * device_align; // Much larger buffer
    void* aligned_ptr = std::aligned_alloc(device_align, buffer_size);
    ASSERT_NE(aligned_ptr, nullptr);
    
    BumpAllocator pool(reinterpret_cast<std::byte*>(aligned_ptr), buffer_size);
    
    // Each allocation will be aligned to its requirement
    auto span1 = pool.allocate<int>(device, 1);     // 4 bytes data, aligned to int_align
    auto span2 = pool.allocate<int>(device, 1);     // 4 bytes data, aligned to int_align  
    auto span3 = pool.allocate<double>(device, 1);  // 8 bytes data, aligned to double_align
    auto span4 = pool.allocate<char>(device, 100);  // 100 bytes data, aligned to char_align
    
    EXPECT_EQ(span1.size(), 1);
    EXPECT_EQ(span2.size(), 1);
    EXPECT_EQ(span3.size(), 1);
    EXPECT_EQ(span4.size(), 100);
    
    // Verify proper alignment for each allocation
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(span1.data()) % int_align, 0);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(span2.data()) % int_align, 0);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(span3.data()) % double_align, 0);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(span4.data()) % char_align, 0);
    
    std::free(aligned_ptr);
}

// Test the relationship between allocation_size and actual allocations
TEST_F(BumpAllocatorTest, AllocationSizeVsActualBehavior) {
    auto device_align = std::max((size_t)16, device.get_property(DeviceProperty::MEM_BASE_ADDR_ALIGN) / 8);
    
    // allocation_size is used for planning - it tells you the worst-case
    // space needed if this was the only allocation
    auto int_align = std::max(device_align, static_cast<std::uintptr_t>(alignof(int)));
    auto predicted_size = BumpAllocator::allocation_size<int>(device, 1);
    EXPECT_EQ(predicted_size, int_align);
    
    // But actual allocations can be more efficient by packing multiple
    // allocations into the same aligned region
    void* aligned_ptr = std::aligned_alloc(int_align, 4 * int_align);
    ASSERT_NE(aligned_ptr, nullptr);
    
    BumpAllocator pool(reinterpret_cast<std::byte*>(aligned_ptr), 4 * int_align);
    
    // Track how much space is actually consumed
    std::byte* start_ptr = reinterpret_cast<std::byte*>(aligned_ptr);
    
    auto span1 = pool.allocate<int>(device, 1);
    auto span2 = pool.allocate<int>(device, 1);
    
    // The allocations should be aligned, but the distance between them
    // depends on the alignment requirements
    EXPECT_GE(reinterpret_cast<std::byte*>(span2.data()) - reinterpret_cast<std::byte*>(span1.data()), sizeof(int));
    
    // Both allocations should fit within the buffer since we allocated plenty of space
    EXPECT_LT(reinterpret_cast<std::byte*>(span2.data()) + sizeof(int) - start_ptr, 4 * int_align);
    
    std::free(aligned_ptr);
    
    // This explains why allocation_size is conservative - it's for planning
    // buffer sizes where you don't know the exact allocation pattern
}

// Test complex real-world scenario similar to syevx usage
TEST_F(BumpAllocatorTest, SyevxLikeUsagePattern) {
    // Simulate syevx buffer size calculation pattern
    const size_t n = 32;
    const size_t neigs = 8;
    const size_t extra_directions = 4;
    const size_t block_vectors = neigs + extra_directions;
    const size_t batch_size = 4;
    
    // Calculate required buffer size (similar to syevx_buffer_size)
    size_t total_size = 0;
    
    // Main data arrays
    total_size += BumpAllocator::allocation_size<float>(device, n * block_vectors * 3 * batch_size) * 4; // Sdata, ASdata, S_newdata, Stempdata
    total_size += BumpAllocator::allocation_size<float>(device, block_vectors * block_vectors * 3 * 3 * batch_size); // StASdata
    total_size += BumpAllocator::allocation_size<float>(device, block_vectors * block_vectors * 3 * batch_size); // C_pdata
    total_size += BumpAllocator::allocation_size<float>(device, block_vectors * 3 * batch_size); // lambdas
    total_size += BumpAllocator::allocation_size<float>(device, neigs * batch_size) * 2; // residuals + best_residuals
    
    // Pointer arrays for batched operations
    if (batch_size > 1) {
        total_size += BumpAllocator::allocation_size<float*>(device, batch_size) * 21;
    }
    
    // Workspace for sub-operations (simplified)
    total_size += BumpAllocator::allocation_size<std::byte>(device, 1024); // syev workspace
    total_size += BumpAllocator::allocation_size<std::byte>(device, 2048); // ortho workspace
    
    // Create allocator with calculated size.
    // Add (device_align - 1) padding bytes so the BumpAllocator can absorb any initial
    // misalignment of the allocation returned by new[] (which only guarantees
    // alignof(std::max_align_t), not the device's MEM_BASE_ADDR_ALIGN).
    auto device_align = std::max((size_t)16, device.get_property(DeviceProperty::MEM_BASE_ADDR_ALIGN) / 8);
    auto large_buffer = std::make_unique<std::byte[]>(total_size + device_align - 1);
    BumpAllocator pool(large_buffer.get(), total_size + device_align - 1);
    
    // Perform allocations in the same order as syevx
    auto Sdata = pool.allocate<float>(device, n * block_vectors * 3 * batch_size);
    auto ASdata = pool.allocate<float>(device, n * block_vectors * 3 * batch_size);
    auto S_newdata = pool.allocate<float>(device, n * block_vectors * 3 * batch_size);
    auto Stempdata = pool.allocate<float>(device, n * block_vectors * 3 * batch_size);
    auto StASdata = pool.allocate<float>(device, block_vectors * block_vectors * 3 * 3 * batch_size);
    auto C_pdata = pool.allocate<float>(device, block_vectors * block_vectors * 3 * batch_size);
    auto lambdas = pool.allocate<float>(device, block_vectors * 3 * batch_size);
    auto residuals = pool.allocate<float>(device, neigs * batch_size);
    auto best_residuals = pool.allocate<float>(device, neigs * batch_size);
    
    // Pointer arrays
    std::vector<Span<float*>> pointer_arrays;
    if (batch_size > 1) {
        for (int i = 0; i < 21; ++i) {
            pointer_arrays.push_back(pool.allocate<float*>(device, batch_size));
        }
    }
    
    // Workspaces
    auto syev_workspace = pool.allocate<std::byte>(device, 1024);
    auto ortho_workspace = pool.allocate<std::byte>(device, 2048);
    
    // Verify all allocations succeeded
    EXPECT_EQ(Sdata.size(), n * block_vectors * 3 * batch_size);
    EXPECT_EQ(ASdata.size(), n * block_vectors * 3 * batch_size);
    EXPECT_EQ(S_newdata.size(), n * block_vectors * 3 * batch_size);
    EXPECT_EQ(Stempdata.size(), n * block_vectors * 3 * batch_size);
    EXPECT_EQ(StASdata.size(), block_vectors * block_vectors * 3 * 3 * batch_size);
    EXPECT_EQ(C_pdata.size(), block_vectors * block_vectors * 3 * batch_size);
    EXPECT_EQ(lambdas.size(), block_vectors * 3 * batch_size);
    EXPECT_EQ(residuals.size(), neigs * batch_size);
    EXPECT_EQ(best_residuals.size(), neigs * batch_size);
    EXPECT_EQ(syev_workspace.size(), 1024);
    EXPECT_EQ(ortho_workspace.size(), 2048);
    
    // Verify no overlaps between major allocations
    EXPECT_TRUE(reinterpret_cast<char*>(ASdata.data()) >= 
                reinterpret_cast<char*>(Sdata.data()) + Sdata.size() * sizeof(float));
    EXPECT_TRUE(reinterpret_cast<char*>(S_newdata.data()) >= 
                reinterpret_cast<char*>(ASdata.data()) + ASdata.size() * sizeof(float));
}

// Test disjoint allocation_size and allocate calls (key test case)
TEST_F(BumpAllocatorTest, DisjointAllocationSizeAndAllocate) {
    // This test specifically addresses the scenario where allocation_size
    // is called in one context and allocate is called in another
    
    const size_t n = 64;
    const size_t batch = 8;
    
    // Phase 1: Calculate sizes in "planning" phase (like buffer size calculation functions)
    size_t int_array_size = BumpAllocator::allocation_size<int>(device, n * batch);
    size_t float_array_size = BumpAllocator::allocation_size<float>(device, n * n * batch);
    size_t double_array_size = BumpAllocator::allocation_size<double>(device, n * batch);
    size_t complex_array_size = BumpAllocator::allocation_size<std::complex<float>>(device, n * n * batch);
    size_t pointer_array_size = BumpAllocator::allocation_size<float*>(device, batch);
    size_t workspace_size = BumpAllocator::allocation_size<std::byte>(device, 4096);
    
    size_t total_calculated_size = int_array_size + float_array_size + double_array_size + 
                                   complex_array_size + pointer_array_size + workspace_size;
    
    // Phase 2: Create allocator with calculated size in different context.
    // Add (device_align - 1) padding bytes so the BumpAllocator can absorb any initial
    // misalignment from new[] (which only guarantees alignof(std::max_align_t)).
    auto device_align = std::max((size_t)16, device.get_property(DeviceProperty::MEM_BASE_ADDR_ALIGN) / 8);
    auto planned_buffer = std::make_unique<std::byte[]>(total_calculated_size + device_align - 1);
    BumpAllocator pool(planned_buffer.get(), total_calculated_size + device_align - 1);
    
    // Phase 3: Perform actual allocations in execution phase
    auto int_array = pool.allocate<int>(device, n * batch);
    auto float_array = pool.allocate<float>(device, n * n * batch);
    auto double_array = pool.allocate<double>(device, n * batch);
    auto complex_array = pool.allocate<std::complex<float>>(device, n * n * batch);
    auto pointer_array = pool.allocate<float*>(device, batch);
    auto workspace = pool.allocate<std::byte>(device, 4096);
    
    // Verify all allocations succeeded with correct sizes
    EXPECT_EQ(int_array.size(), n * batch);
    EXPECT_EQ(float_array.size(), n * n * batch);
    EXPECT_EQ(double_array.size(), n * batch);
    EXPECT_EQ(complex_array.size(), n * n * batch);
    EXPECT_EQ(pointer_array.size(), batch);
    EXPECT_EQ(workspace.size(), 4096);
    
    // Verify proper alignment
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(int_array.data()) % 
              std::max(device_align, static_cast<std::uintptr_t>(alignof(int))), 0);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(float_array.data()) % 
              std::max(device_align, static_cast<std::uintptr_t>(alignof(float))), 0);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(double_array.data()) % 
              std::max(device_align, static_cast<std::uintptr_t>(alignof(double))), 0);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(complex_array.data()) % 
              std::max(device_align, static_cast<std::uintptr_t>(alignof(std::complex<float>))), 0);
}

// Test edge cases and error conditions
TEST_F(BumpAllocatorTest, EdgeCasesAndErrors) {
    auto device_align = std::max((size_t)16, device.get_property(DeviceProperty::MEM_BASE_ADDR_ALIGN) / 8);
    
    // Test with properly aligned small buffer
    const size_t small_size = 128; // Should work if aligned
    void* aligned_ptr = std::aligned_alloc(device_align, small_size + device_align);
    ASSERT_NE(aligned_ptr, nullptr);
    
    BumpAllocator aligned_pool(reinterpret_cast<std::byte*>(aligned_ptr), small_size + device_align);
    
    // Should work with aligned buffer
    auto aligned_span = aligned_pool.allocate<double>(device, 1);
    EXPECT_EQ(aligned_span.size(), 1);
    auto double_align = std::max(device_align, static_cast<std::uintptr_t>(alignof(double)));
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(aligned_span.data()) % double_align, 0);
    
    std::free(aligned_ptr);
    
    // Test with intentionally unaligned buffer start but sufficient total size
    const size_t large_size = 16384; // 16KB
    auto unaligned_buffer = std::make_unique<char[]>(large_size + device_align);
    char* unaligned_ptr = unaligned_buffer.get() + 1; // Intentionally misalign by 1 byte
    
    BumpAllocator unaligned_pool(unaligned_ptr, large_size);
    
    // Should still work because allocator will find aligned address within buffer
    auto unaligned_test_span = unaligned_pool.allocate<double>(device, 1);
    EXPECT_EQ(unaligned_test_span.size(), 1);
    // Result should still be aligned even though buffer start wasn't
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(unaligned_test_span.data()) % double_align, 0);
}

// Test allocation size consistency
TEST_F(BumpAllocatorTest, AllocationSizeConsistency) {
    // Test that allocation_size and actual allocation consume the same amount
    const size_t test_size = 100;
    
    // Create two identical pools
    auto buffer1 = std::make_unique<std::byte[]>(buffer_size);
    auto buffer2 = std::make_unique<std::byte[]>(buffer_size);
    BumpAllocator pool1(buffer1.get(), buffer_size);
    BumpAllocator pool2(buffer2.get(), buffer_size);
    
    // Get predicted size
    size_t predicted_size = BumpAllocator::allocation_size<float>(device, test_size);
    
    // Allocate in first pool and measure consumed space
    void* start_ptr = buffer1.get();
    auto span1 = pool1.allocate<float>(device, test_size);
    
    // Allocate something else to see where the next allocation would go
    auto span2 = pool1.allocate<char>(device, 1);
    
    // Calculate actual consumed space
    size_t actual_consumed = reinterpret_cast<char*>(span2.data()) - 
                            reinterpret_cast<char*>(start_ptr);
    
    // The actual consumed space should be at least the predicted size
    // (it might be more due to alignment of the second allocation)
    EXPECT_GE(actual_consumed, predicted_size);
    
    // Verify the span has correct size
    EXPECT_EQ(span1.size(), test_size);
}

// Test pointer array allocations (common in batched operations)
TEST_F(BumpAllocatorTest, PointerArrayAllocations) {
    BumpAllocator pool(buffer.get(), buffer_size);
    
    const size_t batch_size = 8;
    
    // Allocate pointer arrays like in syevx
    auto ptr_array1 = pool.allocate<float*>(device, batch_size);
    auto ptr_array2 = pool.allocate<double*>(device, batch_size);
    auto ptr_array3 = pool.allocate<std::complex<float>*>(device, batch_size);
    
    EXPECT_EQ(ptr_array1.size(), batch_size);
    EXPECT_EQ(ptr_array2.size(), batch_size);
    EXPECT_EQ(ptr_array3.size(), batch_size);
    
    // Verify alignment based on actual requirements
    auto device_align = std::max((size_t)16, device.get_property(DeviceProperty::MEM_BASE_ADDR_ALIGN) / 8);
    auto ptr_align = std::max(device_align, static_cast<std::uintptr_t>(alignof(float*)));
    
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr_array1.data()) % ptr_align, 0);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr_array2.data()) % ptr_align, 0);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr_array3.data()) % ptr_align, 0);
}

// Test workspace allocation patterns
TEST_F(BumpAllocatorTest, WorkspaceAllocationPatterns) {
    BumpAllocator pool(buffer.get(), buffer_size);
    
    // Allocate various workspace types
    auto byte_workspace1 = pool.allocate<std::byte>(device, 8192);
    auto byte_workspace2 = pool.allocate<std::byte>(device, 4096);
    auto temp_storage = pool.allocate<float>(device, 1024);
    auto byte_workspace3 = pool.allocate<std::byte>(device, 2048);
    
    EXPECT_EQ(byte_workspace1.size(), 8192);
    EXPECT_EQ(byte_workspace2.size(), 4096);
    EXPECT_EQ(temp_storage.size(), 1024);
    EXPECT_EQ(byte_workspace3.size(), 2048);
    
    // All should be properly aligned to their respective requirements
    auto device_align = std::max((size_t)16, device.get_property(DeviceProperty::MEM_BASE_ADDR_ALIGN) / 8);
    auto byte_align = std::max(device_align, static_cast<std::uintptr_t>(alignof(std::byte)));
    auto float_align = std::max(device_align, static_cast<std::uintptr_t>(alignof(float)));
    
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(byte_workspace1.data()) % byte_align, 0);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(byte_workspace2.data()) % byte_align, 0);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(temp_storage.data()) % float_align, 0);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(byte_workspace3.data()) % byte_align, 0);
}

// Test realistic buffer size calculation and usage
TEST_F(BumpAllocatorTest, RealisticBufferSizeCalculationAndUsage) {
    // Simulate realistic parameters
    const size_t n = 128;
    const size_t batch_size = 16;
    
    // Calculate buffer size using allocation_size like real functions do
    size_t total_size = 0;
    total_size += BumpAllocator::allocation_size<float>(device, n * n * batch_size);  // Matrix A
    total_size += BumpAllocator::allocation_size<float>(device, n * batch_size);     // Vector b  
    total_size += BumpAllocator::allocation_size<float>(device, n * batch_size);     // Vector x
    total_size += BumpAllocator::allocation_size<std::byte>(device, 16384);          // Workspace
    total_size += BumpAllocator::allocation_size<float*>(device, batch_size * 3);    // Pointer arrays
    
    // Create allocator with calculated size.
    // Add (device_align - 1) padding bytes to absorb initial alignment gap from new[].
    auto device_align = std::max((size_t)16, device.get_property(DeviceProperty::MEM_BASE_ADDR_ALIGN) / 8);
    auto realistic_buffer = std::make_unique<std::byte[]>(total_size + device_align - 1);
    BumpAllocator pool(realistic_buffer.get(), total_size + device_align - 1);
    
    // Perform allocations in realistic order
    auto matrix_A = pool.allocate<float>(device, n * n * batch_size);
    auto vector_b = pool.allocate<float>(device, n * batch_size);
    auto vector_x = pool.allocate<float>(device, n * batch_size);
    auto workspace = pool.allocate<std::byte>(device, 16384);
    auto ptr_arrays = pool.allocate<float*>(device, batch_size * 3);
    
    // All allocations should succeed
    EXPECT_EQ(matrix_A.size(), n * n * batch_size);
    EXPECT_EQ(vector_b.size(), n * batch_size);
    EXPECT_EQ(vector_x.size(), n * batch_size);
    EXPECT_EQ(workspace.size(), 16384);
    EXPECT_EQ(ptr_arrays.size(), batch_size * 3);
}

// Test allocation ordering consistency  
TEST_F(BumpAllocatorTest, AllocationOrderingConsistency) {
    // Test that allocation order doesn't affect alignment or success
    BumpAllocator pool1(buffer.get(), buffer_size);
    BumpAllocator pool2(buffer.get(), buffer_size);
    
    // Same allocations in different order
    auto a1 = pool1.allocate<int>(device, 100);
    auto b1 = pool1.allocate<double>(device, 50);
    auto c1 = pool1.allocate<std::byte>(device, 1024);
    
    auto c2 = pool2.allocate<std::byte>(device, 1024);
    auto a2 = pool2.allocate<int>(device, 100);
    auto b2 = pool2.allocate<double>(device, 50);
    
    // All should have correct sizes
    EXPECT_EQ(a1.size(), 100);
    EXPECT_EQ(b1.size(), 50);
    EXPECT_EQ(c1.size(), 1024);
    EXPECT_EQ(a2.size(), 100);
    EXPECT_EQ(b2.size(), 50);
    EXPECT_EQ(c2.size(), 1024);
    
    // All should be properly aligned to their respective requirements
    auto device_align = std::max((size_t)16, device.get_property(DeviceProperty::MEM_BASE_ADDR_ALIGN) / 8);
    auto int_align = std::max(device_align, static_cast<std::uintptr_t>(alignof(int)));
    auto double_align = std::max(device_align, static_cast<std::uintptr_t>(alignof(double)));
    auto byte_align = std::max(device_align, static_cast<std::uintptr_t>(alignof(std::byte)));
    
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(a1.data()) % int_align, 0);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(b1.data()) % double_align, 0);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(c1.data()) % byte_align, 0);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(a2.data()) % int_align, 0);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(b2.data()) % double_align, 0);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(c2.data()) % byte_align, 0);
}

// Test multiple types in sequence
TEST_F(BumpAllocatorTest, MultipleTypesInSequence) {
    BumpAllocator pool(buffer.get(), buffer_size);
    
    // Allocate different types in sequence
    auto chars = pool.allocate<char>(device, 100);
    auto shorts = pool.allocate<short>(device, 50);
    auto ints = pool.allocate<int>(device, 25);
    auto longs = pool.allocate<long>(device, 12);
    auto floats = pool.allocate<float>(device, 25);
    auto doubles = pool.allocate<double>(device, 12);
    auto complexf = pool.allocate<std::complex<float>>(device, 10);
    auto complexd = pool.allocate<std::complex<double>>(device, 5);
    
    // Verify all allocations
    EXPECT_EQ(chars.size(), 100);
    EXPECT_EQ(shorts.size(), 50);
    EXPECT_EQ(ints.size(), 25);
    EXPECT_EQ(longs.size(), 12);
    EXPECT_EQ(floats.size(), 25);
    EXPECT_EQ(doubles.size(), 12);
    EXPECT_EQ(complexf.size(), 10);
    EXPECT_EQ(complexd.size(), 5);
    
    // Verify no overlaps (basic sanity check)
    EXPECT_TRUE(reinterpret_cast<char*>(shorts.data()) >= 
                reinterpret_cast<char*>(chars.data()) + chars.size());
    EXPECT_TRUE(reinterpret_cast<char*>(ints.data()) >= 
                reinterpret_cast<char*>(shorts.data()) + shorts.size() * sizeof(short));
}

// Test large allocations
TEST_F(BumpAllocatorTest, LargeAllocations) {
    const size_t large_size = 10 * 1024 * 1024; // 10MB
    auto large_buffer = std::make_unique<std::byte[]>(large_size);
    BumpAllocator pool(large_buffer.get(), large_size);
    
    // Allocate a large chunk
    const size_t elements = 1024 * 1024; // 1M elements
    auto large_span = pool.allocate<float>(device, elements);
    EXPECT_EQ(large_span.size(), elements);
    EXPECT_NE(large_span.data(), nullptr);
    
    // Should still be able to allocate more
    auto small_span = pool.allocate<int>(device, 100);
    EXPECT_EQ(small_span.size(), 100);
}

// Test Queue vs Device consistency
TEST_F(BumpAllocatorTest, QueueVsDeviceConsistency) {
    BumpAllocator pool(buffer.get(), buffer_size);
    
    // Allocate using device
    auto span1 = pool.allocate<double>(device, 50);
    
    // Allocate using queue
    auto span2 = pool.allocate<double>(*queue, 50);
    
    // Both should work and be properly aligned
    EXPECT_EQ(span1.size(), 50);
    EXPECT_EQ(span2.size(), 50);
    
    auto device_align = std::max((size_t)16, device.get_property(DeviceProperty::MEM_BASE_ADDR_ALIGN) / 8);
    auto alignment = std::max(device_align, static_cast<std::uintptr_t>(alignof(double)));
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(span1.data()) % alignment, 0);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(span2.data()) % alignment, 0);
    
    // Allocation sizes should be consistent
    EXPECT_EQ(BumpAllocator::allocation_size<double>(device, 50),
              BumpAllocator::allocation_size<double>(*queue, 50));
}

// ---- sizing mode -----------------------------------------------------------

// Replays an allocation sequence against whichever pool it is handed. This is
// the shape every converted *_buffer_size / implementation pair takes: one
// layout function, called once in sizing mode and once for real.
static void replay_layout(BumpAllocator& pool, const Device& device) {
    const size_t n = 97;      // deliberately not a multiple of any alignment
    const size_t batch = 5;
    pool.allocate<float>(device, n * n * batch);
    pool.allocate<int>(device, n * batch);
    pool.allocate<std::byte>(device, 1);
    pool.allocate<double>(device, n);
    pool.allocate<float*>(device, batch);
    pool.allocate<std::complex<double>>(device, n * batch);
    pool.allocate<char>(device, 3);
    pool.allocate<std::byte>(device, 4096);
    // Deliberately ends on an extent that is not a multiple of the alignment:
    // that is the case where "bytes consumed" and "bytes required" diverge.
    pool.allocate<float>(device, 5);
}

// The contract: a real pool given required_bytes() satisfies the same sequence.
//
// Note the subtlety this is guarding. allocate() checks the alignment-rounded
// size against the bytes left, but advances the cursor only by the raw extent,
// so "bytes the sequence consumes" is strictly smaller than "bytes the sequence
// needs to be handed". Sizing mode has to report the latter.
TEST_F(BumpAllocatorTest, MeasuredSizeSufficesForRealAllocation) {
    auto sizer = BumpAllocator::measuring();
    replay_layout(sizer, device);
    const size_t measured = sizer.required_bytes();
    EXPECT_GT(measured, 0u);

    // A real pool of exactly that size, based at a device-aligned address.
    auto device_align = std::max((size_t)16, device.get_property(DeviceProperty::MEM_BASE_ADDR_ALIGN) / 8);
    void* mem = std::aligned_alloc(device_align, ((measured + device_align - 1) / device_align) * device_align);
    ASSERT_NE(mem, nullptr);

    BumpAllocator real(reinterpret_cast<std::byte*>(mem), measured);
    EXPECT_NO_THROW(replay_layout(real, device));

    std::free(mem);
}

// Callers add a callee's reported size into their own running total and then
// re-serve it with allocate<std::byte>(), which rounds the request up. A sizing
// result that is not itself an alignment multiple therefore under-provisions
// every such caller. Summed allocation_size totals always had this property;
// sizing mode has to keep it.
TEST_F(BumpAllocatorTest, MeasuredSizeIsAnAlignmentMultiple) {
    auto device_align = std::max((size_t)16, device.get_property(DeviceProperty::MEM_BASE_ADDR_ALIGN) / 8);

    auto sizer = BumpAllocator::measuring();
    replay_layout(sizer, device);
    EXPECT_EQ(sizer.required_bytes() % device_align, 0u);

    // Including the degenerate case of a single ragged allocation.
    auto one = BumpAllocator::measuring();
    one.allocate<float>(device, 5);
    EXPECT_EQ(one.required_bytes() % device_align, 0u);
    EXPECT_EQ(one.required_bytes(), BumpAllocator::allocation_size<float>(device, 5));
}

// ...and it is tight: what is left over is under a single alignment quantum,
// so the figure is a real size and not a padded guess.
TEST_F(BumpAllocatorTest, MeasuredSizeIsTight) {
    auto sizer = BumpAllocator::measuring();
    replay_layout(sizer, device);
    const size_t measured = sizer.required_bytes();

    auto device_align = std::max((size_t)16, device.get_property(DeviceProperty::MEM_BASE_ADDR_ALIGN) / 8);
    void* mem = std::aligned_alloc(device_align, ((measured + device_align - 1) / device_align) * device_align);
    ASSERT_NE(mem, nullptr);

    BumpAllocator real(reinterpret_cast<std::byte*>(mem), measured);
    replay_layout(real, device);
    EXPECT_LT(real.remaining().size(), device_align);

    std::free(mem);
}

// Sizing mode is never larger than the hand-summed allocation_size total that
// the *_buffer_size functions use today, so converting a call site can only
// shrink its workspace, never grow it.
TEST_F(BumpAllocatorTest, MeasuredSizeNeverExceedsSummedAllocationSize) {
    const size_t n = 97, batch = 5;
    size_t summed = 0;
    summed += BumpAllocator::allocation_size<float>(device, n * n * batch);
    summed += BumpAllocator::allocation_size<int>(device, n * batch);
    summed += BumpAllocator::allocation_size<std::byte>(device, 1);
    summed += BumpAllocator::allocation_size<double>(device, n);
    summed += BumpAllocator::allocation_size<float*>(device, batch);
    summed += BumpAllocator::allocation_size<std::complex<double>>(device, n * batch);
    summed += BumpAllocator::allocation_size<char>(device, 3);
    summed += BumpAllocator::allocation_size<std::byte>(device, 4096);
    summed += BumpAllocator::allocation_size<float>(device, 5);

    auto sizer = BumpAllocator::measuring();
    replay_layout(sizer, device);
    EXPECT_LE(sizer.required_bytes(), summed);
}

// Sizing mode hands out aligned, non-null, distinct, non-overlapping addresses
// so that views can be built over it -- it just never backs them with memory.
TEST_F(BumpAllocatorTest, MeasuringHandsOutUsableButUnbackedAddresses) {
    auto sizer = BumpAllocator::measuring();
    auto device_align = std::max((size_t)16, device.get_property(DeviceProperty::MEM_BASE_ADDR_ALIGN) / 8);

    auto a = sizer.allocate<float>(device, 10);
    auto b = sizer.allocate<double>(device, 10);
    EXPECT_NE(a.data(), nullptr);
    EXPECT_NE(b.data(), nullptr);
    EXPECT_EQ(a.size(), 10u);
    EXPECT_EQ(b.size(), 10u);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(a.data()) % device_align, 0u);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(b.data()) % device_align, 0u);
    EXPECT_GE(reinterpret_cast<std::byte*>(b.data()),
              reinterpret_cast<std::byte*>(a.data()) + 10 * sizeof(float));

    // Zero-size behaves as it does for a real pool, and costs nothing.
    const size_t before = sizer.required_bytes();
    auto z = sizer.allocate<int>(device, 0);
    EXPECT_EQ(z.data(), nullptr);
    EXPECT_EQ(sizer.required_bytes(), before);
}

TEST_F(BumpAllocatorTest, MeasuringRejectsQueriesThatMeanNothing) {
    auto sizer = BumpAllocator::measuring();
    EXPECT_TRUE(sizer.is_measuring());
    // No real tail exists, so a callee cannot size itself against it.
    EXPECT_THROW((void)sizer.remaining(), std::runtime_error);

    BumpAllocator real(buffer.get(), buffer_size);
    EXPECT_FALSE(real.is_measuring());
    EXPECT_THROW((void)real.required_bytes(), std::runtime_error);
}

// ---- per-queue workspace arena --------------------------------------------

TEST_F(BumpAllocatorTest, WorkspaceLeaseIsUsableAndAligned) {
    auto device_align = std::max((size_t)16, device.get_property(DeviceProperty::MEM_BASE_ADDR_ALIGN) / 8);

    auto lease = queue->workspace(4096);
    ASSERT_NE(lease.data(), nullptr);
    EXPECT_EQ(lease.size(), 4096u);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(lease.data()) % device_align, 0u);

    // It is real, writable, shared memory -- not a sizing-mode stand-in.
    std::memset(lease.data(), 0xAB, lease.size());
    EXPECT_EQ(static_cast<unsigned char>(lease.data()[4095]), 0xABu);

    // And it feeds a BumpAllocator without the pool having to realign it.
    BumpAllocator pool(lease.span());
    EXPECT_NO_THROW(pool.allocate<double>(device, 64));
}

// The point of the arena: repeating the same request must stop allocating.
TEST_F(BumpAllocatorTest, WorkspaceReusesMemoryAcrossLeases) {
    std::byte* first = nullptr;
    {
        auto lease = queue->workspace(8192);
        first = lease.data();
    }
    const size_t cap_after_first = queue->workspace_capacity();
    EXPECT_GT(cap_after_first, 0u);

    for (int i = 0; i < 32; ++i) {
        auto lease = queue->workspace(8192);
        EXPECT_EQ(lease.data(), first);                       // same bytes every time
        EXPECT_EQ(queue->workspace_capacity(), cap_after_first);  // no new allocation
    }
}

// Nested leases must not overlap, and -- the part that is easy to get wrong --
// an inner lease that forces a new block must not invalidate the outer one.
TEST_F(BumpAllocatorTest, NestedLeasesAreDisjointAndOuterSurvivesGrowth) {
    auto outer = queue->workspace(1024);
    std::memset(outer.data(), 0x11, outer.size());

    {
        auto inner = queue->workspace(2048);
        EXPECT_TRUE(inner.data() + inner.size() <= outer.data() ||
                    outer.data() + outer.size() <= inner.data());
        std::memset(inner.data(), 0x22, inner.size());

        // Force a fresh block while both leases are live.
        auto huge = queue->workspace(4 * 1024 * 1024);
        std::memset(huge.data(), 0x33, huge.size());
        EXPECT_TRUE(huge.data() + huge.size() <= outer.data() ||
                    outer.data() + outer.size() <= huge.data());
    }

    // Outer's pointer is still valid and its contents untouched.
    for (size_t i = 0; i < outer.size(); ++i) {
        ASSERT_EQ(static_cast<unsigned char>(outer.data()[i]), 0x11u) << "clobbered at " << i;
    }
}

// A lease released early frees its bytes for the next borrow, and moving a
// lease transfers the obligation rather than releasing twice.
TEST_F(BumpAllocatorTest, WorkspaceLeaseReleaseAndMove) {
    std::byte* p = nullptr;
    {
        auto a = queue->workspace(2048);
        p = a.data();
        a.release();
        EXPECT_EQ(a.data(), nullptr);
        auto b = queue->workspace(2048);
        EXPECT_EQ(b.data(), p);   // reclaimed
    }

    auto a = queue->workspace(2048);
    auto* pa = a.data();
    auto b = std::move(a);
    EXPECT_EQ(b.data(), pa);
    EXPECT_EQ(a.data(), nullptr);
    b.release();
    auto c = queue->workspace(2048);
    EXPECT_EQ(c.data(), pa);      // the moved-from handle did not release it early
}

// Growth is geometric, not one allocation per step, so a caller whose request
// ratchets upward does not leave a trail of blocks.
TEST_F(BumpAllocatorTest, WorkspaceGrowthIsGeometric) {
    Queue q;
    size_t allocations = 0;
    size_t prev_cap = 0;
    for (size_t bytes = 1024; bytes <= 4u * 1024 * 1024; bytes += 4096) {
        auto lease = q.workspace(bytes);
        const size_t cap = q.workspace_capacity();
        if (cap != prev_cap) {
            ++allocations;
            prev_cap = cap;
        }
    }
    // ~1000 distinct sizes; doubling from 64 KiB reaches 4 MiB in far fewer.
    EXPECT_LT(allocations, 20u);
}

// Releasing a lease that is not the innermost one used to rewind the arena's
// cursor over memory a live lease was still pointing at, so the next borrow
// silently aliased it. The arena now refuses to rewind past a live lease: the
// bytes stay reserved until the leases taken after them come back, which is
// wasteful but never wrong.
TEST_F(BumpAllocatorTest, WorkspaceOutOfOrderReleaseDoesNotAliasLiveLease) {
#ifndef NDEBUG
    // The fix asserts on exactly the call this test makes, so in a debug build
    // the scenario cannot be run in-process. Death-testing it would fork a
    // process with the SYCL runtime loaded, which is worse than losing the
    // coverage in a configuration nobody runs the suite in by default.
    GTEST_SKIP() << "arena asserts on out-of-order release when NDEBUG is not defined";
#else
    auto a = queue->workspace(2048);
    auto b = queue->workspace(2048);
    ASSERT_NE(a.data(), nullptr);
    ASSERT_NE(b.data(), nullptr);
    ASSERT_TRUE(a.data() + a.size() <= b.data() || b.data() + b.size() <= a.data());
    std::memset(b.data(), 0x22, b.size());

    // Out of order: b was taken after a and is still live.
    a.release();

    // Big enough that the buggy rewind-to-a would have run straight through b.
    auto c = queue->workspace(4096);
    ASSERT_NE(c.data(), nullptr);
    EXPECT_TRUE(c.data() + c.size() <= b.data() || b.data() + b.size() <= c.data())
        << "workspace handed out bytes that a live lease is still using";

    std::memset(c.data(), 0x33, c.size());
    for (size_t i = 0; i < b.size(); ++i) {
        ASSERT_EQ(static_cast<unsigned char>(b.data()[i]), 0x22u) << "clobbered at " << i;
    }
#endif
}

// The arena otherwise holds its high-water mark until the queue dies, which is
// the wrong trade for a queue that lives as long as the process.
TEST_F(BumpAllocatorTest, WorkspaceTrimFreesBlocksOnlyWhenNothingIsLeased) {
    // Its own queue: trim frees every block, so nothing else may be borrowing.
    Queue q;
    const size_t big = 1u << 20;

    {
        auto lease = q.workspace(big);
        std::memset(lease.data(), 0x5A, lease.size());
        const size_t held = q.workspace_capacity();
        EXPECT_GE(held, big);

        // Refused while the lease is live -- those blocks are what it points at.
        // The bool is the only way a caller can tell "refused" from "freed
        // nothing because there was nothing to free", so pin it.
        EXPECT_FALSE(q.trim_workspace());
        EXPECT_EQ(q.workspace_capacity(), held);
        EXPECT_EQ(static_cast<unsigned char>(lease.data()[big - 1]), 0x5Au);
    }

    EXPECT_GE(q.workspace_capacity(), big);
    EXPECT_TRUE(q.trim_workspace());
    EXPECT_EQ(q.workspace_capacity(), 0u);

    // Trimming hands the memory back; it does not retire the arena.
    auto again = q.workspace(4096);
    EXPECT_NE(again.data(), nullptr);
    EXPECT_GT(q.workspace_capacity(), 0u);
}

// Reassigning a live lease is the one out-of-order release the caller cannot
// avoid -- the right-hand borrow is taken before the left-hand one is released
// -- so it must not abort, and it must not hand the old bytes to the next
// borrow while the new lease still sits above them.
//
// This is also where the deferred-reclaim path gets its only coverage in a
// build without NDEBUG: every other way of provoking it asserts on purpose, so
// WorkspaceOutOfOrderReleaseDoesNotAliasLiveLease below is skipped in a debug
// build. Reassignment is the exempt path, so it can be tested in both.
TEST_F(BumpAllocatorTest, WorkspaceReassignedLeaseDefersReclaimAndDoesNotAlias) {
    Queue q;  // its own queue so workspace_capacity() is only about this test
    const size_t baseline = q.workspace_capacity();

    {
        auto ws = q.workspace(2048);
        ASSERT_NE(ws.data(), nullptr);

        // The old loan is now returned out of order, underneath the new one.
        ws = q.workspace(4096);
        ASSERT_NE(ws.data(), nullptr);
        std::memset(ws.data(), 0x44, ws.size());

        // The returned bytes must not be re-served while ws is live.
        auto c = q.workspace(4096);
        ASSERT_NE(c.data(), nullptr);
        EXPECT_TRUE(c.data() + c.size() <= ws.data() || ws.data() + ws.size() <= c.data())
            << "workspace re-served a deferred loan underneath a live lease";
        std::memset(c.data(), 0x55, c.size());
        for (size_t i = 0; i < ws.size(); ++i) {
            ASSERT_EQ(static_cast<unsigned char>(ws.data()[i]), 0x44u) << "clobbered at " << i;
        }
        // c dies first, then ws: reverse order, which is what finally pops the
        // deferred entry.
    }

    // Deferred, not leaked: once the loans above it came back the arena rewound
    // past it, so a fresh borrow of the original size fits in what is already
    // held rather than opening another block.
    const size_t held = q.workspace_capacity();
    {
        auto again = q.workspace(2048);
        ASSERT_NE(again.data(), nullptr);
    }
    EXPECT_EQ(q.workspace_capacity(), held);
    EXPECT_GT(held, baseline);
}

// A no-pessimisation guard, not a regression test: this passes against the
// pre-fix arena too, because a pure LIFO sequence rewound to the same place
// under the old unconditional rewind. It is here so a future change that makes
// the ordering check defer the *common* case is caught -- reverse-order release
// must stay an immediate rewind, so a repeated nesting pattern keeps landing on
// the same bytes and allocates once.
TEST_F(BumpAllocatorTest, WorkspaceLifoReleaseStillReusesTheSameBytes) {
    std::byte* outer_p = nullptr;
    std::byte* inner_p = nullptr;
    size_t settled = 0;

    for (int i = 0; i < 8; ++i) {
        auto outer = queue->workspace(4096);
        auto inner = queue->workspace(8192);
        if (i == 0) {
            outer_p = outer.data();
            inner_p = inner.data();
            settled = queue->workspace_capacity();
            ASSERT_GT(settled, 0u);
        }
        EXPECT_EQ(outer.data(), outer_p);
        EXPECT_EQ(inner.data(), inner_p);
        EXPECT_EQ(queue->workspace_capacity(), settled);

        // Explicit, in reverse order of acquisition -- the case the arena is
        // built for, and the one scope exit would produce anyway.
        inner.release();
        outer.release();
    }

    // Everything is back, so the next borrow starts where the first one did.
    auto reuse = queue->workspace(4096);
    EXPECT_EQ(reuse.data(), outer_p);
    EXPECT_EQ(queue->workspace_capacity(), settled);
}

// Stress test with many small allocations
TEST_F(BumpAllocatorTest, ManySmallAllocations) {
    BumpAllocator pool(buffer.get(), buffer_size);
    
    std::vector<Span<int>> spans;
    
    // Allocate many small chunks
    for (int i = 0; i < 1000 && i * 4 * sizeof(int) < buffer_size / 2; ++i) {
        auto span = pool.allocate<int>(device, 4);
        EXPECT_EQ(span.size(), 4);
        spans.push_back(span);
    }
    
    // Verify no overlaps in first few allocations
    for (size_t i = 1; i < std::min(spans.size(), size_t(10)); ++i) {
        EXPECT_TRUE(reinterpret_cast<char*>(spans[i].data()) >= 
                    reinterpret_cast<char*>(spans[i-1].data()) + spans[i-1].size() * sizeof(int));
    }
}

// ---------------------------------------------------------------------------
// Queue thread-affinity guard.
//
// A Queue owns an unsynchronised bump arena and a cached last-event; sharing
// one across threads used to corrupt both silently. The contract is now
// enforced with a thread-id compare, and these two tests are what keep it
// enforced. See docs/cpp-api.md "Synchronisation and threading".
// ---------------------------------------------------------------------------
TEST(QueueThreadAffinityTest, WorkspaceFromAnotherThreadThrows) {
    Queue q;
    bool threw = false;
    std::thread t([&] {
        try {
            (void)q.workspace(1024);
        } catch (const std::runtime_error&) {
            threw = true;
        }
    });
    t.join();
    EXPECT_TRUE(threw) << "Queue::workspace() from a foreign thread must throw";
}

TEST(QueueThreadAffinityTest, AttachToCurrentThreadTransfersOwnership) {
    Queue q;
    bool ok = false;
    std::thread t([&] {
        q.attach_to_current_thread();
        try {
            auto lease = q.workspace(1024);
            ok = lease.span().size() >= 1024;
        } catch (const std::runtime_error&) {
            ok = false;
        }
    });
    t.join();
    // The queue now belongs to a thread that has exited; re-take it so the
    // destructor on this thread does not trip the guard.
    q.attach_to_current_thread();
    EXPECT_TRUE(ok) << "attach_to_current_thread() must hand the Queue over";
}
