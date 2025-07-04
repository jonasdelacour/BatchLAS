#include <gtest/gtest.h>
#include <util/mempool.hh>
#include <util/sycl-vector.hh>
#include <util/sycl-span.hh>
#include <util/sycl-device-queue.hh>
#include <vector>
#include <complex>
#include <memory>
#include <cstring>
#include <algorithm>

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
    
    // Create allocator with calculated size
    auto large_buffer = std::make_unique<std::byte[]>(total_size);
    BumpAllocator pool(large_buffer.get(), total_size);
    
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
    
    // Phase 2: Create allocator with calculated size in different context
    auto planned_buffer = std::make_unique<std::byte[]>(total_calculated_size);
    BumpAllocator pool(planned_buffer.get(), total_calculated_size);
    
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
    auto device_align = std::max((size_t)16, device.get_property(DeviceProperty::MEM_BASE_ADDR_ALIGN) / 8);
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
    
    // Create allocator with calculated size
    auto realistic_buffer = std::make_unique<std::byte[]>(total_size);
    BumpAllocator pool(realistic_buffer.get(), total_size);
    
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
