#pragma once

#include <cstddef>
#include <algorithm>
#include <limits>
#include <tuple>
#include <cstdint>
#include "../../src/queue.hh"

namespace batchlas {

/**
 * @brief Kernel types for heuristic selection
 */
enum class KernelType {
    ELEMENTWISE,    ///< Element-wise operations (fill, copy, etc.)
    REDUCTION,      ///< Reduction operations (sum, max, etc.)
    SCAN,           ///< Prefix scan operations
    MEMORY_BOUND,   ///< Memory bandwidth limited
    COMPUTE_BOUND,  ///< Compute intensive
    SPARSE,         ///< Sparse matrix operations
    GEMM,          ///< General matrix multiply
    SMALL_MATRIX,   ///< Small matrix operations
    TASK_BASED     ///< Task-based parallelism, e.g. no fine-grained parallelism, each thread solves one problem
};

/**
 * @brief Compute optimal work-group size based on device characteristics and kernel type
 * 
 * @param device The target device
 * @param kernel_type Type of kernel operation
 * @param problem_size Characteristic problem size (e.g., matrix dimension)
 * @param batch_size Number of matrices in batch
 * @return Optimal work-group size
 */
inline size_t compute_optimal_wg_size(const Device& device, KernelType kernel_type, 
                                      size_t problem_size = 0, size_t batch_size = 1, size_t memory_per_problem = 0) {
    const size_t max_wg_size = device.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE);
    const size_t max_compute_units = device.get_property(DeviceProperty::MAX_COMPUTE_UNITS);
    const DeviceType dev_type = device.type;
    
    // Base work-group size heuristics based on device type
    size_t base_wg_size;
    switch (dev_type) {
        case DeviceType::GPU:
            // GPUs typically benefit from larger work-groups (warp/wavefront multiples)
            base_wg_size = 256;  // Common choice for NVIDIA/AMD GPUs
            break;
        case DeviceType::CPU:
            // CPUs prefer smaller work-groups to avoid oversubscription
            base_wg_size = std::min(size_t(64), max_compute_units * 2);
            break;
        case DeviceType::ACCELERATOR:
            // Other accelerators
            base_wg_size = 128;
            break;
        default:
            base_wg_size = 64;
            break;
    }
    
    // Adjust based on kernel type
    switch (kernel_type) {
        case KernelType::ELEMENTWISE:
            // For element-wise operations, prefer larger work-groups on GPU
            if (dev_type == DeviceType::GPU) {
                base_wg_size = std::min(base_wg_size, size_t(512));
            } else {
                base_wg_size = std::min(base_wg_size, size_t(128));
            }
            break;
            
        case KernelType::REDUCTION:
            // Reductions benefit from power-of-2 work-group sizes
            if (dev_type == DeviceType::GPU) {
                base_wg_size = std::min(base_wg_size, size_t(512));
            } else {
                base_wg_size = std::min(base_wg_size, size_t(64));
            }
            // Ensure power of 2
            base_wg_size = size_t(1) << (31 - __builtin_clzl(base_wg_size));
            break;
            
        case KernelType::SCAN:
            // Scans need careful work-group sizing for efficiency
            base_wg_size = std::min(base_wg_size, size_t(256));
            base_wg_size = size_t(1) << (31 - __builtin_clzl(base_wg_size));
            break;
            
        case KernelType::MEMORY_BOUND:
            // Memory bound operations don't need huge work-groups
            base_wg_size = std::min(base_wg_size, size_t(128));
            break;
            
        case KernelType::COMPUTE_BOUND:
            // Compute bound can use larger work-groups
            if (dev_type == DeviceType::GPU) {
                base_wg_size = std::min(base_wg_size, size_t(1024));
            }
            break;
            
        case KernelType::SPARSE:
            // Sparse operations often have irregular workloads
            base_wg_size = std::min(base_wg_size, size_t(128));
            break;
            
        case KernelType::GEMM:
            // GEMM kernels typically use tiled approaches
            base_wg_size = std::min(base_wg_size, size_t(256));
            break;
            
        case KernelType::SMALL_MATRIX:
            // For small matrices, use smaller work-groups
            base_wg_size = std::min({base_wg_size, problem_size, size_t(64)});
            break;

        case KernelType::TASK_BASED:
            // Each thread handles one task, so smaller work-groups
            base_wg_size = std::min(base_wg_size, size_t(32));
    }
    
    // Final constraint based on problem size
    if (problem_size > 0) {
        base_wg_size = std::min(base_wg_size, problem_size);
    }
    
    // Ensure we don't exceed device limits
    base_wg_size = std::min(base_wg_size, max_wg_size);

    if (memory_per_problem > 0) {

    }
    
    // Ensure at least work-group size of 1
    return std::max(base_wg_size, size_t(1));
}

/**
 * @brief Compute optimal nd_range sizes for batched operations with sophisticated decomposition
 * 
 * @param total_work Total number of work items needed
 * @param device Target device
 * @param kernel_type Type of kernel operation
 * @param batch_size Number of matrices in batch
 * @param elements_per_matrix Number of elements per matrix
 * @param preferred_wg_size Optional preferred work-group size (0 = auto)
 * @return std::tuple<size_t, size_t, bool> {global_size, local_size, use_grid_stride}
 */
inline std::tuple<size_t, size_t, bool> compute_batched_nd_range_sizes(size_t total_work, 
                                                                       const Device& device,
                                                                       KernelType kernel_type,
                                                                       size_t batch_size,
                                                                       size_t problem_size,
                                                                       size_t preferred_wg_size = 0,
                                                                       size_t footprint_per_problem = 0,
                                                                       size_t max_wg_size_for_kernel = 0
                                                                    ) {
    const size_t max_compute_units = device.get_property(DeviceProperty::MAX_COMPUTE_UNITS);
    
    // Check if we need grid-stride approach due to int32 overflow
    const size_t INT32_MAX_SAFE = static_cast<size_t>(std::numeric_limits<int32_t>::max()) / 2;
    bool use_grid_stride = total_work > INT32_MAX_SAFE;
    auto num_cus = device.get_property(DeviceProperty::MAX_COMPUTE_UNITS);
    auto vendor = device.get_vendor();
    auto shedulers_per_cu = 2; //Default to 2 schedulers per CU
    if ((vendor == Vendor::NVIDIA || vendor == Vendor::AMD) && device.type == DeviceType::GPU) {
        shedulers_per_cu = 4; 
        //NVIDIA GPUs have 4 warp engines per CU / SM (Streaming Multiprocessor)
        //AMD GPUs have 4 SIMD lanes per CU
    } else if (vendor == Vendor::INTEL && device.type == DeviceType::GPU) {
        shedulers_per_cu = 8; //Intel GPUs have 8 or 16 "Vector Engines" per CU / Xe Core
    }

    //auto L1_cache_size = device.get_property(DeviceProperty::LOCAL_MEM_SIZE); //Assume local mem is L1 cache size
    auto L2_cache_size = device.get_property(DeviceProperty::GLOBAL_MEM_CACHE_SIZE); //L2 cache size

    

    size_t local_size;
    if (preferred_wg_size > 0) {
        local_size = preferred_wg_size;
    } else {
        local_size = compute_optimal_wg_size(device, kernel_type, problem_size, batch_size);
    }
    
    size_t global_size;

    if (footprint_per_problem > 0) {
        //Estimate how many problems fit in L2 cache
        size_t problems_in_L2 = L2_cache_size / footprint_per_problem;
        global_size = std::min(batch_size, problems_in_L2) * local_size;   
    }
    
    if (use_grid_stride) {
        // Use grid-stride approach: limit global size to prevent overflow
        // Use enough work-groups to saturate the device
        size_t target_workgroups = max_compute_units * 4; // 4x oversubscription
        global_size = target_workgroups * local_size;
        
        // Ensure we don't exceed safe limits
        global_size = std::min(global_size, INT32_MAX_SAFE);
    } else {
        // Batch-aware decomposition strategy
        if (batch_size >= max_compute_units) {
            // Enough matrices for 1 work-group per matrix
            // Each work-group handles problem_size elements
            size_t workgroups_per_matrix = 1;
            size_t target_elements_per_workgroup = problem_size;
            
            // Adjust work-group size if needed
            if (target_elements_per_workgroup > local_size) {
                workgroups_per_matrix = (target_elements_per_workgroup + local_size - 1) / local_size;
            }
            
            global_size = batch_size * workgroups_per_matrix * local_size;
        } else {
            // Not enough matrices: use multiple work-groups per matrix
            size_t workgroups_per_matrix = (max_compute_units + batch_size - 1) / batch_size;
            
            // Don't create more work-groups than needed per matrix
            size_t max_workgroups_per_matrix = (problem_size + local_size - 1) / local_size;
            workgroups_per_matrix = std::min(workgroups_per_matrix, max_workgroups_per_matrix);
            
            global_size = batch_size * workgroups_per_matrix * local_size;
        }
        
        // Ensure global size doesn't exceed total work (with padding)
        size_t min_global_size = ((total_work + local_size - 1) / local_size) * local_size;
        global_size = std::max(global_size, min_global_size);
    }
    
    return {global_size, local_size, use_grid_stride};
}

/**
 * @brief Compute batched matrix decomposition for work-group assignment
 * 
 * @param batch_size Number of matrices in batch
 * @param elements_per_matrix Number of elements per matrix  
 * @param device Target device
 * @param kernel_type Type of kernel operation
 * @param preferred_wg_size Optional preferred work-group size (0 = auto)
 * @return std::tuple<size_t, size_t, size_t> {global_size, local_size, work_groups_per_matrix}
 */
inline std::tuple<size_t, size_t, size_t> compute_batched_matrix_decomposition(
    size_t batch_size,
    size_t elements_per_matrix,
    const Device& device,
    KernelType kernel_type,
    size_t preferred_wg_size = 0)
{
    // ------------------------------------------------------------------
    // Work‑group decomposition heuristic
    //
    //  * If there are at least as many matrices as compute units, launch
    //    **one** work‑group per matrix – the kernel relies on a grid‑stride
    //    loop to cover the remaining elements.
    //
    //  * Otherwise, spread the available compute units across the matrices:
    //      work_groups_per_matrix = ceil(CUs / batch_size)
    //    but never exceed the number actually required to visit every
    //    element (ceil(elements_per_matrix / local_size)).
    //
    //  * Finally cap the global size so it fits in a signed 32‑bit int; if
    //    we have to down‑scale, we keep at least one WG per matrix and rely
    //    on the kernel’s internal grid‑striding to pick up the slack.
    // ------------------------------------------------------------------
    const size_t max_compute_units = device.get_property(DeviceProperty::MAX_COMPUTE_UNITS);

    // 1. Choose a local size (work‑group size)
    size_t local_size = preferred_wg_size > 0
                            ? preferred_wg_size
                            : compute_optimal_wg_size(device, kernel_type, elements_per_matrix, batch_size);

    // 2. How many work‑groups are NEEDED to cover one matrix?
    const size_t required_wgs_per_matrix = (elements_per_matrix + local_size - 1) / local_size;

    // 3. Initial WG-per-matrix choice based on CU / batch ratio
    size_t work_groups_per_matrix;
    if (batch_size >= max_compute_units) {
        // More matrices than compute units → 1 WG per matrix.
        work_groups_per_matrix = 1;
    } else {
        // Spread compute units across matrices.
        const size_t target_wgs_per_matrix = (max_compute_units + batch_size - 1) / batch_size; // ceil
        work_groups_per_matrix = std::min(target_wgs_per_matrix, required_wgs_per_matrix);
        work_groups_per_matrix = std::max(work_groups_per_matrix, size_t(1));
    }

    // 4. Cap global size to stay within 32‑bit limits to avoid SYCL INT overflow.
    const size_t INT32_MAX_SAFE = static_cast<size_t>(std::numeric_limits<int32_t>::max()) - local_size;
    size_t global_size = batch_size * work_groups_per_matrix * local_size;

    if (global_size > INT32_MAX_SAFE) {
        const size_t max_total_wgs   = INT32_MAX_SAFE / local_size;
        const size_t max_wgs_per_mat = std::max<size_t>(1, max_total_wgs / batch_size);
        work_groups_per_matrix       = std::min(work_groups_per_matrix, max_wgs_per_mat);
        global_size                  = batch_size * work_groups_per_matrix * local_size;
    }

    return {global_size, local_size, work_groups_per_matrix};
}

/**
 * @brief Compute optimal global and local work sizes for nd_range kernels with overflow protection
 * 
 * @param total_work Total number of work items needed
 * @param device Target device
 * @param kernel_type Type of kernel operation
 * @param preferred_wg_size Optional preferred work-group size (0 = auto)
 * @return std::pair<size_t, size_t> {global_size, local_size}
 */
inline std::pair<size_t, size_t> compute_nd_range_sizes(size_t total_work, 
                                                        const Device& device,
                                                        KernelType kernel_type,
                                                        size_t preferred_wg_size = 0) {
    size_t local_size;
    if (preferred_wg_size > 0) {
        local_size = preferred_wg_size;
    } else {
        local_size = compute_optimal_wg_size(device, kernel_type, total_work);
    }
    
    // Check for potential overflow
    const size_t max_safe_work_items = std::numeric_limits<int>::max() / 2;
    
    size_t global_size;
    if (total_work > max_safe_work_items) {
        // Use grid-stride approach - limit to a reasonable number of work-groups
        size_t max_work_groups = std::min(device.get_property(DeviceProperty::MAX_COMPUTE_UNITS) * 16, max_safe_work_items / local_size);
        global_size = max_work_groups * local_size;
    } else {
        // Round up global size to be a multiple of local size
        global_size = ((total_work + local_size - 1) / local_size) * local_size;
    }
    
    return {global_size, local_size};
}

/**
 * @brief Compute optimal 2D work-group size for matrix operations
 * 
 * @param rows Number of matrix rows
 * @param cols Number of matrix columns
 * @param device Target device
 * @param kernel_type Type of kernel operation
 * @return std::pair<std::pair<size_t, size_t>, std::pair<size_t, size_t>> {{global_x, global_y}, {local_x, local_y}}
 */
inline std::pair<std::pair<size_t, size_t>, std::pair<size_t, size_t>> 
compute_2d_nd_range_sizes(size_t rows, size_t cols, 
                         const Device& device,
                         KernelType kernel_type) {
    const size_t max_wg_size = device.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE);
    const DeviceType dev_type = device.type;
    
    size_t local_x, local_y;
    
    if (dev_type == DeviceType::GPU) {
        // For GPUs, common choices are 16x16 or 32x8, 8x32 tiles
        if (rows >= 16 && cols >= 16) {
            local_x = 16;
            local_y = 16;
        } else if (cols >= 32) {
            local_x = 32;
            local_y = 8;
        } else if (rows >= 32) {
            local_x = 8;
            local_y = 32;
        } else {
            local_x = 8;
            local_y = 8;
        }
    } else {
        // For CPUs and other devices, use smaller tiles
        local_x = 8;
        local_y = 8;
    }
    
    // Ensure we don't exceed max work-group size
    while (local_x * local_y > max_wg_size) {
        if (local_x >= local_y) {
            local_x /= 2;
        } else {
            local_y /= 2;
        }
    }
    
    // Calculate global sizes (rounded up)
    size_t global_x = ((cols + local_x - 1) / local_x) * local_x;
    size_t global_y = ((rows + local_y - 1) / local_y) * local_y;
    
    return {{global_x, global_y}, {local_x, local_y}};
}

} // namespace batchlas
