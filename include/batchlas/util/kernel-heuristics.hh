#pragma once

#include <cstddef>
#include <algorithm>
#include <limits>
#include <cstdint>
// Installed header: public includes only (src/ is not installed).
#include <batchlas/util/sycl-device-queue.hh>

/// @file
/// @brief Launch-size heuristics used by the library's own kernels. Installed, not API.
/// @ingroup internal_helpers

namespace batchlas {

/// @addtogroup internal_helpers
/// @{

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

/// @brief Result of compute_batched_nd_range_sizes().
struct BatchedNdRangeSizes {
    size_t global_size;     ///< total work-items, a multiple of `local_size`
    size_t local_size;      ///< work-group size
    bool use_grid_stride;   ///< the kernel must grid-stride: `global_size` was capped below the total work
};

/// @brief Result of compute_batched_matrix_decomposition().
struct BatchedMatrixDecomposition {
    size_t global_size;              ///< batch * work_groups_per_matrix * local_size
    size_t local_size;               ///< work-group size
    size_t work_groups_per_matrix;   ///< work-groups assigned to each matrix
};

/**
 * @brief Compute a work-group size from the device type and kernel type
 *
 * Starts from a per-device-type base (GPU 256, CPU min(64, 2 x compute units),
 * accelerator 128), caps it per kernel type, then by `problem_size` (when
 * nonzero) and the device's maximum work-group size. Never returns 0.
 *
 * @param device The target device
 * @param kernel_type Type of kernel operation
 * @param problem_size Characteristic problem size (e.g., matrix dimension); 0 = no cap
 * @param batch_size Number of matrices in batch; currently unused
 * @param memory_per_problem Currently unused
 * @return Work-group size
 * @trap For REDUCTION and SCAN the "power of two" rounding is `1 << (31 - __builtin_clzl(x))`,
 *       which assumes a 32-bit `long`. On LP64 the shift count is negative (undefined
 *       behaviour), so the result is NOT guaranteed to be a power of two.
 */
inline size_t compute_optimal_wg_size(const Device& device, KernelType kernel_type, 
                                      size_t problem_size = 0, size_t batch_size = 1, size_t memory_per_problem = 0) {
    const size_t max_wg_size = device.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE);
    const size_t max_compute_units = device.get_property(DeviceProperty::MAX_COMPUTE_UNITS);
    const DeviceType dev_type = device.type;
    
    size_t base_wg_size;
    switch (dev_type) {
        case DeviceType::GPU:
            base_wg_size = 256;  // Common choice for NVIDIA/AMD GPUs
            break;
        case DeviceType::CPU:
            base_wg_size = std::min(size_t(64), max_compute_units * 2);
            break;
        case DeviceType::ACCELERATOR:
            base_wg_size = 128;
            break;
        default:
            base_wg_size = 64;
            break;
    }
    
    switch (kernel_type) {
        case KernelType::ELEMENTWISE:
            if (dev_type == DeviceType::GPU) {
                base_wg_size = std::min(base_wg_size, size_t(512));
            } else {
                base_wg_size = std::min(base_wg_size, size_t(128));
            }
            break;
            
        case KernelType::REDUCTION:
            if (dev_type == DeviceType::GPU) {
                base_wg_size = std::min(base_wg_size, size_t(512));
            } else {
                base_wg_size = std::min(base_wg_size, size_t(64));
            }
            // "Power of 2" only where long is 32-bit: see the @trap above.
            base_wg_size = size_t(1) << (31 - __builtin_clzl(base_wg_size));
            break;
            
        case KernelType::SCAN:
            base_wg_size = std::min(base_wg_size, size_t(256));
            base_wg_size = size_t(1) << (31 - __builtin_clzl(base_wg_size));
            break;
            
        case KernelType::MEMORY_BOUND:
            base_wg_size = std::min(base_wg_size, size_t(128));
            break;
            
        case KernelType::COMPUTE_BOUND:
            if (dev_type == DeviceType::GPU) {
                base_wg_size = std::min(base_wg_size, size_t(1024));
            }
            break;
            
        case KernelType::SPARSE:
            base_wg_size = std::min(base_wg_size, size_t(128));
            break;
            
        case KernelType::GEMM:
            base_wg_size = std::min(base_wg_size, size_t(256));
            break;
            
        case KernelType::SMALL_MATRIX:
            base_wg_size = std::min({base_wg_size, problem_size, size_t(64)});
            break;

        case KernelType::TASK_BASED:
            base_wg_size = std::min(base_wg_size, size_t(32));
    }
    
    if (problem_size > 0) {
        base_wg_size = std::min(base_wg_size, problem_size);
    }
    
    base_wg_size = std::min(base_wg_size, max_wg_size);

    if (memory_per_problem > 0) {

    }
    
    return std::max(base_wg_size, size_t(1));
}

/**
 * @brief Compute nd_range sizes for a batched operation
 *
 * When `total_work` exceeds INT32_MAX / 2 the result grid-strides with four
 * work-groups per compute unit. Otherwise, with at least as many matrices as
 * compute units, each matrix gets ceil(problem_size / local_size) work-groups;
 * with fewer, the compute units are spread over the matrices, capped at what
 * one matrix needs. The global size is never below `total_work` rounded up.
 *
 * @param total_work Total number of work items needed
 * @param device Target device
 * @param kernel_type Type of kernel operation
 * @param batch_size Number of matrices in batch
 * @param problem_size Elements (or work items) per matrix
 * @param preferred_wg_size Optional preferred work-group size (0 = auto)
 * @param footprint_per_problem Bytes per problem; currently has no effect (its estimate is overwritten)
 * @param max_wg_size_for_kernel Currently unused
 * @return BatchedNdRangeSizes {global_size, local_size, use_grid_stride}
 */
inline BatchedNdRangeSizes compute_batched_nd_range_sizes(size_t total_work,
                                                                             const Device& device,
                                                                             KernelType kernel_type,
                                                                             size_t batch_size,
                                                                             size_t problem_size,
                                                                             size_t preferred_wg_size = 0,
                                                                             size_t footprint_per_problem = 0,
                                                                             size_t max_wg_size_for_kernel = 0
                                                                            ) {
    const size_t max_compute_units = device.get_property(DeviceProperty::MAX_COMPUTE_UNITS);
    
    const size_t INT32_MAX_SAFE = static_cast<size_t>(std::numeric_limits<int32_t>::max()) / 2;
    bool use_grid_stride = total_work > INT32_MAX_SAFE;
    auto num_cus = device.get_property(DeviceProperty::MAX_COMPUTE_UNITS);
    auto vendor = device.get_vendor();
    auto shedulers_per_cu = 2; //Default to 2 schedulers per CU
    if ((vendor == Vendor::NVIDIA || vendor == Vendor::AMD) && device.type == DeviceType::GPU) {
        shedulers_per_cu = 4; 
    } else if (vendor == Vendor::INTEL && device.type == DeviceType::GPU) {
        shedulers_per_cu = 8; //Intel GPUs have 8 or 16 "Vector Engines" per CU / Xe Core
    }

    auto L2_cache_size = device.get_property(DeviceProperty::GLOBAL_MEM_CACHE_SIZE); //L2 cache size

    

    size_t local_size;
    if (preferred_wg_size > 0) {
        local_size = preferred_wg_size;
    } else {
        local_size = compute_optimal_wg_size(device, kernel_type, problem_size, batch_size);
    }
    
    size_t global_size;

    if (footprint_per_problem > 0) {
        size_t problems_in_L2 = L2_cache_size / footprint_per_problem;
        global_size = std::min(batch_size, problems_in_L2) * local_size;   
    }
    
    if (use_grid_stride) {
        size_t target_workgroups = max_compute_units * 4; // 4x oversubscription
        global_size = target_workgroups * local_size;
        
        global_size = std::min(global_size, INT32_MAX_SAFE);
    } else {
        if (batch_size >= max_compute_units) {
            size_t workgroups_per_matrix = 1;
            size_t target_elements_per_workgroup = problem_size;
            
            if (target_elements_per_workgroup > local_size) {
                workgroups_per_matrix = (target_elements_per_workgroup + local_size - 1) / local_size;
            }
            
            global_size = batch_size * workgroups_per_matrix * local_size;
        } else {
            size_t workgroups_per_matrix = (max_compute_units + batch_size - 1) / batch_size;
            
            size_t max_workgroups_per_matrix = (problem_size + local_size - 1) / local_size;
            workgroups_per_matrix = std::min(workgroups_per_matrix, max_workgroups_per_matrix);
            
            global_size = batch_size * workgroups_per_matrix * local_size;
        }
        
        size_t min_global_size = ((total_work + local_size - 1) / local_size) * local_size;
        global_size = std::max(global_size, min_global_size);
    }
    
    return {.global_size = global_size,
            .local_size = local_size,
            .use_grid_stride = use_grid_stride};
}

/**
 * @brief Compute batched matrix decomposition for work-group assignment
 *
 * With at least as many matrices as compute units, one work-group per matrix,
 * and the kernel grid-strides over the rest of the matrix. Otherwise
 * work_groups_per_matrix = ceil(CUs / batch_size), never more than
 * ceil(elements_per_matrix / local_size). The global size is then capped to fit
 * a signed 32-bit int, keeping at least one work-group per matrix; the kernel's
 * grid-stride picks up the slack.
 *
 * @param batch_size Number of matrices in batch
 * @param elements_per_matrix Number of elements per matrix  
 * @param device Target device
 * @param kernel_type Type of kernel operation
 * @param preferred_wg_size Optional preferred work-group size (0 = auto)
 * @return BatchedMatrixDecomposition {global_size, local_size, work_groups_per_matrix}
 */
inline BatchedMatrixDecomposition compute_batched_matrix_decomposition(
    size_t batch_size,
    size_t elements_per_matrix,
    const Device& device,
    KernelType kernel_type,
    size_t preferred_wg_size = 0)
{
    const size_t max_compute_units = device.get_property(DeviceProperty::MAX_COMPUTE_UNITS);

    size_t local_size = preferred_wg_size > 0
                            ? preferred_wg_size
                            : compute_optimal_wg_size(device, kernel_type, elements_per_matrix, batch_size);

    const size_t required_wgs_per_matrix = (elements_per_matrix + local_size - 1) / local_size;

    size_t work_groups_per_matrix;
    if (batch_size >= max_compute_units) {
        work_groups_per_matrix = 1;
    } else {
        const size_t target_wgs_per_matrix = (max_compute_units + batch_size - 1) / batch_size; // ceil
        work_groups_per_matrix = std::min(target_wgs_per_matrix, required_wgs_per_matrix);
        work_groups_per_matrix = std::max(work_groups_per_matrix, size_t(1));
    }

    const size_t INT32_MAX_SAFE = static_cast<size_t>(std::numeric_limits<int32_t>::max()) - local_size;
    size_t global_size = batch_size * work_groups_per_matrix * local_size;

    if (global_size > INT32_MAX_SAFE) {
        const size_t max_total_wgs   = INT32_MAX_SAFE / local_size;
        const size_t max_wgs_per_mat = std::max<size_t>(1, max_total_wgs / batch_size);
        work_groups_per_matrix       = std::min(work_groups_per_matrix, max_wgs_per_mat);
        global_size                  = batch_size * work_groups_per_matrix * local_size;
    }

    return {.global_size = global_size,
            .local_size = local_size,
            .work_groups_per_matrix = work_groups_per_matrix};
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
    
    const size_t max_safe_work_items = std::numeric_limits<int>::max() / 2;
    
    size_t global_size;
    if (total_work > max_safe_work_items) {
        size_t max_work_groups = std::min(device.get_property(DeviceProperty::MAX_COMPUTE_UNITS) * 16, max_safe_work_items / local_size);
        global_size = max_work_groups * local_size;
    } else {
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
        local_x = 8;
        local_y = 8;
    }
    
    while (local_x * local_y > max_wg_size) {
        if (local_x >= local_y) {
            local_x /= 2;
        } else {
            local_y /= 2;
        }
    }
    
    size_t global_x = ((cols + local_x - 1) / local_x) * local_x;
    size_t global_y = ((rows + local_y - 1) / local_y) * local_y;
    
    return {{global_x, global_y}, {local_x, local_y}};
}

/// @}

} // namespace batchlas
