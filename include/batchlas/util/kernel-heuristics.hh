// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <algorithm>
#include <bit>
#include <limits>
#include <cstdint>
// Installed header: public includes only (src/ is not installed).
#include <batchlas/util/sycl-device-queue.hh>

/// @file
/// @brief Launch-size heuristics used by the library's own kernels. Installed, not API.
/// @ingroup api_internal_helpers

namespace batchlas {

/// @addtogroup api_internal_helpers
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
    size_t global_size;              ///< \f$\mathrm{batch\_size} \cdot \mathrm{work\_groups\_per\_matrix} \cdot \mathrm{local\_size}\f$
    size_t local_size;               ///< work-group size
    size_t work_groups_per_matrix;   ///< work-groups assigned to each matrix
};

/**
 * @brief Compute a work-group size from the device type and kernel type
 *
 * Starts from a per-device-type base (GPU 256, CPU \f$\min(64, 2 \cdot \mathrm{CUs})\f$ with
 * \f$\mathrm{CUs}\f$ the compute units, accelerator 128), caps it per kernel type, then by
 * `problem_size` (when nonzero) and the device's maximum work-group size. Never returns 0.
 *
 * @param[in] device              The target device
 * @param[in] kernel_type         Type of kernel operation
 * @param[in] problem_size        Characteristic problem size (e.g., matrix dimension); 0 = no cap
 * @param[in] batch_size          Number of matrices in batch; currently unused
 * @param[in] memory_per_problem  Currently unused
 * @return Work-group size. For REDUCTION and SCAN the capped base is rounded down to a power
 *         of two before the problem-size and device caps, which can still make it a non-power.
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
            base_wg_size = std::bit_floor(base_wg_size);
            break;

        case KernelType::SCAN:
            base_wg_size = std::min(base_wg_size, size_t(256));
            base_wg_size = std::bit_floor(base_wg_size);
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
 * When `total_work` exceeds \f$\mathrm{INT32\_MAX}/2\f$ the result grid-strides with four
 * work-groups per compute unit. Otherwise, with at least as many matrices as
 * compute units, each matrix gets
 * \f$\lceil \mathrm{problem\_size} / \mathrm{local\_size} \rceil\f$ work-groups;
 * with fewer, the compute units are spread over the matrices, capped at what
 * one matrix needs. The global size is never below `total_work` rounded up to a
 * multiple of the local size.
 *
 * @param[in] total_work              Total number of work items needed
 * @param[in] device                  Target device
 * @param[in] kernel_type             Type of kernel operation
 * @param[in] batch_size              Number of matrices in batch
 * @param[in] problem_size            Elements (or work items) per matrix
 * @param[in] preferred_wg_size       Optional preferred work-group size (0 = auto)
 * @param[in] footprint_per_problem   Bytes per problem; currently has no effect (its estimate is overwritten)
 * @param[in] max_wg_size_for_kernel  Currently unused
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
 * \f[
 *   \mathrm{work\_groups\_per\_matrix}
 *     = \max\left(1,\; \min\left(\left\lceil \frac{\mathrm{CUs}}{\mathrm{batch\_size}} \right\rceil,\;
 *                 \left\lceil \frac{\mathrm{elements\_per\_matrix}}{\mathrm{local\_size}} \right\rceil\right)\right),
 * \f]
 * with \f$\mathrm{CUs}\f$ the compute units. The global size is then capped to fit
 * a signed 32-bit int, keeping at least one work-group per matrix; the kernel's
 * grid-stride picks up the slack.
 *
 * @param[in] batch_size           Number of matrices in batch
 * @param[in] elements_per_matrix  Number of elements per matrix
 * @param[in] device               Target device
 * @param[in] kernel_type          Type of kernel operation
 * @param[in] preferred_wg_size    Optional preferred work-group size (0 = auto)
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
 * @param[in] total_work         Total number of work items needed
 * @param[in] device             Target device
 * @param[in] kernel_type        Type of kernel operation
 * @param[in] preferred_wg_size  Optional preferred work-group size (0 = auto)
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

/// @}

} // namespace batchlas
