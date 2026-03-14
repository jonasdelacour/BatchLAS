#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace batchlas {

struct FlatNode {
    int32_t offset = 0;
    int32_t logical_n = 0;
    int32_t left_n = 0;
    int32_t right_n = 0;

    bool is_leaf() const { return left_n == 0 && right_n == 0; }
};

struct FlatLevel {
    int32_t capacity_n = 0;
    std::vector<FlatNode> nodes;
};

struct FlatSchedule {
    std::vector<FlatLevel> levels;     // root -> leaf
    std::vector<int32_t> split_flags;  // length n-1, 1 when a global split boundary exists
};

inline FlatSchedule build_flat_schedule(int64_t n, int64_t leaf_threshold) {
    if (n <= 0) {
        throw std::invalid_argument("build_flat_schedule: n must be positive.");
    }
    if (leaf_threshold <= 0) {
        throw std::invalid_argument("build_flat_schedule: leaf_threshold must be positive.");
    }

    FlatSchedule schedule;
    schedule.split_flags.assign(static_cast<std::size_t>(std::max<int64_t>(0, n - 1)), 0);

    FlatLevel root_level;
    root_level.capacity_n = static_cast<int32_t>(n);
    root_level.nodes.push_back(FlatNode{
        .offset = 0,
        .logical_n = static_cast<int32_t>(n),
        .left_n = 0,
        .right_n = 0,
    });
    schedule.levels.push_back(std::move(root_level));

    while (schedule.levels.back().capacity_n > leaf_threshold) {
        const FlatLevel& current = schedule.levels.back();
        FlatLevel next;
        next.capacity_n = 0;
        next.nodes.reserve(current.nodes.size() * 2);

        for (const FlatNode& node : current.nodes) {
            const int32_t left_n = node.logical_n / 2;
            const int32_t right_n = node.logical_n - left_n;
            if (left_n <= 0 || right_n <= 0) {
                throw std::runtime_error("build_flat_schedule: invalid split produced an empty child.");
            }

            const int32_t boundary = node.offset + left_n - 1;
            if (boundary >= 0 && boundary < static_cast<int32_t>(schedule.split_flags.size())) {
                schedule.split_flags[static_cast<std::size_t>(boundary)] = 1;
            }

            next.nodes.push_back(FlatNode{
                .offset = node.offset,
                .logical_n = left_n,
                .left_n = 0,
                .right_n = 0,
            });
            next.nodes.push_back(FlatNode{
                .offset = node.offset + left_n,
                .logical_n = right_n,
                .left_n = 0,
                .right_n = 0,
            });
            next.capacity_n = std::max(next.capacity_n, std::max(left_n, right_n));
        }

        schedule.levels.push_back(std::move(next));
    }

    for (std::size_t level_ix = 0; level_ix + 1 < schedule.levels.size(); ++level_ix) {
        auto& level = schedule.levels[level_ix];
        const auto& child_level = schedule.levels[level_ix + 1];
        for (std::size_t node_ix = 0; node_ix < level.nodes.size(); ++node_ix) {
            const auto& left = child_level.nodes[node_ix * 2];
            const auto& right = child_level.nodes[node_ix * 2 + 1];
            level.nodes[node_ix].left_n = left.logical_n;
            level.nodes[node_ix].right_n = right.logical_n;
        }
    }

    return schedule;
}

inline std::size_t flat_level_super_batch_size(const FlatLevel& level, std::size_t batch_size) {
    return level.nodes.size() * batch_size;
}

inline std::size_t flat_level_matrix_elements(const FlatLevel& level, std::size_t batch_size) {
    const std::size_t cap = static_cast<std::size_t>(level.capacity_n);
    return cap * cap * flat_level_super_batch_size(level, batch_size);
}

inline std::size_t flat_level_vector_elements(const FlatLevel& level, std::size_t batch_size) {
    const std::size_t cap = static_cast<std::size_t>(level.capacity_n);
    return cap * flat_level_super_batch_size(level, batch_size);
}

inline const FlatLevel& flat_leaf_level(const FlatSchedule& schedule) {
    return schedule.levels.back();
}

} // namespace batchlas
