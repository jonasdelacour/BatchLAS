// SPDX-License-Identifier: MIT
#pragma once
// The host route's ILU(k) symbolic phase as the device::IlukPlan device::iluk_factorize takes.

#include <batchlas/blas/functions/iluk.hh>
#include <batchlas/device/sparse.hh>
#include <batchlas/util/sycl-vector.hh>
#include <src/util/internal-api.hh>

#include <array>

namespace batchlas::sycl_iluk {

// One shared-USM allocation; plan() points into it: it must outlive kernels that captured a plan.
struct IlukPlanStorage {
    static constexpr int kArrays = 15;  // device::IlukPlan's pointers, in its order
    UnifiedVector<int> ints;
    std::array<int, kArrays> offsets{};
    int n = 0, a_nnz = 0, nnz = 0, levels = 0, u_levels = 0;

    device::IlukPlan plan() const {
        const int* p = ints.data();
        const auto at = [&](int k) { return p + offsets[static_cast<std::size_t>(k)]; };
        return {n,     a_nnz, nnz,   levels, u_levels, at(0),  at(1),  at(2),  at(3), at(4),
                at(5), at(6), at(7), at(8),  at(9),    at(10), at(11), at(12), at(13), at(14)};
    }
};

// ILU(levels_of_fill) on item 0's pattern, validated as iluk_factorize (no per-item pattern check).
template <typename T>
BATCHLAS_INTERNAL_API IlukPlanStorage iluk_plan(const MatrixView<T, MatrixFormat::CSR>& A, const ILUKParams<T>& params);

}  // namespace batchlas::sycl_iluk
