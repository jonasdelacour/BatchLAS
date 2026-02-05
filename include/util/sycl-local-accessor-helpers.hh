// SPDX-License-Identifier: MIT
// Copyright (c) 2025 Jonas deLacour
#pragma once

#include <sycl/sycl.hpp>

namespace batchlas {
namespace util {

/**
 * @brief Helper function to get a raw pointer from a SYCL local accessor.
 * 
 * This function provides a convenient wrapper around the verbose
 * local_accessor::get_multi_ptr<sycl::access::decorated::no>().get() syntax,
 * avoiding the deprecation warning for get_pointer().
 * 
 * @tparam T The element type of the local accessor
 * @tparam Dims The dimensionality of the accessor
 * @param accessor The local accessor to get a pointer from
 * @return T* Raw pointer to the local accessor's storage
 */
template <typename T, int Dims = 1>
inline T* get_raw_ptr(const sycl::local_accessor<T, Dims>& accessor) {
    return accessor.template get_multi_ptr<sycl::access::decorated::no>().get();
}

} // namespace util
} // namespace batchlas
