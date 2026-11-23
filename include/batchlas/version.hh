// SPDX-License-Identifier: MIT
#pragma once
/// @file
/// @brief The BatchLAS release this header belongs to, as preprocessor constants.
///
/// Plain macros and no includes, so it works header-only (`-I <checkout>/include`) and in device
/// code. `project(BatchLAS VERSION ...)` in the top-level CMakeLists.txt is the source of the
/// number; configure fails when this file disagrees with it (cmake/BatchLASVersion.cmake). The
/// compatibility promise behind the numbers: @ref guide_cpp_api, "Versioning and compatibility".
/// @ingroup api_config

/// @brief Major version. A change means the API or ABI covered by the promise broke.
#define BATCHLAS_VERSION_MAJOR 1
/// @brief Minor version. A change adds API without breaking any.
#define BATCHLAS_VERSION_MINOR 0
/// @brief Patch version. A change fixes behaviour without changing the API.
#define BATCHLAS_VERSION_PATCH 0
/// @brief `"MAJOR.MINOR.PATCH"`, the same string as the CMake package and `batchlas.__version__`.
#define BATCHLAS_VERSION_STRING "1.0.0"
