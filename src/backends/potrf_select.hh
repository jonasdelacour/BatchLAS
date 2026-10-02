#pragma once

// The potrf selection path built on the descriptor registry (src/backends/potrf_routes.hh and
// src/dispatch/selection/). The CUDA facade (potrf, potrf_buffer_size) runs through it.
// evidence: experiments/kernel_selection/descriptor-registry/README.md

#include "potrf_routes.hh"
#include "potrf_windows.hh"

#include "../dispatch/selection/run.hh"
#include "../util/internal-api.hh"

#include <optional>
#include <string>

namespace batchlas::potrf_v2 {

namespace ds = dispatch::sel;
namespace pr = potrf_routes;

template <Backend B, class T, bool Vendor = dispatch::solver_vendor_available<B>>
using Tbl = pr::PotrfTable<B, T, Vendor>;

template <Backend B, class T, bool Vendor = dispatch::solver_vendor_available<B>>
using Sel = ds::Selection<Tbl<B, T, Vendor>>;

enum class Mode : std::uint8_t { Full, DecideOnly };   // DecideOnly: no workspace query

// Strategy binding for (arch, op, dtype); the override is for A/B and bisects.
enum class StrategyOverride : std::uint8_t { Auto, ModelOnly, WindowsOnly };

// Vendor = false is instantiated everywhere; Vendor = true only where a vendor potrf is linked.
template <Backend B, class T, bool Vendor = dispatch::solver_vendor_available<B>>
BATCHLAS_INTERNAL_API Sel<B, T, Vendor> select(Queue& q, const MatrixView<T, MatrixFormat::Dense>& A,
                                               Uplo uplo, Mode mode = Mode::Full,
                                               StrategyOverride so = StrategyOverride::Auto);

template <Backend B, class T, bool Vendor = dispatch::solver_vendor_available<B>>
BATCHLAS_INTERNAL_API Event run(Queue& q, const Sel<B, T, Vendor>& sel,
                                const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo,
                                Span<std::byte> ws, Span<int32_t> info);

template <Backend B, class T, bool Vendor = dispatch::solver_vendor_available<B>>
BATCHLAS_INTERNAL_API std::string explain(const Sel<B, T, Vendor>& sel);

// What the facade calls: the chosen route's own workspace, and select + run fused.
template <Backend B, class T>
BATCHLAS_INTERNAL_API std::size_t buffer_size(Queue& q, const MatrixView<T, MatrixFormat::Dense>& A,
                                              Uplo uplo);
template <Backend B, class T>
BATCHLAS_INTERNAL_API Event potrf(Queue& q, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo,
                                  Span<std::byte> ws, Span<int32_t> info);

// Test hooks: unhonoured-pin warnings emitted so far, and the pin policy override.
BATCHLAS_INTERNAL_API std::size_t pin_warnings_emitted();
BATCHLAS_INTERNAL_API void set_pin_policy(std::optional<ds::PinPolicy> p);

}  // namespace batchlas::potrf_v2

namespace batchlas::backend {

// The pre-registry facade (resolve + if-chain, max-over-tiers sizing): the non-CUDA potrf, and
// the reference the registry path is compared against. Defined in entry_points/factorization.cc.
template <Backend B, typename T>
BATCHLAS_INTERNAL_API Event potrf_legacy(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                                         Uplo uplo, Span<std::byte> workspace, Span<int32_t> info);
template <Backend B, typename T>
BATCHLAS_INTERNAL_API std::size_t potrf_buffer_size_legacy(
    Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo);

}  // namespace batchlas::backend
