#pragma once

// The NEW potrf selection path (descriptor registry), side by side with the shipped one.
// Sizing and running call the same select(); run() consumes the Selection it returns.

#include "../../../src/backends/potrf_routes.hh"
#include "../../../src/backends/potrf_windows.hh"

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

// Strategy binding for (arch, op, dtype); `strategy` overrides it for A/B and bisects.
enum class StrategyOverride : std::uint8_t { Auto, ModelOnly, WindowsOnly };

template <Backend B, class T, bool Vendor = dispatch::solver_vendor_available<B>>
Sel<B, T, Vendor> select(Queue& q, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo,
                         Span<std::byte> ws = {}, Span<int32_t> info = {},
                         Mode mode = Mode::Full, StrategyOverride so = StrategyOverride::Auto);

template <Backend B, class T, bool Vendor = dispatch::solver_vendor_available<B>>
Event run(Queue& q, const Sel<B, T, Vendor>& sel, const MatrixView<T, MatrixFormat::Dense>& A,
          Uplo uplo, Span<std::byte> ws, Span<int32_t> info);

template <Backend B, class T>
std::size_t potrf_buffer_size(Queue& q, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo);

template <Backend B, class T>
Event potrf(Queue& q, const MatrixView<T, MatrixFormat::Dense>& A, Uplo uplo,
            Span<std::byte> ws, Span<int32_t> info = {});

template <Backend B, class T, bool Vendor = dispatch::solver_vendor_available<B>>
std::string explain(const Sel<B, T, Vendor>& sel);

// Test hooks: warnings emitted so far, and the pin policy (default from BATCHLAS_PIN_POLICY).
std::size_t pin_warnings_emitted();
void set_pin_policy(std::optional<ds::PinPolicy> p);

}  // namespace batchlas::potrf_v2
