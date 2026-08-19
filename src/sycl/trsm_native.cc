// Native batched TRSM — the kernel translation unit.
//
// WP3 step 2. At this step the TU exists and defines exactly one thing: the
// per-type register capacity, and it reports ZERO for every type, meaning "no
// native trsm kernel in this build".
//
// That is not a placeholder in the pejorative sense — it is the step's whole
// point. RouteTable<Op::trsm,T>::supports() reads the capacity through
// TrsmShape::cta_max_n and reports both native routes UNSUPPORTED when it is
// zero (include/batchlas/blas/dispatch/route_trsm.hh), so this file changes no
// behaviour anywhere while establishing:
//
//   * the translation unit and its CMake wiring, so the next step edits an
//     existing file rather than adding one;
//   * the link-time delta. src/sycl/ is its own device-link unit and the
//     library link already measures ~43.4 s with only gemm_kernels.cc in it
//     (scripts/register_probe.sh). WP3_TRSM_SPEC.md:713 budgets against "~30 s",
//     a figure already breached before WP3 began, so the budget has to be a
//     delta against the measured baseline and this is where that starts.
//
// The capacities become real numbers in the next step, and only after
// scripts/register_probe.sh has read the actual register counts of the
// instantiated kernels. The rule there, which is NOT the rule the spec states:
// pass iff `0 bytes spill stores, 0 bytes spill loads` AND
// `registers x work-group size <= 65536`. The spec's `stack frame == 0` gate
// rejects spill-free kernels — 220 of 376 entry functions in this library
// currently carry a non-zero stack frame with zero spills.

#include "trsm_native.hh"

#include <complex>

namespace batchlas::sycl_trsm {

// ZERO EVERYWHERE, ON PURPOSE. See the header. Each of these gets its measured
// value in the step that adds the kernel it describes, one type at a time, with
// the register-probe output quoted at the site.
template <> int trsm_cta_max_n<float>()                { return 0; }
template <> int trsm_cta_max_n<double>()               { return 0; }
template <> int trsm_cta_max_n<std::complex<float>>()  { return 0; }
template <> int trsm_cta_max_n<std::complex<double>>() { return 0; }

} // namespace batchlas::sycl_trsm
