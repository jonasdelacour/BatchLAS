#include <util/minibench.hh>

#include <blas/device.hh>
#include <blas/extensions.hh>
#include <blas/matrix.hh>

#include "bench_utils.hh"

#include <batchlas/backend_config.h>

#include "../src/math-helpers.hh"
#include "../src/queue.hh"

#include <util/bench_structured.hh>
#include <util/group-invoke.hh>
#include <util/sycl-local-accessor-helpers.hh>

#include <algorithm>
#include <array>
#include <chrono>
#include <complex>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

#if __has_include(<sycl/ext/oneapi/experimental/clock.hpp>)
#include <sycl/ext/oneapi/experimental/clock.hpp>
#define BATCHLAS_LATRD_TUNING_HAS_ONEAPI_CLOCK 1
#else
#define BATCHLAS_LATRD_TUNING_HAS_ONEAPI_CLOCK 0
#endif

using namespace batchlas;

namespace {

constexpr int kLatrdDefaultIb = 32;

enum class TimedSection : int {
	PreloadDiag = 0,
	TailUpdate,
	LarfgBuildV,
	Hemv,
	Corrections,
	WFinalize,
	FusedUpdate,
	Total,
	Count,
};

constexpr std::size_t kTimedSectionCount = static_cast<std::size_t>(TimedSection::Count);

constexpr std::array<const char*, kTimedSectionCount> kTimedSectionMetricNames = {
	"%diag",
	"%tail",
	"%larfg",
	"%hemv",
	"%corr",
	"%wfin",
	"%fused",
	"Cyc/mat",
};

template <typename T>
inline T hermitian_diagonal(const T& value) {
	return value;
}

template <typename Real>
inline std::complex<Real> hermitian_diagonal(const std::complex<Real>& value) {
	return std::complex<Real>(value.real(), Real(0));
}

template <typename T, int WG, bool FuseTrailingUpdate, device::DeviceBlasPolicy HemvPolicy>
class LatrdLowerPanelTuningKernel;

template <typename Item>
inline std::uint64_t read_work_group_clock(const Item&) {
#if defined(__SYCL_DEVICE_ONLY__)
	return static_cast<std::uint64_t>(__builtin_readcyclecounter());
#else
	return 0;
#endif
}

inline std::size_t timed_section_offset(int batch_index, TimedSection section) {
	return static_cast<std::size_t>(batch_index) * kTimedSectionCount + static_cast<std::size_t>(section);
}

template <typename T>
inline std::array<double, kTimedSectionCount> reduce_section_cycles(const UnifiedVector<T>& section_cycles, std::size_t batch) {
	std::array<double, kTimedSectionCount> totals{};
	for (std::size_t b = 0; b < batch; ++b) {
		const std::size_t base = b * kTimedSectionCount;
		for (std::size_t section = 0; section < kTimedSectionCount; ++section) {
			totals[section] += static_cast<double>(section_cycles[base + section]);
		}
	}
	return totals;
}

inline device::DeviceBlasPolicy parse_device_blas_policy(int code) {
	switch (code) {
		case 0:
			return device::DeviceBlasPolicy::Auto;
		case 1:
			return device::DeviceBlasPolicy::Generic;
		case 2:
			return device::DeviceBlasPolicy::Subgroup16;
		case 3:
			return device::DeviceBlasPolicy::Subgroup32;
		default:
			throw std::invalid_argument("latrd_lower_panel_tuning_benchmark: invalid hemv policy");
	}
}

template <typename Benchmark>
inline void LatrdLowerPanelTuningBenchSizes(Benchmark* b) {
	for (int n : {64, 128, 256, 512}) {
		for (int hemv_policy : {0, 1, 2, 3}) {
			b->Args({n, 1024, kLatrdDefaultIb, 0, 0, 0, hemv_policy});
			b->Args({n, 1024, kLatrdDefaultIb, 0, 1, 0, hemv_policy});
		}
	}
}

template <typename T, int WG, bool FuseTrailingUpdate, device::DeviceBlasPolicy HemvPolicy>
Event latrd_lower_panel_tuning_batched_wg(Queue& q,
										  const MatrixView<T, MatrixFormat::Dense>& a,
										  const VectorView<T>& e,
										  const VectorView<T>& tau,
										  const MatrixView<T, MatrixFormat::Dense>& w,
										  const Span<std::uint64_t>& section_cycles) {
	constexpr int wg = WG;

	(void)q->submit([&](sycl::handler& h) {
		auto A_view = a.kernel_view();
		auto W_view = w.kernel_view();
		VectorView<T> E_view = e;
		VectorView<T> TAU_view = tau;
		const std::uint64_t* section_base = section_cycles.data();

		const int n = A_view.rows();
		const int batch = A_view.batch_size();
		const int ib = W_view.cols();

		auto v_local = sycl::local_accessor<T, 1>(sycl::range<1>(static_cast<std::size_t>(n)), h);
		auto wcol_local = sycl::local_accessor<T, 1>(sycl::range<1>(static_cast<std::size_t>(n)), h);
		auto vip_local = sycl::local_accessor<T, 1>(sycl::range<1>(static_cast<std::size_t>(ib)), h);
		auto wip_local = sycl::local_accessor<T, 1>(sycl::range<1>(static_cast<std::size_t>(ib)), h);

		h.parallel_for<LatrdLowerPanelTuningKernel<T, WG, FuseTrailingUpdate, HemvPolicy>>(
			sycl::nd_range<1>(sycl::range<1>(static_cast<std::size_t>(batch) * wg), sycl::range<1>(wg)),
			[=](sycl::nd_item<1> it) {
				const int b = static_cast<int>(it.get_group_linear_id());
				if (b >= batch) return;

				const int lid = static_cast<int>(it.get_local_linear_id());
				const sycl::group<1> g = it.get_group();
				T* v_ptr = util::get_raw_ptr(v_local);
				T* wcol_ptr = util::get_raw_ptr(wcol_local);
				T* vip_ptr = util::get_raw_ptr(vip_local);
				T* wip_ptr = util::get_raw_ptr(wip_local);
				std::uint64_t* section_ptr = const_cast<std::uint64_t*>(section_base) + b * static_cast<int>(kTimedSectionCount);

				auto Ab = A_view.batch_item(b);
				auto Wb = W_view.batch_item(b);

				std::uint64_t total_start = 0;
				if (lid == 0) {
					total_start = read_work_group_clock(it);
				}

				for (int i = 0; i < ib; ++i) {
					if (i >= n - 1) break;
					const int tail = n - (i + 1);
					auto a_col_tail = Ab(Slice(i + 1, SliceEnd()), i);
					auto v_tail = VectorView<T>(v_ptr + i + 1, tail);
					auto wcol_tail = VectorView<T>(wcol_ptr + i + 1, tail);

					std::uint64_t section_start = 0;

					if (lid == 0) {
						section_start = read_work_group_clock(it);
					}
					if (i > 0) {
						auto vip_view = VectorView<T>(vip_ptr, i);
						auto wip_view = VectorView<T>(wip_ptr, i);
						batchlas::device::copy(g, Ab(i, Slice(0, i)), vip_view);
						batchlas::device::copy(g, Wb(i, Slice(0, i)), wip_view);
					}
					it.barrier(sycl::access::fence_space::local_space);

					if (i > 0) {
						const auto vip_view = VectorView<T>(vip_ptr, i);
						const auto wip_view = VectorView<T>(wip_ptr, i);
						const T panel_dot = batchlas::device::dotc(g, vip_view, wip_view);
						if (lid == 0) {
							Ab(i, i) = hermitian_diagonal(Ab(i, i) - panel_dot - batchlas::device::detail::conj(panel_dot));
						}
					}
					if (i > 0) {
						it.barrier(sycl::access::fence_space::local_space);
					}
					if (lid == 0) {
						section_ptr[static_cast<int>(TimedSection::PreloadDiag)] += read_work_group_clock(it) - section_start;
						section_start = read_work_group_clock(it);
					}

					if (i > 0) {
						for (int p = 0; p < i; ++p) {
							auto v_prev = Ab(Slice(i + 1, SliceEnd()), p);
							auto w_prev = Wb(Slice(i + 1, SliceEnd()), p);
							batchlas::device::hadamard(g,
													   a_col_tail,
													   [&](T x, T w_prev_entry, T v_prev_entry) {
														   return x - batchlas::device::detail::conj(wip_ptr[p]) * v_prev_entry -
																	  batchlas::device::detail::conj(vip_ptr[p]) * w_prev_entry;
													   },
													   a_col_tail,
													   w_prev,
													   v_prev);
						}
					}
					it.barrier(sycl::access::fence_space::global_space);
					if (lid == 0) {
						section_ptr[static_cast<int>(TimedSection::TailUpdate)] += read_work_group_clock(it) - section_start;
						section_start = read_work_group_clock(it);
					}

					const int x0 = i + 2;
					T alpha_i = i + 1 < n ? Ab(i + 1, i) : T(0);
					const T tau_i = internal::larfg(g, alpha_i, Ab(Slice(x0, SliceEnd()), i));
					if (lid == 0) {
						E_view(i, b) = alpha_i;
						TAU_view(i, b) = tau_i;
						Ab(i + 1, i) = T(1);
					}

					it.barrier(sycl::access::fence_space::local_space);
					if (lid == 0) {
						v_ptr[i + 1] = T(1);
					}
					if (x0 < n) {
						batchlas::device::copy(g, Ab(Slice(x0, SliceEnd()), i), VectorView<T>(v_ptr + x0, n - x0));
					}
					it.barrier(sycl::access::fence_space::local_space);
					if (lid == 0) {
						section_ptr[static_cast<int>(TimedSection::LarfgBuildV)] += read_work_group_clock(it) - section_start;
						section_start = read_work_group_clock(it);
					}

					auto trailing_view = KernelMatrixView<T, MatrixFormat::Dense>(
						Ab.data() + (i + 1) + (i + 1) * Ab.ld(),
						tail,
						tail,
						Ab.ld());
					batchlas::device::hemv<HemvPolicy, Uplo::Lower>(
						g,
						trailing_view,
						v_tail,
						wcol_tail,
						T(1),
						T(0));
					it.barrier(sycl::access::fence_space::local_space);
					if (lid == 0) {
						section_ptr[static_cast<int>(TimedSection::Hemv)] += read_work_group_clock(it) - section_start;
						section_start = read_work_group_clock(it);
					}
					
					for (int p = 0; p < i; ++p) {
						const int pc = p;
						const auto vp_tail = Ab(Slice(i + 1), p);
						const auto wp_tail = Wb(Slice(i + 1), pc);
						const T gamma = batchlas::device::dotc(g, wp_tail, v_tail);
						const T delta = batchlas::device::dotc(g, vp_tail, v_tail);
						for (int r = i + 1 + lid; r < n; r += wg) {
							wcol_local[r] -= vp_tail(r - (i + 1)) * gamma + wp_tail(r - (i + 1)) * delta;
						}
					}
					it.barrier(sycl::access::fence_space::local_space);
					if (lid == 0) {
						section_ptr[static_cast<int>(TimedSection::Corrections)] += read_work_group_clock(it) - section_start;
						section_start = read_work_group_clock(it);
					}

					batchlas::device::scal(g, wcol_tail, tau_i);
					it.barrier(sycl::access::fence_space::local_space);
					const T dot = batchlas::device::dotc(g, v_tail, wcol_tail);
					const T alpha2 = T(-0.5) * tau_i * dot;
					batchlas::device::axpy(g, v_tail, wcol_tail, alpha2);
					it.barrier(sycl::access::fence_space::local_space);
					batchlas::device::copy(g, wcol_tail, Wb(Slice(i + 1, SliceEnd()), i));
					it.barrier(sycl::access::fence_space::global_space);
					if (lid == 0) {
						section_ptr[static_cast<int>(TimedSection::WFinalize)] += read_work_group_clock(it) - section_start;
					}
				}

				if constexpr (FuseTrailingUpdate) {
					const int j2 = ib;
					const int n2 = n - j2;
					if (n2 > 0) {
						std::uint64_t section_start = 0;
						if (lid == 0) {
							section_start = read_work_group_clock(it);
						}
						batchlas::device::her2k<device::DeviceBlasPolicy::Subgroup16,
														Uplo::Lower,
														Transpose::NoTrans>(
							g,
							Ab(Slice(j2), Slice(0, ib)),
							Wb(Slice(j2), Slice(0, ib)),
							Ab(Slice(j2), Slice(j2)),
							T(-1),
							T(1));
						if (lid == 0) {
							section_ptr[static_cast<int>(TimedSection::FusedUpdate)] += read_work_group_clock(it) - section_start;
						}
					}
				}

				if (lid == 0) {
					section_ptr[static_cast<int>(TimedSection::Total)] += read_work_group_clock(it) - total_start;
				}
			});
	});

	return q.get_event();
}

template <typename T, device::DeviceBlasPolicy HemvPolicy>
Event latrd_lower_panel_tuning_dispatch_wg(Queue& q,
										   const MatrixView<T, MatrixFormat::Dense>& a,
										   const VectorView<T>& e,
										   const VectorView<T>& tau,
										   const MatrixView<T, MatrixFormat::Dense>& w,
										   const Span<std::uint64_t>& section_cycles,
										   int32_t wg_hint,
										   bool fuse_trailing_update) {
	const int n = a.rows();
	auto call = [&](auto wg_tag) {
		constexpr int WG = decltype(wg_tag)::value;
		if (fuse_trailing_update) {
			return latrd_lower_panel_tuning_batched_wg<T, WG, true, HemvPolicy>(q, a, e, tau, w, section_cycles);
		}
		return latrd_lower_panel_tuning_batched_wg<T, WG, false, HemvPolicy>(q, a, e, tau, w, section_cycles);
	};

	if (wg_hint == 64) {
		return call(std::integral_constant<int, 64>{});
	}
	if (wg_hint == 128) {
		return call(std::integral_constant<int, 128>{});
	}
	if (wg_hint == 256) {
		return call(std::integral_constant<int, 256>{});
	}
	if (n <= 128) {
		return call(std::integral_constant<int, 64>{});
	}
	if (n <= 256) {
		return call(std::integral_constant<int, 128>{});
	}
	return call(std::integral_constant<int, 256>{});
}

template <typename T>
Event latrd_lower_panel_tuning_launch(Queue& q,
									  const MatrixView<T, MatrixFormat::Dense>& a,
									  const VectorView<T>& e,
									  const VectorView<T>& tau,
									  const MatrixView<T, MatrixFormat::Dense>& w,
									  const Span<std::uint64_t>& section_cycles,
									  int32_t wg_hint,
									  bool fuse_trailing_update,
									  device::DeviceBlasPolicy hemv_policy) {
	switch (hemv_policy) {
		case device::DeviceBlasPolicy::Auto:
			return latrd_lower_panel_tuning_dispatch_wg<T, device::DeviceBlasPolicy::Auto>(q, a, e, tau, w, section_cycles, wg_hint, fuse_trailing_update);
		case device::DeviceBlasPolicy::Generic:
			return latrd_lower_panel_tuning_dispatch_wg<T, device::DeviceBlasPolicy::Generic>(q, a, e, tau, w, section_cycles, wg_hint, fuse_trailing_update);
		case device::DeviceBlasPolicy::Subgroup16:
			return latrd_lower_panel_tuning_dispatch_wg<T, device::DeviceBlasPolicy::Subgroup16>(q, a, e, tau, w, section_cycles, wg_hint, fuse_trailing_update);
		case device::DeviceBlasPolicy::Subgroup32:
			return latrd_lower_panel_tuning_dispatch_wg<T, device::DeviceBlasPolicy::Subgroup32>(q, a, e, tau, w, section_cycles, wg_hint, fuse_trailing_update);
	}
	throw std::invalid_argument("latrd_lower_panel_tuning_benchmark: unsupported hemv policy");
}

inline bool latrd_lower_panel_work_group_clock_supported(const Queue& q) {
	(void)q;
	return true;
}

} // namespace

template <typename T, Backend B>
static void BM_LATRD_LOWER_PANEL_TUNING(minibench::State& state) {
#if BATCHLAS_HAS_CUDA_BACKEND
	const std::size_t n = state.range(0);
	const std::size_t batch = state.range(1);
	const int ib = state.range(2);
	const int j0 = state.range(3);
	const bool fuse_trailing_update = state.range(4) != 0;
	const int32_t wg_hint = static_cast<int32_t>(state.range(5));
	const device::DeviceBlasPolicy hemv_policy = parse_device_blas_policy(state.range(6));

	auto q = std::make_shared<Queue>("gpu", true);
	if (!latrd_lower_panel_work_group_clock_supported(*q)) {
		throw std::runtime_error("latrd_lower_panel_tuning_benchmark: device does not support aspect::ext_oneapi_clock_work_group");
	}

	const double approx_flops = 2.0 * double(n) * double(n) * double(ib) * double(batch);

	auto A = std::make_shared<Matrix<T>>(Matrix<T>::Random(n, n, true, batch, 2026));
	auto e = std::make_shared<Vector<T>>(Vector<T>::zeros(n - 1, batch));
	auto tau = std::make_shared<Vector<T>>(Vector<T>::zeros(n - 1, batch));
	auto W = std::make_shared<Matrix<T>>(Matrix<T>::Zeros(n, std::max(1, ib), batch));
	auto section_cycles = std::make_shared<UnifiedVector<std::uint64_t>>(batch * kTimedSectionCount, 0);
	auto last_section_totals = std::make_shared<std::array<double, kTimedSectionCount>>();

	bench::ManagedInputs managed_inputs(q);
	managed_inputs.pristine(A);
	managed_inputs.pristine(e);
	managed_inputs.pristine(tau);
	managed_inputs.pristine(W);
	managed_inputs.prepare(*section_cycles);

	state.SetPrepare(managed_inputs.make_prepare_once());
	state.SetBeforeEachRun([reset = managed_inputs.make_before_each_run(), section_cycles]() mutable {
		reset();
		section_cycles->fill(0);
	});

	auto kernel_once = [q, A, e, tau, W, section_cycles, j0, ib, wg_hint, fuse_trailing_update, hemv_policy]() mutable {
		auto A_panel = A->view()({j0, SliceEnd()}, {j0, SliceEnd()});
		auto e_panel = VectorView<T>(*e)(Slice(j0, j0 + ib));
		auto tau_panel = VectorView<T>(*tau)(Slice(j0, j0 + ib));
		auto W_panel = W->view()({j0, SliceEnd()}, {0, ib});
		(void)latrd_lower_panel_tuning_launch<T>(*q,
												 A_panel,
												 e_panel,
												 tau_panel,
												 W_panel,
												 section_cycles->to_span(),
												 wg_hint,
												 fuse_trailing_update,
												 hemv_policy);
	};

	state.SetKernel(std::function<void()>(kernel_once));
	state.SetTimedKernelMs([q, kernel_once, section_cycles, last_section_totals, batch]() mutable -> double {
		const auto host_t0 = std::chrono::steady_clock::now();
		Event start = q->get_event();
		kernel_once();
		Event end = q->get_event();
		end.wait();
		const auto host_t1 = std::chrono::steady_clock::now();

		*last_section_totals = reduce_section_cycles(*section_cycles, batch);

		const double prof_ms = bench_event_elapsed_ms(start, end);
		if (prof_ms >= 0.0) {
			return prof_ms;
		}
		return std::chrono::duration<double, std::milli>(host_t1 - host_t0).count();
	});
	state.SetBatchEndWait(q);
	state.SetMetricsFunc([last_section_totals, batch](minibench::Result& res) {
		const double total_cycles = (*last_section_totals)[static_cast<std::size_t>(TimedSection::Total)];
		const double cycles_per_matrix = batch > 0 ? total_cycles / static_cast<double>(batch) : 0.0;
		res.metrics[kTimedSectionMetricNames[static_cast<std::size_t>(TimedSection::Total)]] = cycles_per_matrix;

		double covered_pct = 0.0;
		for (std::size_t section = 0; section < static_cast<std::size_t>(TimedSection::Total); ++section) {
			const double pct = total_cycles > 0.0 ? 100.0 * (*last_section_totals)[section] / total_cycles : 0.0;
			res.metrics[kTimedSectionMetricNames[section]] = pct;
			covered_pct += pct;
		}
		res.metrics["%other"] = std::max(0.0, 100.0 - covered_pct);
	});

	state.SetMetric("GFLOPS", approx_flops * 1e-9, minibench::Rate);
	state.SetMetric("T(µs)/matrix", (1.0 / double(batch)) * 1e6, minibench::Reciprocal);
#else
	(void)state;
	throw std::runtime_error("latrd_lower_panel_tuning_benchmark requires the CUDA SYCL benchmark build path");
#endif
}

BATCHLAS_BENCH_CUDA_ALL_TYPES(BM_LATRD_LOWER_PANEL_TUNING, LatrdLowerPanelTuningBenchSizes);

MINI_BENCHMARK_MAIN();
