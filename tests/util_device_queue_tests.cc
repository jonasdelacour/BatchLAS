// SPDX-License-Identifier: MIT
#include <gtest/gtest.h>
#include <batchlas/util/env.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/kernel-heuristics.hh>
#include <batchlas/backend_config.h>

#include <batchlas/sycl_interop.hh>
#include <batchlas/blas/linalg.hh>
#include <batchlas/blas/matrix.hh>
#include <tests/test_utils.hh>
#include <complex>
#include <iostream>
#include <src/queue.hh>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <vector>
#ifdef BATCHLAS_UTIL_TESTS_HAVE_CUDART
#include <cuda_runtime_api.h>
#endif

TEST(DeviceTest, DefaultConstruction) {
    Device device;
    EXPECT_EQ(device.idx, 0);
    EXPECT_EQ(device.type, DeviceType::HOST);
}

TEST(DeviceTest, IndexAndTypeConstruction) {
    Device device(1, DeviceType::CPU);
    EXPECT_EQ(device.idx, 1);
    EXPECT_EQ(device.type, DeviceType::CPU);
}

TEST(DeviceTest, GetDevices) {
    auto cpus = Device::get_devices(DeviceType::CPU);
    auto gpus = Device::get_devices(DeviceType::GPU);
    auto accelerators = Device::get_devices(DeviceType::ACCELERATOR);
    // Every entry must name its own device: get_devices() once returned idx 0 for
    // all of them, so gpus.at(1) silently was GPU 0.
    for (const auto* list : {&cpus, &gpus, &accelerators}) {
        for (size_t i = 0; i < list->size(); ++i) EXPECT_EQ((*list)[i].idx, i);
    }

    // We can't guarantee specific hardware is available on the test system
    // But we can at least check that the API returns something reasonable
    EXPECT_NO_THROW({
        auto default_device = Device::default_device();
    });
}

TEST(DeviceTest, StringConstruction) {
    // Test might fail if specific hardware isn't available, so we'll wrap in try/catch
    try {
        Device cpu_device("cpu");
        EXPECT_EQ(cpu_device.type, DeviceType::CPU);
    } catch (const std::runtime_error&) {
        // No CPU device available, that's ok for the test
    }
}

TEST(DeviceTest, DeviceProperties) {
    try {
        // Get default device, which should always be available
        Device device = Device::default_device();

        // Test getting various properties
        EXPECT_NO_THROW({
            size_t wg_size = device.get_property(DeviceProperty::MAX_WORK_GROUP_SIZE);
            size_t compute_units = device.get_property(DeviceProperty::MAX_COMPUTE_UNITS);
        });
        // Was a stderr print and a silent 0.
        EXPECT_THROW((void)device.get_property(static_cast<DeviceProperty>(999)), batchlas::invalid_argument);

        // Device name and vendor should return something
        EXPECT_FALSE(device.get_name().empty());
        EXPECT_FALSE(device.get_vendor() == Vendor::OTHER);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Skipping device property tests due to no devices available";
    }
}

TEST(EventTest, Basic) {
    Event event;

    // Basic construction and move operations should work
    EXPECT_NO_THROW({
        Event event2;
        Event event3 = std::move(event2);
    });
}

TEST(QueueTest, DefaultConstruction) {
    EXPECT_NO_THROW({
        Queue queue;
    });
}

TEST(QueueTest, DeviceConstruction) {
    try {
        // Get default device and create queue
        Device device = Device::default_device();

        EXPECT_NO_THROW({
            Queue queue(device);
            EXPECT_EQ(queue.device().idx, device.idx);
            EXPECT_EQ(queue.device().type, device.type);
            EXPECT_TRUE(queue.in_order());
        });

        EXPECT_NO_THROW({
            Queue queue(device, false);
            EXPECT_FALSE(queue.in_order());
        });
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Skipping queue tests due to no devices available";
    }
}

TEST(QueueTest, MoveOperations) {
    try {
        Device device = Device::default_device();

        Queue queue1(device);

        // Test move construction
        EXPECT_NO_THROW({
            Queue queue2 = std::move(queue1);
            EXPECT_EQ(queue2.device().idx, device.idx);
            EXPECT_EQ(queue2.device().type, device.type);
        });

        // Test move assignment
        EXPECT_NO_THROW({
            Queue queue3;
            queue3 = Queue(device);
            EXPECT_EQ(queue3.device().idx, device.idx);
            EXPECT_EQ(queue3.device().type, device.type);
        });
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Skipping queue tests due to no devices available";
    }
}

TEST(QueueTest, GetEvent) {
    try {
        Queue queue(Device::default_device());

        EXPECT_NO_THROW({
            Event event = queue.get_event();
        });
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Skipping event tests due to no devices available";
    }
}

TEST(QueueTest, EnqueueEvent) {
    try {
        Queue queue(Device::default_device());
        Event event = queue.get_event();

        EXPECT_NO_THROW({
            queue.enqueue(event);
        });
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Skipping enqueue tests due to no devices available";
    }
}

TEST(QueueTest, EnqueueMultipleEvents) {
    try {
        Queue queue(Device::default_device());
        std::vector<Event> events;

        for (int i = 0; i < 3; i++) {
            events.push_back(queue.get_event());
        }

        EXPECT_NO_THROW({
            queue.enqueue(events);
        });
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Skipping enqueue tests due to no devices available";
    }
}

TEST(QueueTest, WaitAndThrow) {
    try {
        Queue queue(Device::default_device());

        EXPECT_NO_THROW({
            queue.wait();
            queue.wait_and_throw();
        });
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Skipping wait tests due to no devices available";
    }
}

// Pins the contract that every kernel-geometry knob in src/ops now
// depends on. These call sites each used to carry their own atoi-plus-`> 0`
// parser; the shared helper only preserves them if it agrees on the whole input
// space, and the interesting half of that space (unset, empty, non-positive,
// unparseable, trailing junk) is exactly what no existing test reaches -- the
// knobs the suite already sets are all positive integers, where every candidate
// parser agrees trivially.
TEST(EnvHelpers, PositiveIntOrClampsAndFallsBack) {
    const char* kKey = "BATCHLAS_TEST_ENV_POSITIVE_INT";

    {   // unset
        batchlas::ScopedEnvVar v(kKey, nullptr);
        EXPECT_EQ(batchlas::env_positive_int_or(kKey, 7), 7);
    }
    {   // empty string: stoi throws, so fallback
        batchlas::ScopedEnvVar v(kKey, "");
        EXPECT_EQ(batchlas::env_positive_int_or(kKey, 7), 7);
    }
    {   // zero and negatives are "meaningless geometry", i.e. unset
        batchlas::ScopedEnvVar v(kKey, "0");
        EXPECT_EQ(batchlas::env_positive_int_or(kKey, 7), 7);
    }
    {
        batchlas::ScopedEnvVar v(kKey, "-3");
        EXPECT_EQ(batchlas::env_positive_int_or(kKey, 7), 7);
    }
    {   // unparseable falls back rather than silently reading as 0
        batchlas::ScopedEnvVar v(kKey, "abc");
        EXPECT_EQ(batchlas::env_positive_int_or(kKey, 7), 7);
    }
    {   // stoi takes the leading integer and ignores the tail -- documented here
        // because atoi did the same, so routing the old call sites through this
        // did not change them.
        batchlas::ScopedEnvVar v(kKey, "8junk");
        EXPECT_EQ(batchlas::env_positive_int_or(kKey, 7), 8);
    }
    {
        batchlas::ScopedEnvVar v(kKey, "16");
        EXPECT_EQ(batchlas::env_positive_int_or(kKey, 7), 16);
    }
}

// env_truthy/env_falsy take the VALUE, not the name, and an unset variable is
// neither -- that is what lets a caller tell "forced off" from "not specified".
TEST(EnvHelpers, TruthyAndFalsyAreNotComplements) {
    EXPECT_FALSE(batchlas::env_truthy(nullptr));
    EXPECT_FALSE(batchlas::env_falsy(nullptr));

    EXPECT_TRUE(batchlas::env_truthy("1"));
    EXPECT_TRUE(batchlas::env_truthy("true"));
    EXPECT_TRUE(batchlas::env_truthy("ON"));
    EXPECT_FALSE(batchlas::env_truthy("True"));  // exact spellings only

    EXPECT_TRUE(batchlas::env_falsy("0"));
    EXPECT_TRUE(batchlas::env_falsy("off"));
    EXPECT_FALSE(batchlas::env_falsy("False"));  // exact spellings only
}

// Per-architecture routing keys on this value, and 0 must mean "not CUDA" so every
// other device keeps the as-measured (sm_89) windows.
// evidence: docs/perf/blackwell.md#compute-capability-key
TEST(DeviceTest, CudaComputeCapabilityIsZeroOffCuda) {
    for (const Device& d : Device::get_devices(DeviceType::CPU)) {
        EXPECT_EQ(d.cuda_compute_capability(), 0) << d.get_name();
    }
    for (const Device& d : Device::get_devices(DeviceType::GPU)) {
        if (d.get_vendor() != Vendor::NVIDIA) EXPECT_EQ(d.cuda_compute_capability(), 0) << d.get_name();
    }
}

TEST(DeviceTest, CudaComputeCapabilityOnNvidiaGpu) {
    std::vector<Device> nv;
    for (const Device& d : Device::get_devices(DeviceType::GPU)) {
        if (d.get_vendor() == Vendor::NVIDIA) nv.push_back(d);
    }
    if (nv.empty()) GTEST_SKIP() << "no NVIDIA GPU visible";
#ifdef BATCHLAS_UTIL_TESTS_HAVE_CUDART
    // Independent oracle: the CUDA runtime, not the SYCL version string we parse.
    // Paired per device (the CUDA adapter enumerates in CUDA ordinal order, and the
    // name check proves the pairing), so on a mixed-GPU box one device reporting
    // another's cc still fails.
    int count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
    ASSERT_EQ(static_cast<size_t>(count), nv.size()) << "SYCL and CUDA see different NVIDIA GPUs";
#endif
    for (size_t i = 0; i < nv.size(); ++i) {
        const Device& d = nv[i];
        const int cc = d.cuda_compute_capability();
#if BATCHLAS_HAS_CUDA_BACKEND
        EXPECT_GE(cc, 50) << d.get_name();
        EXPECT_LT(cc, 1000) << d.get_name();
#endif
        EXPECT_EQ(d.cuda_compute_capability(), cc) << "memoized value changed";
#ifdef BATCHLAS_UTIL_TESTS_HAVE_CUDART
        cudaDeviceProp prop{};
        ASSERT_EQ(cudaGetDeviceProperties(&prop, static_cast<int>(i)), cudaSuccess);
        ASSERT_EQ(d.get_name(), std::string(prop.name)) << "SYCL/CUDA ordinal pairing broke at " << i;
        EXPECT_EQ(cc, prop.major * 10 + prop.minor) << d.get_name() << " (CUDA ordinal " << i << ")";
#endif
    }
}

// Straddle both edges of the family window: sm_120 and sm_121 are in; sm_89,
// sm_100 (datacenter Blackwell) and a future sm_130 are out.
TEST(DeviceTest, Sm120FamilyWindow) {
    using batchlas::is_sm120_family;
    static_assert(!is_sm120_family(0) && !is_sm120_family(89) && !is_sm120_family(100));
    EXPECT_FALSE(is_sm120_family(119));
    EXPECT_TRUE(is_sm120_family(120));
    EXPECT_TRUE(is_sm120_family(121));
    EXPECT_TRUE(is_sm120_family(129));
    EXPECT_FALSE(is_sm120_family(130));
}

// A launch the CUDA driver rejects (grid z = 65536 > 65535) must be a catchable sycl::exception
// at the submission on both SYCL implementations, never a std::terminate from a later ~queue
// (acpp reports every pending error from every queue destructor), and the queue must keep working.
// evidence: docs/design/sycl-implementations.md#sycl-impl-asynchronous-errors
namespace {
class OverLimitGridKernel;
class OverLimitGridRawKernel;
class OversizedLocalMemoryKernel;
class AfterFailureKernel;

const sycl::nd_range<3> kOverLimitGrid(sycl::range<3>(65536, 1, 1), sycl::range<3>(1, 1, 1));

std::optional<Queue> cuda_gpu_queue() {
    for (const Device& d : Device::get_devices(DeviceType::GPU)) {
        Queue q(d);
        if (batchlas::impl::is_cuda(batchlas::impl::backend_of(batchlas::sycl_queue(q)))) return q;
    }
    return std::nullopt;
}

// The queue survived: a kernel after the failure runs and its write lands.
void expect_queue_still_runs(Queue& q, int* out) {
    q->parallel_for<AfterFailureKernel>(sycl::nd_range<1>(32, 32), [=](sycl::nd_item<1> it) {
        if (it.get_global_id(0) == 0) *out = 7;
    });
    EXPECT_NO_THROW(q.wait());
    EXPECT_EQ(*out, 7);
}
}  // namespace

TEST(QueueAsyncErrors, OverLimitGridThrowsFromTheLaunch) {
    auto q = cuda_gpu_queue();
    if (!q) GTEST_SKIP() << "no CUDA GPU: the grid limit is the CUDA driver's";
    int* out = sycl::malloc_shared<int>(1, batchlas::sycl_queue(*q));
    *out = 0;
    EXPECT_THROW((*q)->parallel_for<OverLimitGridKernel>(kOverLimitGrid, [=](sycl::nd_item<3>) { *out = 1; }),
                 sycl::exception);
    { sycl::queue copy = batchlas::sycl_queue(*q); }  // acpp: ~queue reports what is pending
    EXPECT_NO_THROW(q->wait());
    EXPECT_EQ(*out, 0) << "the rejected kernel ran";
    expect_queue_still_runs(*q, out);
    sycl::free(out, batchlas::sycl_queue(*q));
}

// 1 MiB of local memory fits no GPU and passes the grid check: on acpp only the launch itself
// fails, so this is the error the post-submission check must turn into an exception.
TEST(QueueAsyncErrors, OversizedLocalMemoryThrowsFromTheLaunch) {
    auto q = cuda_gpu_queue();
    if (!q) GTEST_SKIP() << "no CUDA GPU";
    int* out = sycl::malloc_shared<int>(1, batchlas::sycl_queue(*q));
    *out = 0;
    EXPECT_THROW((*q)->submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 1> slm(sycl::range<1>(std::size_t{1} << 18), h);
        h.parallel_for<OversizedLocalMemoryKernel>(sycl::nd_range<1>(32, 32), [=](sycl::nd_item<1> it) {
            slm[it.get_local_id(0)] = 1;
            sycl::group_barrier(it.get_group());
            if (it.get_global_id(0) == 0) *out = slm[31];
        });
    }), sycl::exception);
    { sycl::queue copy = batchlas::sycl_queue(*q); }
    EXPECT_NO_THROW(q->wait());
    EXPECT_EQ(*out, 0) << "the rejected kernel ran";
    expect_queue_still_runs(*q, out);
    sycl::free(out, batchlas::sycl_queue(*q));
}

// Submitted around BatchLAS's own check: the error a copied queue's destructor collects is
// thrown by the next Queue::wait, and that destructor does not terminate the process.
TEST(QueueAsyncErrors, ErrorCollectedByAQueueCopySurfacesAtTheNextWait) {
    auto q = cuda_gpu_queue();
    if (!q) GTEST_SKIP() << "no CUDA GPU: the grid limit is the CUDA driver's";
    int* out = sycl::malloc_shared<int>(1, batchlas::sycl_queue(*q));
    *out = 0;
    bool thrown = false;
    try {
        sycl::queue& raw = batchlas::sycl_queue(*q);
        raw.parallel_for<OverLimitGridRawKernel>(kOverLimitGrid, [=](sycl::nd_item<3>) { *out = 1; });
        { sycl::queue copy = raw; }
        q->wait();
    } catch (const sycl::exception&) {
        thrown = true;
    }
    EXPECT_TRUE(thrown) << "the rejected launch was reported nowhere";
    EXPECT_EQ(*out, 0) << "the rejected kernel ran";
    expect_queue_still_runs(*q, out);
    sycl::free(out, batchlas::sycl_queue(*q));
}
// get_event() on an in-order queue returns the last RECORDED event. A USM memcpy/memset/fill the
// queue did not record ran on after a caller waited on that event: syev_blocked's final copy
// segfaulted its NETLIB test on both DPC++ and AdaptiveCpp. 256 MiB keeps each op running well
// past the wait; a recorded kernel goes first so get_event() takes the last-event path.
TEST(QueueTest, GetEventCoversUsmMemcpyMemsetAndFill) {
    constexpr size_t n = size_t(1) << 26;
    int devices_run = 0;
    for (const char* dev : {"cpu", "gpu"}) {
        std::optional<Queue> q;
        try {
            q.emplace(Device(dev), true);
        } catch (const std::exception&) {
            continue;
        }
        ++devices_run;
        sycl::queue& sq = batchlas::sycl_queue(*q);
        float* src = sycl::malloc_device<float>(n, sq);
        float* dst = sycl::malloc_host<float>(n, sq);
        int* marker = sycl::malloc_host<int>(1, sq);
        sq.fill(src, 1.5f, n).wait();
        const auto recorded_kernel = [&] {
            (*q)->parallel_for(sycl::range<1>(1), [=](sycl::id<1>) { *marker = 7; });
        };
        const auto check = [&](float want, const char* op) {
            q->get_event().wait();
            EXPECT_EQ(dst[0], want) << op << " on " << dev;
            EXPECT_EQ(dst[n - 1], want) << op << " on " << dev << ": still running after the wait";
            q->wait();
        };

        std::fill(dst, dst + n, 0.0f);
        recorded_kernel();
        (*q)->memcpy(dst, src, n * sizeof(float));
        check(1.5f, "memcpy");

        std::fill(dst, dst + n, 0.0f);
        recorded_kernel();
        (*q)->fill(dst, 2.5f, n);
        check(2.5f, "fill");

        std::fill(dst, dst + n, 0.0f);
        recorded_kernel();
        (*q)->memset(dst, 0xff, n * sizeof(float));
        q->get_event().wait();
        EXPECT_TRUE(std::isnan(dst[n - 1])) << "memset on " << dev << ": still running after the wait";
        q->wait();

        sycl::free(src, sq);
        sycl::free(dst, sq);
        sycl::free(marker, sq);
    }
    if (devices_run == 0) GTEST_SKIP() << "no cpu or gpu device";
}

// An empty nd_range launches nothing, but its event must still follow earlier work on an in-order
// queue, as the launch's would. acpp skips the launch (a 0-block grid errors); returning a
// default event there made a wait on it return while a 256 MiB copy was still running.
class EmptyNdRangeKernel;
TEST(QueueTest, EmptyNdRangeEventFollowsEarlierWork) {
    constexpr size_t n = size_t(1) << 26;
    std::optional<Queue> q;
    try {
        q.emplace(Device("gpu"), true);
    } catch (const std::exception&) {
        GTEST_SKIP() << "no gpu device";
    }
    sycl::queue& sq = batchlas::sycl_queue(*q);
    float* src = sycl::malloc_device<float>(n, sq);
    float* dst = sycl::malloc_host<float>(n, sq);
    sq.fill(src, 1.5f, n).wait();
    std::fill(dst, dst + n, 0.0f);
    (*q)->memcpy(dst, src, n * sizeof(float));
    sycl::event e = (*q)->parallel_for<EmptyNdRangeKernel>(sycl::nd_range<1>(0, 32), [=](sycl::nd_item<1>) {
        dst[0] = -1.0f;
    });
    e.wait();
    EXPECT_EQ(dst[n - 1], 1.5f) << "the copy was still running after a wait on the empty launch";
    EXPECT_EQ(dst[0], 1.5f) << "the empty launch ran a work-item";
    q->wait();
    sycl::free(src, sq);
    sycl::free(dst, sq);
}

// The same through a command group (impl::parallel_for), with a local accessor in the group: the
// kernel path most drivers launch by.
class EmptyNdRangeCgKernel;
TEST(QueueTest, EmptyNdRangeInCommandGroupFollowsEarlierWork) {
    constexpr size_t n = size_t(1) << 26;
    std::optional<Queue> q;
    try {
        q.emplace(Device("gpu"), true);
    } catch (const std::exception&) {
        GTEST_SKIP() << "no gpu device";
    }
    sycl::queue& sq = batchlas::sycl_queue(*q);
    float* src = sycl::malloc_device<float>(n, sq);
    float* dst = sycl::malloc_host<float>(n, sq);
    sq.fill(src, 2.5f, n).wait();
    std::fill(dst, dst + n, 0.0f);
    (*q)->memcpy(dst, src, n * sizeof(float));
    sycl::event e = (*q)->submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> scratch(sycl::range<1>(32), h);
        batchlas::impl::parallel_for<EmptyNdRangeCgKernel>(h, sycl::nd_range<1>(0, 32), [=](sycl::nd_item<1> it) {
            scratch[it.get_local_id(0)] = -1.0f;
            dst[0] = scratch[0];
        });
    });
    e.wait();
    EXPECT_EQ(dst[n - 1], 2.5f) << "the copy was still running after a wait on the empty launch";
    EXPECT_EQ(dst[0], 2.5f) << "the empty launch ran a work-item";
    q->wait();
    sycl::free(src, sq);
    sycl::free(dst, sq);
}

// enqueue() orders every later submission path of an out-of-order queue after the event, with no
// host wait. The event is a kernel spinning on a host flag the host sets only after submitting;
// acpp's barrier node alone ordered nothing, so reads saw 0 and writes were overwritten. A watchdog
// sets the flag after 5 s, so a host wait in enqueue() fails the test instead of hanging. CPU only:
// a GPU kernel need not see a host store while it runs.
class OooEnqueueNamedKernel;
class OooEnqueueNamedTask;
TEST(QueueTest, OutOfOrderEnqueueOrdersLaterSubmissionsWithoutAHostWait) {
    std::optional<Queue> ooo;
    try {
        ooo.emplace(Device("cpu"), false);
    } catch (const std::exception&) {
        GTEST_SKIP() << "no cpu device";
    }
    Queue gate_q(*ooo, true);
    sycl::queue& sq = batchlas::sycl_queue(*ooo);
    constexpr int slots = 9;
    int* gate = sycl::malloc_host<int>(2, sq);
    int* marker = gate + 1;
    int* out = sycl::malloc_host<int>(slots, sq);
    gate[0] = 0;
    *marker = 0;
    std::fill(out, out + slots, -1);
    std::atomic_ref<int> open(gate[0]);

    sycl::event held = gate_q->single_task([=] {
        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::system> flag(gate[0]);
        while (flag.load() == 0) {
        }
        *marker = 1;
        out[6] = out[7] = 0;
    });
    std::atomic<bool> opened_by_watchdog{false};
    std::thread watchdog([&] {
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (open.load() == 0 && std::chrono::steady_clock::now() < until) std::this_thread::yield();
        if (open.exchange(1) == 0) opened_by_watchdog = true;
    });

    Event e = batchlas::event_from_sycl(held);
    ooo->enqueue(e);
    const bool enqueue_waited = open.load() != 0;
    (*ooo)->submit([&](sycl::handler& h) { h.single_task([=] { out[0] = *marker; }); });
    (*ooo)->parallel_for(sycl::range<1>(1), [=](sycl::id<1>) { out[1] = *marker; });
    (*ooo)->parallel_for(size_t(1), [=](size_t) { out[2] = *marker; });
    (*ooo)->parallel_for<OooEnqueueNamedKernel>(sycl::nd_range<1>(1, 1), [=](sycl::nd_item<1>) {
        out[3] = *marker;
    });
    (*ooo)->single_task([=] { out[4] = *marker; });
    (*ooo)->memcpy(out + 5, marker, sizeof(int));
    (*ooo)->fill(out + 6, 9, 1);
    (*ooo)->memset(out + 7, 0x01, sizeof(int));
    (*ooo)->single_task<OooEnqueueNamedTask>([=] { out[8] = *marker; });
    open.store(1);
    watchdog.join();
    ooo->wait();
    gate_q.wait();

    EXPECT_FALSE(enqueue_waited || opened_by_watchdog) << "enqueue() waited on the host";
    const char* path[slots] = {"submit", "parallel_for(range)", "parallel_for(size_t)",
                               "parallel_for<Name>(nd_range)", "single_task", "memcpy",
                               "fill", "memset", "single_task<Name>"};
    const int want[slots] = {1, 1, 1, 1, 1, 1, 9, 0x01010101, 1};
    for (int i = 0; i < slots; ++i) {
        EXPECT_EQ(out[i], want[i]) << path[i] << " ran before the enqueued event";
    }
    sycl::free(gate, sq);
    sycl::free(out, sq);
}

// queue_from_sycl: a root Queue inside the caller's context, so the caller's malloc_shared USM is
// used in place. Non-natural ld and batch stride, complex data, beta != 0.
TEST(SyclInterop, QueueFromSyclSharesContext) {
    std::vector<sycl::device> gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (gpus.empty()) GTEST_SKIP() << "no gpu device";
    // The last gpu: with several visible, a Queue that took the wrong logical index shows up.
    const size_t dev_pos = gpus.size() - 1;
    sycl::device dev = gpus[dev_pos];
    sycl::context ctx(dev);
    Queue q = batchlas::queue_from_sycl(ctx, dev);
    EXPECT_EQ(q.device().idx, dev_pos);
    EXPECT_TRUE(batchlas::sycl_queue(q).get_context() == ctx);
    EXPECT_TRUE(batchlas::sycl_queue(q).get_device() == dev);
    EXPECT_EQ(q.device().type, DeviceType::GPU);
    EXPECT_TRUE(q.in_order());

    using T = std::complex<float>;
    using MV = batchlas::MatrixView<T, batchlas::MatrixFormat::Dense>;
    constexpr int m = 13, n = 9, k = 11, batch = 3;
    constexpr int lda = m + 3, ldb = k + 2, ldc = m + 5;
    constexpr int sa = lda * k + 7, sb = ldb * n + 5, sc = ldc * n + 11;
    sycl::queue plain(ctx, dev);
    T* a = sycl::malloc_shared<T>(size_t(sa) * batch, plain);
    T* b = sycl::malloc_shared<T>(size_t(sb) * batch, plain);
    T* c = sycl::malloc_shared<T>(size_t(sc) * batch, plain);
    T* c0 = sycl::malloc_shared<T>(size_t(sc) * batch, plain);
    ASSERT_TRUE(a && b && c && c0);
    auto fill = [](T* p, size_t len, int seed) {
        for (size_t i = 0; i < len; ++i)
            p[i] = T(float((i * 7 + seed) % 13) / 13.0f - 0.4f, float((i * 5 + 2 * seed) % 11) / 11.0f - 0.3f);
    };
    fill(a, size_t(sa) * batch, 1);
    fill(b, size_t(sb) * batch, 2);
    fill(c, size_t(sc) * batch, 3);
    std::copy(c, c + size_t(sc) * batch, c0);

    MV A(a, m, k, lda, sa, batch), B(b, k, n, ldb, sb, batch), C(c, m, n, ldc, sc, batch), C0(c0, m, n, ldc, sc, batch);
    const T alpha(1.5f, -0.5f), beta(0.25f, 0.75f);
    batchlas::gemm(q, A, B, C, {.alpha = alpha, .beta = beta});
    q.wait();
    const double err = MV::gemm_backward_error(q, A, batchlas::Transpose::NoTrans, B, batchlas::Transpose::NoTrans, C0, C, MV::up(alpha),
                                               MV::up(beta), C.all_items());
    EXPECT_VERIFY(T, batchlas::verify::Check::blas, k, err);

    sycl::free(a, plain);
    sycl::free(b, plain);
    sycl::free(c, plain);
    sycl::free(c0, plain);
}

// Every rejection is a batchlas::device_error. Which cases run depends on the machine: a
// sub-device needs a partitionable gpu, "not in the context" needs a second device (a second gpu
// or a cpu device). The test says which ran.
TEST(SyclInterop, QueueFromSyclRejectsForeignDevice) {
    std::vector<sycl::device> gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (gpus.empty()) GTEST_SKIP() << "no gpu device";
    int ran = 0;
    try {
        auto subs = gpus.front().create_sub_devices<sycl::info::partition_property::partition_equally>(1);
        sycl::context sub_ctx(subs.front());
        EXPECT_THROW(batchlas::queue_from_sycl(sub_ctx, subs.front()), batchlas::device_error);
        ++ran;
    } catch (const sycl::exception&) {
    }
    sycl::context ctx0(gpus[0]);
    if (gpus.size() >= 2) {
        EXPECT_THROW(batchlas::queue_from_sycl(ctx0, gpus[1]), batchlas::device_error);
        ++ran;
    }
    const auto cpus = sycl::device::get_devices(sycl::info::device_type::cpu);
    if (!cpus.empty()) {
        EXPECT_THROW(batchlas::queue_from_sycl(ctx0, cpus.front()), batchlas::device_error);
        ++ran;
    }
    if (ran == 0) GTEST_SKIP() << "one gpu, no sub-device partition, no cpu device: nothing foreign to construct";
    std::cout << "[ INFO ] foreign-device cases run: " << ran << "\n";
}

// REDUCTION and SCAN round the per-device base down to a power of two before the problem-size
// cap. The rounding was `1 << (31 - __builtin_clzl(x))`, a negative shift on LP64 that came out
// as 2^32 x the intended size, so only the problem-size and device caps bounded it.
TEST(KernelHeuristics, ReductionAndScanRoundTheBaseDownToAPowerOfTwo) {
    for (const char* kind : {"cpu", "gpu"}) {
        std::optional<Device> dev;
        try {
            dev.emplace(kind);
        } catch (const std::exception&) {
            continue;
        }
        const size_t max_wg = dev->get_property(DeviceProperty::MAX_WORK_GROUP_SIZE);
        const size_t cus = dev->get_property(DeviceProperty::MAX_COMPUTE_UNITS);
        const bool gpu = dev->type == DeviceType::GPU;
        const size_t base = gpu ? 256 : std::min(size_t(64), cus * 2);
        const struct { batchlas::KernelType kt; size_t cap; const char* name; } cases[] = {
            {batchlas::KernelType::REDUCTION, gpu ? size_t(512) : size_t(64), "REDUCTION"},
            {batchlas::KernelType::SCAN, 256, "SCAN"},
        };
        for (const auto& c : cases) {
            const size_t rounded = std::bit_floor(std::min(base, c.cap));
            for (size_t ps : {size_t(0), rounded + 1, size_t(1000), size_t(1) << 20}) {
                const size_t want = std::max(std::min(rounded, max_wg), size_t(1));
                EXPECT_EQ(batchlas::compute_optimal_wg_size(*dev, c.kt, ps), want)
                    << kind << " " << c.name << " problem_size=" << ps;
            }
            // A problem size at or below the rounded base caps the result, as it did before.
            for (size_t ps : {size_t(1), size_t(3), rounded - 1, rounded}) {
                if (ps == 0) continue;
                EXPECT_EQ(batchlas::compute_optimal_wg_size(*dev, c.kt, ps), std::min(ps, max_wg))
                    << kind << " " << c.name << " problem_size=" << ps;
            }
        }
    }
}
