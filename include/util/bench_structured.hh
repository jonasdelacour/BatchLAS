#pragma once

#include <util/minibench.hh>

#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>

#include <blas/matrix.hh>

#include <chrono>
#include <functional>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// BatchLAS benchmark helpers for structured benchmarking:
// - USM prepare/prefetch hooks
// - device-resident pristine copies for mutating algorithms

inline double bench_event_elapsed_ms(const Event& start, const Event& end) {
    const auto s = start.profiling_command_start_end_ns();
    const auto e = end.profiling_command_start_end_ns();

    if (!s || !e) return -1.0;
    const auto s_end = s->second;
    const auto e_end = e->second;

    if (e_end <= s_end) return -1.0;
    return double(e_end - s_end) * 1e-6; // ns -> ms
}

namespace bench {

struct ManagedInputs {
    explicit ManagedInputs(std::shared_ptr<::Queue> q_) : q(std::move(q_)) {}

    std::shared_ptr<::Queue> q;

    std::vector<std::function<void()>> init_once;
    std::vector<std::function<void()>> prepare_once;

    std::vector<std::function<void()>> before_each;

    template <typename T>
    ManagedInputs& prepare(::Span<T> s) {

        auto queue = q;
        prepare_once.emplace_back([queue, s]() mutable {
            (void)s.set_access_device(*queue);

            (void)s.prefetch(*queue);
        });
        return *this;
    }

    template <typename T>
    ManagedInputs& prepare(const ::UnifiedVector<T>& v) {
        return prepare(v.to_span());
    }

    template <typename T>
    ManagedInputs& prepare(const batchlas::Vector<T>& v) {
        return prepare(batchlas::VectorView<T>(v));
    }

    template <typename T, batchlas::MatrixFormat F>
    ManagedInputs& prepare(const batchlas::Matrix<T, F>& m) {
        return prepare(m.view());
    }

    template <typename T>
    ManagedInputs& prepare(const batchlas::VectorView<T>& v) {
        auto queue = q;
        prepare_once.emplace_back([queue, v]() mutable {
            (void)v.set_access_device(*queue);

            (void)v.prefetch(*queue);
        });
        return *this;
    }

    template <typename T, batchlas::MatrixFormat F>
    ManagedInputs& prepare(const batchlas::MatrixView<T, F>& m) {
        auto queue = q;
        prepare_once.emplace_back([queue, m]() mutable {

            (void)m.set_access_device(*queue);
            (void)m.prefetch(*queue);
        });
        return *this;
    }

    template <typename T>
    ManagedInputs& prepare(const T&) {
        return *this;
    }

    template <typename T>
    ManagedInputs& pristine(std::shared_ptr<batchlas::Vector<T>> v) {
        auto v0 = std::make_shared<batchlas::Vector<T>>(v->size(), v->batch_size(), v->stride(), v->inc());
        auto queue = q;

        init_once.emplace_back([queue, v, v0]() mutable {
            batchlas::VectorView<T>::copy(*queue, batchlas::VectorView<T>(*v0), batchlas::VectorView<T>(*v)).wait();
        });
        before_each.emplace_back([queue, v, v0]() mutable {

            batchlas::VectorView<T>::copy(*queue, batchlas::VectorView<T>(*v), batchlas::VectorView<T>(*v0)).wait();
        });
        prepare(*v);
        prepare(*v0);

        return *this;
    }

    template <typename T, batchlas::MatrixFormat F>
    ManagedInputs& pristine(std::shared_ptr<batchlas::Matrix<T, F>> m) {

    static_assert(F == batchlas::MatrixFormat::Dense,
              "ManagedInputs::pristine(Matrix) currently supports Dense only");
    auto m0 = std::make_shared<batchlas::Matrix<T, F>>(m->rows(), m->cols(), m->batch_size(), m->ld(), m->stride());
    auto queue = q;

        init_once.emplace_back([queue, m, m0]() mutable {
            batchlas::MatrixView<T, F>::copy(*queue, m0->view(), m->view()).wait();
        });
        before_each.emplace_back([queue, m, m0]() mutable {
            batchlas::MatrixView<T, F>::copy(*queue, m->view(), m0->view()).wait();

        });
        prepare(*m);
        prepare(*m0);
        return *this;
    }

    std::function<void()> make_prepare_once() {
        return [mi = *this]() mutable {
            for (auto& f : mi.init_once) f();
            for (auto& f : mi.prepare_once) f();
            mi.q->wait();
        };
    }

    std::function<void()> make_before_each_run() {
        return [mi = *this]() mutable {
            for (auto& f : mi.before_each) f();
        };
    }
};

template <typename F>
inline auto make_event_timed_kernel_ms(std::shared_ptr<::Queue> q, F&& kernel) {
    return [q = std::move(q), kernel = std::forward<F>(kernel)]() mutable -> double {
        const auto host_t0 = std::chrono::steady_clock::now();
        Event start = q->get_event();
        kernel();
        Event end = q->get_event();
        end.wait();
        const auto host_t1 = std::chrono::steady_clock::now();

        const double prof_ms = bench_event_elapsed_ms(start, end);
        if (prof_ms >= 0.0) return prof_ms;
        return std::chrono::duration<double, std::milli>(host_t1 - host_t0).count();
    };
}

namespace detail {
template <typename X>
decltype(auto) kernel_arg(X& x);
} // namespace detail

namespace detail {

template <typename T>
struct Manage {
    std::shared_ptr<T> ptr;
    void register_with(ManagedInputs& mi) const { mi.prepare(*ptr); }
    T& get() const { return *ptr; }
};

template <typename T>
struct Pristine {
    std::shared_ptr<T> ptr;
    void register_with(ManagedInputs& mi) const { mi.pristine(ptr); }
    T& get() const { return *ptr; }
};

template <typename T>
struct Value {
    T value;
    void register_with(ManagedInputs&) const {}
    T& get() { return value; }
    const T& get() const { return value; }
};

template <typename>
struct is_wrapper : std::false_type {};

template <typename T>
struct is_wrapper<Manage<T>> : std::true_type {};

template <typename T>
struct is_wrapper<Pristine<T>> : std::true_type {};

template <typename T>
struct is_wrapper<Value<T>> : std::true_type {};

template <typename T, std::enable_if_t<!is_wrapper<std::decay_t<T>>::value, int> = 0>
inline auto wrap_managed(T&& t) {
    using U = std::decay_t<T>;
    if constexpr (std::is_arithmetic_v<U> || std::is_enum_v<U> || std::is_trivially_copyable_v<U>) {
        return Value<U>{U(std::forward<T>(t))};
    } else {
        return Manage<U>{std::make_shared<U>(std::forward<T>(t))};
    }
}

template <typename T>
inline auto wrap_managed(const Manage<T>& m) {
    return m;
}

template <typename T>
inline auto wrap_managed(Manage<T>&& m) {
    return std::move(m);
}

template <typename T>
inline auto wrap_managed(const Pristine<T>& p) {
    return p;
}

template <typename T>
inline auto wrap_managed(Pristine<T>&& p) {
    return std::move(p);
}

template <typename T>
inline auto wrap_managed(std::shared_ptr<T> p) {
    return Manage<T>{std::move(p)};
}

template <typename T>
inline decltype(auto) kernel_arg(batchlas::Vector<T>& v) {
    return batchlas::VectorView<T>(v);
}

template <typename T>
inline decltype(auto) kernel_arg(const batchlas::Vector<T>& v) {
    return batchlas::VectorView<T>(v);
}

template <typename T, batchlas::MatrixFormat F>
inline decltype(auto) kernel_arg(batchlas::Matrix<T, F>& m) {
    return m.view();
}

template <typename T, batchlas::MatrixFormat F>
inline decltype(auto) kernel_arg(const batchlas::Matrix<T, F>& m) {
    return m.view();
}

template <typename T>
inline decltype(auto) kernel_arg(::UnifiedVector<T>& v) {
    return v.to_span();
}

template <typename T>
inline decltype(auto) kernel_arg(const ::UnifiedVector<T>& v) {
    return v.to_span();
}

template <typename T>
inline decltype(auto) kernel_arg(::Span<T> s) {
    return s;
}

template <typename T>
inline decltype(auto) kernel_arg(batchlas::VectorView<T> v) {
    return v;
}

template <typename T, batchlas::MatrixFormat F>
inline decltype(auto) kernel_arg(batchlas::MatrixView<T, F> m) {
    return m;
}

template <typename X>
inline decltype(auto) kernel_arg(X& x) {
    return (x);
}

} // namespace detail

template <typename T>
inline auto pristine(T& t) {
    using U = std::decay_t<T>;
    return detail::Pristine<U>{std::make_shared<U>(std::move(t))};
}

template <typename T>
inline auto pristine(const T& t) {
    using U = std::decay_t<T>;
    return detail::Pristine<U>{std::make_shared<U>(t)};
}

template <typename T>
inline auto pristine(T&& t) {
    using U = std::decay_t<T>;
    return detail::Pristine<U>{std::make_shared<U>(std::move(t))};
}

template <typename T>
inline auto pristine(std::shared_ptr<T> p) {
    return detail::Pristine<T>{std::move(p)};
}

template <typename T>
inline auto manage(T&& t) {
    return detail::wrap_managed(std::forward<T>(t));
}

template <typename KernelFn, typename... Managed>
struct KernelConfigurator {
    std::shared_ptr<::Queue> q;
    KernelFn kernel;
    std::tuple<Managed...> managed;

    void operator()(minibench::State& state) {
        ManagedInputs mi(q);
        std::apply([&](auto&... m) { (m.register_with(mi), ...); }, managed);

        state.SetPrepare(mi.make_prepare_once());
        state.SetBeforeEachRun(mi.make_before_each_run());

        auto kernel_once = [q = q, kernel = kernel, managed = managed]() mutable {
            std::apply([&](auto&... m) { kernel(m.get()...); }, managed);
        };
        state.SetKernel(std::function<void()>(kernel_once));
        state.SetTimedKernelMs(bench::make_event_timed_kernel_ms(q, kernel_once));
        state.SetBatchEndWait(q);
    }
};

template <typename... Managed>
struct KernelBuilder {
    std::shared_ptr<::Queue> q;
    std::tuple<Managed...> managed;

    template <typename KernelFn>
    auto operator()(KernelFn&& kernel) && -> KernelConfigurator<std::decay_t<KernelFn>, Managed...> {
        return {
            std::move(q),
            std::forward<KernelFn>(kernel),
            std::move(managed),
        };
    }
};

template <typename... ManagedArgs>
inline auto Kernel(std::shared_ptr<::Queue> q, ManagedArgs&&... managed_args)
    -> KernelBuilder<decltype(detail::wrap_managed(std::forward<ManagedArgs>(managed_args)))...> {
    return {
        std::move(q),
        std::make_tuple(detail::wrap_managed(std::forward<ManagedArgs>(managed_args))...)};
}

} // namespace bench

template <typename F>
inline double bench_time_region_ms(F&& f) {
    auto t0 = std::chrono::steady_clock::now();
    f();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}
