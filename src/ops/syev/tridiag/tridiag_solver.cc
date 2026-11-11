// SPDX-License-Identifier: MIT
#include <src/linalg-impl.hh>
#include <batchlas/util/group-collectives.hh>
#include <src/math-helpers.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/util/sycl-span.hh>
#include <src/queue.hh>
#include <src/sycl/local_mem.hh>
#include <batchlas/device/detail/sscp_target.hh>
#include <batchlas/util/mempool.hh>
#include <sycl/sycl.hpp>
#include <array>
#include <complex>
#include <string>
#include <tuple>
#include <batchlas/error.hh>
#include <batchlas/blas/linalg.hh>

#include <src/util/template-instantiations.hh>
#include <src/util/batch_mask.hh>
#include <batchlas/device.hh>

namespace batchlas {

#if BATCHLAS_SYCL_IMPL_ACPP
inline constexpr bool kAcpp = true;
#else
inline constexpr bool kAcpp = false;
#endif

// QR steps allowed per eigenvalue before the item is reported unconverged (LAPACK ?steqr's MAXIT).
inline constexpr int kTridiagMaxIterations = 30;

// The solver's launch is nd_range(batch * 32, 32). Loops stride by the runtime local range and
// barriers fence global and local memory, as the legacy sycl::group_barrier: a constant stride
// of 32 lets the compiler unroll every elementwise loop over the generic D/L/U pointers (+23%
// instructions, +16 registers on sm_89).
using TridiagEx = device::RtGroup<>;

// Q(:, k:k+1) := Q(:, k:k+1) H_k for the n reflectors H_k = I - 2 v_k v_k^T of one QR sweep.
template <typename T>
void apply_all_reflections(const TridiagEx& ex, const Span<T> V, const int n, const int m, const int ld, T* Q) {
    const auto q = device::mat(Q, m, n + 1, ld);
    for (int k = 0; k < n; k++) device::rot<device::alg::RotReflector2>(ex, device::Givens<T>{V[2 * k], V[2 * k + 1]}, q.col(k), q.col(k + 1));
}

// One explicit shifted QR step on the leading n x n block: T - shift = Q R, T := R Q + shift, Q
// the product of n - 1 2x2 reflectors (kept in Vout for the eigenvector update). D, L, U are
// the diagonal, sub- and super-diagonal; U[n+1:] is the second super-diagonal R fills in.
template <typename T>
void T_QTQ(const TridiagEx& ex, const int n, const Span<T> D, const Span<T> L, const Span<T> U, const Span<T> Vout, T shift=0) {
    const auto d = device::vec(D.data(), n);
    auto U2 = [&](int k) -> T& { return U[(n + 1) + k]; };
    // (I - 2 v v^T) on three scattered pairs, unrolled at compile time: the pointers stay in registers.
    auto refl3 = [](const device::Givens<T>& g, const auto& x, const auto& y) BATCHLAS_DEVICE_BODY {
        device::static_for<3>([&](auto i) BATCHLAS_DEVICE_BODY { device::rot<device::alg::RotReflector2>(g, x(i), y(i)); });
    };

    // L[n-1] couples the active block to the deflated part below it; L[n] may be past the end.
    const T l_nm1 = L[n-1];

    ex.barrier(device::Mem::Global);

    device::map(ex, d, [=](T x) { return x - shift; }, d);
    device::map(ex, device::vec(U.data() + (n + 1), n), [] { return T(0); });
    device::copy(ex, device::vec(L.data(), n - 1), device::vec(U.data(), n - 1));
    device::map(ex, device::vec(Vout.data(), 2 * (n - 1)), [] { return T(0); });
    if (ex.leader()) {
        L[n - 1] = T(0);
        U[n - 1] = T(0);
    }

    ex.barrier(device::Mem::Global);

    // Q^T from the left: H_k zeroes L[k] (rows k, k+1; columns k, k+1, k+2).
    if (ex.leader()) {
        for (int k = 0; k < n - 1; k++) {
            if (L[k] != T(0)) {
                const auto v = device::house2(D[k], L[k]);
                Vout[2 * k] = v[0];
                Vout[2 * k + 1] = v[1];
                refl3(device::Givens<T>{v[0], v[1]}, device::refs(D[k], U[k], U2(k)), device::refs(L[k], D[k + 1], U[k + 1]));
            }
        }
    }

    // Q from the right, H_0 on columns 0, 1. Row 1 reads the annihilated L[0] as an exact zero.
    if (ex.leader()) {
        const std::array<T, 2> v{Vout[0], Vout[1]};
        const T t1 = T(0) + D[1] * v[1];
        device::rot<device::alg::RotReflector2>(device::Givens<T>{v[0], v[1]}, D[0], U[0]);
        L[0] -= T(2) * v[0] * t1;
        D[1] -= T(2) * v[1] * t1;
    }

    ex.barrier(device::Mem::Global);

    // H_k on columns k, k+1 (rows k-1, k, k+1).
    if (ex.leader()) {
        for (int k = 1; k < n - 1; k++) {
            const std::array<T, 2> v{Vout[2 * k], Vout[2 * k + 1]};
            refl3(device::Givens<T>{v[0], v[1]}, device::refs(U[k - 1], D[k], L[k]), device::refs(U2(k - 1), U[k], D[k + 1]));
        }
    }

    ex.barrier(device::Mem::Global);

    device::map(ex, d, [=](T x) { return x + shift; }, d);
    device::copy(ex, device::vec(U.data(), n - 1), device::vec(L.data(), n - 1));

    ex.barrier(device::Mem::Global);

    if (ex.leader()) {
        L[n-1] = l_nm1;
    }
    ex.barrier(device::Mem::Global);
}

// Whether the four work arrays fit the local memory a 32-lane work-group may request (on Metal
// the runtime's per-lane scratch comes off the top).
inline bool tridiag_smem_fits(Queue& ctx, size_t bytes) {
    return impl::local_mem_bytes(ctx->get_device(), 32) >= bytes;
}

template <Backend B, typename T>
Event tridiagonal_solver(Queue& ctx,
                         Span<T> alphas,
                         Span<T> betas,
                         Span<typename base_type<T>::type> W,
                         Span<std::byte> workspace,
                         JobType jobz,
                         const MatrixView<T, MatrixFormat::Dense>& Q,
                         size_t n,
                         size_t batch_size) {
    const std::uint8_t* mask = ::batchlas::detail::mask_ptr(::batchlas::detail::call_mask("tridiagonal_solver", Q));
    using real_t = typename base_type<T>::type;
    auto pool = BumpAllocator(workspace);

    auto smem_required = sizeof(T) * (3 * n + 3 * (n + 1));
    bool smem_possible = tridiag_smem_fits(ctx, smem_required);

    auto V_reflectors = pool.allocate<T>(ctx, !smem_possible ? 2 * n * batch_size : 0);
    auto D_global = pool.allocate<T>(ctx, !smem_possible ? (n+1) * batch_size : 0);
    auto U_global = pool.allocate<T>(ctx, !smem_possible ? 2 * (n + 1) * batch_size : 0);
    auto L_global = pool.allocate<T>(ctx, !smem_possible ? n * batch_size : 0);
    UnifiedVector<int32_t> status(batch_size, 0);
    int32_t* status_ptr = status.data();

    ctx->submit([=](sycl::handler& h) {
        auto Vstride = Q.stride();
        auto Vdata = Q.data_ptr();
        auto Vld = Q.ld();

        sycl::local_accessor<T, 1> D_smem(smem_possible ? n + 1 : 0, h);
        sycl::local_accessor<T, 1> L_smem(smem_possible ? n : 0, h);
        sycl::local_accessor<T, 1> U_smem(smem_possible ? 2 * (n + 1) : 0, h);
        sycl::local_accessor<T, 1> V_smem(smem_possible ? 2 * n : 0, h);

        batchlas::impl::parallel_for(h, sycl::nd_range<1>(batch_size*32, 32), [=](sycl::nd_item<1> item) {
            auto bid = item.get_group_linear_id();
            if (mask != nullptr && mask[bid] == 0) return;  // I1: before the first barrier; status stays 0
            const TridiagEx ex{item};
            auto cta = item.get_group();

            bool failed = false;
            auto solve = [&](Span<T> D, Span<T> L, Span<T> U, Span<T> Vlocal) __attribute__((always_inline)) {
                const int nn = int(n);
                const auto d = device::vec(D.data(), nn);
                T* batch_Q = Vdata + bid * Vstride;

                if (jobz == JobType::EigenVectors) device::laset(ex, device::Zero{}, device::One{}, device::mat(batch_Q, nn, nn, int(Vld)));
                // One fused pass with the legacy int64 bound. Two device::copy calls, or one fused
                // for_each_owned over int n, cost 8 more registers for double on sm_89 (58 -> 66).
                for (int i = ex.lid(); i < n; i += ex.width()) {
                    D[i] = alphas.data()[bid * n + i];
                    L[i] = betas.data()[bid * n + i];
                }

                ex.barrier(device::Mem::Global);

                // Every work-item reads the same local values after T_QTQ's closing barrier, so the
                // loop decisions agree across the group. Relative deflation as LAPACK ?steqr.
                const real_t eps = std::numeric_limits<real_t>::epsilon();
                // TRAP (AdaptiveCpp OpenMP device): a data-dependent loop around barriers never ends
                // there, even at batch 1, unless its condition is a group collective. Host only.
                auto uniform = [&](bool c) {
                    if constexpr (kAcpp && B == Backend::NETLIB) return sycl::group_broadcast(cta, c);
                    else return c;
                };
                for (int k = int(n) - 1; k > 0; k--) {
                    int it = 0;
                    while (uniform(std::abs(L[k-1]) > eps * (std::abs(D[k-1]) + std::abs(D[k])))) {
                        if (it++ == kTridiagMaxIterations) { failed = true; break; }
                        const auto [l0, l1] = device::lae2(D[k-1], L[k-1], D[k]);
                        const T shift = std::abs(l0 - D[k]) < std::abs(l1 - D[k]) ? l0 : l1;
                        T_QTQ(ex, k+1, D, L, U, Vlocal, shift);
                        if (jobz == JobType::EigenVectors) {
                            apply_all_reflections(ex, Vlocal, k, nn, int(Vld), batch_Q);
                        }
                    }
                }
                if (ex.leader()) status_ptr[bid] = failed ? 1 : 0;

                device::map(ex, device::vec(W.data() + bid * n, nn), [](T x) -> real_t {
                    if constexpr (device::Arith<T>::is_complex) return x.real();
                    else return x;
                }, d);
            };

            // TRAP (Metal): no generic address space, so a pointer chosen between local and global
            // memory at run time is rejected ("Conflicting address space information in PHI node").
            // There the solve is emitted once per storage; elsewhere once, on selected pointers.
            auto smem = [&]() BATCHLAS_DEVICE_BODY {
                return std::array<Span<T>, 4>{Span(&D_smem[0], n + 1), Span(&L_smem[0], n),
                                               Span(&U_smem[0], 2 * (n + 1)), Span(&V_smem[0], 2 * n)};
            };
            auto gmem = [&]() BATCHLAS_DEVICE_BODY {
                return std::array<Span<T>, 4>{Span(D_global.data() + bid * (n + 1), n + 1),
                                               Span(L_global.data() + bid * n, n),
                                               Span(U_global.data() + bid * 2 * (n + 1), 2 * (n + 1)),
                                               Span(V_reflectors.data() + bid * 2 * n, 2 * n)};
            };
            if (sycl_impl::on_metal()) {
                if (smem_possible) std::apply(solve, smem());
                else std::apply(solve, gmem());
            } else {
                std::apply(solve, smem_possible ? smem() : gmem());
            }
        });
    });

    ctx.wait();
    for (size_t b = 0; b < batch_size; ++b) {
        if (status_ptr[b] != 0) {
            throw batchlas::convergence_error("tridiagonal_solver: an eigenvalue did not converge within " +
                                              std::to_string(kTridiagMaxIterations) + " QR steps");
        }
    }
    return ctx.get_event();
}

template <Backend B, typename T>
size_t tridiagonal_solver_buffer_size(Queue& ctx, size_t n, size_t batch_size, JobType /*jobz*/){
    auto smem_required = sizeof(T) * (3 * n + 3 * (n + 1));
    bool smem_possible = tridiag_smem_fits(ctx, smem_required);
    size_t size = 0;
    size += BumpAllocator::allocation_size<T>(ctx, !smem_possible ? 2*n*batch_size : 0);
    size += BumpAllocator::allocation_size<T>(ctx, !smem_possible ? (n+1)*batch_size : 0);
    size += BumpAllocator::allocation_size<T>(ctx, !smem_possible ? 2*(n+1)*batch_size : 0);
    size += BumpAllocator::allocation_size<T>(ctx, !smem_possible ? n*batch_size : 0);
    return size;
}

#define TRIDAG_INSTANTIATE(back, fp) \
    template Event tridiagonal_solver<back, BATCHLAS_UNPAREN fp>(Queue&, Span<BATCHLAS_UNPAREN fp>, Span<BATCHLAS_UNPAREN fp>, Span<typename base_type<BATCHLAS_UNPAREN fp>::type>, Span<std::byte>, JobType, const MatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>&, size_t, size_t); \
    template size_t tridiagonal_solver_buffer_size<back, BATCHLAS_UNPAREN fp>(Queue&, size_t, size_t, JobType);

// Real only. The complex arms below have never been enabled:
//   TRIDAG_INSTANTIATE(back, std::complex<float>)
//   TRIDAG_INSTANTIATE(back, std::complex<double>)
BATCHLAS_INSTANTIATE_REAL_ALL_BACKENDS(TRIDAG_INSTANTIATE)

//TRIDAG_INSTANTIATE(Backend::CUDA, std::complex<float>)
//TRIDAG_INSTANTIATE(Backend::CUDA, std::complex<double>)

#undef TRIDAG_INSTANTIATE

} // namespace batchlas

