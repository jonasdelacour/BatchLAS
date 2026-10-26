// SPDX-License-Identifier: MIT
// syev (flat-kernel-selection.md §4.3, R1; docs/design/flat-kernel-selection.md#phase-5-syev): select::run
// takes the first entry of the nearest tuned/syev.<device>.txt row that can_run() admits. Cta,
// CtaFused and Jacobi are the three n <= 32 sub-group solvers; Blocked (sytrd_blocked + stedc) and
// TwoStage (sy2sb + sb2st + stedc) serve every n and call their sub-ops directly.

#include <batchlas/backend_config.h>

#include <batchlas/blas/functions/syev.hh>
#include <batchlas/settings.hh>

#include <src/ops/syev/choice.hh>
#include <src/ops/syev/parts.hh>
#include <src/ops/syev/vendor.hh>
#include <src/select/select.hh>
#include <src/util/template-instantiations.hh>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <variant>

namespace batchlas {
namespace ops::syev {

using select::overloaded;

template <typename T>
using MV = MatrixView<T, MatrixFormat::Dense>;
template <typename T>
using Real = typename base_type<T>::type;

// The sub-group solvers' own ceiling (syev_cta.cc, syev_cta_fused.cc, syev_jacobi_cta.cc all
// throw above it): one matrix per 32-lane partition, the CTA invariant of docs/developer/agent-guide.md §11.
inline constexpr std::int64_t kSmallMaxN = 32;

template <typename T>
select::Key key_of(const MV<T>& A, JobType jobz) {
    return {{"jobz", jobz == JobType::EigenVectors ? "V" : "N"}, {"n", A.rows()}, {"batch", A.batch_size()}};
}

// Correctness only (R3): false means the driver would throw or compute garbage. The native
// drivers are GPU kernels on a GPU backend (a NETLIB instantiation would call netlib on device
// memory) and need a square A with n >= 1; the small solvers add n <= 32 and a 32-wide
// sub-group, Blocked and TwoStage a batch >= 1 (the small drivers do not check it). Blocked and
// TwoStage take Upper by mirroring into Lower (uplo_mirror.hh). Every family, the vendor included,
// runs each item at the top-level order, so none takes a heterogeneous batch. Cta and CtaFused also
// exclude ROCm, where steqr_cta gives wrong eigenvalues (gfx1200).
template <Backend B, typename T>
bool can_run(const SyevChoice& c, const select::Device& d, const MV<T>& A) {
    const std::int64_t n = A.rows();
    const bool homogeneous = !A.is_heterogeneous();
    const bool native = B != Backend::NETLIB && d.is_gpu && homogeneous && A.rows() == A.cols() && n >= 1;
    const bool small = native && n <= kSmallMaxN && d.has_sg32;
    const bool cta_steqr = small && B != Backend::ROCM;
    const bool large = native && A.batch_size() >= 1;
    return std::visit(overloaded{
        [&](Cta) { return cta_steqr; },
        [&](CtaFused) { return cta_steqr; },
        [&](Jacobi) { return small; },
        [&](Blocked) { return large; },
        [&](TwoStage) { return large; },
        [&](Vendor) { return d.has_vendor && homogeneous; },
    }, c);
}

// Deliberately slower and more robust than the CTA STEQR defaults: syev runs inside syevx,
// where an inaccurate Ritz solve stagnates the outer iteration.
template <typename T>
SteqrParams<T> cta_steqr_params() {
    SteqrParams<T> params{};
    params.max_sweeps = 400;
    params.cta_shift_strategy = SteqrShiftStrategy::Wilkinson;
    return params;
}

// The family's own need for A as one call (R5).
template <Backend B, typename T>
std::size_t family_workspace(Queue& q, const SyevChoice& c, const MV<T>& A, Span<Real<T>> w, JobType jobz, Uplo uplo) {
    return std::visit(overloaded{
        [&](Cta) { return syev_cta_buffer_size<B, T>(q, A, jobz, cta_steqr_params<T>()); },
        [&](CtaFused) { return syev_cta_fused_buffer_size<B, T>(q, A, jobz, cta_steqr_params<T>()); },
        [&](Jacobi) { return syev_jacobi_cta_buffer_size<B, T>(q, A, jobz); },
        [&](Blocked) { return syev_blocked_buffer_size<B, T>(q, A, jobz, uplo, StedcParams<Real<T>>{}); },
        [&](TwoStage) { return syev_two_stage_buffer_size<B, T>(q, A, jobz, uplo, StedcParams<Real<T>>{}); },
        [&](Vendor) {
            return ops::syev::detail::syev_vendor_buffer_size_or_throw<B, T>(q, A, w, jobz, uplo);
        },
    }, c);
}

// Every arm clears or stores `info` itself, so the launch adds no clear of its own.
template <Backend B, typename T>
Event family_launch(Queue& q, const SyevChoice& c, const MV<T>& A, Span<Real<T>> w, JobType jobz, Uplo uplo,
                    Span<std::byte> ws, Span<int32_t> info) {
    return std::visit(overloaded{
        [&](Cta) { return syev_cta<B, T>(q, A, w, jobz, uplo, ws, cta_steqr_params<T>(), 0, info); },
        [&](CtaFused) { return syev_cta_fused<B, T>(q, A, w, jobz, uplo, ws, cta_steqr_params<T>(), 0, info); },
        [&](Jacobi) { return syev_jacobi_cta<B, T>(q, A, w, jobz, uplo, ws, JacobiParams<T>(), info); },
        [&](Blocked) { return syev_blocked<B, T>(q, A, w, jobz, uplo, ws, StedcParams<Real<T>>{}, info); },
        [&](TwoStage) { return syev_two_stage<B, T>(q, A, w, jobz, uplo, ws, StedcParams<Real<T>>{}, info); },
        [&](Vendor) { return ops::syev::detail::syev_vendor_or_throw<B, T>(q, A, w, jobz, uplo, ws, info); },
    }, c);
}

// The workspace of the part loop below at `part` items a part: the larger of a full part and the last
// one (part + batch % part). The family's size is not monotone in the batch (blocked's inner chunk is
// grid-capped and BATCHLAS_TUNE_L2_CHUNK_ITEMS-forced), so the last, larger part can need less.
template <Backend B, typename T>
std::size_t part_workspace(Queue& q, const SyevChoice& c, const MV<T>& A, Span<Real<T>> w, JobType jobz, Uplo uplo,
                           int part) {
    return std::max(family_workspace<B, T>(q, c, A.batch_range(0, part), w, jobz, uplo),
                    family_workspace<B, T>(q, c, A.batch_range(0, part + A.batch_size() % part), w, jobz, uplo));
}

// Items per part for blocked and two_stage. Both size their workspace for the whole batch (about
// 7 n^2 per item for double eigenvectors: 22.8 GB at n = 480, batch 1819), so a batch whose
// workspace passes the device's scratch budget runs in parts that each fit it. The slope from 1 to 2
// items gives the first guess (half the items that fit: the last part takes the remainder); the size is
// not linear at small batch everywhere (two_stage with the vendor children), so the guess halves until
// both part sizes fit. 0 means one part. evidence: docs/perf/syev.md#syev-vendor-free-double-band
template <Backend B, typename T>
int part_items(Queue& q, const SyevChoice& c, const MV<T>& A, Span<Real<T>> w, JobType jobz, Uplo uplo) {
    const int batch = A.batch_size();
    if (!std::holds_alternative<Blocked>(c) && !std::holds_alternative<TwoStage>(c)) return 0;
    if (batch < 4) return 0;
    const std::size_t override_bytes = detail::part_budget_override();
    const auto budget = override_bytes != 0 ? override_bytes
                                            : static_cast<std::size_t>(
                                                  select::describe(q.device(), q.backend(), false).scratch_budget);
    if (budget == 0 || family_workspace<B, T>(q, c, A, w, jobz, uplo) <= budget) return 0;
    const std::size_t one = family_workspace<B, T>(q, c, A.batch_range(0, 1), w, jobz, uplo);
    const std::size_t two = family_workspace<B, T>(q, c, A.batch_range(0, 2), w, jobz, uplo);
    std::size_t guess = std::size_t(batch) / 2;
    if (two > one) {
        const std::size_t per = two - one, base = one >= per ? one - per : 0;
        guess = std::min(guess, budget > base ? (budget - base) / per / 2 : 1);
    }
    int part = static_cast<int>(std::max<std::size_t>(1, guess));
    while (part > 1 && part_workspace<B, T>(q, c, A, w, jobz, uplo, part) > budget) part /= 2;
    return part;
}

// A part view from batch_range has no pointer array, and the vendor children of the full build read
// one (cuBLAS/cuSOLVER batched calls), so the parts carry their own, after the family's workspace.
inline constexpr std::size_t kPartPtrAlign = 64;

template <typename T>
std::size_t part_ptr_bytes(int part, int batch) {
    return kPartPtrAlign + static_cast<std::size_t>(part + batch % part) * sizeof(T*);
}

// Exactly what the launch below uses.
template <Backend B, typename T>
std::size_t workspace(Queue& q, const SyevChoice& c, const MV<T>& A, Span<Real<T>> w, JobType jobz, Uplo uplo) {
    const int part = part_items<B, T>(q, c, A, w, jobz, uplo);
    if (part == 0) return family_workspace<B, T>(q, c, A, w, jobz, uplo);
    return part_workspace<B, T>(q, c, A, w, jobz, uplo, part) + part_ptr_bytes<T>(part, A.batch_size());
}

// The parts are independent solves sharing one part-sized workspace and pointer array on the in-order queue.
template <Backend B, typename T>
Event launch(Queue& q, const SyevChoice& c, const MV<T>& A, Span<Real<T>> w, JobType jobz, Uplo uplo,
             Span<std::byte> ws, Span<int32_t> info) {
    const int part = part_items<B, T>(q, c, A, w, jobz, uplo);
    if (part == 0) return family_launch<B, T>(q, c, A, w, jobz, uplo, ws, info);
    const int batch = A.batch_size();
    const std::size_t n = static_cast<std::size_t>(A.rows());
    const bool per_item_info = info.size() >= static_cast<std::size_t>(batch);
    const std::size_t fam = part_workspace<B, T>(q, c, A, w, jobz, uplo, part);
    const auto at = reinterpret_cast<std::uintptr_t>(ws.data() + fam);
    T** ptrs = reinterpret_cast<T**>((at + kPartPtrAlign - 1) / kPartPtrAlign * kPartPtrAlign);
    for (int b0 = 0, cb = 0; b0 < batch; b0 += cb) {
        cb = batch - b0 < 2 * part ? batch - b0 : part;
        const MV<T> sub = A.batch_range(b0, cb);
        MV<T> view(sub.data_ptr(), sub.rows(), sub.cols(), sub.ld(), sub.stride(), cb, ptrs);
        if (!sub.active_rows().empty() || !sub.active_cols().empty())
            view = view.with_active_dims(sub.active_rows(), sub.active_cols());
        (void)family_launch<B, T>(q, c, view, w.subspan(std::size_t(b0) * n, std::size_t(cb) * n), jobz, uplo,
                                  ws.subspan(0, fam), per_item_info ? info.subspan(b0, cb) : Span<int32_t>{});
    }
    return q.get_event();
}

// Every driver refuses a non-square A, and the vendor would read rows() as the order.
// Thrown before choose() so neither Auto nor a pin can take it.
template <typename T>
void validate(const MV<T>& A, const char* who) {
    if (A.rows() != A.cols())
        throw batchlas::invalid_argument(std::string(who) + ": A must be square, got " + std::to_string(A.rows()) +
                                         " x " + std::to_string(A.cols()));
}

// Capability only (the Python binding, the LOBPCG accuracy pin): the same can_run, where the only
// backend term is the ROCm exclusion.
template <typename T>
bool supports(const Queue& q, const MV<T>& A, const SyevChoice& c) {
    const select::Device& d = select::describe(q.device(), Backend::Auto, false);
    if (q.backend() == Backend::ROCM) return can_run<Backend::ROCM, T>(c, d, A);
    return can_run<Backend::Auto, T>(c, d, A);
}

}  // namespace ops::syev

template <Backend Back, typename T>
Event syev(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Span<typename base_type<T>::type> eigenvalues,
           JobType jobtype, Uplo uplo, Span<std::byte> workspace, Span<int32_t> info) {
    ops::syev::validate<T>(A, "syev");
    // An empty batch is a no-op under any pin, as gemm's is: it never reaches selection.
    if (A.batch_size() == 0) return ctx.create_event_after_external_work();
    const coverage::Shape shape{.m = A.rows(), .n = A.cols(), .k = A.rows(), .batch = A.batch_size(), .uplo = uplo};
    const select::Key key = ops::syev::key_of<T>(A, jobtype);
    return select::run<Back, T>(
        ops::syev::spec, ctx, key, ops::syev::candidates<T>(),
        [&](const auto& c, const auto& d) { return ops::syev::can_run<Back, T>(c, d, A); }, shape, key,
        [&](const auto& c) {
            if (workspace.size() < ops::syev::workspace<Back, T>(ctx, c, A, eigenvalues, jobtype, uplo))
                throw batchlas::workspace_error("syev: insufficient workspace for chosen provider");
            // Blocked and TwoStage require an in-order queue.
            return select::on_in_order_queue(ctx, [&](Queue& q) {
                // Defect 14: several native drivers read the other triangle. Mirroring the named one
                // (conjugated, real diagonal) first hands every family the same Hermitian matrix.
                // The vendor reads only `uplo`. evidence: docs/design/known-defects.md#defect-14-the-hermitian-drivers-read-the-unreferenced-triangle
                if (!std::holds_alternative<ops::syev::Vendor>(c)) (void)A.hermitize(q, uplo);
                return ops::syev::launch<Back, T>(q, c, A, eigenvalues, jobtype, uplo, workspace, info);
            });
        });
}

template <Backend Back, typename T>
size_t syev_buffer_size(Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A,
                        Span<typename base_type<T>::type> eigenvalues, JobType jobtype, Uplo uplo) {
    ops::syev::validate<T>(A, "syev_buffer_size");
    if (A.batch_size() == 0) return 0;
    const auto c = select::pick<Back, T>(
        ops::syev::spec, ctx, ops::syev::key_of<T>(A, jobtype), ops::syev::candidates<T>(),
        [&](const auto& k, const auto& d) { return ops::syev::can_run<Back, T>(k, d, A); });
    return ops::syev::workspace<Back, T>(ctx, c, A, eigenvalues, jobtype, uplo);
}

namespace ops::syev::detail {

std::size_t& part_budget_override() {
    static std::size_t bytes = 0;
    return bytes;
}

template <typename T>
bool syev_supports_cta(const Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A) {
    return ops::syev::supports<T>(ctx, A, ops::syev::Cta{});
}
template <typename T>
bool syev_supports_blocked(const Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo) {
    return ops::syev::supports<T>(ctx, A, ops::syev::Blocked{});
}
template <typename T>
bool syev_supports_two_stage(const Queue& ctx, const MatrixView<T, MatrixFormat::Dense>& A, Uplo) {
    return ops::syev::supports<T>(ctx, A, ops::syev::TwoStage{});
}

#define SYEV_SUPPORTS_INSTANTIATE(fp)                                                                      \
    template bool syev_supports_cta<fp>(const Queue&, const MatrixView<fp, MatrixFormat::Dense>&);         \
    template bool syev_supports_blocked<fp>(const Queue&, const MatrixView<fp, MatrixFormat::Dense>&, Uplo); \
    template bool syev_supports_two_stage<fp>(const Queue&, const MatrixView<fp, MatrixFormat::Dense>&, Uplo);
SYEV_SUPPORTS_INSTANTIATE(float)
SYEV_SUPPORTS_INSTANTIATE(double)
SYEV_SUPPORTS_INSTANTIATE(std::complex<float>)
SYEV_SUPPORTS_INSTANTIATE(std::complex<double>)
#undef SYEV_SUPPORTS_INSTANTIATE

}  // namespace ops::syev::detail

#define SYEV_INSTANTIATE(B_, fp) \
    BATCHLAS_INSTANTIATE_OP(B_, fp, syev) BATCHLAS_INSTANTIATE_OP(B_, fp, syev_buffer_size)
BATCHLAS_INSTANTIATE_SCALAR_ALL_BACKENDS(SYEV_INSTANTIATE)
#undef SYEV_INSTANTIATE

}  // namespace batchlas
