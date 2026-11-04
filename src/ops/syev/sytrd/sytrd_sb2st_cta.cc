// SPDX-License-Identifier: MIT
// Sub-group (CTA-style) SB2ST implementation.
//
// This translation unit intentionally contains ONLY the sub-group/chunked-partition
// kernel path. The baseline implementation and the public sytrd_sb2st entrypoints
// live in sytrd_sb2st.cc.

#include <src/ops/syev/sytrd/sytrd_sb2st_cta.hh>
#include <src/ops/syev/sytrd/sytrd_sb2st_cta_instantiations.hh>


#include <batchlas/util/group-invoke.hh>
#include <batchlas/device/detail/sg_partition/sg_partition.hh>
#include <batchlas/device/sg_partition.hh>
#include <batchlas/device/lapack/sb2st.hh>

#include <sycl/sycl.hpp>

#include <batchlas/backend_config.h>

#include <src/math-helpers.hh>
#include <batchlas/device/detail/kernel_attrs.hh>
#include <src/sycl/local_mem.hh>
#include <batchlas/device/detail/sscp_target.hh>

#include <array>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace batchlas::detail {

namespace {

namespace v2 = batchlas::device;

template <typename T, int32_t P>
class BtrdLowerSubgroupKernel;

template <typename T, int32_t P>
Event btrd_lower_inplace_subgroup(Queue& q,
                                  KernelMatrixView<T, MatrixFormat::Dense> ab,
                                  int n,
                                  int kd,
                                  VectorView<typename base_type<T>::type> c,
                                  VectorView<T> work,
                                  VectorView<typename base_type<T>::type> d,
                                  VectorView<typename base_type<T>::type> e,
                                  VectorView<T> tau,
                                  const std::uint8_t* mask) {
    static_assert(P == 2 || P == 4 || P == 8 || P == 16 || P == 32);
    constexpr int32_t sg_size = 32;
    static_assert(sg_size % P == 0);

    using Real = typename base_type<T>::type;

    // --- Optional: stage the band matrix AB in local memory (tight-packed ldab = kd+1).
    // This is gated by device local memory size and the per-work-group footprint.
    constexpr int32_t wg_size = 32;
    const size_t lmem_bytes = impl::local_mem_bytes(q->get_device(), wg_size);
    constexpr int32_t probs_per_wg = wg_size / P;

    // Tight-packed band leading dimension in local memory.
    const int ldab_local_host = kd + 1;
    const int cw_len_host = n + kd + 2; // size of C/WORK scratch per problem

    const size_t ab_bytes_per_wg =
        size_t(ldab_local_host) * size_t(n) * sizeof(T) * size_t(probs_per_wg);
    const size_t cw_bytes_per_wg =
        size_t(cw_len_host) * (sizeof(Real) + sizeof(T)) * size_t(probs_per_wg);

    // Leave headroom (25%) for compiler/runtime usage. Unmeasured guess.
    // evidence: docs/perf/sytrd.md#sytrd-open-debts
    const size_t lmem_budget = (lmem_bytes * 3) / 4;

    // The C/WORK rotation scratch is the *first* claim on local memory: it is touched
    // every chase step, whereas AB staging only pays off if the whole band also fits.
    // Both are optional -- global fallbacks (the `c` / `work` parameters) are always
    // allocated by the caller, so exceeding local memory must degrade, not fail to launch.
    const bool use_local_cw = (cw_bytes_per_wg > 0) && (cw_bytes_per_wg <= lmem_budget);
    const bool use_local_ab = (ab_bytes_per_wg > 0) &&
                              ((ab_bytes_per_wg + (use_local_cw ? cw_bytes_per_wg : size_t(0))) <= lmem_budget);

    // local_accessor must be non-zero sized.
    const size_t ab_local_elems = use_local_ab
                                      ? (size_t(ldab_local_host) * size_t(n) * size_t(probs_per_wg))
                                      : size_t(1);
    const size_t cw_local_elems = use_local_cw
                                      ? (size_t(cw_len_host) * size_t(probs_per_wg))
                                      : size_t(1);
    (void)q->submit([&](sycl::handler& h) {
        const int32_t batch = static_cast<int32_t>(ab.batch_size());

        // Tuneable: keep modest to avoid spilling local memory; SB2ST is fairly heavy.
        constexpr int32_t wg_size = 32;
        static_assert(wg_size % P == 0);
        const int32_t probs_per_wg = wg_size / P;
        const int32_t num_wg = (batch + probs_per_wg - 1) / probs_per_wg;
        const int32_t global = num_wg * wg_size;

        sycl::local_accessor<T, 1> ab_local(sycl::range<1>(ab_local_elems), h);
        sycl::local_accessor<Real, 1> c_local(sycl::range<1>(cw_local_elems), h);
        sycl::local_accessor<T, 1> work_local(sycl::range<1>(cw_local_elems), h);

        impl::parallel_for<BtrdLowerSubgroupKernel<T, P>>(h,
            sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(global)),
                              sycl::range<1>(static_cast<size_t>(wg_size))),
            [=](sycl::nd_item<1> it) BATCHLAS_REQD_SG_SIZE(32) {
                using Real = typename base_type<T>::type;
                using Ops = btrd_givens_ops<T>;

                const auto wg = it.get_group();
                const int32_t wg_id = static_cast<int32_t>(wg.get_group_linear_id());
                const auto sg = it.get_sub_group();
                const auto partition = make_partition<P>(sg);

                // NOTE: make_partition<P>(sg) partitions *within a sub-group*.
                // If the work-group contains multiple sub-groups, partition.get_group_linear_id()
                // repeats for each sub-group. Make part_id unique within the whole work-group.
                const int32_t sg_id = static_cast<int32_t>(sg.get_group_linear_id());
                const int32_t parts_per_sg = static_cast<int32_t>(partition.get_group_linear_range());
                const int32_t part_id = sg_id * parts_per_sg + static_cast<int32_t>(partition.get_group_linear_id());
                const v2::SgpPartition<P> pex{sg};  // lane = pex.lid(); the same SubGroupPartition<P>
                const int32_t prob_id = wg_id * probs_per_wg + part_id;
                if (prob_id >= batch) return;
                // I2: an inactive item chases its scratch copy like any other and stores nothing.
                const bool live = mask == nullptr || mask[prob_id] != 0;

                auto ABv = ab.batch_item(prob_id);
                auto Dv = d.batch_item(prob_id);
                auto Ev = e.batch_item(prob_id);
                auto TAUv = tau.batch_item(prob_id);

                const int ldab_l = kd + 1;
                const int cw_len = n + kd + 2;
                T* const ABg = ABv.data();
                auto& D = Dv;
                auto& E = Ev;
                auto& TAU = TAUv;

                // C/WORK live in local memory when the per-work-group footprint fits, otherwise in the
                // caller-provided global scratch (built with inc=1, so the batch base is contiguous).
                auto local_cw = [&]() BATCHLAS_DLA_BODY {
                    return std::pair<Real*, T*>{
                        c_local.template get_multi_ptr<sycl::access::decorated::no>().get() + part_id * cw_len,
                        work_local.template get_multi_ptr<sycl::access::decorated::no>().get() + part_id * cw_len};
                };
                auto global_cw = [&]() BATCHLAS_DLA_BODY {
                    return std::pair<Real*, T*>{c.batch_item(prob_id).data().data(),
                                                work.batch_item(prob_id).data().data()};
                };
                auto local_ab = [&]() BATCHLAS_DLA_BODY -> T* {
                    return ab_local.template get_multi_ptr<sycl::access::decorated::no>().get() + part_id * (ldab_l * n);
                };

                auto zero_cw = [&](Real* C, T* WORK) BATCHLAS_DLA_BODY {
                    v2::for_each_owned(pex, cw_len, [&](int i) BATCHLAS_DLA_BODY {
                        C[i] = Real(0);
                        WORK[i] = T(0);
                    });
                    group_barrier(partition);
                };
                // Copy (r,c) between global (ld = ABv.ld()) and tight-packed local (ld = kd+1) storage.
                auto stage_ab = [&](T* ABl, bool to_local) BATCHLAS_DLA_BODY {
                    if (!to_local) group_barrier(partition);
                    const int ab_elems_l = ldab_l * n;
                    v2::for_each_owned(pex, ab_elems_l, [&](int t) BATCHLAS_DLA_BODY {
                        const int ccol = t / ldab_l;
                        const int rrow = t - ccol * ldab_l;
                        if (to_local) ABl[t] = ABg[rrow + ccol * ABv.ld()];
                        else ABg[rrow + ccol * ABv.ld()] = ABl[t];
                    });
                    group_barrier(partition);
                };

                // The bulge chase proper (device::sb2st_givens_chase), on whichever storage AB/C/WORK
                // point at; indexing is identical.
                auto chase = [&](T* AB, const int ldab, Real* C, T* WORK) BATCHLAS_DLA_BODY {
                    v2::sb2st_givens_chase<Ops>(pex, n, kd, AB, ldab, C, WORK);
                };

                // TRAP (Metal): no generic address space, so a pointer chosen between local and global
                // memory at run time is rejected ("Conflicting address space information in PHI node").
                // There the chase is instantiated once per storage combination; elsewhere the pointers are
                // selected once and the chase is emitted once, as before.
                if (sycl_impl::on_metal()) {
                    auto run = [&](std::pair<Real*, T*> cw) BATCHLAS_DLA_BODY {
                        zero_cw(cw.first, cw.second);
                        if (use_local_ab) {
                            T* const ABl = local_ab();
                            stage_ab(ABl, true);
                            chase(ABl, ldab_l, cw.first, cw.second);
                            stage_ab(ABl, false);
                        } else {
                            chase(ABg, ABv.ld(), cw.first, cw.second);
                        }
                    };
                    if (use_local_cw) run(local_cw());
                    else run(global_cw());
                } else {
                    const auto cw = use_local_cw ? local_cw() : global_cw();
                    zero_cw(cw.first, cw.second);
                    T* AB = ABg;
                    int ldab = ABv.ld();
                    if (use_local_ab) {
                        AB = local_ab();
                        ldab = ldab_l;
                        stage_ab(AB, true);
                    }
                    chase(AB, ldab, cw.first, cw.second);
                    if (use_local_ab) stage_ab(AB, false);
                }

                invoke_one(partition, [&]() {
                    if (live) Ops::finalize_tridiag(ABv, n, D, E, TAU);
                });
            });
    });
    return q.get_event();
}

} // namespace

template <typename T>
Event btrd_lower_inplace_subgroup_dispatch(Queue& q,
                                           KernelMatrixView<T, MatrixFormat::Dense> ab,
                                           int n,
                                           int kd,
                                           VectorView<typename base_type<T>::type> c,
                                           VectorView<T> work,
                                           VectorView<typename base_type<T>::type> d,
                                           VectorView<typename base_type<T>::type> e,
                                           VectorView<T> tau,
                                           const std::uint8_t* mask) {
    // Choose a partition size that keeps enough lanes for typical kd<=32 while
    // allowing smaller partitions when kd is very small.
    if (kd <= 2) {
        return btrd_lower_inplace_subgroup<T, 2>(q, ab, n, kd, c, work, d, e, tau, mask);
    }
    if (kd <= 4) {
        return btrd_lower_inplace_subgroup<T, 4>(q, ab, n, kd, c, work, d, e, tau, mask);
    }
    if (kd <= 8) {
        return btrd_lower_inplace_subgroup<T, 8>(q, ab, n, kd, c, work, d, e, tau, mask);
    }
    if (kd <= 16) {
        return btrd_lower_inplace_subgroup<T, 16>(q, ab, n, kd, c, work, d, e, tau, mask);
    }
    return btrd_lower_inplace_subgroup<T, 32>(q, ab, n, kd, c, work, d, e, tau, mask);
}

// ---- Explicit template instantiations (needed because sytrd_sb2st.cc only sees the declaration) ----
#define BTRD_LOWER_INPLACE_SUBGROUP_DISPATCH_INSTANTIATE(fp, real_fp) \
template Event btrd_lower_inplace_subgroup_dispatch<BATCHLAS_UNPAREN fp>(Queue&, \
                                                                          KernelMatrixView<BATCHLAS_UNPAREN fp, MatrixFormat::Dense>, \
                                                                          int, \
                                                                          int, \
                                                                          VectorView<BATCHLAS_UNPAREN real_fp>, \
                                                                          VectorView<BATCHLAS_UNPAREN fp>, \
                                                                          VectorView<BATCHLAS_UNPAREN real_fp>, \
                                                                          VectorView<BATCHLAS_UNPAREN real_fp>, \
                                                                          VectorView<BATCHLAS_UNPAREN fp>, \
                                                                          const std::uint8_t*);

BATCHLAS_FOR_EACH_SYTRD_SB2ST_CTA_DISPATCH_TYPE(BTRD_LOWER_INPLACE_SUBGROUP_DISPATCH_INSTANTIATE)

#undef BTRD_LOWER_INPLACE_SUBGROUP_DISPATCH_INSTANTIATE

} // namespace batchlas::detail
