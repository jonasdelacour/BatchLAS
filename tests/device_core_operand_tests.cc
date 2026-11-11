// SPDX-License-Identifier: MIT
// device_core_operand_tests: operand.hh additions (design A.2) against host references.
// Guards (agent-guide section 8): complex data with nonzero imaginary parts and Im(diag) != 0
// (load_row<Herm> must drop it); every unit_head form read at the head (whose stored value is
// poisoned NaN and must never be returned), above and below it; lane-cyclic register vectors
// with R = 3 registers on 8- and 32-lane executors, k on both sides of every register boundary;
// neighbour exchange at the lane wrap (lane 0 / lane size - 1); a row window c0 > 0 that
// straddles the triangle's diagonal; a local-memory tile through a decorated pointer. Red set
// (each break made once, 2026-11-04, and this suite failed): unit_head Refl::QL returning v(i)
// below the head instead of zero.
// M0 (2026-11-11, seen red, sm_89): UnitHeadOf::tail with head - k + 1 (UnitHeadOfTail...).
#include <tests/device_core_common.hh>
#include <batchlas/device/operand.hh>
#include <batchlas/device/level1.hh>

#include <cmath>
#include <limits>

using namespace device_core_test;

namespace {

template <class T> T conj_t(const T& v) {
    if constexpr (is_cplx<T>) return std::conj(v);
    else return v;
}
template <class T> class DeviceCoreOperandTest : public DeviceCoreTest<T> {};
TYPED_TEST_SUITE(DeviceCoreOperandTest, AllTypes);

TYPED_TEST(DeviceCoreOperandTest, UnitHeadFormsNeverReadTheHead) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int n = 9, head = 4;
    auto v_h = vals<T>(n, 1);
    v_h[head] = T(std::numeric_limits<R>::quiet_NaN());
    const auto v = batchlas::device::vec(v_h.data(), n);
    const auto qr = batchlas::device::unit_head<batchlas::device::Refl::QR>(v, head);
    const auto ql = batchlas::device::unit_head<batchlas::device::Refl::QL>(v, head);
    const auto keep = batchlas::device::unit_head<batchlas::device::Refl::QL, batchlas::device::DiagFill::Keep>(v, head);
    const auto old = batchlas::device::unit_head(v, head);
    for (int i = 0; i < n; ++i) {
        const T one = T(1), zero = T(0);
        EXPECT_TRUE(same_bits(qr(i), i == head ? one : (i > head ? v_h[i] : zero))) << "QR " << i;
        EXPECT_TRUE(same_bits(ql(i), i == head ? one : (i < head ? v_h[i] : zero))) << "QL " << i;
        EXPECT_TRUE(same_bits(keep(i), i == head ? one : v_h[i])) << "Keep " << i;
        EXPECT_TRUE(same_bits(old(i), qr(i))) << "unit_head(v, k) is Refl::QR, DiagFill::Zero";
    }
}

// Lane-cyclic Frag<T, 3> on a Partition<8> and a SubGroup<32>: element e = lane + P r.
template <class T, int P, class MakeEx> void check_frag_moves(Dev& d, MakeEx make) {
    constexpr int R = 3, len = P * R;
    const auto x_h = vals<T>(32 * R, 2 + P);
    T* x = d.up(x_h);
    T* bc = d.template alloc<T>(32 * len, T(0));
    T* up = d.template alloc<T>(32 * R, T(0));
    T* dn = d.template alloc<T>(32 * R, T(0));
    d.template run<32>(1, 0, [=](sycl::nd_item<1> it, unsigned char*) {
        const auto ex = make(it);
        const int g = static_cast<int>(it.get_local_linear_id());
        const int blk = g - g % P, lane = g % P;
        batchlas::device::Frag<T, R> f, u, w;
        for (int r = 0; r < R; ++r) f[r] = x[(blk + lane) * R + r];  // block-local element lane + P r
        for (int k = 0; k < len; ++k) bc[g * len + k] = batchlas::device::frag_bcast(ex, f, k);
        batchlas::device::neighbor_exchange(ex, f, u, w);
        for (int r = 0; r < R; ++r) {
            up[g * R + r] = u[r];
            dn[g * R + r] = w[r];
        }
    });
    const auto b = d.down(bc, 32 * len), u = d.down(up, 32 * R), w = d.down(dn, 32 * R);
    auto elem = [&](int blk, int e) { return x_h[(blk + e % P) * R + e / P]; };  // element e of the block's vector
    for (int g = 0; g < 32; ++g) {
        const int blk = g - g % P, lane = g % P;
        for (int k = 0; k < len; ++k) EXPECT_TRUE(same_bits(b[g * len + k], elem(blk, k))) << "frag_bcast P " << P << " k " << k;
        for (int r = 0; r < R; ++r) {
            const int e = lane + P * r;
            if (e + 1 < len) EXPECT_TRUE(same_bits(u[g * R + r], elem(blk, e + 1))) << "up P " << P << " e " << e;
            if (e >= 1) EXPECT_TRUE(same_bits(w[g * R + r], elem(blk, e - 1))) << "down P " << P << " e " << e;
        }
    }
}

TYPED_TEST(DeviceCoreOperandTest, FragBcastAndNeighbourExchangeCrossRegisterBoundaries) {
    using T = TypeParam;
    check_frag_moves<T, 8>(this->dev, [](sycl::nd_item<1> it) { return batchlas::device::Partition<8>{it.get_sub_group()}; });
    check_frag_moves<T, 32>(this->dev, [](sycl::nd_item<1> it) { return batchlas::device::SubGroup<32>{it.get_sub_group()}; });
}

TYPED_TEST(DeviceCoreOperandTest, RowMovesOnATrianglePlusLaneVecAndStridedAndLocalViews) {
    using T = TypeParam;
    using R = real_of<T>;
    auto& d = this->dev;
    constexpr int n = 11, ld = n + 3, N = 8, c0 = 3;
    const auto a_h = vals<T>(std::size_t(ld) * n, 3);
    T* a = d.up(a_h);
    T* out = d.template alloc<T>(n * N, T(0));
    T* st = d.up(a_h);
    T* lv = d.template alloc<T>(32, T(0));
    T* tl = d.template alloc<T>(12, T(0));
    const auto lane_h = vals<T>(32, 4);
    T* lin = d.up(lane_h);
    d.template run<32>(1, 64 * sizeof(T), [=](sycl::nd_item<1> it, unsigned char* lm) {
        const int r = static_cast<int>(it.get_local_linear_id());
        const auto A = batchlas::device::mat(a, n, n, ld);
        if (r < n) {
            batchlas::device::Frag<T, N> f;
            batchlas::device::load_row<batchlas::device::Part::Lower, true>(f, A, r, c0, n - c0);
            for (int j = 0; j < N; ++j) out[r * N + j] = f[j];
            batchlas::device::store_row<batchlas::device::Part::StrictUpper>(f, batchlas::device::mat(st, n, n, ld), r, c0, n - c0);
        }
        // LaneVec window [3, 20) of a sub-group: reverse it and find its first negative real part
        const batchlas::device::Partition<32> part{it.get_sub_group()};
        T v = lin[r];
        const auto x = batchlas::device::lane_vec(v, 3, 20);
        batchlas::device::reverse(part, x, true);
        lv[r] = v;
        // a 3 x 4 local tile through a decorated pointer; a row-major strided view of global
        T* slm = reinterpret_cast<T*>(lm);
        const auto p = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::yes>(slm);
        const auto tile = batchlas::device::tile_local<3, 4>(p, 5, 1);
        if (r < 12) tile(r % 3, r / 3) = lin[r];
        sycl::group_barrier(it.get_sub_group());
        const auto rm = batchlas::device::mat_strided(tl, 3, 4, 4, 1);  // row major
        if (r < 12) rm(r % 3, r / 3) = slm[1 + (r / 3) * 5 + r % 3];
    });
    const auto g = d.down(out, n * N), gs = d.down(st, a_h.size()), gl = d.down(lv, 32), gt = d.down(tl, 12);
    for (int r = 0; r < n; ++r)
        for (int j = 0; j < N; ++j) {
            const int c = c0 + j;
            T ref = T(0);
            if (j < n - c0) {
                ref = r >= c ? a_h[c * ld + r] : conj_t(a_h[r * ld + c]);
                if constexpr (is_cplx<T>)
                    if (r == c) ref = T(ref.real(), R(0));
            }
            EXPECT_TRUE(same_bits(g[r * N + j], ref)) << "load_row " << r << "," << c;
        }
    for (int c = 0; c < n; ++c)
        for (int r = 0; r < n; ++r) {
            const bool written = r < c && c >= c0;
            if (!written) {
                EXPECT_TRUE(same_bits(gs[c * ld + r], a_h[c * ld + r])) << "store_row outside " << r << "," << c;
                continue;
            }
            // the row loaded on part Lower holds conj(A(c, r)) above the diagonal
            const T ref = conj_t(a_h[r * ld + c]);
            EXPECT_TRUE(same_bits(gs[c * ld + r], ref)) << "store_row " << r << "," << c;
        }
    for (int l = 0; l < 32; ++l) EXPECT_TRUE(same_bits(gl[l], (l >= 3 && l < 20) ? lane_h[3 + 19 - l] : lane_h[l])) << "LaneVec reverse " << l;
    for (int r = 0; r < 12; ++r) EXPECT_TRUE(same_bits(gt[(r % 3) * 4 + r / 3], lane_h[r])) << "local tile / strided " << r;
}

// UnitHeadOf::tail(k): the same reflector on v(k:), for k below, at and above the head, every
// Refl x DiagFill form, read at every index of the tail (the head's stored value is NaN).
TYPED_TEST(DeviceCoreOperandTest, UnitHeadOfTailKeepsTheReflector) {
    using T = TypeParam;
    using R = real_of<T>;
    constexpr int n = 11, head = 5;
    auto v_h = vals<T>(n, 7);
    v_h[head] = T(std::numeric_limits<R>::quiet_NaN());
    const auto v = batchlas::device::vec(v_h.data(), n);
    const auto qr = batchlas::device::unit_head<batchlas::device::Refl::QR>(v, head);
    const auto ql = batchlas::device::unit_head<batchlas::device::Refl::QL>(v, head);
    const auto qk = batchlas::device::unit_head<batchlas::device::Refl::QR, batchlas::device::DiagFill::Keep>(v, head);
    for (int k : {0, 3, head, head + 2}) {
        const auto tr = qr.tail(k);
        const auto tl = ql.tail(k);
        const auto tk = qk.tail(k);
        EXPECT_EQ(tr.size(), n - k);
        for (int i = 0; i < n - k; ++i) {
            EXPECT_TRUE(same_bits(tr(i), qr(i + k))) << "QR tail " << k << " at " << i;
            EXPECT_TRUE(same_bits(tl(i), ql(i + k))) << "QL tail " << k << " at " << i;
            EXPECT_TRUE(same_bits(tk(i), qk(i + k))) << "Keep tail " << k << " at " << i;
        }
        EXPECT_TRUE(same_bits(tl.tail(1)(0), ql(k + 1))) << "tail of a tail " << k;
    }
}

}  // namespace
