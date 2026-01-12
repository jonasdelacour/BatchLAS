#include <blas/extensions.hh>
#include <blas/functions.hh>
#include <blas/matrix.hh>
#include <util/mempool.hh>

#include <sycl/sycl.hpp>

#include <batchlas/backend_config.h>

#include "../math-helpers.hh"
#include "../queue.hh"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <stdexcept>
#include <type_traits>

namespace batchlas {

namespace {

template <typename T>
inline T conj_if_needed(const T& x) {
    if constexpr (internal::is_complex<T>::value) {
        return T(x.real(), -x.imag());
    } else {
        return x;
    }
}

template <typename T>
inline typename base_type<T>::type real_part(const T& x) {
    if constexpr (internal::is_complex<T>::value) {
        return static_cast<typename base_type<T>::type>(x.real());
    } else {
        return static_cast<typename base_type<T>::type>(x);
    }
}

template <typename T>
inline void enforce_real_diagonal(T& x) {
    if constexpr (internal::is_complex<T>::value) {
        using Real = typename base_type<T>::type;
        x = T(static_cast<Real>(x.real()), Real(0));
    }
}

inline bool env_truthy(const char* v) {
    if (!v) return false;
    const std::string s(v);
    return (s == "1" || s == "true" || s == "TRUE" || s == "on" || s == "ON");
}

inline std::string bandr1_dump_root() {
    if (const char* v = std::getenv("BATCHLAS_DUMP_BANDR1_DIR")) {
        return std::string(v);
    }
    return "output/bandr1_dumps";
}

inline std::string bandr1_dump_dir_for(int32_t sweep_index, int32_t step_in_sweep) {
    std::ostringstream os;
    os << bandr1_dump_root();
    if (sweep_index >= 0) {
        os << "/sweep_" << std::setw(3) << std::setfill('0') << sweep_index;
    }
    if (step_in_sweep >= 0) {
        os << "/step_" << std::setw(4) << std::setfill('0') << step_in_sweep;
    }
    return os.str();
}

template <typename T>
inline void dump_dense_matrix_csv(Queue& ctx,
                                  const MatrixView<T, MatrixFormat::Dense>& mat,
                                  const std::string& tag,
                                  int32_t sweep_index,
                                  int32_t step_in_sweep,
                                  int32_t step_index,
                                  int32_t i1,
                                  int32_t i2,
                                  int32_t j1,
                                  int32_t j2,
                                  int32_t batch_index) {
    const int rows = mat.rows();
    const int cols = mat.cols();
    if (rows <= 0 || cols <= 0) return;
    if (batch_index < 0 || batch_index >= mat.batch_size()) return;

    const std::string dir = bandr1_dump_dir_for(sweep_index, step_in_sweep);
    std::filesystem::create_directories(dir);

    auto src = mat[batch_index];
    T* host = sycl::malloc_host<T>(static_cast<size_t>(rows) * static_cast<size_t>(cols), ctx->get_context());
    if (!host) {
        throw std::runtime_error("bandr1 dump: sycl::malloc_host failed");
    }

    MatrixView<T, MatrixFormat::Dense> host_view(host, rows, cols, rows, rows * cols, /*batch_size=*/1);
    MatrixView<T, MatrixFormat::Dense>::copy(ctx, host_view, src).wait();

    std::ostringstream fname;
    fname << dir << "/" << tag;
    if (sweep_index >= 0) {
        fname << "_sw" << sweep_index;
    }
    if (step_in_sweep >= 0) {
        fname << "_st" << step_in_sweep;
    }
    if (step_index >= 0) {
        fname << "_g" << step_index;
    }
    fname << "_i1" << i1 << "_i2" << i2 << "_j1" << j1 << "_j2" << j2 << "_b" << batch_index << ".csv";

    std::ofstream os(fname.str(), std::ios::out | std::ios::trunc);
    if (!os) {
        sycl::free(host, ctx->get_context());
        return;
    }

    os << "# rows," << rows << ",cols," << cols << "\n";
    os << "# i,j,real,imag\n";
    os << std::setprecision(17);

    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            const T v = host_view(i, j, 0);
            if constexpr (internal::is_complex<T>::value) {
                os << i << "," << j << "," << std::real(v) << "," << std::imag(v) << "\n";
            } else {
                os << i << "," << j << "," << v << "," << 0 << "\n";
            }
        }
    }

    os.close();
    sycl::free(host, ctx->get_context());
}

template <typename T>
inline void band_extract_block(Queue& ctx,
                               const MatrixView<T, MatrixFormat::Dense>& ab,
                               int32_t kd,
                               const MatrixView<T, MatrixFormat::Dense>& out,
                               int32_t r0,
                               int32_t c0) {
    // out(ii,jj) = A(r0+ii, c0+jj), with A represented by lower-band storage in `ab`.
    const int rows = out.rows();
    const int cols = out.cols();
    const int batch = ab.batch_size();

    auto AB = ab.kernel_view();
    auto O = out.kernel_view();

    ctx->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<3>(static_cast<size_t>(batch),
                                       static_cast<size_t>(rows),
                                       static_cast<size_t>(cols)),
                         [=](sycl::id<3> idx) {
                             auto safe_get = [](const auto& M, int row, int col, int b) -> T {
                                 if (b < 0 || b >= M.batch_size_) return T(0);
                                 if (row < 0 || row >= M.rows_) return T(0);
                                 if (col < 0 || col >= M.cols_) return T(0);

                                 const size_t stride = static_cast<size_t>(M.stride_);
                                 const size_t base = static_cast<size_t>(b) * stride;
                                 const size_t off = static_cast<size_t>(col) * static_cast<size_t>(M.ld_) + static_cast<size_t>(row);
                                 const size_t lin = base + off;
                                 const size_t max = static_cast<size_t>(M.batch_size_) * stride;
                                 if (lin >= max) return T(0);
                                 return M.data_[lin];
                             };

                             auto safe_set = [](auto& M, int row, int col, int b, const T& value) {
                                 if (b < 0 || b >= M.batch_size_) return;
                                 if (row < 0 || row >= M.rows_) return;
                                 if (col < 0 || col >= M.cols_) return;

                                 const size_t stride = static_cast<size_t>(M.stride_);
                                 const size_t base = static_cast<size_t>(b) * stride;
                                 const size_t off = static_cast<size_t>(col) * static_cast<size_t>(M.ld_) + static_cast<size_t>(row);
                                 const size_t lin = base + off;
                                 const size_t max = static_cast<size_t>(M.batch_size_) * stride;
                                 if (lin >= max) return;
                                 M.data_[lin] = value;
                             };

                             const int b = static_cast<int>(idx[0]);
                             const int ii = static_cast<int>(idx[1]);
                             const int jj = static_cast<int>(idx[2]);
                             const int i = r0 + ii;
                             const int j = c0 + jj;

                             T val = T(0);
                             if (i >= j) {
                                 const int r = i - j;
                                 if (r <= kd) {
                                     val = safe_get(AB, r, j, b);
                                 }
                             } else {
                                 const int r = j - i;
                                 if (r <= kd) {
                                     val = conj_if_needed(safe_get(AB, r, i, b));
                                 }
                             }
                             if (i == j) {
                                 enforce_real_diagonal(val);
                             }
                             safe_set(O, ii, jj, b, val);
                         });
    });
}

template <typename T>
inline void band_scatter_block_lower(Queue& ctx,
                                     const MatrixView<T, MatrixFormat::Dense>& in,
                                     const MatrixView<T, MatrixFormat::Dense>& ab,
                                     int32_t kd,
                                     int32_t r0,
                                     int32_t c0) {
    // Store only lower-triangle entries of the global block into lower-band `ab`.
    // Global (i,j) stored when i>=j and i-j<=kd.
    const int rows = in.rows();
    const int cols = in.cols();
    const int batch = ab.batch_size();

    auto I = in.kernel_view();
    auto AB = ab.kernel_view();

    ctx->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<3>(static_cast<size_t>(batch),
                                       static_cast<size_t>(rows),
                                       static_cast<size_t>(cols)),
                         [=](sycl::id<3> idx) {
                             const int b = static_cast<int>(idx[0]);
                             const int ii = static_cast<int>(idx[1]);
                             const int jj = static_cast<int>(idx[2]);
                             const int i = r0 + ii;
                             const int j = c0 + jj;
                             if (i < j) return;
                             const int r = i - j;
                             if (r > kd) return;
                             T val = I(ii, jj, b);
                             if (i == j) {
                                 enforce_real_diagonal(val);
                             }
                             AB(r, j, b) = val;
                         });
    });
}

template <typename T>
inline void band_scatter_block(Queue& ctx,
                               const MatrixView<T, MatrixFormat::Dense>& in,
                               const MatrixView<T, MatrixFormat::Dense>& ab,
                               int32_t kd,
                               int32_t r0,
                               int32_t c0) {
    // Scatter a dense block into Hermitian lower-band storage.
    // If (i,j) is above the diagonal, store conj(in) into the swapped lower position.
    const int rows = in.rows();
    const int cols = in.cols();
    const int batch = std::min(in.batch_size(), ab.batch_size());

    auto I = in.kernel_view();
    auto AB = ab.kernel_view();

    ctx->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<3>(static_cast<size_t>(batch),
                                       static_cast<size_t>(rows),
                                       static_cast<size_t>(cols)),
                         [=](sycl::id<3> idx) {
                             const int b = static_cast<int>(idx[0]);
                             const int ii = static_cast<int>(idx[1]);
                             const int jj = static_cast<int>(idx[2]);

                             auto safe_get = [](const auto& M, int row, int col, int bb) -> T {
                                 if (bb < 0 || bb >= M.batch_size()) return T(0);
                                 if (row < 0 || row >= M.rows()) return T(0);
                                 if (col < 0 || col >= M.cols()) return T(0);
                                 const int64_t lin = int64_t(bb) * int64_t(M.stride()) + int64_t(col) * int64_t(M.ld()) + int64_t(row);
                                 const int64_t max_lin = int64_t(M.batch_size()) * int64_t(M.stride());
                                 if (lin < 0 || lin >= max_lin) return T(0);
                                 return M.data()[lin];
                             };

                             auto safe_set = [](const auto& M, int row, int col, int bb, T value) {
                                 if (bb < 0 || bb >= M.batch_size()) return;
                                 if (row < 0 || row >= M.rows()) return;
                                 if (col < 0 || col >= M.cols()) return;
                                 const int64_t lin = int64_t(bb) * int64_t(M.stride()) + int64_t(col) * int64_t(M.ld()) + int64_t(row);
                                 const int64_t max_lin = int64_t(M.batch_size()) * int64_t(M.stride());
                                 if (lin < 0 || lin >= max_lin) return;
                                 M.data()[lin] = value;
                             };

                             const int i = r0 + ii;
                             const int j = c0 + jj;
                             const int diff = i - j;
                             if (diff >= 0) {
                                 if (diff > kd) return;
                                 T val = safe_get(I, ii, jj, b);
                                 if (diff == 0) {
                                     enforce_real_diagonal(val);
                                 }
                                 safe_set(AB, diff, j, b, val);
                             } else {
                                 const int r = -diff;
                                 if (r > kd) return;
                                 // Store A(j,i) in the lower band.
                                 T val = conj_if_needed(safe_get(I, ii, jj, b));
                                 if (r == 0) {
                                     enforce_real_diagonal(val);
                                 }
                                 safe_set(AB, r, i, b, val);
                             }
                         });
    });
}

template <typename T>
inline void band_zero_rows_from(Queue& ctx,
                                const MatrixView<T, MatrixFormat::Dense>& ab,
                                int32_t r0) {
    const int rows = ab.rows();
    const int cols = ab.cols();
    const int batch = ab.batch_size();

    if (r0 >= rows) return;
    auto AB = ab.kernel_view();
    ctx->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<3>(static_cast<size_t>(batch),
                                       static_cast<size_t>(rows - r0),
                                       static_cast<size_t>(cols)),
                         [=](sycl::id<3> idx) {
                             const int b = static_cast<int>(idx[0]);
                             const int rr = static_cast<int>(idx[1]) + r0;
                             const int j = static_cast<int>(idx[2]);
                             AB(rr, j, b) = T(0);
                         });
    });
}

template <typename T>
inline void dense_keep_qr_R_only(Queue& ctx,
                                const MatrixView<T, MatrixFormat::Dense>& a,
                                int32_t r) {
    // After GEQRF, `a` contains Householder vectors below the diagonal.
    // For the similarity update we want the transformed block Q^H * A = R,
    // i.e. an upper-triangular (in the panel columns) block with only the top
    // `r` rows potentially nonzero.
    const int rows = a.rows();
    const int cols = a.cols();
    const int batch = a.batch_size();
    auto A = a.kernel_view();

    ctx->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<3>(static_cast<size_t>(batch),
                                       static_cast<size_t>(rows),
                                       static_cast<size_t>(cols)),
                         [=](sycl::id<3> idx) {
                             const int b = static_cast<int>(idx[0]);
                             const int i = static_cast<int>(idx[1]);
                             const int j = static_cast<int>(idx[2]);
                             if (i >= r || i > j) {
                                 A(i, j, b) = T(0);
                             }
                         });
    });
}

template <typename T>
inline void validate_band_reduction_dims(const MatrixView<T, MatrixFormat::Dense>& ab,
                                        const VectorView<typename base_type<T>::type>& d,
                                        const VectorView<typename base_type<T>::type>& e,
                                        const VectorView<T>& tau,
                                        int32_t kd) {
    const int rows = ab.rows();
    const int n = ab.cols();
    const int batch = ab.batch_size();

    if (kd < 0) {
        throw std::runtime_error("sytrd_band_reduction: kd must be >= 0");
    }
    if (rows != kd + 1) {
        throw std::runtime_error("sytrd_band_reduction: ab.rows() must equal kd+1");
    }
    if (d.size() != n || d.batch_size() != batch) {
        throw std::runtime_error("sytrd_band_reduction: d_out must have size n and matching batch");
    }
    if (e.size() != std::max(0, n - 1) || e.batch_size() != batch) {
        throw std::runtime_error("sytrd_band_reduction: e_out must have size n-1 and matching batch");
    }
    if (tau.size() != std::max(0, n - 1) || tau.batch_size() != batch) {
        throw std::runtime_error("sytrd_band_reduction: tau_out must have size n-1 and matching batch");
    }
}

template <typename T>
inline Span<T> alloc_from_ws(Queue& ctx,
                            Span<std::byte> ws,
                            size_t& byte_offset,
                            size_t count) {
    if (count == 0) return {};

    const std::uintptr_t base_addr = reinterpret_cast<std::uintptr_t>(ws.data());
    const size_t align = BumpAllocator::alignment<T>(ctx.device());

    std::uintptr_t cur = base_addr + byte_offset;
    std::uintptr_t aligned = (cur % align == 0) ? cur : ((cur + align - 1) & ~(align - 1));
    byte_offset = static_cast<size_t>(aligned - base_addr);

    const size_t alloc_bytes = BumpAllocator::allocation_size<T>(ctx.device(), count);
    if (byte_offset + alloc_bytes > ws.size()) {
        throw std::runtime_error("sytrd_band_reduction: insufficient workspace");
    }

    T* ptr = reinterpret_cast<T*>(reinterpret_cast<std::byte*>(ws.data()) + byte_offset);
    byte_offset += alloc_bytes;
    return Span<T>(ptr, count);
}

} // namespace

namespace {

template <Backend B, typename T>
inline void bandr1_one_qr_step(Queue& ctx,
                               const MatrixView<T, MatrixFormat::Dense>& ABw,
                               int32_t kd_work,
                               int32_t b_current,
                               int32_t sweep_index,
                               int32_t step_in_sweep,
                               int32_t step_index,
                               int32_t i1,
                               int32_t i2,
                               int32_t j1,
                               int32_t j2,
                               const MatrixView<T, MatrixFormat::Dense>& Bmat,
                               const MatrixView<T, MatrixFormat::Dense>& Symmat,
                               const MatrixView<T, MatrixFormat::Dense>& Postmat,
                               const MatrixView<T, MatrixFormat::Dense>& Premat,
                               const MatrixView<T, MatrixFormat::Dense>& Rightmat,
                               Span<T> tau_buf,
                               Span<std::byte> ws_backend) {
    const int n = ABw.cols();
    (void)Postmat;

    bool dump_enabled = env_truthy(std::getenv("BATCHLAS_DUMP_BANDR1_STEP"));
    const bool dump_abw_only = env_truthy(std::getenv("BATCHLAS_DUMP_BANDR1_ABW_ONLY"));
    const int dump_step = [&]() -> int {
        if (const char* v = std::getenv("BATCHLAS_DUMP_BANDR1_STEP_INDEX")) {
            try {
                return std::stoi(std::string(v));
            } catch (...) {
                return -1;
            }
        }
        return -1;
    }();
    if (dump_step >= 0 && dump_step != step_index) {
        dump_enabled = false;
    }

    const int dump_sweep = [&]() -> int {
        if (const char* v = std::getenv("BATCHLAS_DUMP_BANDR1_SWEEP_INDEX")) {
            try {
                return std::stoi(std::string(v));
            } catch (...) {
                return -1;
            }
        }
        return -1;
    }();
    if (dump_sweep >= 0 && dump_sweep != sweep_index) {
        dump_enabled = false;
    }

    const int dump_step_in_sweep = [&]() -> int {
        if (const char* v = std::getenv("BATCHLAS_DUMP_BANDR1_STEP_IN_SWEEP")) {
            try {
                return std::stoi(std::string(v));
            } catch (...) {
                return -1;
            }
        }
        return -1;
    }();
    if (dump_step_in_sweep >= 0 && dump_step_in_sweep != step_in_sweep) {
        dump_enabled = false;
    }
    const int dump_batch = [&]() -> int {
        if (const char* v = std::getenv("BATCHLAS_DUMP_BANDR1_BATCH")) {
            try {
                return std::stoi(std::string(v));
            } catch (...) {
                return -1;
            }
        }
        return -1;
    }();

    const Transpose trans_left = internal::is_complex<T>::value ? Transpose::ConjTrans : Transpose::Trans;

    const int m = i2 - i1 + 1;
    const int r = j2 - j1 + 1;
    if (m <= 0 || r <= 0) return;

    if (dump_enabled) {
        if (dump_batch >= 0) {
            dump_dense_matrix_csv<T>(ctx, ABw, "ABw_band_before", sweep_index, step_in_sweep, step_index, i1, i2, j1, j2, dump_batch);
        } else {
            for (int b = 0; b < ABw.batch_size(); ++b) {
                dump_dense_matrix_csv<T>(ctx, ABw, "ABw_band_before", sweep_index, step_in_sweep, step_index, i1, i2, j1, j2, b);
            }
        }
    }

    // Important: do not truncate the panel columns when m < r.
    // QR reflectors (and thus the similarity transform) depend on *all* columns.
    auto Bsub = Bmat({0, m}, {0, r});
    auto Symsub = Symmat({0, m}, {0, m});

    // Post block lives below the diagonal, and in our working representation we
    // store up to kd_work subdiagonals. Even if the *input* band is only b_current
    // wide, QR steps can create fill-in deeper in the band; those entries must be
    // updated as part of A <- A*Q to preserve similarity.
    const int post_r0 = i2 + 1;
    const int post_r1 = std::min(i2 + kd_work, n - 1) + 1;
    const int post_rows = std::max(0, post_r1 - post_r0);

    const int pre_r0 = std::max(0, i1 - kd_work);
    const int pre_rows = std::max(0, i1 - pre_r0);

    const int right_c0 = i2 + 1;
    const int right_c1 = std::min(n, i2 + kd_work + 1);
    const int right_cols = std::max(0, right_c1 - right_c0);

    // Left-of-panel block: rows i1..i2, columns within the work band and strictly
    // left of the QR panel (cols < j1). This must be updated as part of the left
    // multiplication Q^H * A, otherwise later chase steps (where j1 > 0) are not a
    // true similarity transform.
    const int left_c0 = std::max(0, i1 - kd_work);
    const int left_c1 = std::min(j1, i1);
    const int left_cols = std::max(0, left_c1 - left_c0);

    // Mid block is A(i1:i2, j2+1:i1-1). In a true dense similarity transform this region
    // can become dense, but our working representation only stores entries within
    // kd_work below the diagonal. Restrict the update to the in-band portion only.
    //
    // This also ensures the temporary buffer (allocated as m×kd_work) is never
    // sliced wider than its capacity.
    const int mid_c1 = i1;
    const int mid_c0_full = j2 + 1;
    const int mid_c0 = std::max(mid_c0_full, i1 - kd_work);
    const int mid_cols = std::max(0, mid_c1 - mid_c0);

    band_extract_block<T>(ctx, ABw, kd_work, Bsub, i1, j1);

    if (dump_enabled && !dump_abw_only) {
        if (dump_batch >= 0) {
            dump_dense_matrix_csv<T>(ctx, Bsub, "B_before_geqrf", sweep_index, step_in_sweep, step_index, i1, i2, j1, j2, dump_batch);
        } else {
            for (int b = 0; b < ABw.batch_size(); ++b) {
                dump_dense_matrix_csv<T>(ctx, Bsub, "B_before_geqrf", sweep_index, step_in_sweep, step_index, i1, i2, j1, j2, b);
            }
        }
    }

    const int k = std::min(m, r);
    Span<T> tau_span(tau_buf.data(), static_cast<size_t>(k) * static_cast<size_t>(ABw.batch_size()));
    geqrf<B, T>(ctx, Bsub, tau_span, ws_backend).wait();

    if (dump_enabled && !dump_abw_only) {
        if (dump_batch >= 0) {
            dump_dense_matrix_csv<T>(ctx, Bsub, "B_after_geqrf", sweep_index, step_in_sweep, step_index, i1, i2, j1, j2, dump_batch);
        } else {
            for (int b = 0; b < ABw.batch_size(); ++b) {
                dump_dense_matrix_csv<T>(ctx, Bsub, "B_after_geqrf", sweep_index, step_in_sweep, step_index, i1, i2, j1, j2, b);
            }
        }
    }

    band_extract_block<T>(ctx, ABw, kd_work, Symsub, i1, i1);

    if (dump_enabled && !dump_abw_only) {
        if (dump_batch >= 0) {
            dump_dense_matrix_csv<T>(ctx, Symsub, "Sym_before", sweep_index, step_in_sweep, step_index, i1, i2, j1, j2, dump_batch);
        } else {
            for (int b = 0; b < ABw.batch_size(); ++b) {
                dump_dense_matrix_csv<T>(ctx, Symsub, "Sym_before", sweep_index, step_in_sweep, step_index, i1, i2, j1, j2, b);
            }
        }
    }

    ormqr<B, T>(ctx, Bsub, Symsub, Side::Left, trans_left, tau_span, ws_backend).wait();
    ormqr<B, T>(ctx, Bsub, Symsub, Side::Right, Transpose::NoTrans, tau_span, ws_backend).wait();

    if (dump_enabled && !dump_abw_only) {
        if (dump_batch >= 0) {
            dump_dense_matrix_csv<T>(ctx, Symsub, "Sym_after", sweep_index, step_in_sweep, step_index, i1, i2, j1, j2, dump_batch);
        } else {
            for (int b = 0; b < ABw.batch_size(); ++b) {
                dump_dense_matrix_csv<T>(ctx, Symsub, "Sym_after", sweep_index, step_in_sweep, step_index, i1, i2, j1, j2, b);
            }
        }
    }

    band_scatter_block<T>(ctx, Symsub, ABw, kd_work, i1, i1);

    if (left_cols > 0) {
        auto Leftsub = Rightmat({0, m}, {0, left_cols});
        band_extract_block<T>(ctx, ABw, kd_work, Leftsub, i1, left_c0);

        ormqr<B, T>(ctx, Bsub, Leftsub, Side::Left, trans_left, tau_span, ws_backend).wait();

        band_scatter_block<T>(ctx, Leftsub, ABw, kd_work, i1, left_c0);
    }

    // Apply the left-side update to the in-band block between B (cols j1..j2) and Sym (cols i1..i2).
    // This corresponds to the Python reference's `Pre` update: A(i1:i2, j2+1:i1-1) <- Q^H * A.
    if (mid_cols > 0) {
        auto Midsub = Rightmat({0, m}, {0, mid_cols});
        band_extract_block<T>(ctx, ABw, kd_work, Midsub, i1, mid_c0);

        if (dump_enabled && !dump_abw_only) {
            if (dump_batch >= 0) {
                dump_dense_matrix_csv<T>(ctx, Midsub, "Mid_before", sweep_index, step_in_sweep, step_index, i1, i2, j1, j2, dump_batch);
            } else {
                for (int b = 0; b < ABw.batch_size(); ++b) {
                    dump_dense_matrix_csv<T>(ctx, Midsub, "Mid_before", sweep_index, step_in_sweep, step_index, i1, i2, j1, j2, b);
                }
            }
        }

        ormqr<B, T>(ctx, Bsub, Midsub, Side::Left, trans_left, tau_span, ws_backend).wait();

        if (dump_enabled && !dump_abw_only) {
            if (dump_batch >= 0) {
                dump_dense_matrix_csv<T>(ctx, Midsub, "Mid_after", sweep_index, step_in_sweep, step_index, i1, i2, j1, j2, dump_batch);
            } else {
                for (int b = 0; b < ABw.batch_size(); ++b) {
                    dump_dense_matrix_csv<T>(ctx, Midsub, "Mid_after", sweep_index, step_in_sweep, step_index, i1, i2, j1, j2, b);
                }
            }
        }

        band_scatter_block<T>(ctx, Midsub, ABw, kd_work, i1, mid_c0);
    }

    if (right_cols > 0) {
        auto Rightsub = Rightmat({0, m}, {0, right_cols});
        band_extract_block<T>(ctx, ABw, kd_work, Rightsub, i1, right_c0);

        if (dump_enabled && !dump_abw_only) {
            if (dump_batch >= 0) {
                dump_dense_matrix_csv<T>(ctx, Rightsub, "Right_before", sweep_index, step_in_sweep, step_index, i1, i2, j1, j2, dump_batch);
            } else {
                for (int b = 0; b < ABw.batch_size(); ++b) {
                    dump_dense_matrix_csv<T>(ctx, Rightsub, "Right_before", sweep_index, step_in_sweep, step_index, i1, i2, j1, j2, b);
                }
            }
        }

        ormqr<B, T>(ctx, Bsub, Rightsub, Side::Left, trans_left, tau_span, ws_backend).wait();

        if (dump_enabled && !dump_abw_only) {
            if (dump_batch >= 0) {
                dump_dense_matrix_csv<T>(ctx, Rightsub, "Right_after", sweep_index, step_in_sweep, step_index, i1, i2, j1, j2, dump_batch);
            } else {
                for (int b = 0; b < ABw.batch_size(); ++b) {
                    dump_dense_matrix_csv<T>(ctx, Rightsub, "Right_after", sweep_index, step_in_sweep, step_index, i1, i2, j1, j2, b);
                }
            }
        }

        // Rightsub is above the diagonal (rows i1..i2, cols i2+1..), so it is not
        // stored in our lower-band representation. Do NOT hermitian-scatter it into
        // the lower band (that would overwrite the Post block which we update explicitly).
        band_scatter_block_lower<T>(ctx, Rightsub, ABw, kd_work, i1, right_c0);
    }

    if (post_rows > 0) {
        // Use Premat as scratch (it is sized kd_work×m_max).
        auto Postsub = Premat({0, post_rows}, {0, m});
        band_extract_block<T>(ctx, ABw, kd_work, Postsub, post_r0, i1);

        if (dump_enabled && !dump_abw_only) {
            if (dump_batch >= 0) {
                dump_dense_matrix_csv<T>(ctx, Postsub, "Post_before", sweep_index, step_in_sweep, step_index, i1, i2, j1, j2, dump_batch);
            } else {
                for (int b = 0; b < ABw.batch_size(); ++b) {
                    dump_dense_matrix_csv<T>(ctx, Postsub, "Post_before", sweep_index, step_in_sweep, step_index, i1, i2, j1, j2, b);
                }
            }
        }

        ormqr<B, T>(ctx, Bsub, Postsub, Side::Right, Transpose::NoTrans, tau_span, ws_backend).wait();

        if (dump_enabled && !dump_abw_only) {
            if (dump_batch >= 0) {
                dump_dense_matrix_csv<T>(ctx, Postsub, "Post_after", sweep_index, step_in_sweep, step_index, i1, i2, j1, j2, dump_batch);
            } else {
                for (int b = 0; b < ABw.batch_size(); ++b) {
                    dump_dense_matrix_csv<T>(ctx, Postsub, "Post_after", sweep_index, step_in_sweep, step_index, i1, i2, j1, j2, b);
                }
            }
        }

        band_scatter_block<T>(ctx, Postsub, ABw, kd_work, post_r0, i1);
    }

    if (pre_rows > 0) {
        auto Presub = Premat({0, pre_rows}, {0, m});
        band_extract_block<T>(ctx, ABw, kd_work, Presub, pre_r0, i1);

        if (dump_enabled && !dump_abw_only) {
            if (dump_batch >= 0) {
                dump_dense_matrix_csv<T>(ctx, Presub, "Pre_before", sweep_index, step_in_sweep, step_index, i1, i2, j1, j2, dump_batch);
            } else {
                for (int b = 0; b < ABw.batch_size(); ++b) {
                    dump_dense_matrix_csv<T>(ctx, Presub, "Pre_before", sweep_index, step_in_sweep, step_index, i1, i2, j1, j2, b);
                }
            }
        }

        ormqr<B, T>(ctx, Bsub, Presub, Side::Right, Transpose::NoTrans, tau_span, ws_backend).wait();

        if (dump_enabled && !dump_abw_only) {
            if (dump_batch >= 0) {
                dump_dense_matrix_csv<T>(ctx, Presub, "Pre_after", sweep_index, step_in_sweep, step_index, i1, i2, j1, j2, dump_batch);
            } else {
                for (int b = 0; b < ABw.batch_size(); ++b) {
                    dump_dense_matrix_csv<T>(ctx, Presub, "Pre_after", sweep_index, step_in_sweep, step_index, i1, i2, j1, j2, b);
                }
            }
        }

        // Presub is above the diagonal (rows < cols); it is not stored in the lower band.
        // Avoid hermitian-scattering into the lower band (it would overwrite the Mid block).
        band_scatter_block_lower<T>(ctx, Presub, ABw, kd_work, pre_r0, i1);
    }

    dense_keep_qr_R_only<T>(ctx, Bsub, r);

    if (dump_enabled && !dump_abw_only) {
        if (dump_batch >= 0) {
            dump_dense_matrix_csv<T>(ctx, Bsub, "B_after_keepR", sweep_index, step_in_sweep, step_index, i1, i2, j1, j2, dump_batch);
        } else {
            for (int b = 0; b < ABw.batch_size(); ++b) {
                dump_dense_matrix_csv<T>(ctx, Bsub, "B_after_keepR", sweep_index, step_in_sweep, step_index, i1, i2, j1, j2, b);
            }
        }
    }

    band_scatter_block<T>(ctx, Bsub, ABw, kd_work, i1, j1);

    if (dump_enabled) {
        if (dump_batch >= 0) {
            dump_dense_matrix_csv<T>(ctx, ABw, "ABw_band_after", sweep_index, step_in_sweep, step_index, i1, i2, j1, j2, dump_batch);
        } else {
            for (int b = 0; b < ABw.batch_size(); ++b) {
                dump_dense_matrix_csv<T>(ctx, ABw, "ABw_band_after", sweep_index, step_in_sweep, step_index, i1, i2, j1, j2, b);
            }
        }
    }
}

template <Backend B, typename T>
size_t sytrd_band_reduction_single_step_buffer_size_core(Queue& ctx,
                                                         int32_t kd,
                                                         int32_t nb_target,
                                                         int32_t kd_work,
                                                         int32_t n,
                                                         int32_t batch) {
    if (kd <= 1 || n <= 1) return 0;
    kd_work = std::min(kd_work, n - 1);

    const int nb_max = std::min(nb_target, std::max(1, kd - 1));
    const int m_max = kd;

    size_t bytes = 0;
    bytes += BumpAllocator::allocation_size<T>(ctx, static_cast<size_t>(m_max) * static_cast<size_t>(nb_max) * static_cast<size_t>(batch));
    bytes += BumpAllocator::allocation_size<T*>(ctx, static_cast<size_t>(batch));
    bytes += BumpAllocator::allocation_size<T>(ctx, static_cast<size_t>(m_max) * static_cast<size_t>(m_max) * static_cast<size_t>(batch));
    bytes += BumpAllocator::allocation_size<T*>(ctx, static_cast<size_t>(batch));
    bytes += BumpAllocator::allocation_size<T>(ctx, static_cast<size_t>(m_max) * static_cast<size_t>(m_max) * static_cast<size_t>(batch));
    bytes += BumpAllocator::allocation_size<T*>(ctx, static_cast<size_t>(batch));
    bytes += BumpAllocator::allocation_size<T>(ctx, static_cast<size_t>(kd_work) * static_cast<size_t>(m_max) * static_cast<size_t>(batch));
    bytes += BumpAllocator::allocation_size<T*>(ctx, static_cast<size_t>(batch));
    bytes += BumpAllocator::allocation_size<T>(ctx, static_cast<size_t>(m_max) * static_cast<size_t>(kd_work) * static_cast<size_t>(batch));
    bytes += BumpAllocator::allocation_size<T*>(ctx, static_cast<size_t>(batch));
    bytes += BumpAllocator::allocation_size<T>(ctx, static_cast<size_t>(nb_max) * static_cast<size_t>(batch));

    MatrixView<T, MatrixFormat::Dense> dummyB(nullptr, m_max, nb_max, m_max, m_max * nb_max, batch);
    Span<T> dummyTau(nullptr, static_cast<size_t>(nb_max) * static_cast<size_t>(batch));
    bytes += geqrf_buffer_size<B, T>(ctx, dummyB, dummyTau);

    MatrixView<T, MatrixFormat::Dense> dummySym(nullptr, m_max, m_max, m_max, m_max * m_max, batch);
    bytes += ormqr_buffer_size<B, T>(ctx, dummyB, dummySym, Side::Left, Transpose::ConjTrans, dummyTau);
    bytes += ormqr_buffer_size<B, T>(ctx, dummyB, dummySym, Side::Right, Transpose::NoTrans, dummyTau);

    MatrixView<T, MatrixFormat::Dense> dummyPre(nullptr, kd_work, m_max, kd_work, static_cast<int64_t>(kd_work) * m_max, batch);
    bytes += ormqr_buffer_size<B, T>(ctx, dummyB, dummyPre, Side::Right, Transpose::NoTrans, dummyTau);

    MatrixView<T, MatrixFormat::Dense> dummyRight(nullptr, m_max, kd_work, m_max, static_cast<int64_t>(m_max) * kd_work, batch);
    bytes += ormqr_buffer_size<B, T>(ctx, dummyB, dummyRight, Side::Left, Transpose::ConjTrans, dummyTau);

    return bytes;
}

template <Backend B, typename T>
Event sytrd_band_reduction_single_step_core(Queue& ctx,
                                            const MatrixView<T, MatrixFormat::Dense>& ab_in,
                                            const MatrixView<T, MatrixFormat::Dense>& abw_out,
                                            Uplo uplo,
                                            int32_t kd,
                                            const Span<std::byte>& ws,
                                            SytrdBandReductionParams params) {
    if (!ctx.in_order()) {
        throw std::runtime_error("sytrd_band_reduction_single_step: requires an in-order Queue");
    }
    if (uplo != Uplo::Lower) {
        throw std::runtime_error("sytrd_band_reduction_single_step: only Uplo::Lower is implemented");
    }

    const int n = ab_in.cols();
    const int batch = ab_in.batch_size();
    if (ab_in.rows() != kd + 1) {
        throw std::runtime_error("sytrd_band_reduction_single_step: ab_in.rows() must equal kd+1");
    }
    if (abw_out.cols() != n || abw_out.batch_size() != batch) {
        throw std::runtime_error("sytrd_band_reduction_single_step: abw_out must match (n,batch)");
    }

    if (kd <= 1 || n <= 1) {
        // Nothing to do; still populate ABw with the input band.
        // (This makes tests simpler and deterministic.)
        auto ABW = abw_out.kernel_view();
        ctx->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<3>(static_cast<size_t>(batch),
                                           static_cast<size_t>(abw_out.rows()),
                                           static_cast<size_t>(n)),
                             [=](sycl::id<3> idx) {
                                 const int b = static_cast<int>(idx[0]);
                                 const int r = static_cast<int>(idx[1]);
                                 const int j = static_cast<int>(idx[2]);
                                 ABW(r, j, b) = T(0);
                             });
        });
        auto ABin = ab_in.kernel_view();
        ctx->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<3>(static_cast<size_t>(batch),
                                           static_cast<size_t>(std::min<int>(kd + 1, abw_out.rows())),
                                           static_cast<size_t>(n)),
                             [=](sycl::id<3> idx) {
                                 const int b = static_cast<int>(idx[0]);
                                 const int r = static_cast<int>(idx[1]);
                                 const int j = static_cast<int>(idx[2]);
                                 ABW(r, j, b) = ABin(r, j, b);
                             });
        });
        return ctx.get_event();
    }

    int kd_work = params.kd_work;
    if (kd_work <= 0) {
        kd_work = std::min(n - 1, 3 * kd);
    }
    kd_work = std::min(kd_work, n - 1);
    if (abw_out.rows() != kd_work + 1) {
        throw std::runtime_error("sytrd_band_reduction_single_step: abw_out.rows() must equal kd_work+1");
    }

    const int nb_target = std::max<int>(1, params.block_size);
    const int d_per_sweep = params.d;
    if (d_per_sweep < 0) {
        throw std::runtime_error("sytrd_band_reduction_single_step: params.d must be >= 0");
    }

    int max_steps = params.max_steps;
    if (max_steps <= 0) {
        max_steps = 1;
    }

    // Initialize ABw = 0; then copy input band into the top kd+1 rows.
    {
        auto ABW = abw_out.kernel_view();
        ctx->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<3>(static_cast<size_t>(batch),
                                           static_cast<size_t>(kd_work + 1),
                                           static_cast<size_t>(n)),
                             [=](sycl::id<3> idx) {
                                 const int b = static_cast<int>(idx[0]);
                                 const int r = static_cast<int>(idx[1]);
                                 const int j = static_cast<int>(idx[2]);
                                 ABW(r, j, b) = T(0);
                             });
        });

        auto ABin = ab_in.kernel_view();
        ctx->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<3>(static_cast<size_t>(batch),
                                           static_cast<size_t>(kd + 1),
                                           static_cast<size_t>(n)),
                             [=](sycl::id<3> idx) {
                                 const int b = static_cast<int>(idx[0]);
                                 const int r = static_cast<int>(idx[1]);
                                 const int j = static_cast<int>(idx[2]);
                                 ABW(r, j, b) = ABin(r, j, b);
                             });
        });
    }

    // Scratch buffers sized to worst-case within a sweep.
    const int nb_max = std::min(nb_target, std::max(1, kd - 1));
    const int m_max = kd;
    size_t off = 0;
    auto B_buf = alloc_from_ws<T>(ctx, ws, off,
                                  static_cast<size_t>(m_max) * static_cast<size_t>(nb_max) * static_cast<size_t>(batch));
    auto B_ptrs = alloc_from_ws<T*>(ctx, ws, off, static_cast<size_t>(batch));
    auto Sym_buf = alloc_from_ws<T>(ctx, ws, off,
                                    static_cast<size_t>(m_max) * static_cast<size_t>(m_max) * static_cast<size_t>(batch));
    auto Sym_ptrs = alloc_from_ws<T*>(ctx, ws, off, static_cast<size_t>(batch));
    auto Post_buf = alloc_from_ws<T>(ctx, ws, off,
                                     static_cast<size_t>(m_max) * static_cast<size_t>(m_max) * static_cast<size_t>(batch));
    auto Post_ptrs = alloc_from_ws<T*>(ctx, ws, off, static_cast<size_t>(batch));
    auto Pre_buf = alloc_from_ws<T>(ctx, ws, off,
                                    static_cast<size_t>(kd_work) * static_cast<size_t>(m_max) * static_cast<size_t>(batch));
    auto Pre_ptrs = alloc_from_ws<T*>(ctx, ws, off, static_cast<size_t>(batch));
    auto Right_buf = alloc_from_ws<T>(ctx, ws, off,
                                      static_cast<size_t>(m_max) * static_cast<size_t>(kd_work) * static_cast<size_t>(batch));
    auto Right_ptrs = alloc_from_ws<T*>(ctx, ws, off, static_cast<size_t>(batch));
    auto tau_buf = alloc_from_ws<T>(ctx, ws, off, static_cast<size_t>(nb_max) * static_cast<size_t>(batch));

    MatrixView<T, MatrixFormat::Dense> Bmat(B_buf.data(), m_max, nb_max, m_max, m_max * nb_max, batch, B_ptrs.data());
    MatrixView<T, MatrixFormat::Dense> Symmat(Sym_buf.data(), m_max, m_max, m_max, m_max * m_max, batch, Sym_ptrs.data());
    MatrixView<T, MatrixFormat::Dense> Postmat(Post_buf.data(), m_max, m_max, m_max, m_max * m_max, batch, Post_ptrs.data());
    MatrixView<T, MatrixFormat::Dense> Premat(Pre_buf.data(), kd_work, m_max, kd_work, static_cast<int64_t>(kd_work) * m_max, batch, Pre_ptrs.data());
    MatrixView<T, MatrixFormat::Dense> Rightmat(Right_buf.data(), m_max, kd_work, m_max, static_cast<int64_t>(m_max) * kd_work, batch, Right_ptrs.data());

    // Ensure batched pointer arrays are initialized before any backend kernels
    // potentially consume them.
    (void)Bmat.data_ptrs(ctx);
    (void)Symmat.data_ptrs(ctx);
    (void)Postmat.data_ptrs(ctx);
    (void)Premat.data_ptrs(ctx);
    (void)Rightmat.data_ptrs(ctx);

    // Ensure batched pointer arrays are initialized before any backend kernels
    // potentially consume them.
    (void)Bmat.data_ptrs(ctx);
    (void)Symmat.data_ptrs(ctx);
    (void)Postmat.data_ptrs(ctx);
    (void)Premat.data_ptrs(ctx);
    (void)Rightmat.data_ptrs(ctx);

    auto ws_backend = ws.subspan(off);

    // Run up to max_steps chase steps using the same schedule as the full BANDR1 algorithm.
    int b = kd;
    int steps_done = 0;
    const int max_sweeps = (params.max_sweeps < 0) ? std::max(0, kd - 1) : std::max(0, params.max_sweeps);
    for (int sweep = 0; sweep < max_sweeps && b > 1 && steps_done < max_steps; ++sweep) {
        int step_in_sweep = 0;
        const int d_red = (d_per_sweep > 0) ? std::min(d_per_sweep, b - 1)
                                            : std::max(1, b - std::min(nb_target, b - 1));
        const int b_tilde = b - d_red;
        const int nb = std::min(std::max(1, nb_target), b_tilde);

        bool hit_step_limit = false;

        // Match the reference schedule (portable_banded_tridiagonal.py):
        // for each starting panel (j1_start), chase the bulge by updating (j1,j2,i1,i2)
        // along the diagonal chain.
        for (int j1_start = 0; j1_start < std::max(0, n - b_tilde - 1) && steps_done < max_steps; j1_start += nb) {
            int j1 = j1_start;
            int j2 = std::min(j1 + nb - 1, n - 1);
            int i1 = j1 + b_tilde;
            int i2 = std::min(j1 + b + nb - 1, n - 1);

            while (i1 < n && steps_done < max_steps) {
                if (i1 > i2) {
                    break;
                }

                const int m = i2 - i1 + 1;
                const int r = j2 - j1 + 1;
                if (m <= 0 || r <= 0) {
                    break;
                }

                bandr1_one_qr_step<B, T>(ctx, abw_out, kd_work, b, sweep, step_in_sweep, steps_done, i1, i2, j1, j2,
                                         Bmat, Symmat, Postmat, Premat, Rightmat,
                                         tau_buf, ws_backend);
                ++steps_done;
                ++step_in_sweep;

                if (steps_done >= max_steps) {
                    hit_step_limit = true;
                    break;
                }

                const int new_j1 = i1;
                const int new_j2 = std::min(new_j1 + nb - 1, n - 1);
                const int new_i1 = i1 + b;
                const int new_i2 = std::min(i2 + b, n - 1);
                j1 = new_j1;
                j2 = new_j2;
                i1 = new_i1;
                i2 = new_i2;
            }

            if (hit_step_limit) {
                break;
            }
        }

        // If we stopped mid-sweep due to max_steps, do NOT apply sweep-finalization
        // operations that are not similarity transforms (e.g. explicit band truncation).
        if (hit_step_limit) {
            break;
        }

        if (b_tilde + 1 <= kd_work) {
            band_zero_rows_from<T>(ctx, abw_out, b_tilde + 1);
        }
        b = b_tilde;
    }

    return ctx.get_event();
}

} // namespace

namespace {

template <Backend B, typename T>
size_t sytrd_band_reduction_bandr1_buffer_size_core(Queue& ctx,
                                                    const MatrixView<T, MatrixFormat::Dense>& ab_in,
                                                    const VectorView<typename base_type<T>::type>& d_out,
                                                    const VectorView<typename base_type<T>::type>& e_out,
                                                    const VectorView<T>& tau_out,
                                                    int32_t kd,
                                                    int32_t nb_target,
                                                    int32_t kd_work) {
    validate_band_reduction_dims(ab_in, d_out, e_out, tau_out, kd);

    const int n = ab_in.cols();
    const int batch = ab_in.batch_size();

    if (kd <= 1 || n <= 1) {
        return 0;
    }

    kd_work = std::min(kd_work, n - 1);

    const int nb_max = std::min(nb_target, std::max(1, kd - 1));
    const int m_max = kd;

    size_t bytes = 0;

    bytes += BumpAllocator::allocation_size<T>(ctx,
                                               static_cast<size_t>(kd_work + 1) * static_cast<size_t>(n) * static_cast<size_t>(batch));

    bytes += BumpAllocator::allocation_size<T>(ctx, static_cast<size_t>(m_max) * static_cast<size_t>(nb_max) * static_cast<size_t>(batch));
    bytes += BumpAllocator::allocation_size<T*>(ctx, static_cast<size_t>(batch));
    bytes += BumpAllocator::allocation_size<T>(ctx, static_cast<size_t>(m_max) * static_cast<size_t>(m_max) * static_cast<size_t>(batch));
    bytes += BumpAllocator::allocation_size<T*>(ctx, static_cast<size_t>(batch));
    bytes += BumpAllocator::allocation_size<T>(ctx, static_cast<size_t>(m_max) * static_cast<size_t>(m_max) * static_cast<size_t>(batch));
    bytes += BumpAllocator::allocation_size<T*>(ctx, static_cast<size_t>(batch));
    bytes += BumpAllocator::allocation_size<T>(ctx, static_cast<size_t>(kd_work) * static_cast<size_t>(m_max) * static_cast<size_t>(batch));
    bytes += BumpAllocator::allocation_size<T*>(ctx, static_cast<size_t>(batch));
    bytes += BumpAllocator::allocation_size<T>(ctx, static_cast<size_t>(m_max) * static_cast<size_t>(kd_work) * static_cast<size_t>(batch));
    bytes += BumpAllocator::allocation_size<T*>(ctx, static_cast<size_t>(batch));
    bytes += BumpAllocator::allocation_size<T>(ctx, static_cast<size_t>(nb_max) * static_cast<size_t>(batch));

    MatrixView<T, MatrixFormat::Dense> dummyB(nullptr, m_max, nb_max, m_max, m_max * nb_max, batch);
    Span<T> dummyTau(nullptr, static_cast<size_t>(nb_max) * static_cast<size_t>(batch));
    bytes += geqrf_buffer_size<B, T>(ctx, dummyB, dummyTau);

    MatrixView<T, MatrixFormat::Dense> dummySym(nullptr, m_max, m_max, m_max, m_max * m_max, batch);
    bytes += ormqr_buffer_size<B, T>(ctx, dummyB, dummySym, Side::Left, Transpose::ConjTrans, dummyTau);
    bytes += ormqr_buffer_size<B, T>(ctx, dummyB, dummySym, Side::Right, Transpose::NoTrans, dummyTau);
    bytes += ormqr_buffer_size<B, T>(ctx, dummyB, dummySym, Side::Right, Transpose::ConjTrans, dummyTau);

    MatrixView<T, MatrixFormat::Dense> dummyPre(nullptr, kd_work, m_max, kd_work, static_cast<int64_t>(kd_work) * m_max, batch);
    bytes += ormqr_buffer_size<B, T>(ctx, dummyB, dummyPre, Side::Right, Transpose::NoTrans, dummyTau);

    MatrixView<T, MatrixFormat::Dense> dummyRight(nullptr, m_max, kd_work, m_max, static_cast<int64_t>(m_max) * kd_work, batch);
    bytes += ormqr_buffer_size<B, T>(ctx, dummyB, dummyRight, Side::Left, Transpose::ConjTrans, dummyTau);

    return bytes;
}

template <Backend B, typename T>
Event sytrd_band_reduction_bandr1_core(Queue& ctx,
                                       const MatrixView<T, MatrixFormat::Dense>& ab_in,
                                       const VectorView<typename base_type<T>::type>& d_out,
                                       const VectorView<typename base_type<T>::type>& e_out,
                                       const VectorView<T>& tau_out,
                                       Uplo uplo,
                                       int32_t kd,
                                       const Span<std::byte>& ws,
                                       int32_t nb_target,
                                       int32_t kd_work,
                                       int32_t max_sweeps,
                                       int32_t d_per_sweep) {
    validate_band_reduction_dims(ab_in, d_out, e_out, tau_out, kd);

    if (!ctx.in_order()) {
        throw std::runtime_error("sytrd_band_reduction: requires an in-order Queue");
    }
    if (uplo != Uplo::Lower) {
        throw std::runtime_error("sytrd_band_reduction: only Uplo::Lower is implemented");
    }

    const int n = ab_in.cols();
    const int batch = ab_in.batch_size();

    // kd == 0: diagonal-only.
    if (n <= 0) {
        return ctx.get_event();
    }

    if (kd == 0) {
        auto AB = ab_in.kernel_view();
        auto dptr = d_out.data_ptr();
        const int d_inc = d_out.inc();
        const int d_stride = d_out.stride();
        auto eptr = e_out.data_ptr();
        const int e_inc = e_out.inc();
        const int e_stride = e_out.stride();
        auto tauptr = tau_out.data_ptr();
        const int tau_inc = tau_out.inc();
        const int tau_stride = tau_out.stride();

        ctx->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<2>(static_cast<size_t>(batch), static_cast<size_t>(n)), [=](sycl::id<2> idx) {
                const int b = static_cast<int>(idx[0]);
                const int j = static_cast<int>(idx[1]);
                dptr[b * d_stride + j * d_inc] = real_part(AB(0, j, b));
            });
        });

        if (n > 1) {
            ctx->submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<2>(static_cast<size_t>(batch), static_cast<size_t>(n - 1)), [=](sycl::id<2> idx) {
                    const int b = static_cast<int>(idx[0]);
                    const int j = static_cast<int>(idx[1]);
                    eptr[b * e_stride + j * e_inc] = typename base_type<T>::type(0);
                    tauptr[b * tau_stride + j * tau_inc] = T(0);
                });
            });
        }

        return ctx.get_event();
    }

    // kd == 1: already tridiagonal in band storage.
    if (kd == 1) {
        auto AB = ab_in.kernel_view();
        auto dptr = d_out.data_ptr();
        const int d_inc = d_out.inc();
        const int d_stride = d_out.stride();
        auto eptr = e_out.data_ptr();
        const int e_inc = e_out.inc();
        const int e_stride = e_out.stride();
        auto tauptr = tau_out.data_ptr();
        const int tau_inc = tau_out.inc();
        const int tau_stride = tau_out.stride();

        ctx->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<2>(static_cast<size_t>(batch), static_cast<size_t>(n)), [=](sycl::id<2> idx) {
                const int b = static_cast<int>(idx[0]);
                const int j = static_cast<int>(idx[1]);
                dptr[b * d_stride + j * d_inc] = real_part(AB(0, j, b));
            });
        });

        if (n > 1) {
            ctx->submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<2>(static_cast<size_t>(batch), static_cast<size_t>(n - 1)), [=](sycl::id<2> idx) {
                    const int b = static_cast<int>(idx[0]);
                    const int j = static_cast<int>(idx[1]);
                    eptr[b * e_stride + j * e_inc] = internal::abs(AB(1, j, b));
                    tauptr[b * tau_stride + j * tau_inc] = T(0);
                });
            });
        }

        return ctx.get_event();
    }

    kd_work = std::min(kd_work, n - 1);

    size_t off = 0;
    auto ABw_buf = alloc_from_ws<T>(ctx, ws, off,
                                    static_cast<size_t>(kd_work + 1) * static_cast<size_t>(n) * static_cast<size_t>(batch));
    MatrixView<T, MatrixFormat::Dense> ABw(ABw_buf.data(), kd_work + 1, n, kd_work + 1,
                                           (kd_work + 1) * n, batch);

    // Initialize ABw = 0; then copy input band into the top kd+1 rows.
    {
        auto ABW = ABw.kernel_view();
        ctx->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<3>(static_cast<size_t>(batch),
                                           static_cast<size_t>(kd_work + 1),
                                           static_cast<size_t>(n)),
                             [=](sycl::id<3> idx) {
                                 const int b = static_cast<int>(idx[0]);
                                 const int r = static_cast<int>(idx[1]);
                                 const int j = static_cast<int>(idx[2]);
                                 ABW(r, j, b) = T(0);
                             });
        });

        auto ABin = ab_in.kernel_view();
        ctx->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<3>(static_cast<size_t>(batch),
                                           static_cast<size_t>(kd + 1),
                                           static_cast<size_t>(n)),
                             [=](sycl::id<3> idx) {
                                 const int b = static_cast<int>(idx[0]);
                                 const int r = static_cast<int>(idx[1]);
                                 const int j = static_cast<int>(idx[2]);
                                 ABW(r, j, b) = ABin(r, j, b);
                             });
        });
    }

    // Scratch buffers sized to worst-case within a sweep.
    const int nb_max = std::min(nb_target, std::max(1, kd - 1));
    const int m_max = kd;
    auto B_buf = alloc_from_ws<T>(ctx, ws, off,
                                  static_cast<size_t>(m_max) * static_cast<size_t>(nb_max) * static_cast<size_t>(batch));
    auto B_ptrs = alloc_from_ws<T*>(ctx, ws, off, static_cast<size_t>(batch));
    auto Sym_buf = alloc_from_ws<T>(ctx, ws, off,
                                    static_cast<size_t>(m_max) * static_cast<size_t>(m_max) * static_cast<size_t>(batch));
    auto Sym_ptrs = alloc_from_ws<T*>(ctx, ws, off, static_cast<size_t>(batch));
    auto Post_buf = alloc_from_ws<T>(ctx, ws, off,
                                     static_cast<size_t>(m_max) * static_cast<size_t>(m_max) * static_cast<size_t>(batch));
    auto Post_ptrs = alloc_from_ws<T*>(ctx, ws, off, static_cast<size_t>(batch));
    auto Pre_buf = alloc_from_ws<T>(ctx, ws, off,
                                    static_cast<size_t>(kd_work) * static_cast<size_t>(m_max) * static_cast<size_t>(batch));
    auto Pre_ptrs = alloc_from_ws<T*>(ctx, ws, off, static_cast<size_t>(batch));
    auto Right_buf = alloc_from_ws<T>(ctx, ws, off,
                                      static_cast<size_t>(m_max) * static_cast<size_t>(kd_work) * static_cast<size_t>(batch));
    auto Right_ptrs = alloc_from_ws<T*>(ctx, ws, off, static_cast<size_t>(batch));
    auto tau_buf = alloc_from_ws<T>(ctx, ws, off, static_cast<size_t>(nb_max) * static_cast<size_t>(batch));

    MatrixView<T, MatrixFormat::Dense> Bmat(B_buf.data(), m_max, nb_max, m_max, m_max * nb_max, batch, B_ptrs.data());
    MatrixView<T, MatrixFormat::Dense> Symmat(Sym_buf.data(), m_max, m_max, m_max, m_max * m_max, batch, Sym_ptrs.data());
    MatrixView<T, MatrixFormat::Dense> Postmat(Post_buf.data(), m_max, m_max, m_max, m_max * m_max, batch, Post_ptrs.data());
    MatrixView<T, MatrixFormat::Dense> Premat(Pre_buf.data(), kd_work, m_max, kd_work, static_cast<int64_t>(kd_work) * m_max, batch, Pre_ptrs.data());
    MatrixView<T, MatrixFormat::Dense> Rightmat(Right_buf.data(), m_max, kd_work, m_max, static_cast<int64_t>(m_max) * kd_work, batch, Right_ptrs.data());

    // Ensure batched pointer arrays are initialized before any backend kernels
    // potentially consume them.
    (void)Bmat.data_ptrs(ctx);
    (void)Symmat.data_ptrs(ctx);
    (void)Postmat.data_ptrs(ctx);
    (void)Premat.data_ptrs(ctx);
    (void)Rightmat.data_ptrs(ctx);

    auto ws_backend = ws.subspan(off);

    int b = kd;
    int steps_done = 0;

    for (int sweep = 0; sweep < max_sweeps && b > 1; ++sweep) {
        int step_in_sweep = 0;
        // Scheduling constraints from the reference (bandr1):
        //   b_tilde = b - d, with 1 <= d < b
        //   1 <= nb <= b_tilde
        // This ensures the QR panel block B is strictly below the diagonal.
        const int d_red = (d_per_sweep > 0) ? std::min(d_per_sweep, b - 1)
                                            : std::max(1, b - std::min(nb_target, b - 1));
        const int b_tilde = b - d_red;
        const int nb = std::min(std::max(1, nb_target), b_tilde);

        for (int j1_start = 0; j1_start < std::max(0, n - b_tilde - 1); j1_start += nb) {
            int j1 = j1_start;
            int j2 = std::min(j1 + nb - 1, n - 1);
            int i1 = j1 + b_tilde;
            int i2 = std::min(j1 + b + nb - 1, n - 1);

            while (i1 < n) {
                if (i1 > i2) {
                    break;
                }

                const int m = i2 - i1 + 1;
                const int r = j2 - j1 + 1;
                if (m <= 0 || r <= 0) {
                    break;
                }

                bandr1_one_qr_step<B, T>(ctx, ABw, kd_work, b, sweep, step_in_sweep, steps_done, i1, i2, j1, j2,
                                         Bmat, Symmat, Postmat, Premat, Rightmat,
                                         tau_buf, ws_backend);

                ++steps_done;
                ++step_in_sweep;

                const int new_j1 = i1;
                const int new_j2 = std::min(new_j1 + nb - 1, n - 1);
                const int new_i1 = i1 + b;
                const int new_i2 = std::min(i2 + b, n - 1);
                j1 = new_j1;
                j2 = new_j2;
                i1 = new_i1;
                i2 = new_i2;
            }
        }

        if (b_tilde + 1 <= kd_work) {
            band_zero_rows_from<T>(ctx, ABw, b_tilde + 1);
        }

        b = b_tilde;
    }

    {
        auto AB = ABw.kernel_view();
        auto dptr = d_out.data_ptr();
        const int d_inc = d_out.inc();
        const int d_stride = d_out.stride();
        auto eptr = e_out.data_ptr();
        const int e_inc = e_out.inc();
        const int e_stride = e_out.stride();
        auto tauptr = tau_out.data_ptr();
        const int tau_inc = tau_out.inc();
        const int tau_stride = tau_out.stride();

        ctx->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range<2>(static_cast<size_t>(batch), static_cast<size_t>(n)),
                             [=](sycl::id<2> idx) {
                                 const int b = static_cast<int>(idx[0]);
                                 const int j = static_cast<int>(idx[1]);
                                 dptr[b * d_stride + j * d_inc] = real_part(AB(0, j, b));
                             });
        });

        if (n > 1) {
            ctx->submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range<2>(static_cast<size_t>(batch), static_cast<size_t>(n - 1)),
                                 [=](sycl::id<2> idx) {
                                     const int b = static_cast<int>(idx[0]);
                                     const int j = static_cast<int>(idx[1]);
                                     eptr[b * e_stride + j * e_inc] = internal::abs(AB(1, j, b));
                                     tauptr[b * tau_stride + j * tau_inc] = T(0);
                                 });
            });
        }
    }

    return ctx.get_event();
}

} // namespace

template <Backend B, typename T>
size_t sytrd_band_reduction_buffer_size(Queue& ctx,
                                        const MatrixView<T, MatrixFormat::Dense>& ab_in,
                                        const VectorView<typename base_type<T>::type>& d_out,
                                        const VectorView<typename base_type<T>::type>& e_out,
                                        const VectorView<T>& tau_out,
                                        Uplo uplo,
                                        int32_t kd,
                                        int32_t block_size) {
    SytrdBandReductionParams params;
    params.block_size = block_size;
    return sytrd_band_reduction_buffer_size<B, T>(ctx, ab_in, d_out, e_out, tau_out, uplo, kd, params);
}

template <Backend B, typename T>
size_t sytrd_band_reduction_buffer_size(Queue& ctx,
                                        const MatrixView<T, MatrixFormat::Dense>& ab_in,
                                        const VectorView<typename base_type<T>::type>& d_out,
                                        const VectorView<typename base_type<T>::type>& e_out,
                                        const VectorView<T>& tau_out,
                                        Uplo uplo,
                                        int32_t kd,
                                        SytrdBandReductionParams params) {
    validate_band_reduction_dims(ab_in, d_out, e_out, tau_out, kd);

    (void)uplo;
    const int n = ab_in.cols();
    int kd_work = params.kd_work;
    if (kd_work <= 0) {
        kd_work = std::min(n - 1, 3 * kd);
    }
    const int nb_target = std::max<int>(1, params.block_size);
    return sytrd_band_reduction_bandr1_buffer_size_core<B, T>(ctx, ab_in, d_out, e_out, tau_out, kd, nb_target, kd_work);
}

template <Backend B, typename T>
Event sytrd_band_reduction(Queue& ctx,
                           const MatrixView<T, MatrixFormat::Dense>& ab_in,
                           const VectorView<typename base_type<T>::type>& d_out,
                           const VectorView<typename base_type<T>::type>& e_out,
                           const VectorView<T>& tau_out,
                           Uplo uplo,
                           int32_t kd,
                           const Span<std::byte>& ws,
                           int32_t block_size) {
    SytrdBandReductionParams params;
    params.block_size = block_size;
    return sytrd_band_reduction<B, T>(ctx, ab_in, d_out, e_out, tau_out, uplo, kd, ws, params);
}

template <Backend B, typename T>
Event sytrd_band_reduction(Queue& ctx,
                           const MatrixView<T, MatrixFormat::Dense>& ab_in,
                           const VectorView<typename base_type<T>::type>& d_out,
                           const VectorView<typename base_type<T>::type>& e_out,
                           const VectorView<T>& tau_out,
                           Uplo uplo,
                           int32_t kd,
                           const Span<std::byte>& ws,
                           SytrdBandReductionParams params) {
    validate_band_reduction_dims(ab_in, d_out, e_out, tau_out, kd);

    const int n = ab_in.cols();
    const int nb_target = std::max<int>(1, params.block_size);
    int kd_work = params.kd_work;
    if (kd_work <= 0) {
        kd_work = std::min(n - 1, 3 * kd);
    }
    const int max_sweeps = (params.max_sweeps < 0) ? std::max(0, kd - 1) : std::max(0, params.max_sweeps);
    const int d_per_sweep = params.d;
    if (d_per_sweep < 0) {
        throw std::runtime_error("sytrd_band_reduction: params.d must be >= 0");
    }
    return sytrd_band_reduction_bandr1_core<B, T>(ctx, ab_in, d_out, e_out, tau_out, uplo, kd, ws,
                                                  nb_target, kd_work, max_sweeps, d_per_sweep);
}

template <Backend B, typename T>
Event sytrd_band_reduction_single_step(Queue& ctx,
                                       const MatrixView<T, MatrixFormat::Dense>& ab_in,
                                       const MatrixView<T, MatrixFormat::Dense>& abw_out,
                                       Uplo uplo,
                                       int32_t kd,
                                       const Span<std::byte>& ws,
                                       SytrdBandReductionParams params) {
    return sytrd_band_reduction_single_step_core<B, T>(ctx, ab_in, abw_out, uplo, kd, ws, params);
}

template <Backend B, typename T>
size_t sytrd_band_reduction_single_step_buffer_size(Queue& ctx,
                                                    const MatrixView<T, MatrixFormat::Dense>& ab_in,
                                                    const MatrixView<T, MatrixFormat::Dense>& abw_out,
                                                    Uplo uplo,
                                                    int32_t kd,
                                                    SytrdBandReductionParams params) {
    (void)uplo;
    if (ab_in.rows() != kd + 1) {
        throw std::runtime_error("sytrd_band_reduction_single_step_buffer_size: ab_in.rows() must equal kd+1");
    }
    const int n = ab_in.cols();
    const int batch = ab_in.batch_size();
    int kd_work = params.kd_work;
    if (kd_work <= 0) {
        kd_work = std::min(n - 1, 3 * kd);
    }
    kd_work = std::min(kd_work, n - 1);
    if (abw_out.rows() != kd_work + 1 || abw_out.cols() != n || abw_out.batch_size() != batch) {
        throw std::runtime_error("sytrd_band_reduction_single_step_buffer_size: abw_out must be (kd_work+1) x n with matching batch");
    }
    const int nb_target = std::max<int>(1, params.block_size);
    return sytrd_band_reduction_single_step_buffer_size_core<B, T>(ctx, kd, nb_target, kd_work, n, batch);
}

#define SYTRD_BAND_REDUCTION_INSTANTIATE(back, fp) \
    template Event sytrd_band_reduction<back, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const VectorView<typename base_type<fp>::type>&, \
        const VectorView<typename base_type<fp>::type>&, \
        const VectorView<fp>&, \
        Uplo, \
        int32_t, \
        const Span<std::byte>&, \
        int32_t); \
    template Event sytrd_band_reduction<back, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const VectorView<typename base_type<fp>::type>&, \
        const VectorView<typename base_type<fp>::type>&, \
        const VectorView<fp>&, \
        Uplo, \
        int32_t, \
        const Span<std::byte>&, \
        SytrdBandReductionParams); \
    template size_t sytrd_band_reduction_buffer_size<back, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const VectorView<typename base_type<fp>::type>&, \
        const VectorView<typename base_type<fp>::type>&, \
        const VectorView<fp>&, \
        Uplo, \
        int32_t, \
        int32_t);

#define SYTRD_BAND_REDUCTION_SINGLE_STEP_INSTANTIATE(back, fp) \
    template Event sytrd_band_reduction_single_step<back, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Uplo, \
        int32_t, \
        const Span<std::byte>&, \
        SytrdBandReductionParams); \
    template size_t sytrd_band_reduction_single_step_buffer_size<back, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        Uplo, \
        int32_t, \
        SytrdBandReductionParams);

#define SYTRD_BAND_REDUCTION_BUFFER_PARAMS_INSTANTIATE(back, fp) \
    template size_t sytrd_band_reduction_buffer_size<back, fp>( \
        Queue&, \
        const MatrixView<fp, MatrixFormat::Dense>&, \
        const VectorView<typename base_type<fp>::type>&, \
        const VectorView<typename base_type<fp>::type>&, \
        const VectorView<fp>&, \
        Uplo, \
        int32_t, \
        SytrdBandReductionParams);

#if BATCHLAS_HAS_CUDA_BACKEND
SYTRD_BAND_REDUCTION_INSTANTIATE(Backend::CUDA, float)
SYTRD_BAND_REDUCTION_BUFFER_PARAMS_INSTANTIATE(Backend::CUDA, float)
SYTRD_BAND_REDUCTION_SINGLE_STEP_INSTANTIATE(Backend::CUDA, float)
SYTRD_BAND_REDUCTION_INSTANTIATE(Backend::CUDA, double)
SYTRD_BAND_REDUCTION_BUFFER_PARAMS_INSTANTIATE(Backend::CUDA, double)
SYTRD_BAND_REDUCTION_SINGLE_STEP_INSTANTIATE(Backend::CUDA, double)
SYTRD_BAND_REDUCTION_INSTANTIATE(Backend::CUDA, std::complex<float>)
SYTRD_BAND_REDUCTION_BUFFER_PARAMS_INSTANTIATE(Backend::CUDA, std::complex<float>)
SYTRD_BAND_REDUCTION_SINGLE_STEP_INSTANTIATE(Backend::CUDA, std::complex<float>)
SYTRD_BAND_REDUCTION_INSTANTIATE(Backend::CUDA, std::complex<double>)
SYTRD_BAND_REDUCTION_BUFFER_PARAMS_INSTANTIATE(Backend::CUDA, std::complex<double>)
SYTRD_BAND_REDUCTION_SINGLE_STEP_INSTANTIATE(Backend::CUDA, std::complex<double>)
#endif

#if BATCHLAS_HAS_ROCM_BACKEND
SYTRD_BAND_REDUCTION_INSTANTIATE(Backend::ROCM, float)
SYTRD_BAND_REDUCTION_BUFFER_PARAMS_INSTANTIATE(Backend::ROCM, float)
SYTRD_BAND_REDUCTION_SINGLE_STEP_INSTANTIATE(Backend::ROCM, float)
SYTRD_BAND_REDUCTION_INSTANTIATE(Backend::ROCM, double)
SYTRD_BAND_REDUCTION_BUFFER_PARAMS_INSTANTIATE(Backend::ROCM, double)
SYTRD_BAND_REDUCTION_SINGLE_STEP_INSTANTIATE(Backend::ROCM, double)
SYTRD_BAND_REDUCTION_INSTANTIATE(Backend::ROCM, std::complex<float>)
SYTRD_BAND_REDUCTION_BUFFER_PARAMS_INSTANTIATE(Backend::ROCM, std::complex<float>)
SYTRD_BAND_REDUCTION_SINGLE_STEP_INSTANTIATE(Backend::ROCM, std::complex<float>)
SYTRD_BAND_REDUCTION_INSTANTIATE(Backend::ROCM, std::complex<double>)
SYTRD_BAND_REDUCTION_BUFFER_PARAMS_INSTANTIATE(Backend::ROCM, std::complex<double>)
SYTRD_BAND_REDUCTION_SINGLE_STEP_INSTANTIATE(Backend::ROCM, std::complex<double>)
#endif

#if BATCHLAS_HAS_HOST_BACKEND
SYTRD_BAND_REDUCTION_INSTANTIATE(Backend::NETLIB, float)
SYTRD_BAND_REDUCTION_BUFFER_PARAMS_INSTANTIATE(Backend::NETLIB, float)
SYTRD_BAND_REDUCTION_SINGLE_STEP_INSTANTIATE(Backend::NETLIB, float)
SYTRD_BAND_REDUCTION_INSTANTIATE(Backend::NETLIB, double)
SYTRD_BAND_REDUCTION_BUFFER_PARAMS_INSTANTIATE(Backend::NETLIB, double)
SYTRD_BAND_REDUCTION_SINGLE_STEP_INSTANTIATE(Backend::NETLIB, double)
SYTRD_BAND_REDUCTION_INSTANTIATE(Backend::NETLIB, std::complex<float>)
SYTRD_BAND_REDUCTION_BUFFER_PARAMS_INSTANTIATE(Backend::NETLIB, std::complex<float>)
SYTRD_BAND_REDUCTION_SINGLE_STEP_INSTANTIATE(Backend::NETLIB, std::complex<float>)
SYTRD_BAND_REDUCTION_INSTANTIATE(Backend::NETLIB, std::complex<double>)
SYTRD_BAND_REDUCTION_BUFFER_PARAMS_INSTANTIATE(Backend::NETLIB, std::complex<double>)
SYTRD_BAND_REDUCTION_SINGLE_STEP_INSTANTIATE(Backend::NETLIB, std::complex<double>)
#endif

#undef SYTRD_BAND_REDUCTION_INSTANTIATE
#undef SYTRD_BAND_REDUCTION_BUFFER_PARAMS_INSTANTIATE
#undef SYTRD_BAND_REDUCTION_SINGLE_STEP_INSTANTIATE

} // namespace batchlas
