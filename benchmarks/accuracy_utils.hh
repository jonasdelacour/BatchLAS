// SPDX-License-Identifier: MIT
#pragma once

#include <batchlas/blas/functions.hh>
#include <batchlas/blas/linalg.hh>
#include <batchlas/error.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <src/queue.hh>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace batchlas::accuracy {

inline std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

inline SteqrShiftStrategy parse_shift_strategy(const std::string& value) {
    const auto key = to_lower(value);
    if (key == "lapack") return SteqrShiftStrategy::Lapack;
    if (key == "wilkinson") return SteqrShiftStrategy::Wilkinson;
    throw batchlas::invalid_argument("Invalid --cta-shift value (use lapack or wilkinson)");
}

inline constexpr std::uint64_t kFnvOffset = 1469598103934665603ull;

template <typename Word>
inline std::uint64_t fnv1a(std::uint64_t h, const Word& w) {
    unsigned char bytes[sizeof(Word)];
    std::memcpy(bytes, &w, sizeof(Word));
    for (unsigned char c : bytes) h = (h ^ c) * 1099511628211ull;
    return h;
}

inline std::string hex_hash(std::uint64_t h) {
    std::ostringstream s;
    s << std::hex << std::setw(16) << std::setfill('0') << h;
    return s.str();
}

// FNV-1a over the bit patterns of item b's rows x cols entries, column-major. Call after the
// producing queue has been waited on.
template <typename Real>
inline std::vector<std::string> item_hashes(const MatrixView<Real, MatrixFormat::Dense>& A) {
    std::vector<std::string> out(static_cast<std::size_t>(A.batch_size()));
    const Real* base = A.data_ptr();
    for (int b = 0; b < A.batch_size(); ++b) {
        std::uint64_t h = kFnvOffset;
        for (int j = 0; j < A.cols(); ++j) {
            for (int i = 0; i < A.rows(); ++i) {
                h = fnv1a(h, base[static_cast<std::size_t>(b) * A.stride() + static_cast<std::size_t>(j) * A.ld() + i]);
            }
        }
        out[static_cast<std::size_t>(b)] = hex_hash(h);
    }
    return out;
}

template <typename Real>
inline std::vector<std::string> item_hashes(const VectorView<Real>& v) {
    std::vector<std::string> out(static_cast<std::size_t>(v.batch_size()));
    const Real* base = v.data_ptr();
    for (int b = 0; b < v.batch_size(); ++b) {
        std::uint64_t h = kFnvOffset;
        for (int i = 0; i < v.size(); ++i) {
            h = fnv1a(h, base[static_cast<std::size_t>(b) * v.stride() + static_cast<std::size_t>(i) * v.inc()]);
        }
        out[static_cast<std::size_t>(b)] = hex_hash(h);
    }
    return out;
}

// The input is built on the device, so two SYCL implementations can round it differently:
// scripts/impl_diff.py pairs samples only where this matches.
template <typename Real>
inline std::vector<std::string> input_hashes(const MatrixView<Real, MatrixFormat::Dense>& A) {
    return item_hashes(A);
}

// One line per (impl, n, dtype) folding every row's w_hash and z_hash in sample order, so a base
// run and a branch run compare by a single line: "hash_total,impl,n,dtype,items,w,z".
struct RunHashes {
    struct Total { std::uint64_t w = kFnvOffset, z = kFnvOffset; std::size_t items = 0; };
    std::vector<std::pair<std::string, Total>> totals;

    void add(const std::string& key, const std::string& w_hash, const std::string& z_hash) {
        auto it = std::find_if(totals.begin(), totals.end(), [&](const auto& t) { return t.first == key; });
        if (it == totals.end()) it = totals.insert(totals.end(), {key, Total{}});
        for (char c : w_hash) it->second.w = fnv1a(it->second.w, c);
        for (char c : z_hash) it->second.z = fnv1a(it->second.z, c);
        ++it->second.items;
    }

    void print(std::ostream& os) const {
        for (const auto& [key, t] : totals)
            os << "hash_total," << key << "," << t.items << "," << hex_hash(t.w) << "," << hex_hash(t.z) << "\n";
    }
};

// --inputs DIR: the first run to reach a chunk writes DIR/<first sample>.bin and every later run
// reads it back over its own generated matrix, so two trees see the same bits.
template <typename Real>
inline void share_inputs(const std::string& dir, int first_sample, const MatrixView<Real, MatrixFormat::Dense>& A) {
    if (dir.empty()) return;
    std::filesystem::create_directories(dir);
    const auto path = std::filesystem::path(dir) / (std::to_string(first_sample) + ".bin");
    const bool load = std::filesystem::exists(path);
    std::fstream f(path, std::ios::binary | (load ? std::ios::in : std::ios::out));
    Real* base = A.data_ptr();
    const auto bytes = static_cast<std::streamsize>(sizeof(Real) * static_cast<std::size_t>(A.rows()));
    for (int b = 0; f && b < A.batch_size(); ++b) {
        for (int j = 0; f && j < A.cols(); ++j) {
            char* col = reinterpret_cast<char*>(base + static_cast<std::size_t>(b) * A.stride() +
                                                static_cast<std::size_t>(j) * A.ld());
            if (load) {
                f.read(col, bytes);
            } else {
                f.write(col, bytes);
            }
        }
    }
    if (!f) throw batchlas::error("share_inputs: cannot " + std::string(load ? "read " : "write ") + path.string());
}

template <typename Real>
inline void extract_tridiagonal(Queue& q,
                                const MatrixView<Real, MatrixFormat::Dense>& dense,
                                Vector<Real>& d,
                                Vector<Real>& e) {
    const int n = dense.rows();
    const int batch = dense.batch_size();
    auto a_view = dense.kernel_view();
    auto d_ptr = d.data_ptr();
    auto e_ptr = e.data_ptr();
    const int d_inc = d.inc();
    const int e_inc = e.inc();
    const int d_stride = d.stride();
    const int e_stride = e.stride();

    q->parallel_for(sycl::range<1>(static_cast<size_t>(batch * n)), [=](sycl::id<1> idx) {
        const int linear = static_cast<int>(idx[0]);
        const int b = linear / n;
        const int i = linear - b * n;
        d_ptr[b * d_stride + i * d_inc] = a_view(i, i, b);
        if (i < n - 1) {
            e_ptr[b * e_stride + i * e_inc] = a_view(i + 1, i, b);
        }
    });
    q.wait();
}

template <Backend B, typename Real>
inline UnifiedVector<typename base_type<Real>::type> orthogonality_residuals(
    Queue& q,
    const Matrix<Real, MatrixFormat::Dense>& vectors) {
    const int dimension = vectors.cols();
    const int batch = vectors.batch_size();
    auto gram_minus_i = Matrix<Real>::Identity(dimension, batch);
    (void)gemm<B, Real>(q,
                  vectors,
                  vectors,
                  gram_minus_i,
                  Real(1),
                  Real(-1),
                  Transpose::Trans,
                  Transpose::NoTrans);
    q.wait();
    return norm(q, gram_minus_i, NormType::Frobenius);
}

} // namespace batchlas::accuracy
