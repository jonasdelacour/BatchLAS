#pragma once

// What a call throws when nothing in this build can serve it: no native kernel for
// the shape and no vendor library compiled in. SYCL-free, so the op enum can key
// installed headers too.

#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

#include <batchlas/blas/enums.hh>

namespace batchlas {

// The public ops that can lack an implementation in a build (one per
// include/batchlas/blas/functions/*.hh).
enum class Op : uint8_t {
    gemm, gemv, trsm, trmm, symm, hemm, syrk, herk, syr2k, her2k,
    potrf, getrf, getrs, getri, geqrf, orgqr, ormqr, syev, gesvd, spmm,
    gesv, posv,
    COUNT
};

enum class ScalarKind : uint8_t { F32, F64, C32, C64 };

template <typename T>
inline constexpr ScalarKind scalar_kind_of =
    std::is_same_v<T, float>                ? ScalarKind::F32 :
    std::is_same_v<T, double>               ? ScalarKind::F64 :
    std::is_same_v<T, std::complex<float>>  ? ScalarKind::C32 :
                                              ScalarKind::C64;

inline constexpr std::string_view to_string(ScalarKind s) {
    switch (s) {
        case ScalarKind::F32: return "float";
        case ScalarKind::F64: return "double";
        case ScalarKind::C32: return "complex<float>";
        case ScalarKind::C64: return "complex<double>";
    }
    return "?";
}

inline constexpr std::string_view op_name(Op o) {
    switch (o) {
        case Op::gemm:  return "gemm";   case Op::gemv:  return "gemv";
        case Op::trsm:  return "trsm";   case Op::trmm:  return "trmm";
        case Op::symm:  return "symm";   case Op::hemm:  return "hemm";
        case Op::syrk:  return "syrk";   case Op::herk:  return "herk";
        case Op::syr2k: return "syr2k";  case Op::her2k: return "her2k";
        case Op::potrf: return "potrf";  case Op::getrf: return "getrf";
        case Op::getrs: return "getrs";  case Op::getri: return "getri";
        case Op::geqrf: return "geqrf";  case Op::orgqr: return "orgqr";
        case Op::ormqr: return "ormqr";  case Op::syev:  return "syev";
        case Op::gesvd: return "gesvd";  case Op::spmm:  return "spmm";
        case Op::gesv:  return "gesv";   case Op::posv:  return "posv";
        case Op::COUNT: return "?";
    }
    return "?";
}

// The usual cause is a deliberate -DBATCHLAS_ENABLE_VENDOR_BLAS=OFF, so the message
// says which build switch brings an implementation back.
class NoRouteError : public std::runtime_error {
public:
    NoRouteError(Op op, Backend backend, ScalarKind scalar, std::string detail)
        : std::runtime_error(build_message(op, scalar, detail)),
          op_(op), backend_(backend), scalar_(scalar) {}

    Op op() const { return op_; }
    Backend backend() const { return backend_; }
    ScalarKind scalar() const { return scalar_; }

private:
    static std::string build_message(Op op, ScalarKind scalar, const std::string& detail) {
        std::string m = "BatchLAS: no route for ";
        m += op_name(op);
        m += "<";
        m += to_string(scalar);
        m += "> on this backend";
        if (!detail.empty()) {
            m += " (" + detail + ")";
        }
        m += ".\n";
        m += "  This build has no vendor library for that op, and BatchLAS has no\n"
             "  native kernel for it yet. If you configured with\n"
             "  -DBATCHLAS_ENABLE_VENDOR_BLAS=OFF, re-enabling it restores this op;\n"
             "  otherwise the vendor library was not found at configure time.";
        return m;
    }

    Op op_;
    Backend backend_;
    ScalarKind scalar_;
};

} // namespace batchlas
