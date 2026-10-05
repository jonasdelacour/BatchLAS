// Cross-check probe (not committed): one Auto gesvd call for the cell in $PROBE_CELL,
// "dtype herm jobu jobvh m n batch", so the coverage `reached` row names what Auto chose.
#include <gtest/gtest.h>

#include <batchlas/blas/functions/gesvd.hh>
#include <batchlas/util/sycl-device-queue.hh>
#include <batchlas/util/sycl-vector.hh>
#include <batchlas/blas/matrix.hh>

#include <complex>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>

using namespace batchlas;

namespace {

SvdVectors job(const std::string& j) { return j == "N" ? SvdVectors::None : (j == "A" ? SvdVectors::All : SvdVectors::Thin); }

template <typename T>
void cell(const std::string& h, SvdVectors ju, SvdVectors jv, int m, int n, int batch) {
    Queue q(Device::default_device(), true);
    const int k = std::min(m, n);
    auto A = Matrix<T, MatrixFormat::Dense>::Random(m, n, h != "N", batch, 4242);
    Matrix<T, MatrixFormat::Dense> U(ju == SvdVectors::None ? 1 : m, ju == SvdVectors::All ? m : (ju == SvdVectors::Thin ? k : 1), batch);
    Matrix<T, MatrixFormat::Dense> Vh(jv == SvdVectors::All ? n : (jv == SvdVectors::Thin ? k : 1), jv == SvdVectors::None ? 1 : n, batch);
    UnifiedVector<typename base_type<T>::type> s(std::size_t(k) * batch);
    std::optional<Uplo> herm;
    if (h == "L") herm = Uplo::Lower;
    if (h == "U") herm = Uplo::Upper;
    try {
        const std::size_t bytes = herm ? gesvd_buffer_size<Backend::CUDA, T>(q, A.view(), s.to_span(), U.view(), Vh.view(), ju, jv, *herm)
                                       : gesvd_buffer_size<Backend::CUDA, T>(q, A.view(), s.to_span(), U.view(), Vh.view(), ju, jv);
        UnifiedVector<std::byte> ws(std::max<std::size_t>(bytes, 1));
        Span<std::byte> w(ws.data(), bytes);
        if (herm) (void)gesvd<Backend::CUDA, T>(q, A.view(), s.to_span(), U.view(), Vh.view(), ju, jv, *herm, w);
        else (void)gesvd<Backend::CUDA, T>(q, A.view(), s.to_span(), U.view(), Vh.view(), ju, jv, w);
        q.wait_and_throw();
        std::cout << "PROBE ok\n";
    } catch (const std::exception& e) {
        std::cout << "PROBE threw: " << e.what() << "\n";
    }
}

}  // namespace

TEST(GesvdProbe, Cell) {
    const char* env = std::getenv("PROBE_CELL");
    if (!env) GTEST_SKIP();
    std::istringstream in(env);
    std::string dt, h, ju, jv;
    int m, n, batch;
    in >> dt >> h >> ju >> jv >> m >> n >> batch;
    if (dt == "float") cell<float>(h, job(ju), job(jv), m, n, batch);
    else if (dt == "double") cell<double>(h, job(ju), job(jv), m, n, batch);
    else if (dt == "cfloat") cell<std::complex<float>>(h, job(ju), job(jv), m, n, batch);
    else cell<std::complex<double>>(h, job(ju), job(jv), m, n, batch);
}
