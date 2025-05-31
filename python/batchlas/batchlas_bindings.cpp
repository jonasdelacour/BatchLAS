#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <batchlas.hh>
#include <blas/enums.hh>
#include <blas/matrix_handle_new.hh>
#include <blas/extensions_new.hh>
#include <blas/functions_matrixview.hh>
#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>
#include <util/sycl-vector.hh>
#include <vector>
#include <memory>
#include <cstring> // for memcpy

namespace py = pybind11;

// Class to hold GPU memory for a matrix
template <typename T>
class GpuMatrix {
public:
    GpuMatrix(Queue& queue, int64_t batch_size, int64_t rows, int64_t cols) 
        : batch_size_(batch_size), rows_(rows), cols_(cols), 
          data_(batch_size * rows * cols), queue_(queue) {}
    
    // Copy data from host to device
    void copy_from_host(py::array_t<T> array) {
        auto buf = array.request();
        T* host_ptr = static_cast<T*>(buf.ptr);

        std::memcpy(data_.data(), host_ptr, batch_size_ * rows_ * cols_ * sizeof(T));
    }
    
    // Copy data from device to host
    void copy_to_host(py::array_t<T> array) {
        auto buf = array.request();
        T* host_ptr = static_cast<T*>(buf.ptr);
        
        // Since UnifiedVector uses USM, we can directly copy from it using memcpy
        std::memcpy(host_ptr, data_.data(), batch_size_ * rows_ * cols_ * sizeof(T));
    }
    
    // Get a view of the matrix for use with BatchLAS functions
    batchlas::MatrixView<T, batchlas::MatrixFormat::Dense> get_view() {
        // Create a handle first
        int ld = rows_;  // leading dimension (row stride for column-major)
        int stride = rows_ * cols_;  // batch stride
        
        // Create a vector of pointers to each batch
        data_ptrs_.resize(batch_size_);
        for (int i = 0; i < batch_size_; ++i) {
            data_ptrs_[i] = data_.data() + i * stride;
        }
        
        // Create Span for the pointers
        span_ptrs_ = Span<T*>(data_ptrs_.data(), batch_size_);
        
        // Create the matrix view
        return batchlas::MatrixView<T, batchlas::MatrixFormat::Dense>(
            data_.data(), rows_, cols_, ld, stride, batch_size_, span_ptrs_.data());
    }
    
private:
    int64_t batch_size_;
    int64_t rows_;
    int64_t cols_;
    UnifiedVector<T> data_;
    Queue& queue_;
    UnifiedVector<T*> data_ptrs_;
    Span<T*> span_ptrs_;
};

// Class to hold GPU memory for a vector
template <typename T>
class GpuVector {
public:
    GpuVector(Queue& queue, int64_t batch_size, int64_t length) 
        : batch_size_(batch_size), length_(length), 
          data_(batch_size * length), queue_(queue) {}
    
    // Copy data from host to device
    void copy_from_host(py::array_t<T, py::array::c_style> array) {
        auto buf = array.request();
        T* host_ptr = static_cast<T*>(buf.ptr);
        
        // Since UnifiedVector uses USM, we can directly copy to it using memcpy
        std::memcpy(data_.data(), host_ptr, batch_size_ * length_ * sizeof(T));
    }
    
    // Copy data from device to host
    void copy_to_host(py::array_t<T, py::array::c_style> array) {
        auto buf = array.request();
        T* host_ptr = static_cast<T*>(buf.ptr);
        
        // Since UnifiedVector uses USM, we can directly copy from it using memcpy
        std::memcpy(host_ptr, data_.data(), batch_size_ * length_ * sizeof(T));
    }
    
    // Get a handle of the vector for use with BatchLAS functions
    batchlas::VectorView<T> get_handle() {
        int incx = 1;  // Increment within each vector
        int stride = length_;  // Stride between batch elements
        
        // Create the vector handle
        return batchlas::VectorView<T>(
            data_.data(), length_, incx, stride, batch_size_);
    }
    
private:
    int64_t batch_size_;
    int64_t length_;
    UnifiedVector<T> data_;
    Queue& queue_;
};

PYBIND11_MODULE(batchlas, m) {
    m.doc() = "Python bindings for BatchLAS library";
    
    // Define enums
    py::enum_<batchlas::Backend>(m, "Backend")
        .value("AUTO", batchlas::Backend::AUTO)
        .value("CUDA", batchlas::Backend::CUDA)
        .value("ROCM", batchlas::Backend::ROCM)
        .value("MKL", batchlas::Backend::MKL)
        .value("MAGMA", batchlas::Backend::MAGMA)
        .value("SYCL", batchlas::Backend::SYCL)
        .value("NETLIB", batchlas::Backend::NETLIB)
        .export_values();
        
    py::enum_<batchlas::Transpose>(m, "Transpose")
        .value("NoTrans", batchlas::Transpose::NoTrans)
        .value("Trans", batchlas::Transpose::Trans)
        .export_values();
        
    py::enum_<batchlas::Side>(m, "Side")
        .value("Left", batchlas::Side::Left)
        .value("Right", batchlas::Side::Right)
        .export_values();
        
    py::enum_<batchlas::Uplo>(m, "Uplo")
        .value("Upper", batchlas::Uplo::Upper)
        .value("Lower", batchlas::Uplo::Lower)
        .export_values();
        
    py::enum_<batchlas::Diag>(m, "Diag")
        .value("NonUnit", batchlas::Diag::NonUnit)
        .value("Unit", batchlas::Diag::Unit)
        .export_values();
        
    py::enum_<batchlas::JobType>(m, "JobType")
        .value("EigenVectors", batchlas::JobType::EigenVectors)
        .value("NoEigenVectors", batchlas::JobType::NoEigenVectors)
        .export_values();
        
    py::enum_<batchlas::ComputePrecision>(m, "ComputePrecision")
        .value("Default", batchlas::ComputePrecision::Default)
        .value("F32", batchlas::ComputePrecision::F32)
        .value("F64", batchlas::ComputePrecision::F64)
        .value("F16", batchlas::ComputePrecision::F16)
        .value("BF16", batchlas::ComputePrecision::BF16)
        .value("TF32", batchlas::ComputePrecision::TF32)
        .export_values();
    
    py::enum_<batchlas::OrthoAlgorithm>(m, "OrthoAlgorithm")
        .value("CGS2", batchlas::OrthoAlgorithm::CGS2)
        .value("Chol2", batchlas::OrthoAlgorithm::Chol2)
        .value("Cholesky", batchlas::OrthoAlgorithm::Cholesky)
        .value("ShiftChol3", batchlas::OrthoAlgorithm::ShiftChol3)
        .value("Householder", batchlas::OrthoAlgorithm::Householder)
        .value("SVQB", batchlas::OrthoAlgorithm::SVQB)
        .export_values();
    
    // Create Device and Queue classes
    py::enum_<DeviceType>(m, "DeviceType")
        .value("CPU", DeviceType::CPU)
        .value("GPU", DeviceType::GPU)
        .value("ACCELERATOR", DeviceType::ACCELERATOR)
        .value("HOST", DeviceType::HOST)
        .export_values();
        
    py::class_<Device>(m, "Device")
        .def(py::init<>(), "Create a default device")
        .def(py::init<size_t, DeviceType>(), "Create a device with index and type")
        .def(py::init<std::string>(), "Create a device from a string description (cpu, gpu, or accelerator)")
        .def_static("default_device", &Device::default_device, "Get the default device")
        .def("get_name", &Device::get_name, "Get the device name")
        .def("get_vendor", &Device::get_vendor, "Get the device vendor");
        
    py::class_<Queue>(m, "Queue")
        .def(py::init<>(), "Create a default device queue using automatic device selection")
        .def(py::init<Device>(), "Create a device queue with specific device")
        .def(py::init<Device, bool>(), "Create a device queue with specific device and in-order execution flag");
    
    // Add gemm function
    m.def("gemm", [](Queue& queue, 
                    py::array_t<float> a, 
                    py::array_t<float> b, 
                    py::array_t<float> c,
                    float alpha, 
                    float beta,
                    batchlas::Transpose transA,
                    batchlas::Transpose transB,
                    int64_t batch_size,
                    batchlas::Backend backend) {
        
        // Get array dimensions
        auto buf_a = a.request();
        auto buf_b = b.request();
        auto buf_c = c.request();
        
        // Determine matrix dimensions based on transpose flags
        int64_t m, n, k;
        int64_t rows_a, cols_a, rows_b, cols_b, rows_c, cols_c;
        
        rows_a = buf_a.shape[1];
        cols_a = buf_a.shape[2];
        rows_b = buf_b.shape[1];
        cols_b = buf_b.shape[2];
        rows_c = buf_c.shape[1];
        cols_c = buf_c.shape[2];
        
        if (transA == batchlas::Transpose::NoTrans) {
            m = rows_a;
            k = cols_a;
        } else {
            m = cols_a;
            k = rows_a;
        }
        
        if (transB == batchlas::Transpose::NoTrans) {
            n = cols_b;
            // Verify inner dimensions match
            if (k != rows_b) {
                throw std::runtime_error("Matrix dimensions do not match for gemm operation");
            }
        } else {
            n = rows_b;
            // Verify inner dimensions match
            if (k != cols_b) {
                throw std::runtime_error("Matrix dimensions do not match for gemm operation");
            }
        }
        
        // Verify output matrix dimensions
        if (rows_c != m || cols_c != n) {
            throw std::runtime_error("Output matrix dimensions do not match expected result dimensions");
        }
        
        // Create GPU matrix objects and copy data to the device
        GpuMatrix<float> gpu_a(queue, batch_size, rows_a, cols_a);
        GpuMatrix<float> gpu_b(queue, batch_size, rows_b, cols_b);
        GpuMatrix<float> gpu_c(queue, batch_size, rows_c, cols_c);
        
        gpu_a.copy_from_host(a);
        gpu_b.copy_from_host(b);
        gpu_c.copy_from_host(c);
        
        // Get the views for BatchLAS functions
        auto mat_a = gpu_a.get_view();
        auto mat_b = gpu_b.get_view();
        auto mat_c = gpu_c.get_view();
        
        // Call the C++ function with the specified backend
        if (backend == batchlas::Backend::AUTO) {
            batchlas::gemm<batchlas::Backend::CUDA>(queue, mat_a, mat_b, mat_c, alpha, beta, transA, transB);
        }
        // Or explicitly with the CUDA backend when requested
        else if (backend == batchlas::Backend::CUDA) {
            batchlas::gemm<batchlas::Backend::CUDA>(queue, mat_a, mat_b, mat_c, alpha, beta, transA, transB);
        }
        // Add other backends as needed
        else {
            throw std::runtime_error("Unsupported backend specified");
        }
        // Wait for the operation to complete
        queue.wait();
        
        // Copy the result back to host
        gpu_c.copy_to_host(c);
        
    }, "General matrix multiplication: C = alpha*op(A)*op(B) + beta*C",
       py::arg("queue"), py::arg("a"), py::arg("b"), py::arg("c"), 
       py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
       py::arg("transA") = batchlas::Transpose::NoTrans, 
       py::arg("transB") = batchlas::Transpose::NoTrans,
       py::arg("batch_size") = 1,
       py::arg("backend") = batchlas::Backend::AUTO);
       
    // Add gemv function
    m.def("gemv", [](Queue& queue, 
                    py::array_t<float> a, 
                    py::array_t<float> x, 
                    py::array_t<float> y,
                    float alpha, 
                    float beta,
                    batchlas::Transpose transA,
                    int64_t batch_size,
                    batchlas::Backend backend) {
        
        // Get array dimensions
        auto buf_a = a.request();
        auto buf_x = x.request();
        auto buf_y = y.request();
        
        int64_t rows_a = buf_a.shape[1];
        int64_t cols_a = buf_a.shape[2];
        int64_t len_x = buf_x.shape[1];
        int64_t len_y = buf_y.shape[1];
        
        // Check dimensions
        if (transA == batchlas::Transpose::NoTrans) {
            if (cols_a != len_x || rows_a != len_y) {
                throw std::runtime_error("Matrix/vector dimensions don't match for gemv operation");
            }
        } else {
            if (rows_a != len_x || cols_a != len_y) {
                throw std::runtime_error("Matrix/vector dimensions don't match for gemv operation");
            }
        }
        
        // Create GPU objects and copy data to the device
        GpuMatrix<float> gpu_a(queue, batch_size, rows_a, cols_a);
        GpuVector<float> gpu_x(queue, batch_size, len_x);
        GpuVector<float> gpu_y(queue, batch_size, len_y);
        
        gpu_a.copy_from_host(a);
        gpu_x.copy_from_host(x);
        gpu_y.copy_from_host(y);
        
        // Get the views/handles for BatchLAS functions
        auto mat_a = gpu_a.get_view();
        auto vec_x = gpu_x.get_handle();
        auto vec_y = gpu_y.get_handle();
        
        // Call the C++ function with the specified backend
        if (backend == batchlas::Backend::AUTO) {
            batchlas::gemv<batchlas::Backend::CUDA>(queue, mat_a, vec_x, vec_y, alpha, beta, transA);
        }
        // Or explicitly with the CUDA backend when requested
        else if (backend == batchlas::Backend::CUDA) {
            batchlas::gemv<batchlas::Backend::CUDA>(queue, mat_a, vec_x, vec_y, alpha, beta, transA);
        }
        // Add other backends as needed
        else {
            throw std::runtime_error("Unsupported backend specified");
        }
        // Wait for the operation to complete
        queue.wait();
        
        // Copy the result back to host
        gpu_y.copy_to_host(y);
        
    }, "General matrix-vector multiplication: y = alpha*op(A)*x + beta*y",
       py::arg("queue"), py::arg("a"), py::arg("x"), py::arg("y"), 
       py::arg("alpha") = 1.0f, py::arg("beta") = 0.0f,
       py::arg("transA") = batchlas::Transpose::NoTrans,
       py::arg("batch_size") = 1,
       py::arg("backend") = batchlas::Backend::AUTO);
       
    // Add ortho function for orthonormalization
    m.def("ortho", [](Queue& queue,
                       py::array_t<float> a,
                       batchlas::OrthoAlgorithm algo,
                       batchlas::Backend backend) {
        
        // Get array dimensions
        auto buf_a = a.request();
        auto batch_size = buf_a.ndim > 2 ? buf_a.shape[2] : 1;
        // Get the shape of the first matrix in the batch to determine dimensions
        int64_t rows_a = buf_a.shape[0];
        int64_t cols_a = buf_a.shape[1];
        
        // Create GPU matrix object and copy data to the device
        GpuMatrix<float> gpu_a(queue, batch_size, rows_a, cols_a);

        gpu_a.copy_from_host(a);
        
        // Get the view for BatchLAS function
        auto mat_a = gpu_a.get_view();
        
        // Create a workspace buffer - assume 1MB is enough for now
        // In a production environment, you might want to query the required size
        size_t workspace_size = batchlas::ortho_buffer_size<batchlas::Backend::CUDA>(queue, mat_a, batchlas::Transpose::NoTrans, algo);

        UnifiedVector<std::byte> workspace(workspace_size);
        Span<std::byte> workspace_span(workspace.data(), workspace_size);
        
        // Call the C++ function with CUDA backend
        if (backend == batchlas::Backend::AUTO || backend == batchlas::Backend::CUDA) {
            batchlas::ortho<batchlas::Backend::CUDA>(
                queue, 
                mat_a, 
                batchlas::Transpose::NoTrans, 
                workspace_span,
                algo);
        } else {
            throw std::runtime_error("Only CUDA backend is currently supported for ortho");
        }
        
        // Wait for the operation to complete
        queue.wait();
        
        // Copy the result back to host
        gpu_a.copy_to_host(a);
        
    }, "Orthonormalize matrix columns using specified algorithm",
       py::arg("queue"), py::arg("a"),
       py::arg("algo") = batchlas::OrthoAlgorithm::Chol2,
       py::arg("backend") = batchlas::Backend::AUTO);
       
    // Add syevx function for eigenvalue computation
    m.def("syevx", [](Queue& queue,
                       py::array_t<float> a,
                       py::array_t<float> w,
                       py::array_t<float> z,
                       int64_t batch_size,
                       batchlas::JobType jobz,
                       batchlas::Backend backend) {
        
        // Get array dimensions
        auto buf_a = a.request();
        auto buf_w = w.request();
        auto buf_z = z.request();
        
        int64_t rows_a = buf_a.shape[1];
        int64_t cols_a = buf_a.shape[2];
        int64_t len_w = buf_w.shape[1];
        int64_t rows_z = buf_z.shape[1];
        int64_t cols_z = buf_z.shape[2];
        
        // Check dimensions
        if (rows_a != cols_a) {
            throw std::runtime_error("Input matrix must be square for syevx");
        }
        
        int64_t n = rows_a;
        int64_t neigs = len_w; // Number of eigenvalues to compute
        
        // Check output dimensions
        if (jobz == batchlas::JobType::EigenVectors && (rows_z < n || cols_z < neigs)) {
            throw std::runtime_error("Eigenvector matrix dimensions too small");
        }
        
        // Create GPU objects and copy data to the device
        GpuMatrix<float> gpu_a(queue, batch_size, rows_a, cols_a);
        GpuVector<float> gpu_w(queue, batch_size, len_w);
        GpuMatrix<float> gpu_z(queue, batch_size, rows_z, cols_z);
        
        gpu_a.copy_from_host(a);
        gpu_w.copy_from_host(w);
        
        // Only copy the eigenvector matrix if we're computing eigenvectors
        if (jobz == batchlas::JobType::EigenVectors) {
            gpu_z.copy_from_host(z);
        }
        
        // Get the views/handles for BatchLAS functions
        auto mat_a = gpu_a.get_view();
        auto vec_w = gpu_w.get_handle();
        auto mat_z = gpu_z.get_view();
        
        // Create a workspace buffer - assume 2MB is enough for now
        // In a production environment, you might want to query the required size
        size_t workspace_size = 2 * 1024 * 1024;
        UnifiedVector<std::byte> workspace(workspace_size);
        Span<std::byte> workspace_span(workspace.data(), workspace_size);
        
        // Create default parameters
        batchlas::SyevxParams<float> params;
        
        // Call the C++ function with CUDA backend
        if (backend == batchlas::Backend::AUTO || backend == batchlas::Backend::CUDA) {
            /* batchlas::syevx<batchlas::Backend::CUDA>(
                queue, 
                mat_a, 
                Span<float>(vec_w.data_, neigs * batch_size),
                neigs,
                workspace_span,
                jobz,
                mat_z); */
        } else {
            throw std::runtime_error("Only CUDA backend is currently supported for syevx");
        }
        
        // Wait for the operation to complete
        queue.wait();
        
        // Copy the results back to host
        gpu_w.copy_to_host(w);
        if (jobz == batchlas::JobType::EigenVectors) {
            gpu_z.copy_to_host(z);
        }
        
    }, "Compute eigenvalues and optionally eigenvectors of symmetric matrix",
       py::arg("queue"), py::arg("a"), py::arg("w"), py::arg("z"),
       py::arg("batch_size") = 1,
       py::arg("jobz") = batchlas::JobType::EigenVectors,
       py::arg("backend") = batchlas::Backend::AUTO);
}