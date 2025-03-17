#pragma once
#include <complex>
#include <memory>
#include <array>
#include <util/sycl-device-queue.hh>
#include <util/sycl-span.hh>
#include <util/sycl-vector.hh>
#include "enums.hh"

namespace batchlas {
    namespace detail {
        // Tag types for overload resolution
        template<Backend B> struct backend_tag {};
        struct fallback_tag : backend_tag<Backend::AUTO> {};
    }

    struct BackendSelector {
        // Count available backends at compile time
        static constexpr size_t num_backends = 1   // NETLIB is always available
            #ifdef USE_CUDA
                + 1
            #endif
            #ifdef USE_ROCM
                + 1
            #endif
            #ifdef USE_MKL
                + 1
            #endif
            #ifdef USE_MAGMA
                + 1
            #endif
            #ifdef USE_SYCL
                + 1
            #endif
        ;

        static constexpr std::array<Backend, num_backends> available_backends = {{
                #ifdef USE_CUDA
                    Backend::CUDA,
                #endif
                #ifdef USE_ROCM
                    Backend::ROCM,
                #endif
                #ifdef USE_MKL
                    Backend::MKL,
                #endif
                #ifdef USE_MAGMA
                    Backend::MAGMA,
                #endif
                #ifdef USE_SYCL
                    Backend::SYCL,
                #endif
                Backend::NETLIB  // Always available as fallback
        }};

        static constexpr Backend get(SyclQueue& ctx) {
            auto device = ctx.device();
            switch (device.type) {
                case DeviceType::CPU:
                    #ifdef USE_MKL
                        return Backend::MKL;
                    #else
                        return Backend::NETLIB;
                    #endif

                case DeviceType::GPU:
                    if (device.get_vendor() == "NVIDIA Corporation") {
                        #ifdef USE_CUDA
                            return Backend::CUDA;
                        #elif defined(USE_MAGMA)
                            return Backend::MAGMA;
                        #else
                            return Backend::SYCL;  // SYCL fallback for NVIDIA
                        #endif
                    }
                    else if (device.get_vendor() == "Advanced Micro Devices, Inc.") {
                        #ifdef USE_ROCM
                            return Backend::ROCM;
                        #else
                            return Backend::SYCL;  // SYCL fallback for AMD
                        #endif
                    }
                    return Backend::SYCL;  // SYCL fallback for other GPUs

                case DeviceType::ACCELERATOR:
                    return Backend::SYCL;  // SYCL is primary backend for accelerators

                case DeviceType::HOST:
                    return Backend::NETLIB;

                default:
                    return Backend::AUTO;
            }
        }

        template <typename... Args>
        static constexpr auto select(SyclQueue& ctx, Args&&... args) {
            constexpr size_t num_args = sizeof...(Args);
            static_assert(num_args > 0, "At least one argument must be provided");
            
            // Convert args to array for indexed access
            std::array<std::common_type_t<Args...>, num_args> implementations{std::forward<Args>(args)...};
            
            // Get the backend for this device
            Backend backend = get(ctx);
            
            // Find the implementation index based on backend priority
            size_t impl_index = 0;
            for(size_t i = 0; i < available_backends.size() && i < num_args; ++i) {
                if(available_backends[i] == backend) {
                    impl_index = i;
                    break;
                }
            }
            
            // If no matching backend found, use last argument as fallback
            if(impl_index >= num_args) {
                impl_index = num_args - 1;
            }
            
            return implementations[impl_index];
        }
    };

    //Forward declarations
    template <typename T, BatchType BT>
    struct DenseMatHandle;

    template <typename T, Format F, BatchType BT>
    struct SparseMatHandle;

    template <typename T>
    struct BackendDenseMatrixHandle;

    template <typename T, Format F>
    struct BackendSparseMatrixHandle;

    template <typename T>
    struct BackendDenseVectorHandle;

    template <typename T>
    struct BackendSparseVectorHandle;
    
    template <typename T, BatchType BT>
    struct DenseMatView;

    // Dense matrix handle for single matrix operations
    template <typename T>
    struct DenseMatHandle<T, BatchType::Single> {
        DenseMatHandle(T* data, int rows, int cols, int ld);
        ~DenseMatHandle();
        void init(SyclQueue& ctx);
        void init_backend();
        // Accessors...
        T* data_;
        int rows_, cols_, ld_;
        Layout layout_ = Layout::ColMajor; //Most backends don't support row-major dense matrices

        BackendDenseMatrixHandle<T>* operator->();
        BackendDenseMatrixHandle<T>& operator*();
        
        DenseMatView<T, BatchType::Single> operator()() const {
            return DenseMatView<T, BatchType::Single>(*this);
        }

        DenseMatView<T, BatchType::Single> operator()(int rows, int cols, int ld) const {
            return DenseMatView<T, BatchType::Single>(*this, rows, cols, ld);
        }

        private:
            std::unique_ptr<BackendDenseMatrixHandle<T>> backend_handle_;
    };

    // Dense matrix handle for batched operations
    template <typename T>
    struct DenseMatHandle<T, BatchType::Batched> {
        DenseMatHandle(T* data, int rows, int cols, int ld, int stride, int batch_size);
        ~DenseMatHandle();
        void init(SyclQueue& ctx);
        void init_backend();

        // Accessors...
        T* data_;
        SyclVector<T*> data_ptrs_;
        int rows_, cols_, ld_, stride_, batch_size_;
        Layout layout_ = Layout::ColMajor; //Most backends don't support row-major dense matrices

        BackendDenseMatrixHandle<T>* operator->();
        BackendDenseMatrixHandle<T>& operator*();

        DenseMatView<T, BatchType::Batched> operator()() const {
            return DenseMatView<T, BatchType::Batched>(*this);
        }

        DenseMatView<T, BatchType::Batched> operator()(int rows, int cols = -1, int ld = -1, int stride = -1) const {
            return DenseMatView<T, BatchType::Batched>(*this, rows, cols > 0 ? cols : cols_, ld > 0 ? ld : ld_, stride > 0 ? stride : stride_, batch_size_);
        }

        private:
            std::unique_ptr<BackendDenseMatrixHandle<T>> backend_handle_;
    };

    // Matrix view for batched operations
    template <typename T>
    struct DenseMatView<T, BatchType::Batched> {
        DenseMatView(T* data, int rows, int cols, int ld, int stride, int batch_size, Span<T*> data_ptrs);
        DenseMatView(const DenseMatView<T, BatchType::Batched>& view);
        DenseMatView(DenseMatView<T, BatchType::Batched>&& view);
        DenseMatView<T, BatchType::Batched>& operator=(const DenseMatView<T, BatchType::Batched>& view);
        DenseMatView<T, BatchType::Batched>& operator=(DenseMatView<T, BatchType::Batched>&& view);
        // Allow lvalue reference construction but explicitly delete rvalue reference constructor
        DenseMatView(const DenseMatHandle<T, BatchType::Batched>& handle);
        // Allow for the view to be a reinterpreted view of the matrices
        DenseMatView(const DenseMatHandle<T, BatchType::Batched>& handle, int rows, int cols, int ld, int stride, int batch_size);
        DenseMatView(DenseMatHandle<T, BatchType::Batched>&& handle) = delete;

        // Allow lvalue reference assignment but explicitly delete rvalue reference assignment
        DenseMatView<T, BatchType::Batched>& operator=(const DenseMatHandle<T, BatchType::Batched>& handle);
        DenseMatView<T, BatchType::Batched>& operator=(DenseMatHandle<T, BatchType::Batched>&& handle) = delete;
        
        ~DenseMatView();
        void init(SyclQueue& ctx);
        void init_backend();
        // Accessors...
        T* data_; 
        Span<T*> data_ptrs_;
        int rows_, cols_, ld_, stride_, batch_size_;
        Layout layout_ = Layout::ColMajor; //Most backends don't support row-major dense matrices

        BackendDenseMatrixHandle<T>* operator->();
        BackendDenseMatrixHandle<T>& operator*();

        private:
            std::shared_ptr<BackendDenseMatrixHandle<T>> backend_handle_;
    };
    
    // Matrix view for single matrix operations
    template <typename T>
    struct DenseMatView<T, BatchType::Single> {
        DenseMatView(T* data, int rows, int cols, int ld);
        DenseMatView(const DenseMatView<T, BatchType::Single>& view);
        DenseMatView(DenseMatView<T, BatchType::Single>&& view);
        DenseMatView<T, BatchType::Single>& operator=(const DenseMatView<T, BatchType::Single>& view);
        DenseMatView<T, BatchType::Single>& operator=(DenseMatView<T, BatchType::Single>&& view);
        
        //These handles are already non-owning, but for consistency, delete rvalue reference constructor
        DenseMatView(const DenseMatHandle<T, BatchType::Single>& handle);
        DenseMatView(const DenseMatHandle<T, BatchType::Single>& handle, int rows, int cols, int ld);
        DenseMatView(DenseMatHandle<T, BatchType::Single>&& handle) = delete;

        //These handles are already non-owning, but for consistency, delete rvalue reference assignment
        DenseMatView<T, BatchType::Single>& operator=(const DenseMatHandle<T, BatchType::Single>& handle);
        DenseMatView<T, BatchType::Single>& operator=(DenseMatHandle<T, BatchType::Single>&& handle) = delete;

        ~DenseMatView();
        void init(SyclQueue& ctx);
        void init_backend();

        // Accessors...
        T* data_;
        int rows_, cols_, ld_;
        Layout layout_ = Layout::ColMajor;

        BackendDenseMatrixHandle<T>* operator->();
        BackendDenseMatrixHandle<T>& operator*();

        private:
            std::shared_ptr<BackendDenseMatrixHandle<T>> backend_handle_;
    };

    template <typename T, BatchType BT>
    auto create_view(T* data, int rows, int cols, int ld, int stride, int batch_size, Span<T*> data_ptrs){
        if constexpr (BT == BatchType::Single) {
            return DenseMatView<T, BT>(data, rows, cols, ld);
        } else {
            return DenseMatView<T, BT>(data, rows, cols, ld, stride, batch_size, data_ptrs);
        }
    }

    // Deduction guides for DenseMatView
    template <typename T>
    DenseMatView(const DenseMatHandle<T, BatchType::Single>&) -> DenseMatView<T, BatchType::Single>;

    template <typename T>
    DenseMatView(const DenseMatHandle<T, BatchType::Batched>&) -> DenseMatView<T, BatchType::Batched>;

    // CSR sparse matrix handle for single matrix
    template <typename T>
    struct SparseMatHandle<T, Format::CSR, BatchType::Single> {
        /**
         * @brief Constructs a single CSR sparse matrix handle
         * @param data Array of non-zero values [nnz]
         * @param row_offsets Array of row offsets [rows + 1]
         * @param col_indices Array of column indices [nnz]
         * @param nnz Number of non-zero elements
         * @param rows Number of rows
         * @param cols Number of columns
         * @param layout Layout of the matrix
         */
        SparseMatHandle(T* data, int* row_offsets, int* col_indices, int nnz, int rows, int cols, Layout layout = Layout::RowMajor);
        ~SparseMatHandle();
        void init(SyclQueue& ctx);
        void init_backend();

        // Raw pointers to externally owned memory
        T* data_;              // [nnz] non-zero values
        int* row_offsets_;     // [rows + 1] row offsets
        int* col_indices_;     // [nnz] column indices
        int nnz_, rows_, cols_;
        Layout layout_ = Layout::RowMajor;

        BackendSparseMatrixHandle<T, Format::CSR>* operator->();
        BackendSparseMatrixHandle<T, Format::CSR>& operator*();

        private:
            std::unique_ptr<BackendSparseMatrixHandle<T, Format::CSR>> backend_handle_;
    };

    // CSR sparse matrix handle for batched operations
    template <typename T>
    struct SparseMatHandle<T, Format::CSR, BatchType::Batched> {
        /**
         * @brief Constructs a batched CSR sparse matrix handle
         * @param data Array of non-zero values [batch_size * stride]
         * @param row_offsets Array of row offsets [batch_size * (rows + 1)]
         * @param col_indices Array of column indices [batch_size * nnz]
         * @param nnz Number of non-zero elements per matrix
         * @param rows Number of rows per matrix
         * @param cols Number of columns per matrix
         * @param stride Stride between consecutive matrices in data array
         * @param batch_size Number of matrices in batch
         */
        SparseMatHandle(T* data, int* row_offsets, int* col_indices, 
            int nnz, int rows, int cols, int stride, int batch_size);

        ~SparseMatHandle();
        void init(SyclQueue& ctx);
        void init_backend();

        // Raw pointers to externally owned memory
        T* data_;              // [batch_size * stride] non-zero values
        int* row_offsets_;     // [batch_size * (rows + 1)] row offsets
        int* col_indices_;     // [batch_size * nnz] column indices
        
        int nnz_, rows_, cols_, stride_, batch_size_;
        Layout layout_ = Layout::RowMajor;

        BackendSparseMatrixHandle<T, Format::CSR>* operator->();
        BackendSparseMatrixHandle<T, Format::CSR>& operator*();

        private:
            std::unique_ptr<BackendSparseMatrixHandle<T, Format::CSR>> backend_handle_;
    };

    // Vector handles
    template <typename T, BatchType BT>
    struct DenseVecHandle;

    template <typename T, BatchType BT>
    struct SparseVecHandle;

    template <typename T>
    struct DenseVecHandle<T, BatchType::Single> {
        DenseVecHandle(T* data, int size, int ldc) : data_(data), size_(size), ldc_(ldc) {}
        // Accessors...
        T* data_;
        SyclVector<T*> data_ptrs_;
        int size_, ldc_;

        private:
            std::unique_ptr<BackendDenseVectorHandle<T>> backend_handle_;
    };

    template <typename T>
    struct DenseVecHandle<T, BatchType::Batched> {
        DenseVecHandle(T* data, int size, int ldc, int stride, int batch_size) 
            : data_(data), size_(size), ldc_(ldc), stride_(stride), batch_size_(batch_size) {}
        // Accessors...
        T* data_;
        int size_, ldc_, stride_, batch_size_;

        private:
            std::unique_ptr<BackendDenseVectorHandle<T>> backend_handle_;
    };

    template <typename T>
    struct SparseVecHandle<T, BatchType::Single> {
        SparseVecHandle(T* data, int* indices, int size) : data_(data), indices_(indices), size_(size) {}
        // Accessors...
        T* data_;
        int* indices_;
        int size_;

        private:
            std::unique_ptr<BackendDenseVectorHandle<T>> backend_handle_;
    };

    template <typename T>
    struct SparseVecHandle<T, BatchType::Batched> {
        SparseVecHandle(T* data, int* indices, int size, int stride, int batch_size) 
            : data_(data), indices_(indices), size_(size), stride_(stride), batch_size_(batch_size) {}
        // Accessors...
        T* data_;
        int* indices_;
        int size_, stride_, batch_size_;

        private:
            std::unique_ptr<BackendSparseVectorHandle<T>> backend_handle_;
    };

    // Utility functions
    //Uniform accessor for data
    template <template <typename, BatchType> class Handle, typename T, BatchType BT>
    auto get_data(Handle<T,BT>& handle) {
        return handle.data_;
    }
    
    template <template <typename, BatchType> class Handle, typename T, BatchType BT, std::enable_if_t<BT == BatchType::Batched, int> = 0>
    auto get_ptr_arr(SyclQueue& ctx, Handle<T,BT>& handle) {
        handle.init(ctx);
        return handle.data_ptrs_.data();
    }

    //Uniform accessor for data
    template <template <typename, Format, BatchType> class Handle, typename T, Format F, BatchType BT>
    auto get_data(Handle<T,F,BT>& handle) {
        return handle.data_;
    }

    template <template <typename, BatchType> class Handle, typename T, BatchType BT>
    auto get_batch_size(Handle<T,BT>& handle) {
        if constexpr (BT == BatchType::Batched) {
            return handle.batch_size_;
        } else {
            return 1;
        }
    }

    namespace detail {
        // Type trait to check for complex or floating point types
        template<typename T>
        struct is_complex_or_floating_point : 
            std::bool_constant<std::is_floating_point_v<T> || 
                            std::is_same_v<T, std::complex<float>> || 
                            std::is_same_v<T, std::complex<double>>> {};

        template<Backend B>
        [[noreturn]] void throw_unsupported() {
            throw std::runtime_error("Operation not supported for selected backend: " + std::to_string(static_cast<int>(B)));
        }

        template<typename T>
        using enable_if_scalar_t = typename std::enable_if<
            is_complex_or_floating_point<T>::value
        >::type;
    }
}