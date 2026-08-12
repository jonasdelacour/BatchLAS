// A faithful standalone replica of what BatchLAS ACTUALLY runs today for
// double / complex on the vendor-independent (custom SYCL) path.
//
// src/sycl/gemm_kernels.cc:
//     if constexpr (std::is_same_v<T, double>) {
//         return max_dim <= 32 ? KernelVariant::Direct : KernelVariant::Tiled16;
//     }
//     return max_dim <= 64 ? KernelVariant::Direct : KernelVariant::Tiled16;
//
// i.e. every non-float type falls off the register-kernel ladder entirely and
// lands on launch_tiled<T,16> -> launch_tiled_general, reproduced verbatim
// below from src/sycl/gemm/tiled_general.hh. This is the incumbent the
// candidates have to beat to be worth anything; cuBLAS is the separate, harder
// bar. Without this number the comparison has no floor.
//
// Reproduced exactly, including the three things that make it slow, because
// "correcting" them would measure a kernel that is not in the tree:
//   1. ONE accumulator per thread (T sum), so every MAC needs two shared loads.
//   2. local_id(2) maps to `col`, so adjacent lanes address C an ldc apart --
//      the scattered-epilogue defect, which only costs at beta != 0.
//   3. std::complex operator*, which emits __mulsc3/__muldc3 in device code.
//
// Build:
//   /opt/dpcpp-cuda/bin/clang++ -O3 -std=c++20 -fsycl \
//       -fsycl-targets=nvidia_gpu_sm_89 --cuda-path=/usr/local/cuda-13.2 \
//       -Xcuda-ptxas -v tiled16_incumbent.cpp -o tiled16_incumbent
#include <sycl/sycl.hpp>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include <random>
#include <cmath>

template <typename T, int Tile> class GemmTiledGeneralKernel;

template <typename T, int Tile>
sycl::event launch_tiled_general(sycl::queue& q, int m, int n, int k, int batch,
                                 const T* a_ptr, int lda, int stride_a,
                                 const T* b_ptr, int ldb, int stride_b,
                                 T* c_ptr, int ldc, int stride_c,
                                 T alpha, T beta) {
    const sycl::range<3> local(1, Tile, Tile);
    const sycl::range<3> global((size_t)batch,
                                (size_t)(((m + Tile - 1) / Tile) * Tile),
                                (size_t)(((n + Tile - 1) / Tile) * Tile));
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<T, 1> tile_a(sycl::range<1>(Tile * Tile), h);
        sycl::local_accessor<T, 1> tile_b(sycl::range<1>(Tile * Tile), h);
        h.parallel_for<GemmTiledGeneralKernel<T, Tile>>(
            sycl::nd_range<3>(global, local), [=](sycl::nd_item<3> item) {
            const int bid = (int)item.get_group(0);
            const int local_row = (int)item.get_local_id(1);
            const int local_col = (int)item.get_local_id(2);
            const int row = (int)(item.get_group(1) * Tile + local_row);
            const int col = (int)(item.get_group(2) * Tile + local_col);
            if (bid >= batch) return;
            const int batch_a = bid * stride_a;
            const int batch_b = bid * stride_b;
            const int batch_c = bid * stride_c;
            T sum = T(0);
            for (int kk0 = 0; kk0 < k; kk0 += Tile) {
                const int tile_col = kk0 + local_col;
                const int tile_row = kk0 + local_row;
                // NoTrans/NoTrans column-major accessor: A(row, tile_col)
                tile_a[local_row * Tile + local_col] = (row < m && tile_col < k)
                    ? a_ptr[batch_a + (size_t)tile_col * lda + row] : T(0);
                tile_b[local_row * Tile + local_col] = (col < n && tile_row < k)
                    ? b_ptr[batch_b + (size_t)col * ldb + tile_row] : T(0);
                item.barrier(sycl::access::fence_space::local_space);
                for (int t = 0; t < Tile && kk0 + t < k; ++t) {
                    sum += tile_a[local_row * Tile + t] * tile_b[t * Tile + local_col];
                }
                item.barrier(sycl::access::fence_space::local_space);
            }
            if (row < m && col < n) {
                const T prior = c_ptr[batch_c + (size_t)col * ldc + row];
                c_ptr[batch_c + (size_t)col * ldc + row] = alpha * sum + beta * prior;
            }
        });
    });
}

template <class T> struct Nm { static const char* s(); static double fl(); };
template <> const char* Nm<double>::s() { return "double"; }
template <> double Nm<double>::fl() { return 2.0; }
template <> const char* Nm<float>::s() { return "float"; }
template <> double Nm<float>::fl() { return 2.0; }
template <> const char* Nm<std::complex<float>>::s() { return "cfloat"; }
template <> double Nm<std::complex<float>>::fl() { return 8.0; }
template <> const char* Nm<std::complex<double>>::s() { return "cdouble"; }
template <> double Nm<std::complex<double>>::fl() { return 8.0; }

template <class T> T mk(double re, double im);
template <> double mk<double>(double re, double) { return re; }
template <> float mk<float>(double re, double) { return (float)re; }
template <> std::complex<float> mk(double re, double im) { return {(float)re,(float)im}; }
template <> std::complex<double> mk(double re, double im) { return {re,im}; }

template <class T> void fill(std::vector<T>& v, unsigned seed) {
    std::mt19937 g(seed); std::uniform_real_distribution<double> d(-1,1);
    for (auto& x : v) x = mk<T>(d(g), d(g));
}

template <class T>
int run(sycl::queue& q, int m,int n,int k,int batch,double beta_r,int iters,int warmup) {
    const size_t na=(size_t)m*k*batch, nb=(size_t)k*n*batch, nc=(size_t)m*n*batch;
    std::vector<T> hA(na), hB(nb), hC(nc);
    fill(hA,1); fill(hB,2); fill(hC,3);
    T* dA=sycl::malloc_device<T>(na,q); T* dB=sycl::malloc_device<T>(nb,q);
    T* dC=sycl::malloc_device<T>(nc,q);
    q.memcpy(dA,hA.data(),na*sizeof(T)).wait();
    q.memcpy(dB,hB.data(),nb*sizeof(T)).wait();
    q.memcpy(dC,hC.data(),nc*sizeof(T)).wait();
    const T alpha = mk<T>(1.25,-0.5);
    const T beta  = mk<T>(beta_r, 0.5*beta_r);

    for (int i=0;i<warmup;i++)
        launch_tiled_general<T,16>(q,m,n,k,batch,dA,m,m*k,dB,k,k*n,dC,m,m*n,alpha,beta);
    q.wait();
    auto t0=std::chrono::high_resolution_clock::now();
    for (int i=0;i<iters;i++)
        launch_tiled_general<T,16>(q,m,n,k,batch,dA,m,m*k,dB,k,k*n,dC,m,m*n,alpha,beta);
    q.wait();
    auto t1=std::chrono::high_resolution_clock::now();
    double ms=std::chrono::duration<double,std::milli>(t1-t0).count()/iters;
    double fl=Nm<T>::fl()*(double)m*n*k*batch;
    printf("RESULT lib=tiled16 dtype=%s m=%d n=%d k=%d batch=%d beta=%g ms=%.4f tflops=%.4f\n",
           Nm<T>::s(),m,n,k,batch,beta_r,ms,fl/(ms*1e-3)/1e12);
    sycl::free(dA,q); sycl::free(dB,q); sycl::free(dC,q);
    return 0;
}

int main(int argc,char**argv){
    int m=512,n=512,k=512,batch=128,iters=20,warmup=5; double beta=1.0;
    std::string dt="double";
    for(int i=1;i<argc;i++){ std::string a=argv[i];
        if(a=="--m")m=atoi(argv[++i]); else if(a=="--n")n=atoi(argv[++i]);
        else if(a=="--k")k=atoi(argv[++i]); else if(a=="--batch")batch=atoi(argv[++i]);
        else if(a=="--beta")beta=atof(argv[++i]); else if(a=="--iters")iters=atoi(argv[++i]);
        else if(a=="--warmup")warmup=atoi(argv[++i]); else if(a=="--dtype")dt=argv[++i];
        else { fprintf(stderr,"unknown %s\n",a.c_str()); return 2; } }
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
    if(dt=="double") return run<double>(q,m,n,k,batch,beta,iters,warmup);
    if(dt=="float") return run<float>(q,m,n,k,batch,beta,iters,warmup);
    if(dt=="cfloat") return run<std::complex<float>>(q,m,n,k,batch,beta,iters,warmup);
    if(dt=="cdouble") return run<std::complex<double>>(q,m,n,k,batch,beta,iters,warmup);
    fprintf(stderr,"bad dtype\n"); return 2;
}
