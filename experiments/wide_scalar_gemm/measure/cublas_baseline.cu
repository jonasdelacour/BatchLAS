// cuBLAS strided-batched GEMM baseline, deliberately mirroring the candidate
// harnesses' timing discipline so the ratio means something:
//   * column-major, no transpose, lda=m ldb=k ldc=m, stride = m*k etc.
//   * beta is a CLI argument and DEFAULTS TO 1, because a beta=0 measurement
//     cannot see an epilogue that has to read C back.
//   * warmup iterations run before the timed loop (cuBLAS picks a kernel and
//     the clocks come up on the first call; timing that fabricates a loss).
//   * one wall-clock span over `iters` back-to-back launches, divided by iters
//     -- the same thing the candidates report -- with a device sync at both
//     ends and nothing else in the span.
//   * FLOP count 2*m*n*k*batch for real, 8*m*n*k*batch for complex, matching
//     the candidates so the TFLOP/s numbers are directly comparable.
//
// Build:
//   /usr/local/cuda-13.2/bin/nvcc -O3 -arch=sm_89 cublas_baseline.cu \
//       -o cublas_baseline -lcublas
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>
#include <random>

#define CU(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e_)); exit(1);} } while(0)
#define CB(x) do { cublasStatus_t s_=(x); if(s_!=CUBLAS_STATUS_SUCCESS){ \
    fprintf(stderr,"cuBLAS %s @%d: %d\n",#x,__LINE__,(int)s_); exit(1);} } while(0)

struct Opt { int m=512,n=512,k=512,batch=128,iters=20,warmup=5; double beta=1.0;
             std::string dtype="double"; };

template <class T>
static void fill(std::vector<T>& v, unsigned seed);

template <> void fill<double>(std::vector<double>& v, unsigned seed) {
    std::mt19937 g(seed); std::uniform_real_distribution<double> d(-1,1);
    for (auto& x : v) x = d(g);
}
template <> void fill<float>(std::vector<float>& v, unsigned seed) {
    std::mt19937 g(seed); std::uniform_real_distribution<float> d(-1,1);
    for (auto& x : v) x = d(g);
}
template <> void fill<cuDoubleComplex>(std::vector<cuDoubleComplex>& v, unsigned seed) {
    std::mt19937 g(seed); std::uniform_real_distribution<double> d(-1,1);
    for (auto& x : v) { x.x = d(g); x.y = d(g); }
}
template <> void fill<cuComplex>(std::vector<cuComplex>& v, unsigned seed) {
    std::mt19937 g(seed); std::uniform_real_distribution<float> d(-1,1);
    for (auto& x : v) { x.x = d(g); x.y = d(g); }
}

template <class T, class Call>
static void run(const Opt& o, const char* label, double flops_per_mac, Call call) {
    const long long m=o.m, n=o.n, k=o.k, b=o.batch;
    const long long sa=m*k, sb=k*n, sc=m*n;
    std::vector<T> hA(sa*b), hB(sb*b), hC(sc*b);
    fill(hA, 1); fill(hB, 2); fill(hC, 3);

    T *dA,*dB,*dC;
    CU(cudaMalloc(&dA, sizeof(T)*sa*b));
    CU(cudaMalloc(&dB, sizeof(T)*sb*b));
    CU(cudaMalloc(&dC, sizeof(T)*sc*b));
    CU(cudaMemcpy(dA,hA.data(),sizeof(T)*sa*b,cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dB,hB.data(),sizeof(T)*sb*b,cudaMemcpyHostToDevice));
    CU(cudaMemcpy(dC,hC.data(),sizeof(T)*sc*b,cudaMemcpyHostToDevice));

    cublasHandle_t h; CB(cublasCreate(&h));
    // Strict FP32: no TF32 substitution. The 80 TFLOP/s figure people quote is
    // TF32; comparing an FP32 candidate against a TF32 baseline is not a
    // comparison. For fp64/complex this is a no-op.
    CB(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));

    for (int i=0;i<o.warmup;i++) call(h,dA,dB,dC);
    CU(cudaDeviceSynchronize());

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i=0;i<o.iters;i++) call(h,dA,dB,dC);
    CU(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double,std::milli>(t1-t0).count()/o.iters;
    double fl = flops_per_mac*(double)m*(double)n*(double)k*(double)b;
    printf("RESULT lib=cublas dtype=%s m=%d n=%d k=%d batch=%d beta=%g ms=%.4f tflops=%.3f\n",
           label,o.m,o.n,o.k,o.batch,o.beta, ms, fl/(ms*1e-3)/1e12);
    CB(cublasDestroy(h));
    CU(cudaFree(dA)); CU(cudaFree(dB)); CU(cudaFree(dC));
}

int main(int argc,char**argv){
    Opt o;
    for(int i=1;i<argc;i++){ std::string a=argv[i];
        if(a=="--m") o.m=atoi(argv[++i]);
        else if(a=="--n") o.n=atoi(argv[++i]);
        else if(a=="--k") o.k=atoi(argv[++i]);
        else if(a=="--batch") o.batch=atoi(argv[++i]);
        else if(a=="--beta") o.beta=atof(argv[++i]);
        else if(a=="--iters") o.iters=atoi(argv[++i]);
        else if(a=="--warmup") o.warmup=atoi(argv[++i]);
        else if(a=="--dtype") o.dtype=argv[++i];
        else { fprintf(stderr,"unknown arg %s\n",a.c_str()); return 2; }
    }
    const long long m=o.m,n=o.n,k=o.k;
    if(o.dtype=="double"){
        double al=1.25, be=o.beta;
        run<double>(o,"double",2.0,[&](cublasHandle_t h,double*A,double*B,double*C){
            CB(cublasDgemmStridedBatched(h,CUBLAS_OP_N,CUBLAS_OP_N,o.m,o.n,o.k,
                &al,A,o.m,m*k,B,o.k,k*n,&be,C,o.m,m*n,o.batch)); });
    } else if(o.dtype=="float"){
        float al=1.25f, be=(float)o.beta;
        run<float>(o,"float",2.0,[&](cublasHandle_t h,float*A,float*B,float*C){
            CB(cublasSgemmStridedBatched(h,CUBLAS_OP_N,CUBLAS_OP_N,o.m,o.n,o.k,
                &al,A,o.m,m*k,B,o.k,k*n,&be,C,o.m,m*n,o.batch)); });
    } else if(o.dtype=="cfloat"){
        cuComplex al=make_cuComplex(1.25f,-0.5f);
        cuComplex be=make_cuComplex((float)o.beta,(float)(0.5*o.beta));
        run<cuComplex>(o,"cfloat",8.0,[&](cublasHandle_t h,cuComplex*A,cuComplex*B,cuComplex*C){
            CB(cublasCgemmStridedBatched(h,CUBLAS_OP_N,CUBLAS_OP_N,o.m,o.n,o.k,
                &al,A,o.m,m*k,B,o.k,k*n,&be,C,o.m,m*n,o.batch)); });
    } else if(o.dtype=="cdouble"){
        cuDoubleComplex al=make_cuDoubleComplex(1.25,-0.5);
        cuDoubleComplex be=make_cuDoubleComplex(o.beta,0.5*o.beta);
        run<cuDoubleComplex>(o,"cdouble",8.0,[&](cublasHandle_t h,cuDoubleComplex*A,cuDoubleComplex*B,cuDoubleComplex*C){
            CB(cublasZgemmStridedBatched(h,CUBLAS_OP_N,CUBLAS_OP_N,o.m,o.n,o.k,
                &al,A,o.m,m*k,B,o.k,k*n,&be,C,o.m,m*n,o.batch)); });
    } else { fprintf(stderr,"bad dtype\n"); return 2; }
    return 0;
}
