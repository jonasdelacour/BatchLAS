# BatchLAS documentation {#mainpage}

BatchLAS is a SYCL-first library of **batched** dense and sparse linear algebra:
thousands of small-to-medium problems solved in one call, with native SYCL
kernels competing against vendor libraries (cuBLAS/cuSOLVER/cuSPARSE, rocBLAS,
oneMKL, netlib) through measured routing windows.

This site is both the reference manual and the project's **record of evidence
and decisions**. Code comments carry invariants and a pointer
`evidence: docs/<page>.md#<section>`; the measurement, the rejected
alternatives and the reasoning live on the page it points to.

| If you want to… | Go to |
| --- | --- |
| call BatchLAS from C++ or Python | @ref guide_index |
| look up a function, type or option | @ref api "API reference" (grouped by operation family) |
| know why the code is shaped the way it is | @ref architecture_index |
| read the mathematics behind an algorithm | @ref algorithms_index |
| know which kernel runs for a shape, and the numbers behind that choice | @ref perf_evidence |
| find a raw measurement grid, or what depends on a piece of evidence | @ref results_index |
| build, test, benchmark, tune or document | @ref developer_index |

A minimal call:

```cpp
#include <batchlas.hh>
using namespace batchlas;

Queue ctx(Device::default_device());          // the backend comes from the device
gemm(ctx, A.view(), B.view(), C.view(), {.alpha = 2.0f});
potrf(ctx, S.view(), PotrfOptions{.uplo = Uplo::Upper});
ctx.wait();                                   // nothing is readable before this
```

Every entry point enqueues and returns an Event; matrices are column-major and
must live in device-accessible (USM) memory; one Queue per thread. The full
contract is in @ref md_docs_2cpp-api.

@subpage guide_index
@subpage architecture_index
@subpage algorithms_index
@subpage perf_evidence
@subpage results_index
@subpage developer_index
