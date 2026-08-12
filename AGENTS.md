BatchLAS Agent Environment Guide

TL;DR  Build-time dependencies: CMake ≥3.17 (≥3.21 if you want the cmake --preset workflow), a C++20 compiler with SYCL 2020 support **that has a backend for your GPU vendor**, netlib LAPACK/LAPACKE + CBLAS, and oneDPL headers. Runtime dependencies are the same plus a GPU your SYCL runtime actually exposes. The configuration that is actually exercised here is a CUDA-enabled DPC++ (self-built intel/llvm, installed at /opt/dpcpp-cuda) against CUDA 13.2 and an NVIDIA RTX 4090 (sm_89) on Ubuntu 22.04 — see the "Tested platforms" table in README.md.

⚠️  **NVIDIA targets need a CUDA-capable DPC++.** The stock `intel-oneapi-compiler-dpcpp-cpp` package in §2 has **no CUDA adapter**. On an NVIDIA machine it configures cleanly, with no warning and no error, and builds a CPU-only library. Use either

* Intel oneAPI plus the Codeplay **oneAPI for NVIDIA GPUs** plugin, or
* a self-built `intel/llvm` configured with `--cuda` (this is what `/opt/dpcpp-cuda` is).

Check before you build: `sycl-ls` must list a `[cuda:gpu]` entry. And read the configure output — **`-- Using SYCL targets: spir64_x86_64` means you are about to build, and benchmark, a CPU-only build**, whatever GPUs are in the box. A CUDA-enabled configure names the architecture instead, e.g. `nvidia_gpu_sm_89`. Forcing `-DBATCHLAS_ENABLE_CUDA=ON` under a compiler with no CUDA adapter still does not give you CUDA, but it no longer does so silently: `ON` now means "require it", and the configure aborts with a `FATAL_ERROR` naming the missing `[cuda:gpu]` entry (see "Common CMake options" in README.md). It used to configure for `nvidia_gpu_sm_50`, compile and link with exit 0, and fail at run time with `No kernel named ... was found`. The default `AUTO` does not go down that path at all — with no `[cuda:gpu]` it simply builds the CPU-only library described above.

⸻

1. Prerequisite Packages

Component	Debian/Ubuntu (apt)	Fedora/RHEL-like (dnf)	Arch Linux (pacman)	Source build (fallback)
BLAS/LAPACK (Fortran APIs)	libblas-dev liblapack-dev	blas-devel lapack-devel	blas lapack	see §4
C interface (CBLAS & LAPACKE headers)	liblapacke-dev	lapack-devel	lapacke	see §4
Build tools	build-essential cmake git	@development-tools cmake git	base-devel cmake git	—

Why not just libopenblas-dev? Ubuntu’s OpenBLAS package omits lapacke.h; you still need liblapacke-dev for the C interface, or build LAPACKE yourself. This is a packaging decision, not a BatchLAS bug.

⸻

2. Installing a SYCL Compiler

2a. Intel® oneAPI DPC++/C++ (icpx) — Intel GPUs and CPU-only builds

This is the easy path, and it is the WRONG path if you are targeting NVIDIA: the package below ships no CUDA adapter. See the warning in the TL;DR, and §2b.

# 1. Add Intel's APT repo and key (root)
wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB |
  sudo gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
  | sudo tee /etc/apt/sources.list.d/oneAPI.list

sudo apt update

# 2. Install the minimal SYCL compiler package
sudo apt install intel-oneapi-compiler-dpcpp-cpp      # SYCL 2025.x

# (optional) Classic compilers for C/C++ & Fortran
sudo apt install intel-oneapi-compiler-dpcpp-cpp-and-cpp-classic \
                 intel-oneapi-compiler-fortran

After installation, configure the environment for each shell session:

source /opt/intel/oneapi/setvars.sh   # sets PATH, LD_LIBRARY_PATH, MKLROOT, etc.

You do not need the entire intel-basekit; the single compiler package is enough to build BatchLAS for CPU or an Intel GPU.

2b. CUDA-capable DPC++ — NVIDIA GPUs

Two options, both giving a clang-family `clang++`/`icpx` that emits NVPTX:

* **Codeplay oneAPI for NVIDIA GPUs**: install Intel oneAPI as in §2a, then the Codeplay plugin matching your oneAPI version. It adds the CUDA UR adapter, after which `sycl-ls` reports `[cuda:gpu]`.
* **Self-built `intel/llvm`**: clone https://github.com/intel/llvm and configure with `--cuda` (plus `--cmake-opt=-DCMAKE_INSTALL_PREFIX=<prefix>`). This is what this machine uses; it lives at `/opt/dpcpp-cuda` and is the compiler every number in the repository was measured with.

Either way you also need a CUDA Toolkit (13.2 here) and you point CMake at the compiler explicitly:

cmake -S . -B build -DCMAKE_CXX_COMPILER=/opt/dpcpp-cuda/bin/clang++

Note for consumers of an installed BatchLAS: the *whole* consuming project has to be configured with this same compiler. See "Consuming BatchLAS from CMake" in README.md for why.

2c. oneDPL

Several sources include `<oneapi/dpl/...>` unconditionally, so oneDPL headers are a hard dependency even for a CUDA build. A self-built `intel/llvm` (`/opt/dpcpp-cuda`) does not bundle them. The build looks under `/opt/intel/oneapi/dpl/latest/include`, which is what `sudo apt install intel-oneapi-dpl` gives you; oneDPL is header-only, so a clone of https://github.com/oneapi-src/oneDPL works too as long as its `include/` ends up at that path.

⸻

3. Verifying the Toolchain

icpx --version          # or: /opt/dpcpp-cuda/bin/clang++ --version
cmake --version         # ≥3.17 (≥3.21 for cmake --preset)
sycl-ls                 # must list your GPU, e.g. a [cuda:gpu] entry
pkg-config --exists lapacke cblas && echo "LAPACKE & CBLAS found"


⸻

4. Building netlib LAPACKE/CBLAS from Source (if distro packages are unavailable)

git clone https://github.com/Reference-LAPACK/lapack.git
cd lapack
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DLAPACKE=ON -DBUILD_SHARED_LIBS=ON
cmake --build build -j $(nproc)
sudo cmake --install build   # installs liblapacke.so, libcblas.so, headers

Add the install prefix (e.g. /usr/local/lib) to LD_LIBRARY_PATH or run sudo ldconfig so that the linker can locate the libraries.

⸻

5. CMake Configuration Hints

BatchLAS searches for LAPACKE via find_package(LAPACK REQUIRED COMPONENTS CBLAS LAPACKE) and for SYCL via find_package(SYCL REQUIRED) (provided by Intel’s compiler). If your LAPACKE install lives outside standard prefixes, set:

export CMAKE_PREFIX_PATH="/opt/netlib:$CMAKE_PREFIX_PATH"


⸻

6. Quick Smoke Test

cmake -B build .

# Iterating on one algorithm: this builds the library and only this one test
# binary. Do not build the default target while iterating — it also builds the
# other 48 test executables, which you are not about to run.
cmake --build build --target stedc_tests -j"$(nproc)"
ctest --test-dir build -R '^stedc_tests$' --output-on-failure

# Before pushing, build and run everything.
cmake --build build -j"$(nproc)"
ctest --test-dir build

# Also before pushing, if you touched cmake/, include/ or the install rules:
# the packaging gate. These are the same checks CI runs (.github/workflows/ci.yml).
sh .github/ci/run_local_checks.sh

# The --package mode is the one CI cannot run, because it needs a real install
# from a real SYCL build. Give run_local_checks.sh a prefix and it adds it:
cmake --install build --prefix /tmp/batchlas-prefix
sh .github/ci/run_local_checks.sh /tmp/batchlas-prefix

# The end-to-end version of the same thing — install, then configure, build and
# run examples/consumer/ as a standalone outside project:
ctest --test-dir build -R '^consumer_package_tests$' --output-on-failure


⸻

Known Pitfalls
	•	Silent CPU-only build on an NVIDIA box: the compiler has no CUDA adapter. `-- Using SYCL targets: spir64_x86_64` in the configure output is the tell. See the TL;DR and §2b.
	•	Missing lapacke.h: install liblapacke-dev even if you already have libopenblas-dev.
	•	Multiple BLAS providers: choose the backend with sudo update-alternatives --config libblas.so.3.
	•	device not found at runtime: ensure your GPU driver and its SYCL adapter match the compiler version — the Level-Zero runtime for Intel, the CUDA adapter and driver for NVIDIA.
	•	libsycl.so.9: cannot open shared object file when running anything: the DPC++ runtime is not on the loader path. Export LD_LIBRARY_PATH=<dpcpp-prefix>/lib, or add it to /etc/ld.so.conf.d and run ldconfig.

⸻

License

SPDX-License-Identifier: MIT