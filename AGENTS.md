BatchLAS Agent Environment Guide

TL;DR  Build-time dependencies: CMake ≥3.18, a C++20 compiler with SYCL 2020 support (tested with Intel® oneAPI icpx 2025.1), and netlib LAPACK/LAPACKE + CBLAS. Runtime dependencies are the same plus an OpenCL-capable or Level-Zero GPU (Intel GPUs tested).

⸻

1. Prerequisite Packages

Component	Debian/Ubuntu (apt)	Fedora/RHEL-like (dnf)	Arch Linux (pacman)	Source build (fallback)
BLAS/LAPACK (Fortran APIs)	libblas-dev liblapack-dev	blas-devel lapack-devel	blas lapack	see §4
C interface (CBLAS & LAPACKE headers)	liblapacke-dev	lapack-devel	lapacke	see §4
Build tools	build-essential cmake git	@development-tools cmake git	base-devel cmake git	—

Why not just libopenblas-dev? Ubuntu’s OpenBLAS package omits lapacke.h; you still need liblapacke-dev for the C interface, or build LAPACKE yourself. This is a packaging decision, not a BatchLAS bug.

⸻

2. Installing Intel® oneAPI DPC++/C++ Compiler (icpx)

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

You do not need the entire intel-basekit; the single compiler package is enough to build BatchLAS.

⸻

3. Verifying the Toolchain

icpx --version          # should print 2025.x or newer
cmake --version         # ≥3.18
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
cmake --build build -j
ctest --test-dir build


⸻

Known Pitfalls
	•	Missing lapacke.h: install liblapacke-dev even if you already have libopenblas-dev.
	•	Multiple BLAS providers: choose the backend with sudo update-alternatives --config libblas.so.3.
	•	device not found at runtime: ensure your Intel GPU driver or Level-Zero runtime matches the compiler version.

⸻

License

SPDX-License-Identifier: MIT