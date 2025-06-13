# AGENT INSTRUCTIONS

This repository requires netlib's LAPACKE and Intel's oneAPI DPC++ compiler. These packages are not vendored with the repo, so they must be installed before building or testing.

## Installing netlib LAPACKE

On Ubuntu/Debian systems:

```bash
sudo apt-get update
sudo apt-get install -y liblapacke-dev libblas-dev
```

These packages provide the reference CBLAS and LAPACKE libraries expected by the CMake build scripts.

## Installing the Intel oneAPI DPC++ compiler (icpx)

The project expects an Intel SYCL compiler. Install the minimal oneAPI package that supplies `icpx`:

```bash
wget https://apt.repos.intel.com/setup/intel-oneapi-repo.sh
sudo bash intel-oneapi-repo.sh
sudo apt-get update
sudo apt-get install -y intel-oneapi-compiler-dpcpp-cpp
```

After installation, configure the environment:

```bash
source /opt/intel/oneapi/setvars.sh
```

This script sets variables such as `MKLROOT` which are needed when building BatchLAS.
