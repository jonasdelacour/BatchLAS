#include "bindings/init.hh"

#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_batchlas, module) {
    module.doc() = "Hidden pybind11 extension for BatchLAS";

    batchlas::python::init_types(module);
    batchlas::python::init_blas_ops(module);
    batchlas::python::init_factorization_ops(module);
    batchlas::python::init_spectral_ops(module);
    batchlas::python::init_misc_ops(module);
}
