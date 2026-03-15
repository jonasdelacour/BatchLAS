#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace batchlas::python {

void init_types(py::module_& module);
void init_blas_ops(py::module_& module);
void init_factorization_ops(py::module_& module);
void init_spectral_ops(py::module_& module);
void init_misc_ops(py::module_& module);

}  // namespace batchlas::python
