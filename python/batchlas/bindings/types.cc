#include "init.hh"
#include "support.hh"

namespace batchlas::python {

namespace {

py::tuple dense_shape(const DenseMatrix& matrix) {
    if (matrix.batch_size() == 1) {
        return py::make_tuple(matrix.rows(), matrix.cols());
    }
    return py::make_tuple(matrix.batch_size(), matrix.rows(), matrix.cols());
}

py::tuple vector_shape(const DenseVector& vector) {
    if (vector.batch_size() == 1) {
        return py::make_tuple(vector.size());
    }
    return py::make_tuple(vector.batch_size(), vector.size());
}

py::tuple sparse_shape(const SparseMatrix& matrix) {
    if (matrix.batch_size() == 1) {
        return py::make_tuple(matrix.rows(), matrix.cols());
    }
    return py::make_tuple(matrix.batch_size(), matrix.rows(), matrix.cols());
}

}  // namespace

void init_types(py::module_& module) {
    py::class_<DenseMatrix>(module, "_DenseMatrix")
        .def_property_readonly("dtype", [](const DenseMatrix& matrix) {
            return dtype_name(matrix.dtype());
        })
        .def_property_readonly("rows", &DenseMatrix::rows)
        .def_property_readonly("cols", &DenseMatrix::cols)
        .def_property_readonly("batch_size", &DenseMatrix::batch_size)
        .def_property_readonly("heterogeneous", &DenseMatrix::is_heterogeneous)
        .def_property_readonly("shape", &dense_shape)
        .def("to_python", &dense_matrix_to_python);

    py::class_<DenseVector>(module, "_DenseVector")
        .def_property_readonly("dtype", [](const DenseVector& vector) {
            return dtype_name(vector.dtype());
        })
        .def_property_readonly("size", &DenseVector::size)
        .def_property_readonly("batch_size", &DenseVector::batch_size)
        .def_property_readonly("shape", &vector_shape)
        .def("to_python", &dense_vector_to_python);

    py::class_<SparseMatrix>(module, "_SparseMatrix")
        .def_property_readonly("dtype", [](const SparseMatrix& matrix) {
            return dtype_name(matrix.dtype());
        })
        .def_property_readonly("rows", &SparseMatrix::rows)
        .def_property_readonly("cols", &SparseMatrix::cols)
        .def_property_readonly("batch_size", &SparseMatrix::batch_size)
        .def_property_readonly("shape", &sparse_shape)
        .def("to_python", &sparse_matrix_to_python);

    py::class_<ILUKHandle>(module, "_ILUKPreconditioner")
        .def_property_readonly("dtype", [](const ILUKHandle& handle) {
            return dtype_name(handle.dtype());
        })
        .def_property_readonly("n", &ILUKHandle::n)
        .def_property_readonly("batch_size", &ILUKHandle::batch_size)
        .def("metadata", &iluk_metadata);

    module.def("_dense_from_numpy", [](const py::array& array) {
        return dense_matrix_from_numpy(array);
    });
    module.def("_dense_from_sequence", [](const py::sequence& items) {
        return heterogeneous_dense_matrix_from_sequence(items);
    });
    module.def("_vector_from_numpy", [](const py::array& array) {
        return dense_vector_from_numpy(array);
    });
    module.def("_sparse_from_python", [](const py::object& object) {
        return sparse_matrix_from_python(object);
    });
    module.def("_sparse_from_sequence", [](const py::sequence& items) {
        return sparse_matrix_batch_from_sequence(items);
    });

    module.def("available_backends", &available_backends);
    module.def("available_devices", &available_devices);
    module.def("compiled_features", &compiled_features);
}

}  // namespace batchlas::python
