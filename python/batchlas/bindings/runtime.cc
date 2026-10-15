#include <python/batchlas/bindings/init.hh>
#include <python/batchlas/bindings/support.hh>

#include <batchlas/error.hh>
#include <batchlas/no_route.hh>
#include <batchlas/settings.hh>

#include <cstdlib>

namespace batchlas::python {

namespace {

// Owned for the life of the process, like the Queue cache: the translator runs
// long after init and at interpreter shutdown, so it must not look classes up.
PyObject* base_exception = nullptr;
PyObject* invalid_argument_error = nullptr;
PyObject* out_of_range_error = nullptr;
PyObject* batchlas_error = nullptr;
PyObject* unsupported_error = nullptr;
PyObject* device_error = nullptr;
PyObject* workspace_error = nullptr;
PyObject* convergence_error = nullptr;
PyObject* internal_error = nullptr;
PyObject* api_misuse_error = nullptr;
PyObject* no_route_error = nullptr;

PyObject* new_exception(py::module_& module, const char* name, const char* doc, py::tuple bases) {
    const std::string qualified = std::string("batchlas.") + name;
    PyObject* cls = PyErr_NewExceptionWithDoc(qualified.c_str(), doc, bases.ptr(), nullptr);
    if (cls == nullptr) throw py::error_already_set();
    module.add_object(name, py::handle(cls));
    return cls;
}

py::tuple bases_of(std::initializer_list<PyObject*> classes) {
    py::tuple bases(classes.size());
    std::size_t index = 0;
    for (PyObject* cls : classes) bases[index++] = py::reinterpret_borrow<py::object>(cls);
    return bases;
}

const char* numpy_dtype_name(ScalarKind scalar) {
    switch (scalar) {
        case ScalarKind::F32: return "float32";
        case ScalarKind::F64: return "float64";
        case ScalarKind::C32: return "complex64";
        case ScalarKind::C64: return "complex128";
    }
    return "unknown";
}

void raise_no_route(const NoRouteError& error) {
    py::object instance = py::reinterpret_borrow<py::object>(no_route_error)(error.what());
    instance.attr("op") = std::string(op_name(error.op()));
    instance.attr("backend") = lower_copy(std::string(to_string(error.backend())));
    instance.attr("dtype") = numpy_dtype_name(error.scalar());
    PyErr_SetObject(no_route_error, instance.ptr());
}

void translate(std::exception_ptr pointer) {
    try {
        if (pointer) std::rethrow_exception(pointer);
    } catch (const NoRouteError& error) {
        raise_no_route(error);
    } catch (const batchlas::unsupported& error) {
        PyErr_SetString(unsupported_error, error.what());
    } catch (const batchlas::device_error& error) {
        PyErr_SetString(device_error, error.what());
    } catch (const batchlas::workspace_error& error) {
        PyErr_SetString(workspace_error, error.what());
    } catch (const batchlas::convergence_error& error) {
        PyErr_SetString(convergence_error, error.what());
    } catch (const batchlas::internal_error& error) {
        PyErr_SetString(internal_error, error.what());
    } catch (const batchlas::api_misuse& error) {
        PyErr_SetString(api_misuse_error, error.what());
    } catch (const batchlas::error& error) {
        PyErr_SetString(batchlas_error, error.what());
    } catch (const std::invalid_argument& error) {
        // Not only batchlas::invalid_argument: a bad BATCHLAS_<OP>_ROUTE pin throws
        // the plain std one (src/select/select.hh).
        PyErr_SetString(invalid_argument_error, error.what());
    } catch (const std::out_of_range& error) {
        PyErr_SetString(out_of_range_error, error.what());
    } catch (const batchlas::exception& error) {
        PyErr_SetString(base_exception, error.message());
    }
}

void init_exceptions(py::module_& module) {
    base_exception = new_exception(module, "BatchLASException",
                                   "Root of every BatchLAS exception (batchlas::exception).",
                                   bases_of({PyExc_Exception}));
    invalid_argument_error = new_exception(module, "InvalidArgumentError",
                                           "batchlas::invalid_argument, or any std::invalid_argument "
                                           "such as a bad route pin. A ValueError.",
                                           bases_of({base_exception, PyExc_ValueError}));
    out_of_range_error = new_exception(module, "OutOfRangeError", "batchlas::out_of_range. An IndexError.",
                                       bases_of({base_exception, PyExc_IndexError}));
    batchlas_error = new_exception(module, "BatchLASError", "batchlas::error. A RuntimeError.",
                                   bases_of({base_exception, PyExc_RuntimeError}));
    unsupported_error = new_exception(module, "UnsupportedError",
                                      "batchlas::unsupported: the call is valid but this build cannot serve it.",
                                      bases_of({batchlas_error}));
    device_error = new_exception(module, "DeviceError", "batchlas::device_error.", bases_of({batchlas_error}));
    workspace_error =
        new_exception(module, "WorkspaceError", "batchlas::workspace_error.", bases_of({batchlas_error}));
    convergence_error =
        new_exception(module, "ConvergenceError", "batchlas::convergence_error.", bases_of({batchlas_error}));
    internal_error = new_exception(module, "InternalError", "batchlas::internal_error.", bases_of({batchlas_error}));
    api_misuse_error =
        new_exception(module, "ApiMisuseError", "batchlas::api_misuse.", bases_of({batchlas_error}));
    no_route_error = new_exception(module, "NoRouteError",
                                   "batchlas::NoRouteError: nothing in this build can serve the call. "
                                   "Attributes: op, backend, dtype.",
                                   bases_of({unsupported_error}));
    py::register_exception_translator(&translate);
}

}  // namespace

void init_runtime(py::module_& module) {
    init_exceptions(module);

    module.def("_reload_settings", [] { batchlas::detail::reload_settings(); });

    // os.environ alone is not enough for an embedding host whose C environment
    // is not the interpreter's, so the pin also goes through ::setenv.
    module.def("_setenv", [](const std::string& name, const py::object& value) {
        if (value.is_none()) {
            ::unsetenv(name.c_str());
        } else {
            ::setenv(name.c_str(), value.cast<std::string>().c_str(), 1);
        }
    });

    module.def("_route_pin", [](const std::string& op) -> py::object {
        const EnvValue& value = batchlas::settings().routing.route(op);
        if (!value.is_set()) return py::none();
        return py::str(value.value());
    });

    py::list route_ops;
    for (std::string_view op : RoutingSettings::ops) route_ops.append(std::string(op));
    module.attr("_route_ops") = py::tuple(route_ops);
}

}  // namespace batchlas::python
