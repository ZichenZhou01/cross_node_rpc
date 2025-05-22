#include <pybind11/pybind11.h>
#include "rpc_wrapper_binding.h"


namespace py = pybind11;

PYBIND11_MODULE(rpc_rdma, m) {
    m.doc() = "RDMA-based RPC using pybind11";

    // Define Python functions
    m.def("start_server", &start_server, py::arg("port") = "7471", "Start the RDMA RPC server in a background thread");
    m.def("stop_server", &stop_server, "Stop the RDMA RPC server if running");
    // Define Python classes
    py::class_<RpcClientWrapper>(m, "Client")
        .def(py::init<const std::string&, const std::string&>(), py::arg("host"), py::arg("port") = "7471")
        .def("echo", &RpcClientWrapper::echo, "Send an echo RPC and receive the response")
        .def("add", &RpcClientWrapper::add, "Send an add RPC and receive the sum")
        .def("mul", &RpcClientWrapper::mul, "Send an mul RPC and receive the multiplication result");
}