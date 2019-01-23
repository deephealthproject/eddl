#include "/usr/local/include/python2.7/pybind11/pybind11.h"
#include "tensor.h"

namespace py = pybind11;

using namespace pybind11::literals;

PYBIND11_MODULE(eddl, m) {
  py::class_<Tensor>(m, "Tensor")
    .def(py::init<const int,const int>());
}


