#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "eddl.h"

namespace py = pybind11;

Tensor* tensor_from_npy(py::array_t<float, py::array::c_style | py::array::forcecast> array, int dev){
    // Read input arrays
    py::buffer_info buf = array.request();
    auto size = (int)buf.size;
    auto *ptr = (float *)buf.ptr;

    // Cast pybind shape
    vector<int> shape;
    for(size_t d : buf.shape){
        shape.push_back((int)d);
    }

    // Build tensor
    Tensor* T = new Tensor(shape, dev);
    std::copy(ptr, ptr+size, T->ptr);

    return T;
}


py::array_t<float, py::array::c_style | py::array::forcecast> tensor_getdata(Tensor* T){
    py::array_t<float> result = py::array_t<float>(T->size);

    // Read input arrays
    py::buffer_info buf = result.request();
    auto *ptr = (float *)buf.ptr;

    // Copy and resize
    std::copy(T->ptr, T->ptr+T->size, ptr);
    result.resize(T->shape);

    return result;
}

// Inner name of the shared library (the python import must much this name and the filename.so)
PYBIND11_MODULE(_C, m) {
    // Constants
    m.attr("DEV_CPU") = DEV_CPU;
    m.attr("DEV_GPU") = DEV_GPU;
    m.attr("DEV_FPGA") = DEV_FPGA;

    // Tensors
    py::class_<Tensor> (m, "Tensor", py::buffer_protocol())
        .def(py::init<>())
        .def(py::init<vector<int>, int>())
        .def_readonly("device", &Tensor::device)
        .def_readonly("ndim", &Tensor::ndim)
        .def_readonly("size", &Tensor::size)
        .def_readonly("shape", &Tensor::shape);
    m.def("tensor_from_npy", &tensor_from_npy);
    m.def("tensor_getdata", &tensor_getdata);

    py::class_<Net>(m, "Model")
        .def("summary", &Net::summary)
        .def("plot", &Net::plot)
        .def("train_batch_ni", &Net::train_batch_ni);

    // Optimizer
    py::class_<optim> (m, "Optim");
    // Optimizer: SGD
    py::class_<sgd, optim> (m, "SGD")
        .def(py::init<float, float, float, bool>());

    // Loss
    py::class_<Loss> (m, "Loss");
    // Loss: Cross Entropy
    py::class_<LCrossEntropy, Loss> (m, "LCrossEntropy")
    .def(py::init<>());
    // Loss: Soft Cross Entropy
    py::class_<LSoftCrossEntropy, Loss> (m, "LSoftCrossEntropy")
        .def(py::init<>());
    // Loss: Mean Squared Error
    py::class_<LMeanSquaredError, Loss> (m, "LMeanSquaredError")
    .def(py::init<>());

    // Metric
    py::class_<Metric> (m, "Metric");
    // Metric: Categorical Accuracy
    py::class_<MCategoricalAccuracy, Metric> (m, "MCategoricalAccuracy")
        .def(py::init<>());
    // Metric: Mean Squared Error
    py::class_<MMeanSquaredError, Metric> (m, "MMeanSquaredError")
    .def(py::init<>());

    // Computing service
    py::class_<CompServ>(m, "CompServ");

    // EDDL
    py::class_<EDDL>(m, "EDDL")
        .def(py::init<>())
        .def("CS_CPU", &EDDL::CS_CPU)
        .def("build", &EDDL::build2)
        .def("get_model_mlp", &EDDL::get_model_mlp)
        .def("get_model_cnn", &EDDL::get_model_cnn);
}