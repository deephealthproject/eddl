// The MIT License (MIT)
//
// Copyright (c) 2019 PRHLT Research Group. Inc. http://www.prhlt.upv.es
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.


/////////////////////////////////////////////////////////////////////////////
// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// Main authors and developers:
//      Salva Carrión: salcarpo@prhlt.upv.es
//
//
// Collaborators:
//
// To collaborate please contact salcarpo@prhlt.upv.es
//
/////////////////////////////////////////////////////////////////////////////


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "apis/eddl.h"

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
        .def_readonly("shape", &Tensor::shape)
        .def_buffer([](Tensor &t) -> py::buffer_info {
	  std::vector<ssize_t> strides(t.ndim);
	  ssize_t S = sizeof(float);
	  for (int i = t.ndim - 1; i >=0; --i) {
	    strides[i] = S;
	    S *= t.shape[i];
	  }
	  return py::buffer_info(
            t.ptr,
            sizeof(float),
            py::format_descriptor<float>::format(),
            t.ndim,
            t.shape,
	    strides
	  );
	});
    m.def("tensor_from_npy", &tensor_from_npy);
    m.def("tensor_getdata", &tensor_getdata);

    py::class_<Layer>(m, "Layer")
        .def_readonly("name", &Layer::name)
        .def_readonly("input", &Layer::input)
        .def_readonly("output", &Layer::output)
        .def_readonly("target", &Layer::target)
        .def_readonly("delta", &Layer::delta)
        .def_readonly("orig", &Layer::orig)
        .def_readonly("params", &Layer::params)
        .def_readonly("gradients", &Layer::gradients)
        .def_readonly("parent", &Layer::parent)
        .def_readonly("child", &Layer::child);

    py::class_<Net>(m, "Model")
        .def_readonly("name", &Net::name)
        .def_readonly("dev", &Net::dev)
        .def_readonly("layers", &Net::layers)
        .def_readonly("input_layers", &Net::lin)
        .def_readonly("output_layers", &Net::lout)
        .def_readonly("losses", &Net::losses)
        .def_readonly("metrics", &Net::metrics)
        .def_readonly("fiterr", &Net::fiterr)
        //.def_readonly("optimizer", &Net::optimizer)
        .def("clean_fiterr", &Net::clean_fiterr)
        .def("summary", &Net::summary)
        .def("plot", &Net::plot)
        .def("train_batch_ni", &Net::train_batch_ni)
        .def("evaluate", &Net::evaluate);

    // Optimizer
    py::class_<Optimizer> (m, "Optimizer")
        .def_readonly("name", &Optimizer::name);

    // Optimizer: SGD
    py::class_<SGD, Optimizer> (m, "SGD")
        .def(py::init<float, float, float, bool>());

    // Loss
    py::class_<Loss> (m, "Loss")
        .def(py::init<std::string>())
        .def_readonly("name", &Loss::name);

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
    py::class_<Metric> (m, "Metric")
        .def(py::init<std::string>())
        .def_readonly("name", &Metric::name);

    // Metric: Categorical Accuracy
    py::class_<MCategoricalAccuracy, Metric> (m, "MCategoricalAccuracy")
        .def(py::init<>());
    // Metric: Mean Squared Error
    py::class_<MMeanSquaredError, Metric> (m, "MMeanSquaredError")
        .def(py::init<>());

    // Computing service
    py::class_<CompServ>(m, "CompServ")
        .def_readonly("type", &CompServ::type)
        .def_readonly("local_threads", &CompServ::local_threads)
        .def_readonly("local_gpus", &CompServ::local_gpus)
        .def_readonly("local_fpgas", &CompServ::local_fpgas);

    // EDDL
    py::class_<EDDL>(m, "EDDL")
        .def(py::init<>())
        .def("CS_CPU", &EDDL::CS_CPU)
        .def("build", &EDDL::build2)
        .def("get_model_mlp", &EDDL::get_model_mlp)  // Temp. for debugging
        .def("get_model_cnn", &EDDL::get_model_cnn);  // Temp. for debugging
}
