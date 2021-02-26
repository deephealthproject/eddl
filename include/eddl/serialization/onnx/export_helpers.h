#if defined(cPROTO)
#ifndef EDDL_EXPORT_HELPERS_H
#define EDDL_EXPORT_HELPERS_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/net/net.h"

// Builds the onnx model from the net
onnx::ModelProto build_onnx_model(Net *net, bool gradients);

// Builds the graph of the ModelProto from the net
void set_graph(onnx::ModelProto *model, Net *net, bool gradients);

// Fixes the input shape for recurrent models
void prepare_recurrent_input(string input_name, string output_name, vector<int> input_shape, onnx::GraphProto *graph);

// Fixes the output shape for recurrent models
void prepare_recurrent_output(string input_name, string output_name, vector<int> output_shape, onnx::GraphProto *graph);

#endif // EDDL_EXPORT_HELPERS_H
#endif // cPROTO
