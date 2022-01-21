#if defined(cPROTO)
#ifndef EDDL_EXPORT_HELPERS_H
#define EDDL_EXPORT_HELPERS_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/net/net.h"
#include "eddl/layers/core/layer_core.h"
#include "eddl/serialization/onnx/utils_onnx.h"

// Builds the onnx model from the net
onnx::ModelProto build_onnx_model(Net *net, bool gradients, int seq_len = 0);

// Builds the graph of the ModelProto from the net
void set_graph(onnx::ModelProto *model, Net *net, bool gradients, int seq_len = 0);

// Check if the inputs of a model are for a recurrent encoder, decoder or normal net (without time steps)
// Returns a map: key=input_layer_name, value=INPUT_TYPE
map<string, INPUT_TYPE> check_inputs_for_enc_or_dec(Net *net);

// Returns the layer type depending on if it is the input of a recurrent encoder, decoder or normal net
INPUT_TYPE get_input_type(LInput *l);

// Fixes the input shape for recurrent models
void prepare_recurrent_input(string input_name, string output_name, vector<int> input_shape, onnx::GraphProto *graph);

// Fixes the output shape for recurrent models
void prepare_recurrent_output(string input_name, string output_name, vector<int> output_shape, onnx::GraphProto *graph);

#endif // EDDL_EXPORT_HELPERS_H
#endif // cPROTO
