#if defined(cPROTO)
#ifndef EDDL_LSTM_ONNX_H
#define EDDL_LSTM_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/recurrent/layer_recurrent.h"
#include "eddl/serialization/onnx/utils_onnx.h"

/*
 * ONNX IMPORT
 */

// OPSET: 7, 1
Layer* build_lstm_layer(onnx::NodeProto *node,
                         map<string, vector<float>> &map_init_values,
                         map<string, vector<int>> &map_init_dims,
                         map<string, vector<onnx::NodeProto *>> &input_node_map,
                         map<string, Layer *> &output_node_map,
                         LOG_LEVEL log_level,
                         int dev,
                         int mem);

/*
 * ONNX EXPORT
 */

// OPSET: 7, 1
void build_lstm_node(LLSTM *layer, onnx::GraphProto *graph, bool gradients = false);

/*
 * DISTRIBUTED TRAINING
 */

vector<Tensor *> get_lstm_tensors(onnx::NodeProto &node,
                                  map<string, vector<float>> &map_init_values,
                                  map<string, vector<int>> &map_init_dims);

#endif // EDDL_LSTM_ONNX_H
#endif // cPROTO
