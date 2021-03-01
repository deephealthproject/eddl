#if defined(cPROTO)
#ifndef EDDL_GRU_ONNX_H
#define EDDL_GRU_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/recurrent/layer_recurrent.h"
#include "eddl/serialization/onnx/utils_onnx.h"

/*
 * ONNX IMPORT
 */

// OPSET: 7, 3, 1
Layer* build_gru_layer(onnx::NodeProto *node,
                       map<string, vector<float>> &map_init_values,
                       map<string, vector<int>> &map_init_dims,
                       map<string, vector<onnx::NodeProto *>> &input_node_map,
                       map<string, Layer *> &output_node_map,
                       vector<string> &inputs2remove,
                       LOG_LEVEL log_level,
                       int dev,
                       int mem);

/*
 * ONNX EXPORT
 */

// OPSET: 7, 3, 1
void build_gru_node(LGRU *layer, onnx::GraphProto *graph);

#endif // EDDL_GRU_ONNX_H
#endif // cPROTO
