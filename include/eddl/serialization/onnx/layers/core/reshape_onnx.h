#if defined(cPROTO)
#ifndef EDDL_RESHAPE_ONNX_H
#define EDDL_RESHAPE_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/serialization/onnx/utils_onnx.h"
#include "eddl/layers/core/layer_core.h"

/*
 * ONNX IMPORT
 */

// OPSET: 13, 5
Layer* build_reshape_layer(onnx::NodeProto *node,
                           map<string, onnx::NodeProto *> &constant_node_map, 
                           map<string, vector<float>> &map_init_values,
                           map<string, vector<int>> &map_init_dims,
                           map<string, vector<onnx::NodeProto *>> &input_node_map,
                           map<string, Layer *> &output_node_map,
                           int dev,
                           int mem);

// OPSET: 13, 11, 9, 1
Layer* build_flatten_layer(onnx::NodeProto *node,
                           map<string, Layer *> &output_node_map,
                           int dev,
                           int mem);

/*
 * ONNX EXPORT
 */

// OPSET: 13, 5
void build_reshape_node(LReshape *layer, onnx::GraphProto *graph);

#endif // EDDL_RESHAPE_ONNX_H
#endif // cPROTO
