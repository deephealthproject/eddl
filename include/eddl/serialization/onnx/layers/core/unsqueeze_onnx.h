#if defined(cPROTO)
#ifndef EDDL_UNSQUEEZE_ONNX_H
#define EDDL_UNSQUEEZE_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/serialization/onnx/utils_onnx.h"
#include "eddl/layers/core/layer_core.h"

/*
 * ONNX IMPORT
 */

// OPSET: 11, 1
Layer* build_unsqueeze_layer(onnx::NodeProto *node,
                             map<string, vector<float>> &map_init_values,
                             map<string, vector<int>> &map_init_dims,
                             map<string, Layer *> &output_node_map,
                             LOG_LEVEL log_level,
                             int dev,
                             int mem);

/*
 * ONNX EXPORT
 */

// OPSET: 11, 1
void build_unsqueeze_node(LUnsqueeze *layer, onnx::GraphProto *graph);

void unsqueeze_node_builder(string node_name, string input, string output, vector<int> axes, onnx::GraphProto *graph);

#endif // EDDL_UNSQUEEZE_ONNX_H
#endif // cPROTO
