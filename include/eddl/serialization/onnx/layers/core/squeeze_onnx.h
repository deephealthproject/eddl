#if defined(cPROTO)
#ifndef EDDL_SQUEEZE_ONNX_H
#define EDDL_SQUEEZE_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/serialization/onnx/utils_onnx.h"
#include "eddl/layers/core/layer_core.h"

/*
 * ONNX IMPORT
 */

// OPSET: 11, 1
Layer* build_squeeze_layer(onnx::NodeProto *node,
                           map<string, Layer *> &output_node_map,
                           LOG_LEVEL log_level,
                           int dev,
                           int mem);

/*
 * ONNX EXPORT
 */

// OPSET: 11, 1
void build_squeeze_node(LSqueeze *layer, onnx::GraphProto *graph);

void squeeze_node_builder(string node_name, string input, string output, vector<int> axes, onnx::GraphProto *graph);

#endif // EDDL_SQUEEZE_ONNX_H
#endif // cPROTO
