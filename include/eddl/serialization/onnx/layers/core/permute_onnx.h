#if defined(cPROTO)
#ifndef EDDL_PERMUTE_ONNX_H
#define EDDL_PERMUTE_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/serialization/onnx/utils_onnx.h"
#include "eddl/layers/core/layer_core.h"

/*
 * ONNX IMPORT
 */

// OPSET: 13, 1
Layer* build_permute_layer(onnx::NodeProto *node,
                           map<string, Layer *> &output_node_map,
                           bool recurrent_net,
                           LOG_LEVEL log_level,
                           int dev,
                           int mem);

/*
 * ONNX EXPORT
 */

// OPSET: 13, 1
void build_permute_node(LPermute *layer, onnx::GraphProto *graph);

#endif // EDDL_PERMUTE_ONNX_H
#endif // cPROTO
