#if defined(cPROTO)
#ifndef EDDL_CLAMP_ONNX_H
#define EDDL_CLAMP_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/operators/layer_operators.h"

/*
 * ONNX IMPORT
 */

// OPSET: 13, 12, 11
Layer* build_clamp_layer(onnx::NodeProto *node,
                         map<string, vector<float>> &map_init_values,
                         map<string, Layer *> &output_node_map,
                         int dev,
                         int mem);

/*
 * ONNX EXPORT
 */

// OPSET: 13, 12, 11
void build_clip_node(LClamp *layer, onnx::GraphProto *graph);

#endif // EDDL_CLAMP_ONNX_H
#endif // cPROTO
