#if defined(cPROTO)
#ifndef EDDL_SELECT_ONNX_H
#define EDDL_SELECT_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/core/layer_core.h"

/*
 * ONNX IMPORT
 */

// OPSET: 13, 11, 10
Layer* build_select_layer(onnx::NodeProto *node,
                          map<string, vector<float>> &map_init_values,
                          map<string, Layer *> &output_node_map,
                          int dev,
                          int mem);

/*
 * ONNX EXPORT
 */

// OPSET: 13, 11, 10
void build_select_node(LSelect *layer, onnx::GraphProto *graph);

#endif // EDDL_SELECT_ONNX_H
#endif // cPROTO
