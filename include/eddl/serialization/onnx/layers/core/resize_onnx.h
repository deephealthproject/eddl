#if defined(cPROTO)
#ifndef EDDL_RESIZE_ONNX_H
#define EDDL_RESIZE_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/core/layer_core.h"

/*
 * ONNX IMPORT
 */

// OPSET: 13
Layer* build_resize_layer(onnx::NodeProto *node,
                          map<string, vector<float>> &map_init_values,
                          map<string, Layer *> &output_node_map,
                          int dev,
                          int mem);

/*
 * ONNX EXPORT
 */

// OPSET: 13
void build_resize_node(LResize *layer, onnx::GraphProto *graph);

#endif // EDDL_RESIZE_ONNX_H
#endif // cPROTO
