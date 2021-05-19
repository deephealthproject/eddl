#if defined(cPROTO)
#ifndef EDDL_PAD_ONNX_H
#define EDDL_PAD_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/da/layer_da.h"

/*
 * ONNX IMPORT
 */

// OPSET: 13, 11, 2, 1
Layer* build_pad_layer(onnx::NodeProto *node,
                       map<string, vector<float>> &map_init_values,
                       map<string, Layer *> &output_node_map,
                       int dev,
                       int mem);

/*
 * ONNX EXPORT
 */

// OPSET: 13, 11
void build_pad_node(LPad *layer, onnx::GraphProto *graph);

#endif // EDDL_PAD_ONNX_H
#endif // cPROTO
