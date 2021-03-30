#if defined(cPROTO)
#ifndef EDDL_EXPAND_ONNX_H
#define EDDL_EXPAND_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/auxiliar/layer_auxiliar.h"

/*
 * ONNX IMPORT
 */

// OPSET: 13, 8 
Layer* build_expand_layer(onnx::NodeProto *node,
                          map<string, vector<float>> &map_init_values,
                          map<string, Layer *> &output_node_map,
                          int dev,
                          int mem);

/*
 * ONNX EXPORT
 */

// OPSET: 13, 8
void build_expand_node(LExpand *layer, onnx::GraphProto *graph);

#endif // EDDL_EXPAND_ONNX_H
#endif // cPROTO
