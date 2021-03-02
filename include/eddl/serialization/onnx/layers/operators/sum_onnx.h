#if defined(cPROTO)
#ifndef EDDL_SUM_ONNX_H
#define EDDL_SUM_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/operators/layer_operators.h"

/*
 * ONNX IMPORT
 */

// OPSET: 13, 8, 6, 1
Layer* build_sum_layer(onnx::NodeProto *node,
                       map<string, vector<float>> &map_init_values,
                       map<string, vector<int>> &map_init_dims,
                       map<string, Layer *> &output_node_map,
                       int dev,
                       int mem);

/*
 * ONNX EXPORT
 */

// OPSET: 13, 8, 6, 1
void build_sum_node(LSum *layer, onnx::GraphProto *graph);

#endif // EDDL_SUM_ONNX_H
#endif // cPROTO
