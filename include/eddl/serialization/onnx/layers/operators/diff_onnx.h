#if defined(cPROTO)
#ifndef EDDL_DIFF_ONNX_H
#define EDDL_DIFF_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/operators/layer_operators.h"

/*
 * ONNX IMPORT
 */

// OPSET: 13, 7
Layer* build_diff_layer(onnx::NodeProto *node,
                        map<string, vector<float>> &map_init_values,
                        map<string, Layer *> &output_node_map,
                        int dev,
                        int mem);

/*
 * ONNX EXPORT
 */

// OPSET: 13, 7
void build_sub_node(LDiff *layer, onnx::GraphProto *graph);

#endif // EDDL_DIFF_ONNX_H
#endif // cPROTO
