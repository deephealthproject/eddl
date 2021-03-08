#if defined(cPROTO)
#ifndef EDDL_CONV_ONNX_H
#define EDDL_CONV_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/conv/layer_conv.h"

/*
 * ONNX IMPORT
 */

// To export Conv1D, Conv2D and Conv3D we use the same function because
// in onnx they use the same operator type ("Conv")
// OPSET: 11, 1
Layer* build_conv_layer(onnx::NodeProto *node,
                        map<string, vector<float>> &map_init_values,
                        map<string, vector<int>> &map_init_dims,
                        map<string, Layer *> &output_node_map,
                        int dev,
                        int mem);

/*
 * ONNX EXPORT
 */

// OPSET: 11, 1
void build_conv_node(LConv *layer, onnx::GraphProto *graph, bool gradients);

#endif // EDDL_CONV_ONNX_H
#endif // cPROTO
