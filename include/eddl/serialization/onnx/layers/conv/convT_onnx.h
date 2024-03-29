#if defined(cPROTO)
#ifndef EDDL_CONVT_ONNX_H
#define EDDL_CONVT_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/conv/layer_conv.h"

/*
 * ONNX IMPORT
 */

// To export ConvT1D, ConvT2D and ConvT3D we use the same function because
// in onnx they use the same operator type ("ConvTranspose")
// OPSET: 11, 1
Layer* build_convT_layer(onnx::NodeProto *node,
                         map<string, vector<float>> &map_init_values,
                         map<string, vector<int>> &map_init_dims,
                         map<string, Layer *> &output_node_map,
                         int dev,
                         int mem);

/*
 * ONNX EXPORT
 */

// OPSET: 11, 1
void build_convT_node(LConvT2D *layer, onnx::GraphProto *graph, bool gradients);

/*
 * DISTRIBUTED TRAINING
 */

vector<Tensor *> get_convT_tensors(onnx::NodeProto &node,
                                   map<string, vector<float>> &map_init_values,
                                   map<string, vector<int>> &map_init_dims);

#endif // EDDL_CONVT_ONNX_H
#endif // cPROTO
