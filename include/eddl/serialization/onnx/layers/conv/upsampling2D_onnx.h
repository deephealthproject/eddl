#if defined(cPROTO)
#ifndef EDDL_UPSAMPLING2D_ONNX_H
#define EDDL_UPSAMPLING2D_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/conv/layer_conv.h"

/*
 * ONNX EXPORT
 */

// OPSET: 9 (Op deprecated in ONNX)
Layer* build_upsampling2D_layer(onnx::NodeProto *node,
                                map<string, vector<float>> &map_init_values,
                                map<string, vector<int>> &map_init_dims,
                                map<string, Layer *> &output_node_map,
                                int dev,
                                int mem);

/*
 * ONNX EXPORT
 */

// OPSET: 9
void build_upsample_node(LUpSampling *layer, onnx::GraphProto *graph);

#endif // EDDL_UPSAMPLING2D_ONNX_H
#endif // cPROTO
