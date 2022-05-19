#if defined(cPROTO)
#ifndef EDDL_DEQUANTIZE_LINEAR_ONNX_H
#define EDDL_DEQUANTIZE_LINEAR_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/auxiliar/layer_auxiliar.h"
#include "eddl/layers/core/layer_core.h"
#include "eddl/serialization/onnx/utils_onnx.h"

/*
 * ONNX IMPORT
 */

// OPSET: 13, 8 
Layer* build_dequantize_linear_layer(onnx::NodeProto *node,
                          map<string, vector<float>> &map_init_values,
			              map<string, vector<int>> &map_init_dims,
                          map<string, Layer *> &output_node_map,
                          int dev,
                          int mem);

/*
 * ONNX EXPORT
 */

// OPSET: 13, 8
void build_dequantize_linear_node(LDequantizeLinear *layer, onnx::GraphProto *graph);

#endif // EDDL_DEQUANTIZE_LINEAR_ONNX_H
#endif // cPROTO
