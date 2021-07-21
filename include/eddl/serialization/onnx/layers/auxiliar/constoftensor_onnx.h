#if defined(cPROTO)
#ifndef EDDL_CONSTOFTENSOR_ONNX_H
#define EDDL_CONSTOFTENSOR_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/auxiliar/layer_auxiliar.h"
#include "eddl/serialization/onnx/utils_onnx.h"

/*
 * ONNX IMPORT
 */

// OPSET: 13, 12, 11, 9, 1
Layer* build_constoftensor_layer(onnx::NodeProto *node,
                                 map<string, vector<float>> &map_init_values,
                                 map<string, Layer *> &output_node_map,
                                 int dev,
                                 int mem);

/*
 * ONNX EXPORT
 */

// OPSET: 13, 12, 11, 9, 1
void build_constant_node(LConstOfTensor *layer, onnx::GraphProto *graph);

#endif // EDDL_CONSTOFTENSOR_ONNX_H
#endif // cPROTO
