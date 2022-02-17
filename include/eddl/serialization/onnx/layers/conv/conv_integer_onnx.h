#if defined(cPROTO)
#ifndef EDDL_CONV_INTEGER_ONNX_H
#define EDDL_CONV_INTEGER_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/conv/layer_conv.h"
#include "eddl/serialization/onnx/utils_onnx.h"

/*
 * ONNX IMPORT
 */

// OPSET: ??, ??
Layer* build_conv_integer_layer(onnx::NodeProto *node,
                        map<string, vector<float>> &map_init_values,
                        map<string, vector<int>> &map_init_dims,
                        map<string, Layer *> &output_node_map,
                        LOG_LEVEL log_level,
                        int dev,
                        int mem);

/*
 * ONNX EXPORT
 */

// OPSET: 11, 1
void build_conv_integer_node(LConv *layer, onnx::GraphProto *graph, bool gradients);


#endif // EDDL_CONV_INTEGER_ONNX_H
#endif // cPROTO
