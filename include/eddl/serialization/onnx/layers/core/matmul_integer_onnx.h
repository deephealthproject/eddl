#if defined(cPROTO)
#ifndef EDDL_MATMUL_INTEGER_ONNX_H
#define EDDL_MATMUL_INTEGER_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/core/layer_core.h"
#include "eddl/serialization/onnx/utils_onnx.h"

/*
 * ONNX EXPORT
 */

// OPSET:
Layer* build_matmul_integer_layer(onnx::NodeProto *node,
                         map<string, vector<float>> &map_init_values,
                         map<string, vector<int>> &map_init_dims,
                         map<string, Layer *> &output_node_map,
                         LOG_LEVEL log_level,
                         int dev,
                         int mem);

/*
 * ONNX EXPORT
 */

#endif // EDDL_DENSE_ONNX_H
#endif // cPROTO
