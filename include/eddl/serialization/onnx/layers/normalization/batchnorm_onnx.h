#if defined(cPROTO)
#ifndef EDDL_BATCHNORM_ONNX_H
#define EDDL_BATCHNORM_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/normalization/layer_normalization.h"

/*
 * ONNX IMPORT
 */

// OPSET: 9
Layer* build_batchnorm_layer(onnx::NodeProto *node,
                             map<string, vector<float>> &map_init_values,
                             map<string, vector<int>> &map_init_dims,
                             map<string, Layer *> &output_node_map,
                             int dev,
                             int mem);

/*
 * ONNX EXPORT
 */

// OPSET: 9
void build_batchnorm_node(LBatchNorm *layer, onnx::GraphProto *graph);

#endif // EDDL_BATCHNORM_ONNX_H
#endif // cPROTO
