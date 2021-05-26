#if defined(cPROTO)
#ifndef EDDL_SPLIT_ONNX_H
#define EDDL_SPLIT_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/core/layer_core.h"

/*
 * ONNX IMPORT
 */

// OPSET: 11, 2, 1
Layer* build_split_layer(onnx::NodeProto *node,
                         map<string, vector<float>> &map_init_values,
                         map<string, Layer *> &output_node_map,
                         int dev,
                         int mem);

/*
 * ONNX EXPORT
 */

// This layer is exported using Slice operators because the Split layer in EDDL creates Select layers
// that are the ones detected and exported.

#endif // EDDL_SPLIT_ONNX_H
#endif // cPROTO
