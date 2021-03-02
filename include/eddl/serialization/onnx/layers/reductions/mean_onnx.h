#if defined(cPROTO)
#ifndef EDDL_MEAN_ONNX_H
#define EDDL_MEAN_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/reductions/layer_reductions.h"

/*
 * ONNX IMPORT
 */

// OPSET: 13, 11, 1
Layer* build_rmean_layer(onnx::NodeProto *node,
                         map<string, Layer *> &output_node_map,
                         int dev,
                         int mem);

/*
 * ONNX EXPORT
 */

// OPSET: 13, 11, 1
void build_rmean_node(LRMean *layer, onnx::GraphProto *graph);

#endif // EDDL_MEAN_ONNX_H
#endif // cPROTO
