#if defined(cPROTO)
#ifndef EDDL_ARGMAX_ONNX_H
#define EDDL_ARGMAX_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/reductions/layer_reductions.h"

/*
 * ONNX IMPORT
 */

// OPSET: 13, 12, 11, 1
Layer* build_rargmax_layer(onnx::NodeProto *node,
                           map<string, Layer *> &output_node_map,
                           bool is_recurrent,
                           int dev,
                           int mem);

/*
 * ONNX EXPORT
 */

// OPSET: 13, 12, 11, 1
void build_rargmax_node(LRArgmax *layer, onnx::GraphProto *graph);

#endif // EDDL_ARGMAX_ONNX_H
#endif // cPROTO
