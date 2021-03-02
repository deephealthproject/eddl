#if defined(cPROTO)
#ifndef EDDL_RSUM_ONNX_H
#define EDDL_RSUM_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/reductions/layer_reductions.h"

/*
 * ONNX IMPORT
 */

// OPSET: 11, 1
Layer* build_rsum_layer(onnx::NodeProto *node,
                        map<string, Layer *> &output_node_map,
                        int dev,
                        int mem);

/*
 * ONNX EXPORT
 */

// OPSET: 11, 1
void build_rsum_node(LRSum *layer, onnx::GraphProto *graph);

#endif // EDDL_RSUM_ONNX_H
#endif // cPROTO
