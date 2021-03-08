#if defined(cPROTO)
#ifndef EDDL_MAX_ONNX_H
#define EDDL_MAX_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/reductions/layer_reductions.h"

/*
 * ONNX IMPORT
 */

// OPSET: 13, 12, 11, 1
Layer* build_rmax_layer(onnx::NodeProto *node,
                        map<string, Layer *> &output_node_map,
                        int dev,
                        int mem);

/*
 * ONNX EXPORT
 */

// OPSET: 13, 12, 11, 1
void build_rmax_node(LRMax *layer, onnx::GraphProto *graph);

#endif // EDDL_MAX_ONNX_H
#endif // cPROTO
