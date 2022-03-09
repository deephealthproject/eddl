#if defined(cPROTO)
#ifndef EDDL_CONCAT_ONNX_H
#define EDDL_CONCAT_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/merge/layer_merge.h"

/*
 * ONNX EXPORT
 */

// OPSET: 13, 11, 4, 1
Layer* build_concat_layer(onnx::NodeProto *node,
                          map<string, Layer *> &output_node_map,
                          bool is_recurrent,
                          int dev,
                          int mem);

/*
 * ONNX EXPORT
 */

// OPSET: 13, 11, 4, 1
void build_concat_node(LConcat *layer, onnx::GraphProto *graph, int seq_len = 0);

#endif // EDDL_CONCAT_ONNX_H
#endif // cPROTO
