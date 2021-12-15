#if defined(cPROTO)
#ifndef EDDL_BYPASS_ONNX_H
#define EDDL_BYPASS_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/serialization/onnx/utils_onnx.h"
#include "eddl/layers/core/layer_core.h"

/*
 * ONNX EXPORT
 */

Layer* build_lrn_layer(onnx::NodeProto *node,
                       map<string, Layer *> &output_node_map,
                       LOG_LEVEL log_level,
                       int dev,
                       int mem);
/*
 * ONNX EXPORT
 */

// OPSET: 16, 14, 13, 1
void build_identity_node(LBypass *layer, onnx::GraphProto *graph);

#endif // EDDL_BYPASS_ONNX_H
#endif // cPROTO
