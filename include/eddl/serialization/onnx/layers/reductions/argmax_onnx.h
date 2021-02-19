#if defined(cPROTO)
#ifndef EDDL_ARGMAX_ONNX_H
#define EDDL_ARGMAX_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/reductions/layer_reductions.h"

/*
 * ONNX EXPORT
 */

// OPSET: 13, 12, 11, 1
void build_rargmax_node(LRArgmax *layer, onnx::GraphProto *graph);

#endif // EDDL_ARGMAX_ONNX_H
#endif // cPROTO
