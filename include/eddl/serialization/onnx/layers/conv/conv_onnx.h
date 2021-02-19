#if defined(cPROTO)
#ifndef EDDL_CONV_ONNX_H
#define EDDL_CONV_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/conv/layer_conv.h"

/*
 * ONNX EXPORT
 */

// OPSET: 11, 1
void build_conv_node(LConv *layer, onnx::GraphProto *graph, bool gradients);

#endif // EDDL_CONV_ONNX_H
#endif // cPROTO
