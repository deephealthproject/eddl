#if defined(cPROTO)
#ifndef EDDL_CONV1D_ONNX_H
#define EDDL_CONV1D_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/conv/layer_conv.h"

/*
 * ONNX EXPORT
 */

// OPSET: 11, 1 (
//   Note: same operator ('Conv') than Conv2D and Conv3D, but the handle is different
void build_conv1D_node(LConv1D *layer, onnx::GraphProto *graph, bool gradients);

#endif // EDDL_CONV1D_ONNX_H
#endif // cPROTO
