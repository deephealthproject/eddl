#if defined(cPROTO)
#ifndef EDDL_CONV3D_ONNX_H
#define EDDL_CONV3D_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/conv/layer_conv.h"

/*
 * ONNX EXPORT
 */

// OPSET: 11, 1
//   Note: same operator ('Conv') than Conv2D and Conv1D, but the handle is different
void build_conv3D_node(LConv3D *layer, onnx::GraphProto *graph, bool gradients);

#endif // EDDL_CONV3D_ONNX_H
#endif // cPROTO
