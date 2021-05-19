#if defined(cPROTO)
#ifndef EDDL_CONVT3D_ONNX_H
#define EDDL_CONVT3D_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/conv/layer_conv.h"

/*
 * ONNX EXPORT
 */

// OPSET: 11, 1
//   Note: same operator ('ConvTranspose') than ConvT2D, but the handle is different
void build_convT3D_node(LConvT3D *layer, onnx::GraphProto *graph, bool gradients);

#endif // EDDL_CONVT3D_ONNX_H
#endif // cPROTO
