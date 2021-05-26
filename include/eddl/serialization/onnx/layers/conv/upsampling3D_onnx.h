#if defined(cPROTO)
#ifndef EDDL_UPSAMPLING3D_ONNX_H
#define EDDL_UPSAMPLING3D_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/core/layer_core.h"

/*
 * ONNX IMPORT
 */

// Handled with the LScale layer import function

/*
 * ONNX EXPORT
 */

// OPSET: 13
void build_resize_node_from_upsampling3D(LUpSampling3D *layer, onnx::GraphProto *graph);

#endif // EDDL_UPSAMPLING3D_ONNX_H
#endif // cPROTO
