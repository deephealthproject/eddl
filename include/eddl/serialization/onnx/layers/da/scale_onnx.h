#if defined(cPROTO)
#ifndef EDDL_SCALE_ONNX_H
#define EDDL_SCALE_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/da/layer_da.h"

/*
 * ONNX EXPORT
 */

// OPSET: 13
void build_resize_node(LScale *layer, onnx::GraphProto *graph);

#endif // EDDL_SCALE_ONNX_H
#endif // cPROTO
