#if defined(cPROTO)
#ifndef EDDL_RESHAPE_ONNX_H
#define EDDL_RESHAPE_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/core/layer_core.h"

/*
 * ONNX EXPORT
 */

// OPSET: 13, 5
void build_reshape_node(LReshape *layer, onnx::GraphProto *graph);

#endif // EDDL_RESHAPE_ONNX_H
#endif // cPROTO
