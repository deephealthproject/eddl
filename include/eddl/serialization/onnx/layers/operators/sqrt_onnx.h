#if defined(cPROTO)
#ifndef EDDL_SQRT_ONNX_H
#define EDDL_SQRT_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/operators/layer_operators.h"

/*
 * ONNX EXPORT
 */

// OPSET: 13, 6
void build_sqrt_node(LSqrt *layer, onnx::GraphProto *graph);

#endif // EDDL_SQRT_ONNX_H
#endif // cPROTO
