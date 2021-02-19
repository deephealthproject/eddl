#if defined(cPROTO)
#ifndef EDDL_DIV_ONNX_H
#define EDDL_DIV_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/operators/layer_operators.h"

/*
 * ONNX EXPORT
 */

// OPSET: 13, 7
void build_div_node(LDiv *layer, onnx::GraphProto *graph);

#endif // EDDL_DIV_ONNX_H
#endif // cPROTO
