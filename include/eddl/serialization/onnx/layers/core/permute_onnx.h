#if defined(cPROTO)
#ifndef EDDL_PERMUTE_ONNX_H
#define EDDL_PERMUTE_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/core/layer_core.h"

/*
 * ONNX EXPORT
 */

// OPSET: 13, 1
void build_permute_node(LPermute *layer, onnx::GraphProto *graph);

#endif // EDDL_PERMUTE_ONNX_H
#endif // cPROTO
