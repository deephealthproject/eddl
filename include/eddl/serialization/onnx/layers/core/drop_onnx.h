#if defined(cPROTO)
#ifndef EDDL_DROP_ONNX_H
#define EDDL_DROP_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/core/layer_core.h"

/*
 * ONNX EXPORT
 */

// OPSET: 10, 7
void build_dropout_node(LDropout *layer, onnx::GraphProto *graph);

#endif // EDDL_DROP_ONNX_H
#endif // cPROTO
