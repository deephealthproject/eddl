#if defined(cPROTO)
#ifndef EDDL_LOG_ONNX_H
#define EDDL_LOG_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/operators/layer_operators.h"

/*
 * ONNX EXPORT
 */

// OPSET: 13, 6
void build_log_node(LLog *layer, onnx::GraphProto *graph);

#endif // EDDL_LOG_ONNX_H
#endif // cPROTO
