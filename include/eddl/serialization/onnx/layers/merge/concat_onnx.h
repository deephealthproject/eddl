#if defined(cPROTO)
#ifndef EDDL_CONCAT_ONNX_H
#define EDDL_CONCAT_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/merge/layer_merge.h"

/*
 * ONNX EXPORT
 */

// OPSET: 13, 11, 4, 1
void build_concat_node(LConcat *layer, onnx::GraphProto *graph);

#endif // EDDL_CONCAT_ONNX_H
#endif // cPROTO
