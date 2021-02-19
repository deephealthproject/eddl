#if defined(cPROTO)
#ifndef EDDL_ADD_ONNX_H
#define EDDL_ADD_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/merge/layer_merge.h"

/*
 * ONNX EXPORT
 */

// OPSET: 13, 7
void build_add_node(LAdd *layer, onnx::GraphProto *graph);

#endif // EDDL_ADD_ONNX_H
#endif // cPROTO
