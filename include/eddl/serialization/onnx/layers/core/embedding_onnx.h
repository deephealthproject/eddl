#if defined(cPROTO)
#ifndef EDDL_EMBEDDING_ONNX_H
#define EDDL_EMBEDDING_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/core/layer_core.h"

/*
 * ONNX EXPORT
 */

// Implemented with Gather Op for OPSET: 13, 11, 1
void build_embedding_node(LEmbedding *layer, onnx::GraphProto *graph);

#endif // EDDL_EMBEDDING_ONNX_H
#endif // cPROTO
