#if defined(cPROTO)
#ifndef EDDL_EXPORT_HELPERS_H
#define EDDL_EXPORT_HELPERS_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/core/layer_core.h"

/*
 * ONNX EXPORT
 */

// OPSET: 13, 1
void build_identity_node(string node_name, string input, string output, onnx::GraphProto *graph);

// OPSET: 13, 9, 6
void build_cast_node(string node_name, string input, string output, int cast_type, onnx::GraphProto *graph);

// OPSET: 13, 11, 1
void build_gather_node(string node_name, string input, string output, LEmbedding *layer, onnx::GraphProto *graph);

#endif // EDDL_EXPORT_HELPERS_H
#endif // cPROTO
