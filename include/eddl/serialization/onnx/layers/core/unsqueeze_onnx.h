#if defined(cPROTO)
#ifndef EDDL_UNSQUEEZE_ONNX_H
#define EDDL_UNSQUEEZE_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/core/layer_core.h"

/*
 * ONNX EXPORT
 */

// OPSET: 11, 1
void build_unsqueeze_node(string node_name, string input, string output, vector<int> axes, onnx::GraphProto *graph);

#endif // EDDL_UNSQUEEZE_ONNX_H
#endif // cPROTO
