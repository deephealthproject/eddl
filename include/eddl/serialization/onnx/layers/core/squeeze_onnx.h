#if defined(cPROTO)
#ifndef EDDL_SQUEEZE_ONNX_H
#define EDDL_SQUEEZE_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/core/layer_core.h"

/*
 * ONNX EXPORT
 */

// OPSET: 11, 1
void build_squeeze_node(string node_name, string input, string output, vector<int> axes, onnx::GraphProto *graph);

#endif // EDDL_SQUEEZE_ONNX_H
#endif // cPROTO
