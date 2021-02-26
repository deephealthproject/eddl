#if defined(cPROTO)
#ifndef EDDL_MAXPOOL1D_ONNX_H
#define EDDL_MAXPOOL1D_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/pool/layer_pool.h"

/*
 * ONNX EXPORT
 */

// OPSET: 12, 11, 10, 8, 1
void build_maxpool1D_node(LMaxPool1D *layer, onnx::GraphProto *graph);

#endif // EDDL_MAXPOOL1D_ONNX_H
#endif // cPROTO
