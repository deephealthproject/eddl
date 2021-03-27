#if defined(cPROTO)
#ifndef EDDL_AVGPOOL1D_ONNX_H
#define EDDL_AVGPOOL1D_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/pool/layer_pool.h"

/*
 * ONNX EXPORT
 */

// OPSET: 12, 11, 10, 8, 1
void build_averagepool1D_node(LAveragePool1D *layer, onnx::GraphProto *graph);

#endif // EDDL_AVGPOOL1D_ONNX_H
#endif // cPROTO
