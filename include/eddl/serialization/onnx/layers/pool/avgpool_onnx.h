#if defined(cPROTO)
#ifndef EDDL_AVGPOOL_ONNX_H
#define EDDL_AVGPOOL_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/pool/layer_pool.h"

/*
 * ONNX EXPORT
 */

// OPSET: 11, 10, 7, 1
void build_averagepool_node(LAveragePool *layer, onnx::GraphProto *graph);

#endif // EDDL_AVGPOOL_ONNX_H
#endif // cPROTO
