#if defined(cPROTO)
#ifndef EDDL_GRU_ONNX_H
#define EDDL_GRU_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/recurrent/layer_recurrent.h"

/*
 * ONNX EXPORT
 */

// OPSET: 7, 3, 1
void build_gru_node(LGRU *layer, onnx::GraphProto *graph);

#endif // EDDL_GRU_ONNX_H
#endif // cPROTO
