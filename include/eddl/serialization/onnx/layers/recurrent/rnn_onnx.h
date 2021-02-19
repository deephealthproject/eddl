#if defined(cPROTO)
#ifndef EDDL_RNN_ONNX_H
#define EDDL_RNN_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/recurrent/layer_recurrent.h"

/*
 * ONNX EXPORT
 */

// OPSET: 7, 1
void build_rnn_node(LRNN *layer, onnx::GraphProto *graph);

#endif // EDDL_RNN_ONNX_H
#endif // cPROTO
