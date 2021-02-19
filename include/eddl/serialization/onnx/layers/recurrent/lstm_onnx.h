#if defined(cPROTO)
#ifndef EDDL_LSTM_ONNX_H
#define EDDL_LSTM_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/recurrent/layer_recurrent.h"

/*
 * ONNX EXPORT
 */

// OPSET: 7, 1
void build_lstm_node(LLSTM *layer, onnx::GraphProto *graph);

#endif // EDDL_LSTM_ONNX_H
#endif // cPROTO
