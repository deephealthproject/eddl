#if defined(cPROTO)
#ifndef EDDL_ACTIVATION_ONNX_H
#define EDDL_ACTIVATION_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/core/layer_core.h"

/*
 * ONNX EXPORT
 */

// OPSET: 14, 13, 6
void build_relu_node(LActivation *layer, onnx::GraphProto *graph);

// OPSET: 13, 6
void build_sigmoid_node(LActivation *layer, onnx::GraphProto *graph);

// OPSET: 6
void build_hard_sigmoid_node(LActivation *layer, onnx::GraphProto *graph);

// OPSET: 13, 6
void build_tanh_node(LActivation *layer, onnx::GraphProto *graph);

// Not in ONNX: Custom operator
void build_exponential_node(LActivation *layer, onnx::GraphProto *graph);

// Not in ONNX: Custom operator
void build_linear_node(LActivation *layer, onnx::GraphProto *graph);

// OPSET: 6
void build_leaky_relu_node(LActivation *layer, onnx::GraphProto *graph);

// OPSET: 10
void build_thresholded_relu_node(LActivation *layer, onnx::GraphProto *graph);

// OPSET: 6
void build_elu_node(LActivation *layer, onnx::GraphProto *graph);

// OPSET: 6
void build_selu_node(LActivation *layer, onnx::GraphProto *graph);

// OPSET: 13, 11, 1
void build_softmax_node(LActivation *layer, onnx::GraphProto *graph);

// OPSET: 1
void build_softsign_node(LActivation *layer, onnx::GraphProto *graph);

// OPSET: 1
void build_softplus_node(LActivation *layer, onnx::GraphProto *graph);

#endif // EDDL_ACTIVATION_ONNX_H
#endif // cPROTO
