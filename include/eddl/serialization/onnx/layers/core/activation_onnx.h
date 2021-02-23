#if defined(cPROTO)
#ifndef EDDL_ACTIVATION_ONNX_H
#define EDDL_ACTIVATION_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/core/layer_core.h"

/*
 * ONNX IMPORT
 */

// OPSET: 14, 13, 6
Layer* build_relu_layer(onnx::NodeProto *node,
                        map<string, Layer *> &output_node_map,
                        int dev,
                        int mem);

// OPSET: 13, 6
Layer* build_sigmoid_layer(onnx::NodeProto *node,
                           map<string, Layer *> &output_node_map,
                           int dev,
                           int mem);

// OPSET: 6
Layer* build_hard_sigmoid_layer(onnx::NodeProto *node,
                               map<string, Layer *> &output_node_map,
                               int dev,
                               int mem);

// OPSET: 13, 6
Layer* build_tanh_layer(onnx::NodeProto *node,
                        map<string, Layer *> &output_node_map,
                        int dev,
                        int mem);

// Not in ONNX: Custom operator
Layer* build_exponential_layer(onnx::NodeProto *node,
                               map<string, Layer *> &output_node_map,
                               int dev,
                               int mem);

// Not in ONNX: Custom operator
Layer* build_linear_layer(onnx::NodeProto *node,
                          map<string, Layer *> &output_node_map,
                          int dev,
                          int mem);

// OPSET: 6
Layer* build_leaky_relu_layer(onnx::NodeProto *node,
                              map<string, Layer *> &output_node_map,
                              int dev,
                              int mem);

// OPSET: 10
Layer* build_thresholded_relu_layer(onnx::NodeProto *node,
                                    map<string, Layer *> &output_node_map,
                                    int dev,
                                    int mem);

// OPSET: 6
Layer* build_elu_layer(onnx::NodeProto *node,
                       map<string, Layer *> &output_node_map,
                       int dev,
                       int mem);

// OPSET: 6
Layer* build_selu_layer(onnx::NodeProto *node,
                       map<string, Layer *> &output_node_map,
                       int dev,
                       int mem);

// OPSET: 1
Layer* build_softsign_layer(onnx::NodeProto *node,
                            map<string, Layer *> &output_node_map,
                            int dev,
                            int mem);

// OPSET: 1
Layer* build_softplus_layer(onnx::NodeProto *node,
                            map<string, Layer *> &output_node_map,
                            int dev,
                            int mem);

// OPSET: 11, 1
Layer* build_softmax_layer(onnx::NodeProto *node,
                           map<string, Layer *> &output_node_map,
                           int dev,
                           int mem);

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
