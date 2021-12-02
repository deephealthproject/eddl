#if defined(cPROTO)
#ifndef EDDL_DENSE_ONNX_H
#define EDDL_DENSE_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/core/layer_core.h"
#include "eddl/serialization/onnx/utils_onnx.h"

/*
 * ONNX EXPORT
 */

// OPSET: 13, 11
Layer* build_dense_layer(onnx::NodeProto *node,
                         map<string, vector<float>> &map_init_values,
                         map<string, vector<int>> &map_init_dims,
                         map<string, Layer *> &output_node_map,
                         LOG_LEVEL log_level,
                         int dev,
                         int mem);

/*
 * ONNX EXPORT
 */

// OPSET: 13, 11
void build_gemm_node(LDense *layer, onnx::GraphProto *graph, bool gradients);

// MatMul OPSET: 13, 9, 1 - Add OPSET: 13, 7
//   Note: For recurrent nets we export the dense layer with a matrix multiplication
//         followed by an additon in case of using bias
void build_dense_with_matmul_node(LDense *layer, onnx::GraphProto *graph, bool gradients);

/*
 * DISTRIBUTED TRAINING
 */

void update_dense_weights(LDense *layer, vector<Tensor *> weights);

void apply_grads_to_dense(LDense *layer, vector<Tensor *> grads);

#endif // EDDL_DENSE_ONNX_H
#endif // cPROTO
