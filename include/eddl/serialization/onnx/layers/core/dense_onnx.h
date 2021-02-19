#if defined(cPROTO)
#ifndef EDDL_DENSE_ONNX_H
#define EDDL_DENSE_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/core/layer_core.h"

/*
 * ONNX EXPORT
 */

// OPSET: 13, 11
void build_gemm_node(LDense *layer, onnx::GraphProto *graph, bool gradients);

// MatMul OPSET: 13, 9, 1 - Add OPSET: 13, 7
//   Note: For recurrent nets we export the dense layer with a matrix multiplication
//         followed by an additon in case of using bias
void build_dense_with_matmul_node(LDense *layer, onnx::GraphProto *graph, bool gradients);

#endif // EDDL_DENSE_ONNX_H
#endif // cPROTO
