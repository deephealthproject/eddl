#if defined(cPROTO)
#ifndef EDDL_EMBEDDING_ONNX_H
#define EDDL_EMBEDDING_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/core/layer_core.h"

/*
 * ONNX EXPORT
 */

// Implemented with Gather Op for OPSET: 13, 11, 1
void build_embedding_node(LEmbedding *layer, onnx::GraphProto *graph, bool gradients = false);

/*
 * DISTRIBUTED TRAINING
 */

vector<Tensor *> get_embedding_tensors(onnx::NodeProto &node,
                                       map<string, vector<float>> &map_init_values,
                                       map<string, vector<int>> &map_init_dims);

#endif // EDDL_EMBEDDING_ONNX_H
#endif // cPROTO
