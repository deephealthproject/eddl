#if defined(cPROTO)
#ifndef EDDL_ADD_ONNX_H
#define EDDL_ADD_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/serialization/onnx/utils_onnx.h"
#include "eddl/layers/merge/layer_merge.h"

/*
 * ONNX EXPORT
 */

// OPSET: 13, 7
Layer* build_add_layer(onnx::NodeProto *node,
                       map<string, vector<float>> &map_init_values,
                       map<string, vector<int>> &map_init_dims,
                       map<string, Layer *> &output_node_map,
                       LOG_LEVEL log_level,
                       int dev,
                       int mem);
/*
 * ONNX EXPORT
 */

// OPSET: 13, 7
void build_add_node(LAdd *layer, onnx::GraphProto *graph, int seq_len);

/*
 * DISTRIBUTED TRAINING
 */

vector<Tensor *> get_add_tensors(onnx::NodeProto &node,
                                 map<string, vector<float>> &map_init_values,
                                 map<string, vector<int>> &map_init_dims);

#endif // EDDL_ADD_ONNX_H
#endif // cPROTO
