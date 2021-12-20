#if defined(cPROTO)
#ifndef EDDL_MATMUL_ONNX_H
#define EDDL_MATMUL_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/serialization/onnx/utils_onnx.h"
#include "eddl/layers/merge/layer_merge.h"

/*
 * ONNX IMPORT
 */

// OPSET: 13, 9, 1
Layer* build_matmul_layer(onnx::NodeProto *node,
                          map<string, vector<float>> &map_init_values,
                          map<string, vector<int>> &map_init_dims,
                          map<string, Layer *> &output_node_map,
                          int dev,
                          int mem);

/*
 * DISTRIBUTED TRAINING
 */

vector<Tensor *> get_matmul_tensors(onnx::NodeProto &node,
                                    map<string, vector<float>> &map_init_values,
                                    map<string, vector<int>> &map_init_dims);
#endif // EDDL_MATMUL_ONNX_H
#endif // cPROTO
