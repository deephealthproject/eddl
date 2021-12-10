#if defined(cPROTO)
#ifndef EDDL_EXPORT_NODES_H
#define EDDL_EXPORT_NODES_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/serialization/onnx/utils_onnx.h"
#include "eddl/layers/core/layer_core.h"

/*
 * ONNX IMPORT
 */

// We skip this layer when found
Layer* handle_identity_node(onnx::NodeProto *node,
                            map<string, Layer *> &output_node_map,
                            LOG_LEVEL log_level,
                            int dev,
                            int mem);

// We skip this layer when found
Layer* handle_cast_node(onnx::NodeProto *node,
                        map<string, Layer *> &output_node_map,
                        LOG_LEVEL log_level,
                        int dev,
                        int mem);

// OPSET: 13, 11, 1
Layer* handle_gather_node(onnx::NodeProto *node,
                          map<string, vector<float>> &map_init_values,
                          map<string, vector<int>> &map_init_dims,
                          map<string, Layer *> &output_node_map,
                          LOG_LEVEL log_level,
                          int dev,
                          int mem);

/*
 * ONNX EXPORT
 */

// OPSET: 13, 1
void build_identity_node(string node_name, string input, string output, onnx::GraphProto *graph);

// OPSET: 13, 9, 6
void build_cast_node(string node_name, string input, string output, int cast_type, onnx::GraphProto *graph);

// OPSET: 13, 11, 1
void build_gather_node(string node_name, string input, string output, LEmbedding *layer, onnx::GraphProto *graph, bool gradients = false);

#endif // EDDL_EXPORT_NODES_H
#endif // cPROTO
