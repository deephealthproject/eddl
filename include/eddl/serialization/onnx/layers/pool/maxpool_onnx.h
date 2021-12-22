#if defined(cPROTO)
#ifndef EDDL_MAXPOOL_ONNX_H
#define EDDL_MAXPOOL_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/pool/layer_pool.h"
#include "eddl/serialization/onnx/utils_onnx.h"

/*
 * ONNX IMPORT
 */

// OPSET: 12, 11, 10, 8, 1
Layer* build_maxpool_layer(onnx::NodeProto *node,
                           map<string, Layer *> &output_node_map,
                           LOG_LEVEL log_level,
                           int dev,
                           int mem);

// OPSET: 1
Layer* build_globalmaxpool_layer(onnx::NodeProto *node,
                                 map<string, Layer *> &output_node_map,
                                 int dev,
                                 int mem);

/*
 * ONNX EXPORT
 */

// OPSET: 12, 11, 10, 8, 1
void build_maxpool_node(LMaxPool *layer, onnx::GraphProto *graph);

#endif // EDDL_MAXPOOL_ONNX_H
#endif // cPROTO
