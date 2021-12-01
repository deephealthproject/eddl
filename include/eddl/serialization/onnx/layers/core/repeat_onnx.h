#if defined(cPROTO)
#ifndef EDDL_REPEAT_ONNX_H
#define EDDL_REPEAT_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/serialization/onnx/utils_onnx.h"
#include "eddl/layers/core/layer_core.h"

/*
 * ONNX IMPORT
 */

// OPSET: 13, 6
Layer* build_repeat_layer(onnx::NodeProto *node,
                          map<string, onnx::NodeProto *> &constant_node_map, 
                          map<string, vector<float>> &map_init_values,
                          map<string, Layer *> &output_node_map,
                          LOG_LEVEL log_level,
                          int dev,
                          int mem);

/*
 * ONNX EXPORT
 */

// OPSET: 13, 6
void build_tile_node(LRepeat *layer, onnx::GraphProto *graph);

#endif // EDDL_REPEAT_ONNX_H
#endif // cPROTO
