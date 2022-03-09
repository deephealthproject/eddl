#if defined(cPROTO)
#ifndef EDDL_TOPK_ONNX_H
#define EDDL_TOPK_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/auxiliar/layer_auxiliar.h"

/*
 * ONNX IMPORT
 */

// OPSET: ????? 
Layer* build_topk_layer(onnx::NodeProto *node,
                          map<string, vector<float>> &map_init_values,
			  map<string, vector<int>> &map_init_dims,
                          map<string, Layer *> &output_node_map,
                          int dev,
                          int mem);

/*
 * ONNX EXPORT
 */

// OPSET: ???
void build_topk_node(LTopK *layer, onnx::GraphProto *graph);

#endif // EDDL_TOPK_ONNX_H
#endif // cPROTO
