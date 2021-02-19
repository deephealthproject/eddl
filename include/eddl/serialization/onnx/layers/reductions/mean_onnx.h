#if defined(cPROTO)
#ifndef EDDL_MEAN_ONNX_H
#define EDDL_MEAN_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/reductions/layer_reductions.h"

/*
 * ONNX EXPORT
 */

// OPSET: 13, 11, 1
void build_rmean_node(LRMean *layer, onnx::GraphProto *graph);

#endif // EDDL_MEAN_ONNX_H
#endif // cPROTO
