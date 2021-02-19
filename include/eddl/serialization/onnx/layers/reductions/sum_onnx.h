#if defined(cPROTO)
#ifndef EDDL_SUM_ONNX_H
#define EDDL_SUM_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/reductions/layer_reductions.h"

/*
 * ONNX EXPORT
 */

// OPSET: 11, 1
void build_rsum_node(LRSum *layer, onnx::GraphProto *graph);

#endif // EDDL_SUM_ONNX_H
#endif // cPROTO
