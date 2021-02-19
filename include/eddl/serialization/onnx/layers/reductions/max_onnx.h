#if defined(cPROTO)
#ifndef EDDL_MAX_ONNX_H
#define EDDL_MAX_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/reductions/layer_reductions.h"

/*
 * ONNX EXPORT
 */

// OPSET: 13, 12, 11, 1
void build_rmax_node(LRMax *layer, onnx::GraphProto *graph);

#endif // EDDL_MAX_ONNX_H
#endif // cPROTO
