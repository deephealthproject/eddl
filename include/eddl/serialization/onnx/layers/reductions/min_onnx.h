#if defined(cPROTO)
#ifndef EDDL_MIN_ONNX_H
#define EDDL_MIN_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/reductions/layer_reductions.h"

/*
 * ONNX EXPORT
 */

// OPSET: 13, 12, 11, 1
void build_rmin_node(LRMin *layer, onnx::GraphProto *graph);

#endif // EDDL_MIN_ONNX_H
#endif // cPROTO
