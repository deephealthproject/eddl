#if defined(cPROTO)
#ifndef EDDL_UPSAMPLING_ONNX_H
#define EDDL_UPSAMPLING_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/conv/layer_conv.h"

/*
 * ONNX EXPORT
 */

// OPSET: 9
void build_upsample_node(LUpSampling *layer, onnx::GraphProto *graph);

#endif // EDDL_UPSAMPLING_ONNX_H
#endif // cPROTO
