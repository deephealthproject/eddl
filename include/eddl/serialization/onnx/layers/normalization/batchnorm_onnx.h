#if defined(cPROTO)
#ifndef EDDL_BATCHNORM_ONNX_H
#define EDDL_BATCHNORM_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/normalization/layer_normalization.h"

/*
 * ONNX EXPORT
 */

// OPSET: 9
void build_batchnorm_node(LBatchNorm *layer, onnx::GraphProto *graph);

#endif // EDDL_BATCHNORM_ONNX_H
#endif // cPROTO
