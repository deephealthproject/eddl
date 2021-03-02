#if defined(cPROTO)
#ifndef EDDL_POW_ONNX_H
#define EDDL_POW_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/operators/layer_operators.h"

/*
 * ONNX IMPORT
 */

/*
// OPSET: 13, 12, 7 (TODO: Implement LPow)
Layer* build_pow_layer(onnx::NodeProto *node,
                       map<string, Layer *> &output_node_map,
                       int dev,
                       int mem);
*/

/*
 * ONNX EXPORT
 */

// OPSET: 13, 12, 7
// TODO: Implement layer LPow
//void build_pow_node( LPow *layer, onnx::GraphProto *graph );

#endif // EDDL_POW_ONNX_H
#endif // cPROTO
