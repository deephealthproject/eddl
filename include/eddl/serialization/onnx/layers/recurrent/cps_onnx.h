#if defined(cPROTO)
#ifndef EDDL_CPS_ONNX_H
#define EDDL_CPS_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/layers/recurrent/layer_recurrent.h"

/*
 * ONNX EXPORT
 */

// Not an ONNX operator. Built from unsqueeze and identity operators.
void handle_copy_states(LCopyStates *layer, onnx::GraphProto *graph);

#endif // EDDL_CPS_ONNX_H
#endif // cPROTO
