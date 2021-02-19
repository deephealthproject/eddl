#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/recurrent/cps_onnx.h"
#include "eddl/serialization/onnx/layers/core/unsqueeze_onnx.h"
#include "eddl/serialization/onnx/layers/onnx_helpers/export_helpers.h"

void handle_copy_states(LCopyStates *layer, onnx::GraphProto *graph)
{
  string parent_name = layer->parent[0]->name;
  string child_name = layer->child[0]->name;

  // Set the node to copy the hidden (h) state
  string node_name = parent_name + "_to_" + child_name + "_CopyState_h";
  string input_name = parent_name;
  string output_name = layer->name + "_h";
  /*
   * Add an Unsqueeze layer to reshape the h state to the desired shape for LSTM.
   *
   *   Note: The h state coming from the previous LSTM has been squeezed, so we
   *         have to unsqueeze it to get the desired shape for the decoder LSTM
   */
  build_unsqueeze_node(
      layer->name + "_h_unsqueeze", // node name
      input_name,                   // input name
      output_name,                  // Output name
      {0},                          // axes to squeeze
      graph);

  // Set the node to copy the cell (c) state in case of LSTM
  if (LLSTM *l = dynamic_cast<LLSTM *>(layer->parent[0]))
  {
    node_name = parent_name + "_to_" + child_name + "_CopyState_c";
    input_name = parent_name + "_Y_c";
    output_name = layer->name + "_c";
    build_identity_node(node_name, input_name, output_name, graph);
  }
}

#endif // defined(cPROTO)
