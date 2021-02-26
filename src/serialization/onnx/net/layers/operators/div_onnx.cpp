#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/operators/div_onnx.h"

// ONNX import
Layer* build_div_layer(onnx::NodeProto *node,
                       map<string, Layer *> &output_node_map,
                       int dev,
                       int mem)
{
  string first_operator_name = node->input(0);
  Layer *first_operator = output_node_map[first_operator_name];

  string second_operator_name = node->input(1);
  Layer *second_operator = output_node_map[second_operator_name];

  return new LDiv(first_operator, second_operator, node->name(), dev, mem);
}

// ONNX export
void build_div_node(LDiv *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Div");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

#endif // defined(cPROTO)
