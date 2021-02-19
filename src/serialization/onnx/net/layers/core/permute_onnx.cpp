#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/core/permute_onnx.h"

void build_permute_node(LPermute *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Transpose");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);

  // Attr perm
  onnx::AttributeProto *perm_attr = node->add_attribute();
  perm_attr->set_name("perm");
  perm_attr->set_type(onnx::AttributeProto::INTS);
  perm_attr->add_ints(0); // Set the batch size position. It must not be permuted in EDDL
  for (int i : layer->sd->dims)
  {
    perm_attr->add_ints(i + 1); // Add 1 to fix the batch dim adition
  }
}

#endif // defined(cPROTO)
