#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/merge/concat_onnx.h"

void build_concat_node(LConcat *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Concat");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);

  // Attr axis
  onnx::AttributeProto *concat_axis = node->add_attribute();
  concat_axis->set_name("axis");
  concat_axis->set_type(onnx::AttributeProto::INT);
  concat_axis->set_i(1);
}

#endif // defined(cPROTO)