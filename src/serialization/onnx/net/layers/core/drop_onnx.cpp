#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/core/drop_onnx.h"

void build_dropout_node(LDropout *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Dropout");
  node->set_name(layer->name);
  // Set the inputs of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);

  // Attr ratio
  onnx::AttributeProto *momentum_attr = node->add_attribute();
  momentum_attr->set_name("ratio");
  momentum_attr->set_type(onnx::AttributeProto::FLOAT);
  momentum_attr->set_f(layer->df);
}

#endif // defined(cPROTO)
