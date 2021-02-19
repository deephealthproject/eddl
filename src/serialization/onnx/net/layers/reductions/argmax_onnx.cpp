#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/reductions/argmax_onnx.h"

void build_rargmax_node(LRArgmax *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("ArgMax");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }

  // Attr axis
  onnx::AttributeProto *axis_attr = node->add_attribute();
  axis_attr->set_name("axis");
  axis_attr->set_type(onnx::AttributeProto::INT);
  axis_attr->set_i(layer->axis[0] + 1);

  // Attr keepdims
  onnx::AttributeProto *keepdims_attr = node->add_attribute();
  keepdims_attr->set_name("keepdims");
  keepdims_attr->set_type(onnx::AttributeProto::INT);
  keepdims_attr->set_i((int)layer->keepdims);

  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

#endif // defined(cPROTO)
