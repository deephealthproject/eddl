#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/reductions/min_onnx.h"

void build_rmin_node(LRMin *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("ReduceMin");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }

  // Attr axes
  onnx::AttributeProto *axes_attr = node->add_attribute();
  axes_attr->set_name("axes");
  axes_attr->set_type(onnx::AttributeProto::INTS);
  for (int ax : layer->axis)
    axes_attr->add_ints(ax + 1);

  // Attr keepdims
  onnx::AttributeProto *keepdims_attr = node->add_attribute();
  keepdims_attr->set_name("keepdims");
  keepdims_attr->set_type(onnx::AttributeProto::INT);
  keepdims_attr->set_i(layer->keepdims);

  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

#endif // defined(cPROTO)
