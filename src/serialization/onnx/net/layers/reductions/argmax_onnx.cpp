#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/reductions/argmax_onnx.h"

// ONNX import
Layer* build_rargmax_layer(onnx::NodeProto *node,
                           map<string, Layer *> &output_node_map,
                           int dev,
                           int mem)
{
  int axis = 1;
  bool keepdims = 1;
  for (int j = 0; j < node->attribute_size(); j++)
  {
    onnx::AttributeProto attribute = node->attribute(j);
    string attr_name = attribute.name();
    if (!attr_name.compare("axis"))
    {
      axis = attribute.i();
    }
    else if (!attr_name.compare("keepdims"))
    {
      keepdims = attribute.i();
    }
    //else if (!attr_name.compare("select_last_index")) {  Not implemented in EDDL
    //}
    else
      printf("Error with Argmax attributes. Attribute name is: %s\n", attr_name.c_str());
  }

  string parent_name = node->input(0);
  Layer *parent = output_node_map[parent_name];

  // Prepare the axis for EDDL. Because in EDDL you can't reduce the batch axis (0).
  if (axis > 0)
    axis--;
  else if (axis == 0)
    msg("You can't select the batch axis in Arg Max layer.", "ONNX::ImportNet");
  else
  {
    // From negative to positive axis value
    int parent_out_rank = parent->getShape().size();
    axis = parent_out_rank + axis;

    axis--;
  }

  return new LRArgmax(parent, {axis}, keepdims, node->name(), dev, mem);
}

// ONNX export
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
