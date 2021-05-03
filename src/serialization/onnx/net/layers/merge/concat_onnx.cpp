#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/merge/concat_onnx.h"

// ONNX import
Layer* build_concat_layer(onnx::NodeProto *node,
                          map<string, Layer *> &output_node_map,
                          int dev,
                          int mem)
{
  int axis = 1;
  for (int j = 0; j < node->attribute_size(); j++)
  {
    onnx::AttributeProto attribute = node->attribute(j);
    string attr_name = attribute.name();
    if (!attr_name.compare("axis"))
    {
      axis = attribute.i();
    }
    else
      printf("Error with concat attributes. Attribute name is: %s\n", attr_name.c_str());
  }
  vector<Layer *> parents;
  string parent_name;
  for (int j = 0; j < node->input_size(); j++)
  {
    parent_name = node->input(j);
    parents.push_back(output_node_map[parent_name]);
  }

  // Convert axis to a positive value
  if (axis < 0)
  {
    int n_input_dims = parents[0]->output->getShape().size();
    axis = n_input_dims + axis;
    // Example:
    //   parent_out: [b, c, h, w] -> n_input_dims = 4
    //   axis = -3 (to select "c" dimension)
    //   Compute new value: axis = 4 + (-3) = 1
  }

  // Sanity check
  if (axis == 0)
    msg("Error in Concat layer " + node->name() + ". Concat by the dimension 0 is not allowed.", "ONNX::ImportNet");

  axis = axis - 1; // Convert axis to avoid batch dimension

  return new LConcat(parents, axis, node->name(), dev, mem);
}

// ONNX export
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
  concat_axis->set_i(layer->axis + 1);
}

#endif // defined(cPROTO)
