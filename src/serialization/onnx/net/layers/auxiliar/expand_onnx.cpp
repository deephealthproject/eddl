#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/auxiliar/expand_onnx.h"

// ONNX import
Layer* build_expand_layer(onnx::NodeProto *node,
                          map<string, vector<float>> &map_init_values,
                          map<string, Layer *> &output_node_map,
                          int dev,
                          int mem)
{
  string parent_name = node->input(0);
  Layer *parent = output_node_map[parent_name];
  vector<int> parent_shape = parent->output->getShape();

  string shape_name = node->input(1);
  vector<float> shape_values = map_init_values[shape_name];

  int size = -1; // New target value to expand the axes

  // Look for an axis with dim=1 to expand and the new dim for it
  for (int i = 1; i < parent_shape.size(); ++i)
  {
    if (parent_shape[i] == 1)
    {
      if (size == -1)
        size = static_cast<int>(shape_values[i]);
      else if (size != shape_values[i])
        msg("Error: In node " + node->name() + ", detected more than one axis with dim 1 to expand but with different "
            "target sizes to expand them. All the dim=1 axes must be expanded to the same size.", "[ONNX::ImportNet]");
    }
  }

  if (size == -1)
    msg("Error in Expand node: There aren't axes with dimension 1 in the input shape to expand", "[ONNX::ImportNet]");

  return new LExpand(parent, size, node->name(), dev, mem);
}

// ONNX export
void build_expand_node(LExpand *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto* node = graph->add_node();
  node->set_op_type("Expand");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer* parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  string shape_name(layer->name + "_shape");
  node->add_input(shape_name);

  // Create the shape initializer
  onnx::TensorProto *shape_data = graph->add_initializer();
  shape_data->set_name(shape_name);
  shape_data->set_data_type(onnx::TensorProto::INT64);
  shape_data->add_dims(layer->sd->oshape.size());
  for (int i : layer->sd->oshape)
      shape_data->add_int64_data(i);

  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

#endif // defined(cPROTO)
