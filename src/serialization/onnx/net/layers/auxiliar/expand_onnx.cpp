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

  string shape_name = node->input(1);
  vector<float> *shape_values = &(map_init_values[shape_name]);

  int size = 30; // "{5, 5, 1, 5, 1}" => "{5, 5, 30, 5, 30}"
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
//  shape_data->add_dims(layer->new_shape.size());
//  for (int i : layer->new_shape)
//      shape_data->add_int64_data(i);

  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

#endif // defined(cPROTO)
