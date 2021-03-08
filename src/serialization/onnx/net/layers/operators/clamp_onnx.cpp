#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/operators/clamp_onnx.h"

// ONNX import
Layer* build_clamp_layer(onnx::NodeProto *node,
                         map<string, vector<float>> &map_init_values,
                         map<string, Layer *> &output_node_map,
                         int dev,
                         int mem)
{
  string parent_name = node->input(0);
  Layer *parent = output_node_map[parent_name];

  float min_value = numeric_limits<float>::min();
  float max_value = numeric_limits<float>::max();
  // Look for min and max values of clamp operation
  if (node->input_size() > 1)
  {
    string min_name = node->input(1);
    if(map_init_values.count(min_name))
      min_value = map_init_values[min_name][0];
    else
      msg("Error: min value for Clamp not found", "ONNX::ImportNet");

    if (node->input_size() > 2)
    {
      string max_name = node->input(2);
      if(map_init_values.count(max_name))
        max_value = map_init_values[max_name][0];
      else
        msg("Error: max value for Clamp not found", "ONNX::ImportNet");
    }
  }

  return new LClamp(parent, min_value, max_value, node->name(), dev, mem);
}

// ONNX export
void build_clip_node(LClamp *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Clip");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  string min_input_name(layer->name + "_min");
  string max_input_name(layer->name + "_max");
  node->add_input(min_input_name);
  node->add_input(max_input_name);
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);

  // Create min value initializer
  onnx::TensorProto *min_value = graph->add_initializer();
  min_value->set_name(min_input_name);
  min_value->set_data_type(onnx::TensorProto::FLOAT);
  min_value->add_float_data(layer->min);
  // Create max value initializer
  onnx::TensorProto *max_value = graph->add_initializer();
  max_value->set_name(max_input_name);
  max_value->set_data_type(onnx::TensorProto::FLOAT);
  max_value->add_float_data(layer->max);
}

#endif // defined(cPROTO)
