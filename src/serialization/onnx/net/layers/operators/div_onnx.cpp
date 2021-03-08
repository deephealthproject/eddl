#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/operators/div_onnx.h"

// ONNX import
Layer* build_div_layer(onnx::NodeProto *node,
                       map<string, vector<float>> &map_init_values,
                       map<string, Layer *> &output_node_map,
                       int dev,
                       int mem)
{
  string first_operator_name = node->input(0);
  Layer *first_operator = nullptr;
  float first_operator_value;
  bool first_is_value = false;
  // Check if the second input is a constant value
  if (map_init_values.count(first_operator_name))
  {
    vector<float> dividends = map_init_values[first_operator_name];
    if (dividends.size() == 1)
    {
      first_is_value = true;
      first_operator_value = dividends[0];
    }
    else
      msg("Error: The dividend input of the Div layer " + node->name() + " is not valid", "ONNX::ImportNet");
  }
  else
    first_operator = output_node_map[first_operator_name];

  string second_operator_name = node->input(1);
  Layer *second_operator = nullptr;
  float second_operator_value;
  bool second_is_value = false;
  // Check if the second input is a constant value
  if (map_init_values.count(second_operator_name))
  {
    vector<float> divisors = map_init_values[second_operator_name];
    if (divisors.size() == 1)
    {
      second_is_value = true;
      second_operator_value = divisors[0];
    }
    else
      msg("Error: The divisor input of the Div layer " + node->name() + " is not valid", "ONNX::ImportNet");
  }
  else
    second_operator = output_node_map[second_operator_name];
  
  if (first_is_value)
    return new LDiv(first_operator_value, second_operator, node->name(), dev, mem);
  else if (second_is_value)
    return new LDiv(first_operator, second_operator_value, node->name(), dev, mem);
  else
    return new LDiv(first_operator, second_operator, node->name(), dev, mem);
}

// ONNX export
void build_div_node(LDiv *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Div");
  node->set_name(layer->name);

  // Add inputs to the node
  if (!layer->binary)
  {
    if (layer->left) // Add the value first because is the dividend
    {
      string value_name(layer->name + "_value");
      node->add_input(value_name); // Add the value initializer as input
      // Create the value initializer
      onnx::TensorProto *div_value = graph->add_initializer();
      div_value->set_name(value_name);
      div_value->set_data_type(onnx::TensorProto::FLOAT);
      div_value->add_float_data(layer->val);
    }
    // Set the inputs names of the node from the parents of the layer
    for (Layer *parentl : layer->parent)
    {
      node->add_input(parentl->name);
    }
    if (!layer->left) // Add the value second because is the divisor
    {
      string value_name(layer->name + "_value");
      node->add_input(value_name); // Add the value initializer as input
      // Create the value initializer
      onnx::TensorProto *div_value = graph->add_initializer();
      div_value->set_name(value_name);
      div_value->set_data_type(onnx::TensorProto::FLOAT);
      div_value->add_float_data(layer->val);
    }
  }

  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

#endif // defined(cPROTO)
