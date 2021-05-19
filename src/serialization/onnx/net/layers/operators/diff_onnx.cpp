#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/operators/diff_onnx.h"

// ONNX import
Layer* build_diff_layer(onnx::NodeProto *node,
                        map<string, vector<float>> &map_init_values,
                        map<string, Layer *> &output_node_map,
                        int dev,
                        int mem)
{
  string first_operator_name = node->input(0);
  string second_operator_name = node->input(1);

  if(map_init_values.count(first_operator_name)) // k - T
  {
    // Read the input scalar
    vector<float> scalars = map_init_values[first_operator_name];
    if (scalars.size() != 1)
    {
      msg("Error: The first input operand of the Diff layer " + node->name() + " is not valid", "ONNX::ImportNet");
      return nullptr;
    }
    Layer *second_operator = output_node_map[second_operator_name];
    return new LDiff(scalars[0], second_operator, node->name(), dev, mem);
  }
  else if(map_init_values.count(second_operator_name)) // T - k
  {
    // Read the input scalar
    vector<float> scalars = map_init_values[second_operator_name];
    if (scalars.size() != 1)
    {
      msg("Error: The second input operand of the Diff layer " + node->name() + " is not valid", "ONNX::ImportNet");
      return nullptr;
    }
    Layer *first_operator = output_node_map[first_operator_name];
    return new LDiff(first_operator, scalars[0], node->name(), dev, mem);
  }
  else // T - T
  {
    Layer *first_operator = output_node_map[first_operator_name];
    Layer *second_operator = output_node_map[second_operator_name];
    return new LDiff(first_operator, second_operator, node->name(), dev, mem);
  }
}

// ONNX export
void build_sub_node(LDiff *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Sub");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  if (layer->binary)
  {
    for (Layer *parentl : layer->parent)
      node->add_input(parentl->name);
  }
  else
  {
    // Prepare the scalar operator
    string value_name(layer->name + "_value");
    node->add_input(value_name); // Add the value initializer as input
    // Create the value initializer
    onnx::TensorProto *diff_value = graph->add_initializer();
    diff_value->set_name(value_name);
    diff_value->set_data_type(onnx::TensorProto::FLOAT);
    diff_value->add_float_data(layer->val);
    if (layer->left)
    {
      node->add_input(layer->parent[0]->name);
      node->add_input(value_name);
    }
    else
    {
      node->add_input(value_name);
      node->add_input(layer->parent[0]->name);
    }
  }

  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

#endif // defined(cPROTO)
