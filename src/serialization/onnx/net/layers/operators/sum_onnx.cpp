#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/operators/mult_onnx.h"
#include "eddl/layers/normalization/layer_normalization.h"
#include "eddl/serialization/onnx/utils_onnx.h"

// ONNX import
Layer* build_sum_layer(onnx::NodeProto *node,
                       map<string, vector<float>> &map_init_values,
                       map<string, vector<int>> &map_init_dims,
                       map<string, Layer *> &output_node_map,
                       int dev,
                       int mem)
{
  string first_operator_name = node->input(0);
  Layer *first_operator = output_node_map[first_operator_name];

  string second_operator_name = node->input(1);
  if(map_init_dims.count(second_operator_name))
  {
    vector<float> scalars = map_init_values[second_operator_name];
    if (scalars.size() == 1)
    {
      return new LSum(first_operator, scalars[0], node->name(), dev, mem);
    }
    else
    {
      msg("Error: The second input summand of the Sum layer " + node->name() + " is not valid", "ONNX::ImportNet");
      return nullptr;
    }
  }

  Layer *second_operator = output_node_map[second_operator_name];

  return new LSum(first_operator, second_operator, node->name(), dev, mem);
}

// ONNX export
void build_sum_node(LSum *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Sum");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);

  if (!layer->binary)
  {
    string value_name(layer->name + "_value");
    node->add_input(value_name); // Add the value initializer as input
    // Create the value initializer
    onnx::TensorProto *sum_value = graph->add_initializer();
    sum_value->set_name(value_name);
    sum_value->set_data_type(onnx::TensorProto::FLOAT);
    sum_value->add_float_data(layer->val);
  }
}

#endif // defined(cPROTO)
