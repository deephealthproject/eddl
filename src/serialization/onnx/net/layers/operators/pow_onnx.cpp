#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/operators/pow_onnx.h"

// ONNX import
Layer* build_pow_layer(onnx::NodeProto *node,
                       map<string, vector<float>> &map_init_values,
                       map<string, Layer *> &output_node_map,
                       int dev,
                       int mem)
{
  string base_name = node->input(0);
  Layer *base = output_node_map[base_name];

  string exponent_name = node->input(1);
  float exponent = 1.0;
  if(map_init_values.count(exponent_name))
  {
    vector<float> exponents = map_init_values[exponent_name];
    if (exponents.size() == 1)
    {
      exponent = exponents[0];
    }
    else
    {
      msg("Error: The exponent of the Pow layer " + node->name() + " is not valid", "ONNX::ImportNet");
      return nullptr;
    }
  }
  else
  {
    msg("Error: Exponent of the Pow layer " + node->name() + " not found", "ONNX::ImportNet");
    return nullptr;
  }

  return new LPow(base, exponent, node->name(), dev, mem);
}

// ONNX export
void build_pow_node(LPow *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto* node = graph->add_node();
  node->set_op_type("Pow");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer* parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  string exp_name(layer->name + "_exponent");
  node->add_input(exp_name); // Add the exponent initializer as input

  // Create the exponent value initializer
  onnx::TensorProto *exp_value = graph->add_initializer();
  exp_value->set_name(exp_name);
  exp_value->set_data_type(onnx::TensorProto::FLOAT);
  exp_value->add_float_data(layer->exponent);

  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

#endif // defined(cPROTO)
