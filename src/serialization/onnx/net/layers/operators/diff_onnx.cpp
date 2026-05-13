#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/operators/diff_onnx.h"
// ONNX import
Layer* build_diff_layer(onnx::NodeProto *node,
                        map<string, vector<float>> &map_init_values,
                        map<string, vector<int>> &map_init_dims,
                        map<string, Layer *> &output_node_map,
                        int dev,
                        int mem)
{
  string first_operator_name = node->input(0);
  string second_operator_name = node->input(1);
  vector<float> first_operator_scalars;
  vector<float> second_operator_scalars;

  if (map_init_values.count(first_operator_name)) 
  {
    // los valores que nos interesan están aquí
    first_operator_scalars = map_init_values[first_operator_name];
  }
  else if (map_init_values.count(second_operator_name)) {
    second_operator_scalars = map_init_values[second_operator_name];
  }

  if (!first_operator_scalars.empty() && first_operator_scalars.size() == 1)         // k - T
  {
    Layer *second_operator = output_node_map[second_operator_name];
    return new LDiff(first_operator_scalars[0], second_operator, node->name(), dev, mem);
  } 
  else if (!second_operator_scalars.empty() && second_operator_scalars.size() == 1)  // T - k
  {
    Layer *first_operator = output_node_map[first_operator_name];
    return new LDiff(first_operator, second_operator_scalars[0], node->name(), dev, mem);
  }
  else if (!first_operator_scalars.empty() && first_operator_scalars.size() != 1)
  {
    // Este caso se da cuando el primer operador es un Tensor constante (no escalar)
    Layer *first_operator  = output_node_map[first_operator_name];
    Layer *second_operator = output_node_map[second_operator_name];   
    vector<int> shape_first_op  = map_init_dims[first_operator_name];
    vector<int> shape_second_op = second_operator->getShape();
    Tensor* constant_tensor;

    // Same shape and shame batch size
    if (shape_first_op.size() == shape_second_op.size() && shape_first_op[0] == shape_second_op[0]) {
      constant_tensor = new Tensor(first_operator_scalars, shape_first_op);
    } 
    else if (shape_first_op.size() == shape_second_op.size() && shape_first_op[0] != shape_second_op[0]) {
      // Assuming that the batch size is the one different, but check the other dimensions anyway
      for (size_t i = 1; i < shape_first_op.size(); ++i) {
        if (shape_first_op[i] != shape_second_op[i])
          msg("Error: The second input operand of the Diff layer " + node->name() + " is not valid", "ONNX::ImportNet");
      }

      constant_tensor = new Tensor(first_operator_scalars, shape_first_op);
      constant_tensor = Tensor::repeat(constant_tensor, shape_second_op[0], 0);
    }
    else {
      msg("Error: The second input operand of the Diff layer " + node->name() + " is not valid", "ONNX::ImportNet");
    }

    return new LDiff(constant_tensor, second_operator, node->name(), dev, mem);
  }
  // else if (!second_operator_scalars.empty() && second_operator_scalars.size() != 1)
  // {
  //   std::cout << "LDiff build_diff_layer 4ª cuando el segundo operador es un Tensor constante (no escalar) //TODO" << std::endl;
  //   // Este caso se da cuando el primer operador es un Tensor constante (no escalar)
  //   // Layer *second_operator = output_node_map[second_operator_name];
  //   // vector<int> shape = second_operator->getShape();
  //   // Tensor* constant_tensor = new Tensor(first_operator_scalars, shape);
  //   // return new LDiff(constant_tensor, second_operator, node->name(), dev, mem);
  // }
  else // T - T
  {
    Layer *first_operator = output_node_map[first_operator_name];
    Layer *second_operator = output_node_map[second_operator_name];
    // first_operator->info();
    // second_operator->info();
    return new LDiff(first_operator, second_operator, node->name(), dev, mem);
  } 

  // if(map_init_values.count(first_operator_name)) // k - T
  // {
  //   // Read the input scalar
  //   vector<float> scalars = map_init_values[first_operator_name];
  //   if (scalars.size() != 1)
  //   {
  //     msg("Error: The first input operand of the Diff layer " + node->name() + " is not valid", "ONNX::ImportNet");
  //     return nullptr;
  //   }
  //   Layer *second_operator = output_node_map[second_operator_name];
  //   return new LDiff(scalars[0], second_operator, node->name(), dev, mem);
  // }
  // else if(map_init_values.count(second_operator_name)) // T - k
  // {
  //   // Read the input scalar
  //   vector<float> scalars = map_init_values[second_operator_name];
  //   if (scalars.size() != 1)
  //   {
  //     msg("Error: The second input operand of the Diff layer " + node->name() + " is not valid", "ONNX::ImportNet");
  //     return nullptr;
  //   }
  //   Layer *first_operator = output_node_map[first_operator_name];
  //   return new LDiff(first_operator, scalars[0], node->name(), dev, mem);
  // }
  // else // T - T
  // {
  //   Layer *first_operator = output_node_map[first_operator_name];
  //   Layer *second_operator = output_node_map[second_operator_name];
  //   return new LDiff(first_operator, second_operator, node->name(), dev, mem);
  // }
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
