#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/merge/add_onnx.h"
#include "eddl/layers/core/layer_core.h"
#include "eddl/layers/conv/layer_conv.h"
#include "eddl/layers/normalization/layer_normalization.h"
#include "eddl/layers/auxiliar/layer_auxiliar.h"
#include "eddl/layers/operators/layer_operators.h"

// ONNX import
Layer* build_add_layer(onnx::NodeProto *node,
                       map<string, vector<float>> &map_init_values,
                       map<string, vector<int>> &map_init_dims,
                       map<string, Layer *> &output_node_map,
                       LOG_LEVEL log_level,
                       int dev,
                       int mem)
{
  log_string("Add detected", log_level, LOG_LEVEL::DEBUG);
  vector<Layer *> parents;
  string parent_name;
  bool parameter_input = false;
  int index_parameter = -1; // Possible values 0 and 1, we won't expect parameters in an add with more than two parents
  for (int j = 0; j < node->input_size(); j++)
  {
    parent_name = node->input(j);
    if (output_node_map.count(parent_name))
      parents.push_back(output_node_map[parent_name]);
    else if (map_init_values.count(parent_name))
    {
      parameter_input = true;
      index_parameter = j;
    }
  }
  
  if (parameter_input)
  {
    if (LConv *conv = dynamic_cast<LConv *>(parents[0]))
    {
      ConvolDescriptor *cd = conv->cd;
      string bias_name = node->input(index_parameter);
      vector<float> *bias = &(map_init_values[bias_name]);
      vector<int> bias_shape;
      bias_shape.push_back(bias->size());
      Tensor *bias_tensor = new Tensor(bias_shape, nullptr, dev);
      COPY_FROM_VECTOR_PTR_TO_TENSOR(bias, bias_tensor);
      if (!cd->use_bias)
      {
        cd->use_bias = true; // We need to enable the bias
        Tensor::copy(bias_tensor, cd->bias);
      }
      else
      {
        Tensor *auxiliar_tensor = Tensor::add(cd->bias, bias_tensor);
        Tensor::copy(auxiliar_tensor, cd->bias);
        delete auxiliar_tensor;
      }
      delete bias_tensor;
      return conv;
    }
    else if (LDense *dense = dynamic_cast<LDense *>(parents[0]))
    {
      log_string("Detected a Dense layer as the parent of the Add node.", log_level, LOG_LEVEL::DEBUG);
      string bias_name = node->input(index_parameter);
      vector<float> *bias = &(map_init_values[bias_name]);
      vector<int> bias_dims = map_init_dims[bias_name];
      if (!dense->use_bias)
      {
        log_string("Setting the bias values of the parent Dense to the Add parameters.", log_level, LOG_LEVEL::DEBUG);
        dense->use_bias = true;
        dense->bias = new Tensor(bias_dims, nullptr, dev);
        COPY_FROM_VECTOR_PTR_TO_TENSOR(bias, dense->bias);
        dense->params.push_back(dense->bias);
        dense->gbias = new Tensor(bias_dims, dev);
        dense->gradients.push_back(dense->gbias);
      }
      else
      { // If dense already has a bias, we sum it in top of the bias
        log_string("The parent Dense already has a bias. Adding the parameters of the Add operator to the parent bias.", log_level, LOG_LEVEL::DEBUG);
        Tensor *add_to_bias = new Tensor(bias_dims, nullptr, dev);
        COPY_FROM_VECTOR_PTR_TO_TENSOR(bias, add_to_bias);
        Tensor::add(add_to_bias, dense->bias, dense->bias);
        delete add_to_bias;
      }
      return dense;
    }
    else if (LBatchNorm *bn = dynamic_cast<LBatchNorm *>(parents[0]))
    {
      log_string("Detected a BatchNorm layer as the parent of the Add node.", log_level, LOG_LEVEL::DEBUG);
      string bias_name = node->input(index_parameter);
      vector<float> *bias_weights = &(map_init_values[bias_name]);
      vector<int> bias_dims = map_init_dims[bias_name];
      Tensor *bias_tensor = new Tensor(bias_dims, nullptr, dev);
      COPY_FROM_VECTOR_PTR_TO_TENSOR(bias_weights, bias_tensor);
      Tensor::copy(bias_tensor, bn->bn_b);
      delete bias_tensor;

      return bn;
    }
    else if (map_init_values.count(node->input(index_parameter)))
    {
      log_string("Detected a constant input tensor in the Add node.", log_level, LOG_LEVEL::DEBUG);
      string tensor_name = node->input(index_parameter);
      // Create the tensor with the constant values to add
      vector<float> *tensor_values = &(map_init_values[tensor_name]);
      vector<int> tensor_shape = map_init_dims[tensor_name];

      // Check if the constant tensor has only one number
      int tensor_size = 1;
      for (int i : tensor_shape)
          tensor_size *= i;

      // If it contains just one number we use a Sum layer
      if (tensor_size == 1 && parents.size() == 1)
      {
        log_string("The constant input to the Add node has only one value, going to use a Sum layer...", log_level, LOG_LEVEL::DEBUG);
        return new LSum(parents[0], tensor_values->at(0), node->name(), dev, mem);
      }

      // Check that the tensor shape is valid
      if (tensor_shape.size() < 2)
        msg("Error in Add node " + node->name() + ". The number of dimensions of the constant operator must be 2 or higher", "ONNX::ImportNet");
      else if (tensor_shape[0] != 1)
        msg("Error in Add node " + node->name() + ". The first dimension (batch) of the constant operator must be 1, got " +
            to_string(tensor_shape[0]), "ONNX::ImportNet");
      else
        tensor_shape.erase(tensor_shape.begin()); // Delete the batch dimension

      Tensor *t = new Tensor(tensor_shape, nullptr, dev);
      COPY_FROM_VECTOR_PTR_TO_TENSOR(tensor_values, t);
      // Create an auxiliary layer to get the constant values for the Add layer
      LConstOfTensor *const_layer = new LConstOfTensor(t, tensor_name, dev, mem);
      delete t; // The tensor is cloned inside the LConstOfTensor constructor

      parents.push_back(const_layer); // Add the layer to the parents of the Add layer
    }
    else
      msg("Error in Add node " + node->name() + ". Unable to process input " + node->input(index_parameter), "ONNX::ImportNet");
  }

  LAdd *actual_layer = new LAdd(parents, node->name(), dev, mem);
  log_string("Add layer created", log_level, LOG_LEVEL::DEBUG);
  return actual_layer;
}

// ONNX export
void build_add_node(LAdd *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Add");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

/*
 * DISTRIBUTED TRAINING
 */

vector<Tensor *> get_add_tensors(onnx::NodeProto &node,
                                 map<string, vector<float>> &map_init_values,
                                 map<string, vector<int>> &map_init_dims)
{
  vector<Tensor *> add_tensors;

  bool parameter_input = false;
  int index_parameter = -1; // Possible values 0 and 1, we won't expect parameters in an add with more than two parents
  for (int j = 0; j < node.input_size(); j++)
  {
    string parent_name = node.input(j);
    if (map_init_values.count(parent_name))
    {
      parameter_input = true;
      index_parameter = j;
      break;
    }
  }

  if (parameter_input)
  {
    string weights_name = node.input(index_parameter);
    vector<float> *weights = &(map_init_values[weights_name]);
    vector<int> dims = map_init_dims[weights_name];
    Tensor *weights_tensor = new Tensor(dims, nullptr, DEV_CPU);
    COPY_FROM_VECTOR_PTR_TO_TENSOR(weights, weights_tensor);
    add_tensors.push_back(weights_tensor);
  }

  return add_tensors;
}

#endif // defined(cPROTO)
