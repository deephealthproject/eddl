#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/core/dense_onnx.h"

// ONNX import
Layer* build_dense_layer(onnx::NodeProto *node,
                         map<string, vector<float>> &map_init_values,
                         map<string, vector<int>> &map_init_dims,
                         map<string, Layer *> &output_node_map,
                         LOG_LEVEL log_level,
                         int dev,
                         int mem)
{
  log_string("Dense detected", log_level, LOG_LEVEL::DEBUG);
  bool use_bias = false;
  float alpha;
  float beta;
  int transA = 0;
  int transB = 0;
  vector<int> bias_dims;
  vector<float> *bias;
  for (int j = 0; j < node->attribute_size(); j++)
  {
    onnx::AttributeProto attribute = node->attribute(j);
    string attr_name = attribute.name();
    if (!attr_name.compare("alpha"))
    {
      alpha = attribute.f();
    }
    else if (!attr_name.compare("beta"))
    {
      beta = attribute.f();
    }
    else if (!attr_name.compare("transA"))
    {
      transA = attribute.i();
    }
    else if (!attr_name.compare("transB"))
    {
      transB = attribute.i();
    }
  }

  string parent_name;
  Layer *parent;
  string weights_name;
  string bias_name;
  vector<float> *weights;
  vector<int> dims;

  for (int i = 0; i < 2; i++)
  {
    string input = node->input(i);
    if (!map_init_values.count(input))
    { // parent
      parent_name = node->input(0);
      parent = output_node_map[input];
    }
    else
    { // weights
      weights_name = node->input(i);
      weights = &(map_init_values[input]);
      dims = map_init_dims[input];
    }
  }
  use_bias = node->input_size() > 2;
  int neuronas = 0;
  if (transB)
  {
    neuronas = dims[0];
  }
  else
    neuronas = dims[1];
  string name = node->name();
  Tensor *input_size = parent->output;
  LDense *dense = new LDense(parent, neuronas, use_bias, name, dev, mem);

  Tensor *weights_tensor = new Tensor(dims, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(weights, weights_tensor);

  if (transB)
    weights_tensor->permute_({1, 0});
  Tensor::copy(weights_tensor, dense->W);
  delete weights_tensor;
  if (use_bias)
  {
    bias_name = node->input(2);
    bias = &(map_init_values[bias_name]);
    bias_dims = map_init_dims[bias_name];
    Tensor *bias_tensor = new Tensor(bias_dims, nullptr, dev);
    COPY_FROM_VECTOR_PTR_TO_TENSOR(bias, bias_tensor);
    Tensor::copy(bias_tensor, dense->bias);
    delete bias_tensor;
  }
  return dense;
}

// ONNX export
void build_gemm_node(LDense *layer, onnx::GraphProto *graph, bool gradients)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Gemm");
  node->set_name(layer->name);
  // Set the inputs of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the input params names of the Gemm(Dense) op
  node->add_input(layer->name + "_W");
  if (layer->use_bias)
    node->add_input(layer->name + "_b");
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);

  // Attr alpha
  onnx::AttributeProto *dense_alpha = node->add_attribute();
  dense_alpha->set_name("alpha");
  dense_alpha->set_type(onnx::AttributeProto::FLOAT);
  dense_alpha->set_f(1);

  // Attr beta
  onnx::AttributeProto *dense_beta = node->add_attribute();
  dense_beta->set_name("beta");
  dense_beta->set_type(onnx::AttributeProto::FLOAT);
  dense_beta->set_f(layer->use_bias);

  // Attr transA
  onnx::AttributeProto *dense_transA = node->add_attribute();
  dense_transA->set_name("transA");
  dense_transA->set_type(onnx::AttributeProto::INT);
  dense_transA->set_i(0);

  // Attr transB
  onnx::AttributeProto *dense_transB = node->add_attribute();
  dense_transB->set_name("transB");
  dense_transB->set_type(onnx::AttributeProto::INT);
  dense_transB->set_i(0);

  // Check if we are exporting weights or accumulated gradients
  if (!gradients)
  {
    // Weights input
    onnx::TensorProto *weight = graph->add_initializer();
    weight->set_name(layer->name + "_W");
    weight->set_data_type(onnx::TensorProto::FLOAT);
    weight->mutable_dims()->Add(layer->W->shape.begin(), layer->W->shape.end());      // Set the shape of the weights
    weight->mutable_float_data()->Add(layer->W->ptr, layer->W->ptr + layer->W->size); // Set the weights values
    //weight->mutable_raw_data()->assign( reinterpret_cast<const char*>(layer->W->ptr), sizeof(float) * layer->W->size );
    if (layer->use_bias)
    {
      // Bias input
      onnx::TensorProto *bias = graph->add_initializer();
      bias->set_name(layer->name + "_b");
      bias->set_data_type(onnx::TensorProto::FLOAT);
      bias->mutable_dims()->Add(layer->bias->shape.begin(), layer->bias->shape.end());         // Set the bias shape
      bias->mutable_float_data()->Add(layer->bias->ptr, layer->bias->ptr + layer->bias->size); // Set the bias values
      //bias->mutable_raw_data()->assign( reinterpret_cast<const char*>(layer->bias->ptr), sizeof(float) * layer->bias->size );
    }
  }
  else
  {
    // Accumulated gradients (Weights) input
    onnx::TensorProto *weight = graph->add_initializer();
    weight->set_name(layer->name + "_W");
    weight->set_data_type(onnx::TensorProto::FLOAT);
    weight->mutable_dims()->Add(layer->acc_gW->shape.begin(), layer->acc_gW->shape.end());           // Set the accumulated gradients shape (weights)
    weight->mutable_float_data()->Add(layer->acc_gW->ptr, layer->acc_gW->ptr + layer->acc_gW->size); // Set the accumulated gradients values (weights)
    //weight->mutable_raw_data()->assign( reinterpret_cast<const char*>(layer->acc_gW->ptr), sizeof(float) * layer->acc_gW->size );

    // Check if we are using bias
    if (layer->use_bias)
    {
      // Accumulated gradients (bias) input
      onnx::TensorProto *bias = graph->add_initializer();
      bias->set_name(layer->name + "_b");
      bias->set_data_type(onnx::TensorProto::FLOAT);
      bias->mutable_dims()->Add(layer->acc_gbias->shape.begin(), layer->acc_gbias->shape.end());              // Set the accumulated gradients shape (bias)
      bias->mutable_float_data()->Add(layer->acc_gbias->ptr, layer->acc_gbias->ptr + layer->acc_gbias->size); // Set the accumulated gradients values (bias)
      //bias->mutable_raw_data()->assign( reinterpret_cast<const char*>(layer->acc_gbias->ptr), sizeof(float) * layer->acc_gbias->size );
    }
  }
}

void build_dense_with_matmul_node(LDense *layer, onnx::GraphProto *graph, bool gradients)
{
  /*
   * Build a dense layer by composing a MatMul with an Add
   */

  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("MatMul");
  node->set_name(layer->name + "_MatMul");
  // Set the inputs of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the input param name of the Weight matrix
  node->add_input(layer->name + "_W");
  // Set the name of the output of the node to link with other nodes
  if (layer->use_bias)
    // Output name to link with the Add operator for the bias
    node->add_output(layer->name + "_MatMul");
  else
    // Set the proper output name to connect with the next layer of the model
    node->add_output(layer->name);

  // Set weights for the MatMul
  onnx::TensorProto *weight = graph->add_initializer();
  weight->set_name(layer->name + "_W");
  weight->set_data_type(onnx::TensorProto::FLOAT);
  weight->mutable_dims()->Add(layer->W->shape.begin(), layer->W->shape.end());      // Set the shape of the weights
  weight->mutable_float_data()->Add(layer->W->ptr, layer->W->ptr + layer->W->size); // Set the weights values

  // Create the Add node in case of using bias in the Dense layer
  if (layer->use_bias)
  {
    // Add an empty node to the graph
    onnx::NodeProto *node_bias = graph->add_node();
    node_bias->set_op_type("Add");
    node_bias->set_name(layer->name + "_Add");
    // Take the input from the previous MatMul
    node_bias->add_input(layer->name + "_MatMul");
    // Set the input param name of the Bias matrix
    node_bias->add_input(layer->name + "_b");
    // Set the name of the output of the node to link with other nodes
    node_bias->add_output(layer->name);
    // Set weights for Add (Dense bias)
    onnx::TensorProto *bias = graph->add_initializer();
    bias->set_name(layer->name + "_b");
    bias->set_data_type(onnx::TensorProto::FLOAT);
    bias->mutable_dims()->Add(layer->bias->shape.begin(), layer->bias->shape.end());         // Set the bias shape
    bias->mutable_float_data()->Add(layer->bias->ptr, layer->bias->ptr + layer->bias->size); // Set the bias values
  }
}

#endif // defined(cPROTO)
