#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/core/dense_onnx.h"

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
