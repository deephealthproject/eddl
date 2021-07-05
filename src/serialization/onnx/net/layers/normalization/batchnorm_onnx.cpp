#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/normalization/batchnorm_onnx.h"
#include "eddl/serialization/onnx/utils_onnx.h"

// ONNX import
Layer* build_batchnorm_layer(onnx::NodeProto *node,
                             map<string, vector<float>> &map_init_values,
                             map<string, vector<int>> &map_init_dims,
                             map<string, Layer *> &output_node_map,
                             int dev,
                             int mem)
{
  float epsilon = 1e-03f; // Default value: keep it updated with the default value set in the API
  float momentum = 0.99;  // Default value: keep it updated with the default value set in the API
  for (int j = 0; j < node->attribute_size(); j++)
  { // Set the attributes
  onnx::AttributeProto attribute = node->attribute(j);
  string attr_name = attribute.name();
  if (!attr_name.compare("epsilon"))
    epsilon = attribute.f();
  if (!attr_name.compare("momentum"))
    momentum = attribute.f();
  }

  string parent_name = node->input(0); // Get parent
  Layer *parent = output_node_map[parent_name];
  vector<int> parent_shape = parent->output->shape;

  string scale_name = node->input(1); // Scale parameter
  vector<float> *scale_weights = &(map_init_values[scale_name]);
  vector<int> scale_dims = map_init_dims[scale_name];

  string bias_name = node->input(2); // Bias parameter
  vector<float> *bias_weights = &(map_init_values[bias_name]);
  vector<int> bias_dims = map_init_dims[bias_name];

  string mean_name = node->input(3); // Get weights and dims
  vector<float> *mean_weights = &(map_init_values[mean_name]);
  vector<int> mean_dims = map_init_dims[mean_name];

  string variance_name = node->input(4); // Get weights and dims
  vector<float> *variance_weights = &(map_init_values[variance_name]);
  vector<int> variance_dims = map_init_dims[variance_name];

  string name = node->name();

  bool affine = true; // The ONNX operator description does not have an "affine" attribute. We have to assume that this will be allways true.

  LBatchNorm *actual_layer = new LBatchNorm(parent, momentum, epsilon, affine, name, dev, mem);

  Tensor *scale_tensor = new Tensor(scale_dims, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(scale_weights, scale_tensor);
  Tensor::copy(scale_tensor, actual_layer->bn_g);
  delete scale_tensor;

  Tensor *bias_tensor = new Tensor(bias_dims, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(bias_weights, bias_tensor);
  Tensor::copy(bias_tensor, actual_layer->bn_b);
  delete bias_tensor;

  Tensor *mean_tensor = new Tensor(mean_dims, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(mean_weights, mean_tensor);
  Tensor::copy(mean_tensor, actual_layer->mean);
  delete mean_tensor;

  Tensor *variance_tensor = new Tensor(variance_dims, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(variance_weights, variance_tensor);
  Tensor::copy(variance_tensor, actual_layer->variance);
  delete variance_tensor;

  return actual_layer;
}

// ONNX export
void build_batchnorm_node(LBatchNorm *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("BatchNormalization");
  node->set_name(layer->name);
  // Set the inputs of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  node->add_input(layer->name + "_scale");
  node->add_input(layer->name + "_bias");
  node->add_input(layer->name + "_mean");
  node->add_input(layer->name + "_variance");
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);

  // Attr epsilon
  onnx::AttributeProto *epsilon_attr = node->add_attribute();
  epsilon_attr->set_name("epsilon");
  epsilon_attr->set_type(onnx::AttributeProto::FLOAT);
  epsilon_attr->set_f(layer->epsilon);

  // Attr momentum
  onnx::AttributeProto *momentum_attr = node->add_attribute();
  momentum_attr->set_name("momentum");
  momentum_attr->set_type(onnx::AttributeProto::FLOAT);
  momentum_attr->set_f(layer->momentum);

  int n_features = layer->input->getShape()[1]; // TO-REVIEW 2021-07-05: is this correct in the case layer->input->shape.size() > 2?

  // Scale input
  onnx::TensorProto *scale = graph->add_initializer();
  scale->set_name(layer->name + "_scale");
  scale->set_data_type(onnx::TensorProto::FLOAT);
  scale->add_dims(n_features);

  // Bias input
  onnx::TensorProto *bias = graph->add_initializer();
  bias->set_name(layer->name + "_bias");
  bias->set_data_type(onnx::TensorProto::FLOAT);
  bias->add_dims(n_features);

  // Check if the layer has trainable parameters
  if (layer->affine)
  {
    for (int i = 0; i < n_features; ++i)
    {
      scale->add_float_data(layer->bn_g->ptr[i]);
      bias->add_float_data(layer->bn_b->ptr[i]);
    }
  }
  else
  {
    for (int i = 0; i < n_features; ++i)
    {
      // Set the scale values to 1 (1 is the default value in case of not having trainable parameters)
      scale->add_float_data(1);
      // Set the bias values to 0 (0 is the default value in case of not having trainable parameters)
      bias->add_float_data(0);
    }
  }

  // Mean input
  onnx::TensorProto *mean = graph->add_initializer();
  mean->set_name(layer->name + "_mean");
  mean->set_data_type(onnx::TensorProto::FLOAT);
  mean->add_dims(n_features);
  mean->mutable_float_data()->Add(layer->mean->ptr, layer->mean->ptr + layer->mean->size); // Set the mean values

  // variance input
  onnx::TensorProto *variance = graph->add_initializer();
  variance->set_name(layer->name + "_variance");
  variance->set_data_type(onnx::TensorProto::FLOAT);
  variance->add_dims(n_features);
  variance->mutable_float_data()->Add(layer->variance->ptr, layer->variance->ptr + layer->variance->size); // Set the mean values
}

#endif // defined(cPROTO)
