#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/core/matmul_integer_onnx.h"

// ONNX import
Layer* build_matmul_integer_layer(onnx::NodeProto *node,
                         map<string, vector<float>> &map_init_values,
                         map<string, vector<int>> &map_init_dims,
                         map<string, Layer *> &output_node_map,
                         LOG_LEVEL log_level,
                         int dev,
                         int mem)
{
  log_string("matmul_integer detected", log_level, LOG_LEVEL::DEBUG);
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
  LDense *dense;
  if (parent->output->ndim != 2) {
    printf("flatten added neuronas %d dims0 %d dims1 %d\n", neuronas, dims[0], dims[1]);
    LReshape *l = new LReshape(parent, {1, dims[1]}, "flatten", dev, mem);
    printf("l->output->ndim: %d\n", l->output->ndim);
    dense = new LDense(l, neuronas, use_bias, name, dev, mem);
  } else {
    dense = new LDense(parent, neuronas, use_bias, name, dev, mem);
  }

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

#endif // defined(cPROTO)
