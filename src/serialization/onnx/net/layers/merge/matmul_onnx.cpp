#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/merge/matmul_onnx.h"
#include "eddl/layers/core/layer_core.h"

// ONNX import
Layer* build_matmul_layer(onnx::NodeProto *node,
                          map<string, vector<float>> &map_init_values,
                          map<string, vector<int>> &map_init_dims,
                          map<string, Layer *> &output_node_map,
                          int dev,
                          int mem)
{
  vector<Layer *> parents;
  string parent_name;
  bool dense_detected = false;
  int index_parameter = -1;
  for (int j = 0; j < node->input_size(); j++)
  {
    parent_name = node->input(j);
    if (map_init_values.count(parent_name))
    {
      // Dense detected
      if (dense_detected)
      {
        cerr << "MAT_MUL with two parameters" << endl;
      }
      dense_detected = true;
      index_parameter = j;
    }
    else
      parents.push_back(output_node_map[parent_name]);
  }
  if (dense_detected)
  {
    string weights_name = node->input(index_parameter);
    vector<float> *weights = &(map_init_values[weights_name]);
    vector<int> dims = map_init_dims[weights_name];
    int neuronas = dims[1];
    Layer *parent = parents[1 - index_parameter];
    bool use_bias = false;
    LDense *dense = new LDense(parent, neuronas, use_bias, node->name(), dev, mem);
    Tensor *weights_tensor = new Tensor(dims, nullptr, dev);
    COPY_FROM_VECTOR_PTR_TO_TENSOR(weights, weights_tensor);
    Tensor::copy(weights_tensor, dense->W);
    delete weights_tensor;
    return dense;
  }

  return new LMatMul(parents, node->name(), dev, mem);
}

#endif // defined(cPROTO)
