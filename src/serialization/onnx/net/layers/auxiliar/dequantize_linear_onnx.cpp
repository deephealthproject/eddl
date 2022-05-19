#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/auxiliar/dequantize_linear_onnx.h"

// ONNX import
Layer* build_dequantize_linear_layer(onnx::NodeProto *node,
                                     map<string, vector<float>> &map_init_values,
			                               map<string, vector<int>> &map_init_dims,
                                     map<string, Layer *> &output_node_map,
                                     int dev,
                                     int mem)
{

  printf("build_dequantize_linear layer\n");

  // node name
  string name = node->name();
  cout << "node_name " << name << "\n";

  // x_scale input (1)
  string x_scale_name = node->input(1);
  cout << "x_scale " << x_scale_name << "\n";
  vector<float> *x_scale = &(map_init_values[x_scale_name]);
  vector<int> x_scale_dims = map_init_dims[x_scale_name];
  //
  Tensor *x_scale_tensor = new Tensor(x_scale_dims, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(x_scale, x_scale_tensor);

  // x_zero_point (2)
  string x_zero_point_name = node->input(2);
  cout << "x_zero_point " << x_zero_point_name << "\n";
  vector<float> *x_zero_point = &(map_init_values[x_zero_point_name]);
  vector<int> x_zero_point_dims = map_init_dims[x_zero_point_name];
  //
  Tensor *x_zero_point_tensor = new Tensor(x_zero_point_dims, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(x_zero_point, x_zero_point_tensor);

  // parent
  string parent_name = node->input(0);
  Layer *parent;
  vector<int> shape;
  if (!map_init_values.count(node->input(0))) {
    parent = output_node_map[parent_name];
    shape = parent->output->getShape();
  } else {
    parent = nullptr;
    shape = map_init_dims[node->input(0)];
  }

  printf("parent: %p\n", parent);
  if (parent != nullptr) cout << "parent's name: " << parent->name << "\n";

  LDequantizeLinear *dql = new LDequantizeLinear(parent, shape, name, dev, mem);

  return dql;

}

// ONNX export
void build_dequantize_linear_node(LDequantizeLinear *layer, onnx::GraphProto *graph)
{
  msg("dequantize_linear node export not supported", "[ONNX::ExportNet]");
}

#endif // defined(cPROTO)
