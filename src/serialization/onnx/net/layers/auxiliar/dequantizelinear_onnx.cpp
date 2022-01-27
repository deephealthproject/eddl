#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/auxiliar/dequantizelinear_onnx.h"

// ONNX import
Layer* build_dequantizelinear_layer(onnx::NodeProto *node,
                          map<string, vector<float>> &map_init_values,
			  map<string, vector<int>> &map_init_dims,
                          map<string, Layer *> &output_node_map,
                          int dev,
                          int mem)
{
  string parent_name;
  Layer *parent;
  vector<int> parent_shape;

  string shape_name = node->input(1);
  vector<float> shape_values = map_init_values[shape_name];

 
  string xscale_name;
  string xzeropoint_name;
  float x_scale;
  int x_zero_point;
  // inputs
  for (int i = 0; i < 3; i++) {
    string input = node->input(i);
    if (!map_init_values.count(input)) {
      // parent
      parent_name = node->input(0);
      parent = output_node_map[input];
      parent_shape = parent->output->getShape();
    }else if(i==1){
      // param
      xscale_name = node->input(i);
      vector<float> *input1 = &(map_init_values[xscale_name]);
      vector<int> dims1 = map_init_dims[xscale_name];
      Tensor *input1_tensor = new Tensor(dims1, nullptr, DEV_CPU);
      COPY_FROM_VECTOR_PTR_TO_TENSOR(input1, input1_tensor);
      x_scale = input1_tensor->ptr[0];
    }else if(i==2){
      // param
      xzeropoint_name = node->input(i);
      vector<float> *input2 = &(map_init_values[xzeropoint_name]);
      vector<int> dims2 = map_init_dims[xzeropoint_name];
      Tensor *input2_tensor = new Tensor(dims2, nullptr, DEV_CPU);
      COPY_FROM_VECTOR_PTR_TO_TENSOR(input2, input2_tensor);
      x_zero_point = input2_tensor->ptr[0];
    }
  }


  string name = node->name();

  LDequantizeLinear *dequantize_linear = new LDequantizeLinear(parent, name, dev, mem, x_scale, x_zero_point);

  return dequantize_linear;

}

// ONNX export
void build_dequantizelinear_node(LDequantizeLinear *layer, onnx::GraphProto *graph)
{
  msg("dequantize linear node export not supported", "[ONNX::ExportNet]");
}

#endif // defined(cPROTO)
