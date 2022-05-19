#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/auxiliar/quantize_linear_onnx.h"

// ONNX import
Layer* build_quantize_linear_layer(onnx::NodeProto *node,
                                   map<string, vector<float>> &map_init_values,
			                             map<string, vector<int>> &map_init_dims,
                                   map<string, Layer *> &output_node_map,
                                   int dev,
                                   int mem)
{

  printf("build_quantize_linear layer\n");

  string parent_name;
  Layer *parent;
  vector<int> parent_shape;

  string shape_name = node->input(1);
  vector<float> shape_values = map_init_values[shape_name];

  string y_scale_name;
  vector<float> *y_scale;

  // inputs
  for (int i = 0; i < 2; i++) {
    string input = node->input(i);
    if (!map_init_values.count(input)) {
      // parent
      parent_name = node->input(0);
      parent = output_node_map[input];
      parent_shape = parent->output->getShape();
    } else {
      // param
      y_scale_name = node->input(i);
      y_scale = &(map_init_values[input]);
      //dims = map_init_dims[input];
    }
  }

  string name = node->name();

  LQuantizeLinear *ql = new LQuantizeLinear(parent, name, dev, mem);

  //Tensor *thresholds_tensor = new Tensor(dims, nullptr, dev);
  //COPY_FROM_VECTOR_PTR_TO_TENSOR(thresholds, thresholds_tensor);
  //Tensor::copy(thresholds_tensor, multi_threshold->thresholds);

  return ql;

}

// ONNX export
void build_quantize_linear_node(LQuantizeLinear *layer, onnx::GraphProto *graph)
{
  msg("quantize_linear node export not supported", "[ONNX::ExportNet]");
}

#endif // defined(cPROTO)
