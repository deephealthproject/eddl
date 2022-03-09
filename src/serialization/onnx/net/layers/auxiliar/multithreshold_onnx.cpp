#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/auxiliar/multithreshold_onnx.h"

// ONNX import
Layer* build_multithreshold_layer(onnx::NodeProto *node,
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

  string thresholds_name;
  vector<float> *thresholds;
  vector<int> dims;

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
      thresholds_name = node->input(i);
      thresholds = &(map_init_values[input]);
      dims = map_init_dims[input];
    }
  }

  // attributes
  float out_bias = 0.f;
  float out_scale = 1.f;
  for (int j = 0; j < node->attribute_size(); j++) {
    onnx::AttributeProto attribute = node->attribute(j);
    string attr_name = attribute.name();
    if (!attr_name.compare("out_bias")) {
      printf("get biasi %d\n", attribute.floats_size());
      out_bias = attribute.f();
      printf("out bias = %f\n", out_bias);
    }
    if (!attr_name.compare("out_scale")) {
      printf("get scale %d\n", attribute.floats_size());
      out_scale = attribute.f();
      printf("out scale = %f\n", out_scale);
    }
  }

  string name = node->name();

  LMultiThreshold *multi_threshold = new LMultiThreshold(parent, dims, name, dev, mem, out_bias, out_scale);

  Tensor *thresholds_tensor = new Tensor(dims, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(thresholds, thresholds_tensor);

  Tensor::copy(thresholds_tensor, multi_threshold->thresholds);

  return multi_threshold;

}

// ONNX export
void build_multithreshold_node(LMultiThreshold *layer, onnx::GraphProto *graph)
{
  msg("multithreshold node export not supported", "[ONNX::ExportNet]");
}

#endif // defined(cPROTO)
