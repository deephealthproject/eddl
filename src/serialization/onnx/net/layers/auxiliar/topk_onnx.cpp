#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/auxiliar/topk_onnx.h"

// ONNX import
Layer* build_topk_layer(onnx::NodeProto *node,
                          map<string, vector<float>> &map_init_values,
			  map<string, vector<int>> &map_init_dims,
                          map<string, Layer *> &output_node_map,
                          int dev,
                          int mem)
{
  string parent_name;
  Layer *parent;
  vector<int> parent_shape;
  string k_name;
  int K = 1;
  vector<int> dims;

  string shape_name = node->input(1);
  vector<float> shape_values = map_init_values[shape_name];

  // inputs
  for (int i = 0; i < 2; i++) {
    string input = node->input(i);
    if (!map_init_values.count(input)) {
      // parent
      parent_name = node->input(0);
      parent = output_node_map[input];
      parent_shape = parent->output->getShape();
    } else {
      // K
      k_name = node->input(i);
     //K = (int)*(map_init_values[input]);
      printf(" K = %d\n", K);
      dims = map_init_dims[input];
    }
  }

  // attributes
  int axis = -1;
  int largest = 1;
  int sorted = 1;

  for (int j = 0; j < node->attribute_size(); j++) {
    onnx::AttributeProto attribute = node->attribute(j);
    string attr_name = attribute.name();
    if (!attr_name.compare("axis")) {
      axis = attribute.i();
    }
    if (!attr_name.compare("largest")) {
      largest = attribute.i();
    }
    if (!attr_name.compare("sorted")) {
      sorted = attribute.i();
    }
  }

  string name = node->name();

  return new LTopK(parent, dims, name, dev, mem, axis, largest, sorted, K);
}

// ONNX export
void build_topk_node(LTopK *layer, onnx::GraphProto *graph)
{
  msg("Error: topk export not implemented", "[ONNX::ExportNew]");
}

#endif // defined(cPROTO)
