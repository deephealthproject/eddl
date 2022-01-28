#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/auxiliar/quantizerlinear_onnx.h"

// ONNX import
Layer* build_quantizelinear_layer(onnx::NodeProto *node,
                          map<string, vector<float>> &map_init_values,
			  map<string, vector<int>> &map_init_dims,
                          map<string, Layer *> &output_node_map,
                          int dev,
                          int mem)
{
  string name = node->name();
  cout << "##################################################################" << endl;
  cout << "[DEBUG] Going to import quantize layer \"" << name << "\"" << endl;
  string parent_name;
  Layer *parent;
  vector<int> parent_shape;

  string shape_name = node->input(1);
  vector<float> shape_values = map_init_values[shape_name];
 
  string yscale_name;
  string yzeropoint_name;
  float y_scale;
  int y_zero_point = 0; // 0 is the default value in the ONNX standard

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
      yscale_name = node->input(i);
      vector<float> *input1 = &(map_init_values[yscale_name]);
      y_scale = input1->data()[0];
    }else if(i==2 && node->input_size() > 2){ // y_zero_point is an optional input
      // param
      yzeropoint_name = node->input(i);
      vector<float> *input2 = &(map_init_values[yzeropoint_name]);
      if (input2->size() == 0)
        cerr << "y_zero_point is empty in operator " << name << endl;
      else
        y_zero_point = input2->data()[0];
    }
  }

  LQuantizeLinear *quantize_linear = new LQuantizeLinear(parent, name, dev, mem, y_scale, y_zero_point);
  cout << "[DEBUG] Quantize layer \"" << name << "\" imported!" << endl;
  cout << "##################################################################" << endl;

  return quantize_linear;

}

// ONNX export
void build_quantizelinear_node(LQuantizeLinear *layer, onnx::GraphProto *graph)
{
  msg("quantize linear node export not supported", "[ONNX::ExportNet]");
}

#endif // defined(cPROTO)
