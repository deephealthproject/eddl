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
  string name = node->name();
  cout << "##################################################################" << endl;
  cout << "[DEBUG] Going to import dequantize layer \"" << name << "\"" << endl;
  string parent_name;
  Layer *parent;
  vector<int> parent_shape;

  string shape_name = node->input(1);
  vector<float> shape_values = map_init_values[shape_name];

  string xscale_name;
  string xzeropoint_name;
  float x_scale;
  int x_zero_point = 0; // 0 is the default value in the ONNX standard
  // inputs
  for (int i = 0; i < 3; i++) {
    string input = node->input(i);
    cout << "[DEBUG] Going to check input(" << i << "): " << input << endl;
    if (!map_init_values.count(input)) {
      // parent
      cout << "[DEBUG] Parent " << input << " found!" << endl;
      parent_name = node->input(0);
      parent = output_node_map[input];
      parent_shape = parent->output->getShape();
    //else if (map_init_values.count(input)) {
    //  vector<float> *input0 = &(map_init_values[input]);
    //  val = input0->data()[0];
    }else if(i==1){
      // param
      xscale_name = node->input(i);
      vector<float> *input1 = &(map_init_values[xscale_name]);
      x_scale = input1->data()[0];
    }else if(i==2 && node->input_size() > 2){ // x_zero_point is an optional input
      // param
      xzeropoint_name = node->input(i);
      vector<float> *input2 = &(map_init_values[xzeropoint_name]);
      if (input2->size() == 0)
        cerr << "x_zero_point is empty in operator " << name << endl;
      else
        x_zero_point = input2->data()[0];
    }
  }



  LDequantizeLinear *dequantize_linear = new LDequantizeLinear(parent, name, dev, mem, x_scale, x_zero_point);
  cout << "[DEBUG] Dequantize layer \"" << name << "\" imported!" << endl;
  cout << "##################################################################" << endl;

  return dequantize_linear;

}

// ONNX export
void build_dequantizelinear_node(LDequantizeLinear *layer, onnx::GraphProto *graph)
{
  msg("dequantize linear node export not supported", "[ONNX::ExportNet]");
}

#endif // defined(cPROTO)
