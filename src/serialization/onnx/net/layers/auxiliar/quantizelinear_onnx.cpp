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
  Layer *parent;
  vector<int> parent_shape;

  string shape_name = node->input(1);
  vector<float> shape_values = map_init_values[shape_name];
 
  Tensor *y_scale;
  Tensor *y_zero_point; // 0 is the default value in the ONNX standard

  // inputs
  for (int i = 0; i < 3; i++) {
    string input = node->input(i);
    cout << "[DEBUG] Going to check input(" << i << "): " << input << " of " << node->input_size() << endl;
    if (!map_init_values.count(input)) {
      // parent
      cout << "[DEBUG] Parent " << input << " found!" << endl;
      parent = output_node_map[input];
      parent_shape = parent->output->getShape();
    }else if (map_init_values.count(input) && i == 0) {
      cout << "[DEBUG] Parent " << input << " not found! " << map_init_values.count(input) << endl;
      vector<float> *input0 = &(map_init_values[input]);
      vector<int> dims0;
      if(input0->size()<=1){
         dims0 = {1};
      }else{
         dims0 = map_init_dims[input];
      }  
      Tensor *input0_tensor = new Tensor(dims0, nullptr, dev);
      COPY_FROM_VECTOR_PTR_TO_TENSOR(input0, input0_tensor);
      printf("dims0: %d, size input %d\n", dims0.size(), input0_tensor->size);
      parent = new LConstOfTensor(input0_tensor, node->name(), dev, mem);
      parent->output = input0_tensor;
      parent_shape = parent->output->getShape();
    }else if(i==1){
      // param
      vector<float> *input1 = &(map_init_values[input]);
      vector<int> dims1;
      if(input1->size()<=1){
        dims1 = {1};  
      }else{
         dims1 = map_init_dims[input];
      } 
      y_scale = new Tensor(dims1, nullptr, dev);
      COPY_FROM_VECTOR_PTR_TO_TENSOR(input1, y_scale);
    }else if(i==2 && node->input_size() > 2){ // y_zero_point is an optional input
      // param
      vector<float> *input2 = &(map_init_values[input]);
      vector<int> dims2;
      if (input2->size() <= 1){
        dims2 = {1};  
      }else{
        dims2 = map_init_dims[input];
      }
      y_zero_point = new Tensor(dims2, nullptr, dev);
       COPY_FROM_VECTOR_PTR_TO_TENSOR(input2, y_zero_point);
    }
  }

  // attributes
  int axis = 0;
  if(node->attribute_size()>0){
    for (int j = 0; j < node->attribute_size(); j++) {
      onnx::AttributeProto attribute = node->attribute(j);
      string attr_name = attribute.name();
      if (!attr_name.compare("axis")) {
        printf("get axis %d\n", attribute.floats_size());
        axis = attribute.i();
        printf("axis = %d\n", axis);
      }
    }
    if(axis < 0){
      axis = axis + parent->output->ndim;
      //Negative value means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(input).
    }
  }else{
    axis = -1;
  }

  //falta analizar el PAD, de ahi vienen todos los 0s
  if(name.compare("YoloV3/MobilenetV2/Conv/Pad_quantize")==0){
    parent->output->info();
    parent->output->print();
  }


  LQuantizeLinear *quantize_linear = new LQuantizeLinear(parent, name, dev, mem, y_scale, y_zero_point, axis);
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
