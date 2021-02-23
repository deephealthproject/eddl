#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/conv/upsampling_onnx.h"

// ONNX import
Layer* build_upsampling_layer(onnx::NodeProto *node,
                              map<string, vector<float>> &map_init_values,
                              map<string, vector<int>> &map_init_dims,
                              map<string, Layer *> &output_node_map,
                              int dev,
                              int mem)
{
  string interpolation_mode;
  float batch_scale;
  float channel_scale;
  float height_scale;
  float width_scale;
  for (int j = 0; j < node->attribute_size(); j++)
  { // Set the attributes
    onnx::AttributeProto attribute = node->attribute(j);
    string attr_name = attribute.name();
    if (!attr_name.compare("mode"))
      interpolation_mode = attribute.s();
  }

  string parent_name = node->input(0); // Get parent
  Layer *parent = output_node_map[parent_name];
  vector<int> parent_shape = parent->output->shape;

  string scales_name = node->input(1); // Get scales and dims
  vector<float> *scales = &(map_init_values[scales_name]);
  vector<int> scales_dims = map_init_dims[scales_name];

  if (scales_dims[0] != 4)
  {
    cerr << "Dimensions of upsampling layer in onnx are wrong" << endl;
  }
  batch_scale = scales->at(0);
  channel_scale = scales->at(1);
  height_scale = scales->at(2);
  width_scale = scales->at(3);

  string name = node->name();
  vector<int> size_vector;
  size_vector.push_back((int)height_scale);
  size_vector.push_back((int)width_scale);

  return new LUpSampling(parent, size_vector, interpolation_mode, name, dev, mem);
}

// ONNX export
void build_upsample_node(LUpSampling *layer, onnx::GraphProto *graph)
{
  // Scales input for the upsample node
  onnx::TensorProto *scales = graph->add_initializer();
  scales->set_name(layer->name + "_scales");
  scales->set_data_type(onnx::TensorProto::FLOAT);
  scales->add_dims(2 + layer->size.size()); // (batch_size, channels, height, width)
  // Add the scale factor for the first two dimensions
  for (int i = 0; i < 2; ++i)
  {
    scales->add_float_data(1);
  }
  for (int i = 0; i < layer->size.size(); ++i)
  {
    scales->add_float_data(layer->size[i]);
  }

  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Upsample");
  node->set_name(layer->name);
  // Set the inputs of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the input with the scale values
  node->add_input(layer->name + "_scales");
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);

  // Attr mode
  onnx::AttributeProto *mode_attr = node->add_attribute();
  mode_attr->set_name("mode");
  mode_attr->set_type(onnx::AttributeProto::STRING);
  mode_attr->set_s(layer->interpolation);
}

#endif // defined(cPROTO)
