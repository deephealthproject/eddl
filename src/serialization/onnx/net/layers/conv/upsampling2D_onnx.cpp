#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/conv/upsampling2D_onnx.h"
#include "eddl/layers/core/layer_core.h"

// ONNX import
Layer* build_upsampling_layer(onnx::NodeProto *node,
                              map<string, vector<float>> &map_init_values,
                              map<string, vector<int>> &map_init_dims,
                              map<string, Layer *> &output_node_map,
                              int dev,
                              int mem)
{
  string name = node->name();
  string interpolation_mode;
  float batch_scale;
  float channel_scale;
  float depth_scale; // For 3D input
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

  int upsample_dim;
  int n_dims = scales_dims[0];
  if (n_dims == 4)
    upsample_dim = 2;  // 2D input
  else if (n_dims == 5)
    upsample_dim = 3;  // 3D input
  else
    msg("Error in node " + name + ". Unexpected number of scale values, got " + to_string(n_dims) + ", expected 4 or 5.", "ONNX::ImportNet");

  batch_scale = scales->at(0);
  channel_scale = scales->at(1);
  if (upsample_dim == 2)
  {
    height_scale = scales->at(2);
    width_scale = scales->at(3);
  } else { // 3D
    depth_scale = scales->at(2);
    height_scale = scales->at(3);
    width_scale = scales->at(4);
  }

  vector<int> size_vector;
  if (upsample_dim == 3)
    size_vector.push_back((int)depth_scale);
  size_vector.push_back((int)height_scale);
  size_vector.push_back((int)width_scale);

  if (upsample_dim == 3)
  {
    // To create a UpSampling3D layer we have to provide the target shape not the scale values
    vector<int> new_shape = size_vector;
    for (int i = 2; i < parent_shape.size(); ++i)
      new_shape[i-2] *= parent_shape[i];
    return new LUpSampling3D(parent, new_shape, true, getWrappingMode("nearest"), 0.0, getTransformationMode("half_pixel"), node->name(), DEV_CPU, 0);
  }
  else
    return new LUpSampling(parent, size_vector, interpolation_mode, name, dev, mem);
}

// ONNX export
void build_upsample_node(LUpSampling *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Resize");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
    node->add_input(parentl->name);
  node->add_input(layer->name + "_roi");
  node->add_input(layer->name + "_scales");
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);

  // coordinate_transformation_mode attr
  onnx::AttributeProto *trans_mode_attr = node->add_attribute();
  trans_mode_attr->set_name("coordinate_transformation_mode");
  trans_mode_attr->set_type(onnx::AttributeProto::STRING);
  trans_mode_attr->set_s("half_pixel");

  // coordinate_transformation_mode attr
  onnx::AttributeProto *mode_attr = node->add_attribute();
  mode_attr->set_name("mode");
  mode_attr->set_type(onnx::AttributeProto::STRING);
  mode_attr->set_s("nearest");

  // roi input
  onnx::TensorProto *roi = graph->add_initializer();
  roi->set_name(layer->name + "_roi");
  roi->set_data_type(onnx::TensorProto::FLOAT);
  roi->add_dims(8);
  // Set roi to : [0,0,0,0,1,1,1,1] (To select the full input tensor)
  int parent_dims = layer->parent[0]->output->getShape().size();
  for (int i = 0; i < parent_dims; ++i)
    roi->add_float_data(0);
  for (int i = 0; i < parent_dims; ++i)
    roi->add_float_data(1);

  // scales input
  onnx::TensorProto *scales = graph->add_initializer();
  scales->set_name(layer->name + "_scales");
  scales->set_data_type(onnx::TensorProto::FLOAT);
  scales->add_dims(4);
  scales->add_float_data(1); // Batch
  scales->add_float_data(1); // Channels
  scales->add_float_data(layer->size[0]); // H
  scales->add_float_data(layer->size[1]); // W
}

#endif // defined(cPROTO)
