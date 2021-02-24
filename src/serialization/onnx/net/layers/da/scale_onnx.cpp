#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/da/scale_onnx.h"
#include "eddl/serialization/onnx/utils_onnx.h"

// ONNX import
Layer* build_scale_layer(onnx::NodeProto *node,
                         map<string, vector<float>> &map_init_values,
                         map<string, Layer *> &output_node_map,
                         int dev,
                         int mem)
{
      bool reshape_out = true;
      string da_mode("nearest");
      float constant = 0.0;

      for (int j = 0; j < node->attribute_size(); j++)
      { // Set the attributes
        onnx::AttributeProto attribute = node->attribute(j);
        string attr_name = attribute.name();
        if (!attr_name.compare("coordinate_transformation_mode"))
        {
          if (attribute.s().compare("asymmetric"))
          {
            msg("In Resize operator, the coordinate transformation mode \"" + attribute.s() + "\" is not supported. It must be \"asymmetric\".", "ONNX::ImportNet");
          }
        }
        if (!attr_name.compare("mode"))
        {
          if (attribute.s().compare("nearest"))
          {
            // ONNX only supports: "nearest", "linear" and "cubic".
            msg("In Resize operator, the mode \"" + attribute.s() + "\" is not supported. It must be \"nearest\".", "ONNX::ImportNet");
          }
        }
      }

      string parent_name = node->input(0);
      Layer *parent = output_node_map[parent_name];
      vector<int> new_shape = parent->getShape();

      string weights_name = node->input(2);
      float *dim_scales = new float [(&(map_init_values[weights_name]))->size()];
      COPY_FROM_VECTOR_PTR_TO_FLOAT_PTR(&(map_init_values[weights_name]), dim_scales);

      // Compute new shape by scaling the parent output shape
      for (int i = 0; i < new_shape.size(); ++i)
      {
        new_shape[i] = new_shape[i] * dim_scales[i];
      }

      delete [] dim_scales;

      return new LScale(parent, {new_shape[2], new_shape[3]}, reshape_out, getWrappingMode(da_mode), constant, node->name(), DEV_CPU, 0);
}

// ONNX export
void build_resize_node(LScale *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Resize");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  node->add_input(layer->name + "_roi");
  node->add_input(layer->name + "_scales");
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);

  // coordinate_transformation_mode attr
  onnx::AttributeProto *trans_mode_attr = node->add_attribute();
  trans_mode_attr->set_name("coordinate_transformation_mode");
  trans_mode_attr->set_type(onnx::AttributeProto::STRING);
  trans_mode_attr->set_s("asymmetric");

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
  scales->add_float_data(1);                                                 // Batch
  scales->add_float_data(1);                                                 // Channels
  scales->add_float_data(layer->new_shape[0] / layer->input->getShape()[2]); // H
  scales->add_float_data(layer->new_shape[1] / layer->input->getShape()[3]); // H
}

#endif // defined(cPROTO)
