#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/da/scale_onnx.h"

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
