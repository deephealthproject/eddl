#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/conv/upsampling_onnx.h"

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
