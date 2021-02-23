#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/core/drop_onnx.h"

// ONNX import
Layer* build_dropout_layer(onnx::NodeProto *node,
                           map<string, Layer *> &output_node_map,
                           int dev,
                           int mem)
{
  float ratio = 0.5;
  for (int j = 0; j < node->attribute_size(); j++)
  { // Set the attributes
    onnx::AttributeProto attribute = node->attribute(j);
    string attr_name = attribute.name();
    if (!attr_name.compare("ratio"))
      ratio = attribute.f();
  }

  string parent_name = node->input(0); // Get parent
  Layer *parent = output_node_map[parent_name];
  vector<int> parent_shape = parent->output->shape;

  string name = node->name();
  return new LDropout(parent, ratio, true, name, dev, mem);
}

// ONNX export
void build_dropout_node(LDropout *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Dropout");
  node->set_name(layer->name);
  // Set the inputs of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);

  // Attr ratio
  onnx::AttributeProto *momentum_attr = node->add_attribute();
  momentum_attr->set_name("ratio");
  momentum_attr->set_type(onnx::AttributeProto::FLOAT);
  momentum_attr->set_f(layer->df);
}

#endif // defined(cPROTO)
