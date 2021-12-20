#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/core/bypass_onnx.h"

// ONNX import
Layer* build_lrn_layer(onnx::NodeProto *node,
                       map<string, Layer *> &output_node_map,
                       LOG_LEVEL log_level,
                       int dev,
                       int mem)
{
  string name = node->name();
  log_string("Going to use a Bypass layer to skip the LRN node \"" + name + "\"", log_level, LOG_LEVEL::WARN);
  string parent_name = node->input(0); // Get parent
  Layer *parent = output_node_map[parent_name];

  return new LBypass(parent, name, name, dev, mem);
}

// ONNX export
void build_identity_node(LBypass *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Identity");
  node->set_name(layer->name);
  // Set the inputs of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

#endif // defined(cPROTO)
