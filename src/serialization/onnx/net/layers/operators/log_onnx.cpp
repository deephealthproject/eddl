#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/operators/log_onnx.h"

// ONNX import
Layer* build_log_layer(onnx::NodeProto *node,
                       map<string, Layer *> &output_node_map,
                       int dev,
                       int mem)
{
  string parent_name = node->input(0);
  Layer *parent = output_node_map[parent_name];

  return new LLog(parent, node->name(), dev, mem);
}

// ONNX export
void build_log_node(LLog *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Log");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

#endif // defined(cPROTO)
