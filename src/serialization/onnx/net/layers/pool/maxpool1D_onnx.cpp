#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/pool/maxpool1D_onnx.h"

void build_maxpool1D_node(LMaxPool1D *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("MaxPool");
  node->set_name(layer->name);
  // Set the inputs of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);

  // Attr kernel_shape
  onnx::AttributeProto *max_pool_ks = node->add_attribute();
  max_pool_ks->set_name("kernel_shape");
  max_pool_ks->set_type(onnx::AttributeProto::INTS);
  max_pool_ks->add_ints(layer->pd->kr);

  // Attr pads
  onnx::AttributeProto *max_pool_pads = node->add_attribute();
  max_pool_pads->set_name("pads");
  max_pool_pads->set_type(onnx::AttributeProto::INTS);
  max_pool_pads->add_ints(layer->pd->padrt);
  max_pool_pads->add_ints(layer->pd->padrb);

  // Attr strides
  onnx::AttributeProto *max_pool_strides = node->add_attribute();
  max_pool_strides->set_name("strides");
  max_pool_strides->set_type(onnx::AttributeProto::INTS);
  max_pool_strides->add_ints(layer->pd->sr);
}

#endif // defined(cPROTO)
