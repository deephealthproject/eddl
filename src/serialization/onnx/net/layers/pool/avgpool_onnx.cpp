#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/pool/avgpool_onnx.h"

void build_averagepool_node(LAveragePool *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("AveragePool");
  node->set_name(layer->name);
  // Set the inputs of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);

  // Attr kernel_shape
  onnx::AttributeProto *avg_pool_ks = node->add_attribute();
  avg_pool_ks->set_name("kernel_shape");
  avg_pool_ks->set_type(onnx::AttributeProto::INTS);
  for (int i : layer->pd->ksize)
  {
    avg_pool_ks->add_ints(i);
  }

  // Attr pads
  onnx::AttributeProto *avg_pool_pads = node->add_attribute();
  avg_pool_pads->set_name("pads");
  avg_pool_pads->set_type(onnx::AttributeProto::INTS);
  avg_pool_pads->add_ints(layer->pd->padrt);
  avg_pool_pads->add_ints(layer->pd->padcl);
  avg_pool_pads->add_ints(layer->pd->padrb);
  avg_pool_pads->add_ints(layer->pd->padcr);

  // Attr strides
  onnx::AttributeProto *avg_pool_strides = node->add_attribute();
  avg_pool_strides->set_name("strides");
  avg_pool_strides->set_type(onnx::AttributeProto::INTS);
  avg_pool_strides->add_ints(layer->pd->sr);
  avg_pool_strides->add_ints(layer->pd->sc);
}

#endif // defined(cPROTO)
