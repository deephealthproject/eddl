#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/pool/avgpool_onnx.h"

// ONNX import
Layer* build_averagepool_layer(onnx::NodeProto *node,
                               map<string, Layer *> &output_node_map,
                               int dev,
                               int mem)
{
  int filters;
  vector<int> kernel_shape;
  vector<int> strides;
  vector<int> pads(4, 0); // Default value. 4 zeros
  bool explicit_padding = false;
  int ceil_mode = 0;
  int count_include_pad = 0;
  vector<int> dilations;
  int storage_order = 0;

  for (int j = 0; j < node->attribute_size(); j++)
  { // Set the attributes
    onnx::AttributeProto attribute = node->attribute(j);
    string attr_name = attribute.name();
    if (!attr_name.compare("auto_pad"))
    {
      if (!attribute.s().compare("NOTSET"))
        continue;
      // if(!attribute.s().compare("VALID")) explicit_padding=false;
    }
    //else if (!attr_name.compare("ceil_mode")) { Not in EDDL
    //}
    //else if (!attr_name.compare("count_include_pad")) { Not in EDDL
    //}
    else if (!attr_name.compare("kernel_shape"))
    {
      for (int h = 0; h < attribute.ints_size(); h++)
      {
        kernel_shape.push_back(attribute.ints(h));
      }
    }
    else if (!attr_name.compare("pads"))
    {
      explicit_padding = true;
      for (int h = 0; h < 4; h++)
      {
        pads[h] = attribute.ints(h);
      }
    }
    else if (!attr_name.compare("strides"))
    {
      for (int h = 0; h < attribute.ints_size(); h++)
      {
        strides.push_back(attribute.ints(h));
      }
    }
  }

  string name = node->name();
  string parent_name = node->input(0); // Get parent
  Layer *parent = output_node_map[parent_name];
  vector<int> parent_shape = parent->output->shape;

  return new LAveragePool(parent, new PoolDescriptor(kernel_shape, strides, pads), name, dev, mem);
}

// ONNX import
Layer* build_globalaveragegpool_layer(onnx::NodeProto *node,
                                      map<string, Layer *> &output_node_map,
                                      int dev,
                                      int mem)
{
  string parent_name = node->input(0); // Get parent
  Layer *parent = output_node_map[parent_name];
  vector<int> parent_shape = parent->output->shape;

  int h = parent_shape[2];
  int w = parent_shape[3];

  return new LAveragePool(parent, {h, w}, {1, 1}, "none", node->name(), dev, mem);
}

// ONNX export
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
