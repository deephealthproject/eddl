#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/pool/maxpool_onnx.h"

// ONNX import
Layer* build_maxpool_layer(onnx::NodeProto *node,
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
  vector<int> dilations;
  int storage_order = 0;
  bool pool1d = false;

  for (int j = 0; j < node->attribute_size(); j++)
  { // Set the attributes
    onnx::AttributeProto attribute = node->attribute(j);
    string attr_name = attribute.name();
    if (!attr_name.compare("auto_pad"))
    { // We dont know if it is implemented
      if (!attribute.s().compare("NOTSET"))
        continue;
      // if(!attribute.s().compare("VALID")) explicit_padding=false;
    }
    //else if (!attr_name.compare("ceil_mode")) {
    //}
    //else if (!attr_name.compare("dilations")) {
    //}
    else if (!attr_name.compare("kernel_shape"))
    {
      for (int h = 0; h < attribute.ints_size(); h++)
      {
        kernel_shape.push_back(attribute.ints(h));
      }
      if (attribute.ints_size() == 1)
        pool1d = true;
    }
    else if (!attr_name.compare("pads"))
    {
      explicit_padding = true;
      for (int h = 0; h < attribute.ints_size(); h++)
      {
        pads[h] = attribute.ints(h);
      }
    }
    //else if (!attr_name.compare("storage_order")) {
    //}
    else if (!attr_name.compare("strides"))
    {
      for (int h = 0; h < attribute.ints_size(); h++)
      {
        strides.push_back(attribute.ints(h));
      }
      if (attribute.ints_size() == 1)
        pool1d = true;
    }
  }

  string parent_name = node->input(0); // Get parent
  Layer *parent = output_node_map[parent_name];
  vector<int> parent_shape = parent->output->shape;

  string name = node->name();

  if (parent_shape.size() == 3)
    pool1d = true;

  if (pool1d)
  {
    strides.push_back(1);
    kernel_shape.push_back(1);
    return new LMaxPool1D(parent, new PoolDescriptor(kernel_shape, strides, pads), name, dev, mem);
  }
  else
    return new LMaxPool(parent, new PoolDescriptor(kernel_shape, strides, pads), name, dev, mem);
}

// ONNX import
Layer* build_globalmaxpool_layer(onnx::NodeProto *node,
                                 map<string, Layer *> &output_node_map,
                                 int dev,
                                 int mem)
{
  string parent_name = node->input(0); // Get parent
  Layer *parent = output_node_map[parent_name];
  vector<int> parent_shape = parent->output->shape;

  int h = parent_shape[2];
  int w = parent_shape[3];

  return new LMaxPool(parent, {h, w}, {1, 1}, "none", "gpool", dev, mem);
}

// ONNX export
void build_maxpool_node(LMaxPool *layer, onnx::GraphProto *graph)
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
  for (int i : layer->pd->ksize)
  {
    max_pool_ks->add_ints(i);
  }

  // Attr pads
  onnx::AttributeProto *max_pool_pads = node->add_attribute();
  max_pool_pads->set_name("pads");
  max_pool_pads->set_type(onnx::AttributeProto::INTS);
  max_pool_pads->add_ints(layer->pd->padrt);
  max_pool_pads->add_ints(layer->pd->padcl);
  max_pool_pads->add_ints(layer->pd->padrb);
  max_pool_pads->add_ints(layer->pd->padcr);

  // Attr strides
  onnx::AttributeProto *max_pool_strides = node->add_attribute();
  max_pool_strides->set_name("strides");
  max_pool_strides->set_type(onnx::AttributeProto::INTS);
  max_pool_strides->add_ints(layer->pd->sr);
  max_pool_strides->add_ints(layer->pd->sc);
}

#endif // defined(cPROTO)
