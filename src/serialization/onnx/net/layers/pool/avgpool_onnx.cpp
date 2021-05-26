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
  vector<int> pads = {};
  bool explicit_padding = false;
  int ceil_mode = 0;
  int count_include_pad = 0;
  vector<int> dilations;
  int storage_order = 0;
  int pool_dim = 2;

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
        kernel_shape.push_back(attribute.ints(h));

      if (attribute.ints_size() == 1)
        pool_dim = 1;
      else if (attribute.ints_size() == 3)
        pool_dim = 3;
    }
    else if (!attr_name.compare("pads"))
    {
      explicit_padding = true;
      for (int h = 0; h < attribute.ints_size(); h++)
        pads.push_back(attribute.ints(h));

      // Reorder padding from [x1_begin, x2_begin,..., x1_end, x2_end,...] to [x1_begin, x1_end, x2_begin, x2_end,...]
      if (attribute.ints_size() == 4) // Pool2D
        swap(pads[1], pads[2]);
      else if (attribute.ints_size() == 6) // Pool3D
      {
        swap(pads[1], pads[3]);
        swap(pads[2], pads[3]);
        swap(pads[3], pads[4]);
      }
    }
    else if (!attr_name.compare("strides"))
    {
      for (int h = 0; h < attribute.ints_size(); h++)
        strides.push_back(attribute.ints(h));

      if (attribute.ints_size() == 1)
        pool_dim = 1;
      if (attribute.ints_size() == 3)
        pool_dim = 3;
    }
  }

  string name = node->name();
  string parent_name = node->input(0); // Get parent
  Layer *parent = output_node_map[parent_name];
  vector<int> parent_shape = parent->output->shape;

  // Check if the padding was not provided
  if (pads.size() == 0)
    pads = vector<int>(pool_dim*2, 0); // Set 0 padding

  if (pool_dim == 3)
    return new LAveragePool3D(parent, new PoolDescriptor3D(kernel_shape, strides, pads), name, dev, mem);
  if (pool_dim == 1)
  {
    strides.push_back(1);
    pads.push_back(0); pads.push_back(0);
    kernel_shape.push_back(1);
    return new LAveragePool1D(parent, new PoolDescriptor(kernel_shape, strides, pads), name, dev, mem);
  }
  else
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

  vector<int> pool_size;
  vector<int> pool_stride;
  if (parent_shape.size() == 3)
    return new LAveragePool1D(parent, {parent_shape[2]}, {1}, "none", node->name(), dev, mem);
  else if (parent_shape.size() == 4)
    return new LAveragePool(parent, {parent_shape[2], parent_shape[3]}, {1, 1}, "none", node->name(), dev, mem);
  else if (parent_shape.size() == 5)
    return new LAveragePool3D(parent, {parent_shape[2], parent_shape[3], parent_shape[4]}, {1, 1, 1}, "none", node->name(), dev, mem);

  return nullptr;
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
