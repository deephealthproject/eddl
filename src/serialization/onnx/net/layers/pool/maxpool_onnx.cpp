#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/pool/maxpool_onnx.h"
#include "eddl/layers/da/layer_da.h"

// ONNX import
Layer* build_maxpool_layer(onnx::NodeProto *node,
                           map<string, Layer *> &output_node_map,
                           LOG_LEVEL log_level,
                           int dev,
                           int mem)
{
  int filters;
  vector<int> kernel_shape;
  vector<int> strides;
  vector<int> pads = {};
  bool explicit_padding = false;
  int ceil_mode = 0;
  vector<int> dilations;
  int storage_order = 0;
  int pool_dim = 2;

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
    //else if (!attr_name.compare("storage_order")) {
    //}
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

  if (parent_shape.size() == 3)
    pool_dim = 1;
  else if (parent_shape.size() == 4)
    pool_dim = 2;
  else if (parent_shape.size() == 5)
    pool_dim = 3;

  // Check if the padding was not provided
  if (pads.size() == 0)
    pads = vector<int>(pool_dim*2, 0); // Set 0 padding

  if (pool_dim == 3)
    return new LMaxPool3D(parent, new PoolDescriptor3D(kernel_shape, strides, pads), name, dev, mem);
  else if (pool_dim == 1)
  {
    strides.push_back(1);
    pads.push_back(0); pads.push_back(0);
    kernel_shape.push_back(1);
    return new LMaxPool1D(parent, new PoolDescriptor(kernel_shape, strides, pads), name, dev, mem);
  }
  else
  {
    LMaxPool *pool_layer;
    PoolDescriptor *pd;
    try
    {
      pd = new PoolDescriptor(kernel_shape, strides, pads);
      pool_layer = new LMaxPool(parent, pd, name, dev, mem);
    }
    catch (AsymmetricPaddingException& e)
    {
        log_string("Detected a padding asymmetry in the MaxPool layer \"" + name + "\". Going to add an explicit Pad layer before to fix it.", log_level, LOG_LEVEL::INFO);
        // Remove the invalid pool layer from the parent child vector
        parent->child.pop_back();
        parent->lout--;

        vector<int> asym_pads = e.get_asymmetric_pads(); // Asymmetric paddings to fix
        string pad_layer_name = name + "__asymmetric_padding";
        // Create a parent layer to fix the padding asymmetry
        parent = new LPad(parent, {asym_pads[0], asym_pads[3], asym_pads[1], asym_pads[2]}, 0.0, pad_layer_name, dev, mem);
        // Create again the full MaxPool layer
        vector<int> new_pads = {0, 0, 0, 0};
        pd = new PoolDescriptor(kernel_shape, strides, new_pads, mem);
        pool_layer = new LMaxPool(parent, pd, name, dev, mem);
    }

    return pool_layer;
  }
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

  vector<int> pool_size;
  vector<int> pool_stride;
  if (parent_shape.size() == 3)
    return new LMaxPool1D(parent, {parent_shape[2]}, {1}, "none", node->name(), dev, mem);
  else if (parent_shape.size() == 4)
    return new LMaxPool(parent, {parent_shape[2], parent_shape[3]}, {1, 1}, "none", node->name(), dev, mem);
  else if (parent_shape.size() == 5)
    return new LMaxPool3D(parent, {parent_shape[2], parent_shape[3], parent_shape[4]}, {1, 1, 1}, "none", node->name(), dev, mem);

  return nullptr;
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
