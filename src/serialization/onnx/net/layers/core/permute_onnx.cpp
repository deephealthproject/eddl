#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/core/permute_onnx.h"

// ONNX import
Layer* build_permute_layer(onnx::NodeProto *node,
                           map<string, Layer *> &output_node_map,
                           bool recurrent_net,
                           LOG_LEVEL log_level,
                           int dev,
                           int mem)
{
  log_string("Transpose layer detected", log_level, LOG_LEVEL::DEBUG);
  string name = node->name(); // Name of the layer
  vector<int> perm;
  for (int j = 0; j < node->attribute_size(); j++)
  { // Set the attributes
    onnx::AttributeProto attribute = node->attribute(j);
    string attr_name = attribute.name();
    if (!attr_name.compare("perm"))
    {
      for (int h = 0; h < attribute.ints_size(); h++)
      {
        perm.push_back(attribute.ints(h));
      }
    }
  }
  log_string("perm vector created", log_level, LOG_LEVEL::DEBUG);

  string parent_name;
  parent_name = node->input(0);
  Layer *parent = output_node_map[parent_name];

  if (recurrent_net)
  {
    if (perm.size() > 1)
    {
      if (perm[0] != 0 || perm[1] != 1)
      {
        log_string("Transpose layers in recurrent nets can not swap batch or sequence dimensions. Skiping Transpose layer...", log_level, LOG_LEVEL::DEBUG);
        return parent;
      }
    }
    else
    {
      log_string("WARNING: Transpose layer with permute indices size of " + to_string(perm.size()) + ". Skiping Transpose layer...", log_level, LOG_LEVEL::WARN);
      return parent;
    }
  }

  // EDDL models have to be batch first in shape
  if (perm[0] != 0)
  {
    msg("The perm vector of the operator " + name + " is not valid (perm[0] != 0). EDDL tensors are batch first.", "ONNX::ImportNet");
  }
  else
  {
    // Remove batch dimension to create the Permute layer
    perm.erase(perm.begin());
    // Fix the perm vector after removing batch dim
    for (int i = 0; i < perm.size(); ++i)
      perm[i]--;
  }

  Layer *actual_layer = new LPermute(parent, perm, name, dev, mem);
  log_string("Permute layer created", log_level, LOG_LEVEL::DEBUG);
  return actual_layer;
}

// ONNX export
void build_permute_node(LPermute *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Transpose");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);

  // Attr perm
  onnx::AttributeProto *perm_attr = node->add_attribute();
  perm_attr->set_name("perm");
  perm_attr->set_type(onnx::AttributeProto::INTS);
  perm_attr->add_ints(0); // Set the batch size position. It must not be permuted in EDDL
  for (int i : layer->sd->dims)
  {
    perm_attr->add_ints(i + 1); // Add 1 to fix the batch dim adition
  }
}

#endif // defined(cPROTO)
