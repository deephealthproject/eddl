#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/reductions/sum_onnx.h"

// ONNX import
Layer* build_rsum_layer(onnx::NodeProto *node,
                        map<string, Layer *> &output_node_map,
                        int dev,
                        int mem)
{
  vector<int> axes;
  bool keepdims = 1;
  for (int j = 0; j < node->attribute_size(); j++)
  {
    onnx::AttributeProto attribute = node->attribute(j);
    string attr_name = attribute.name();
    if (!attr_name.compare("keepdims"))
    {
      keepdims = attribute.i();
    }
    else if (!attr_name.compare("axes"))
    {
      for (int h = 0; h < attribute.ints_size(); h++)
      {
        axes.push_back(attribute.ints(h));
      }
    }
    else
      printf("Error with ReduceSum attributes. Attribute name is: %s\n", attr_name.c_str());
  }

  string parent_name = node->input(0);
  Layer *parent = output_node_map[parent_name];

  // Prepare the axes for EDDL. Because in EDDL you can't reduce the batch axis (0).
  for (int i = 0; i < axes.size(); ++i)
  {
    if (axes[i] > 0)
      axes[i]--;
    else if (axes[i] == 0)
      msg("You can't reduce the batch axis in Reduce Sum layer.", "ONNX::ImportNet");
    else
    {
      // From negative to positive axis value
      int parent_out_rank = parent->getShape().size();
      axes[i] += parent_out_rank;

      axes[i]--;
    }
  }

  return new LRSum(parent, axes, keepdims, node->name(), dev, mem);
}

// ONNX export
void build_rsum_node(LRSum *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("ReduceSum");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }

  // Attr axes
  onnx::AttributeProto *axes_attr = node->add_attribute();
  axes_attr->set_name("axes");
  axes_attr->set_type(onnx::AttributeProto::INTS);
  for (int ax : layer->axis)
    axes_attr->add_ints(ax + 1);

  // Attr keepdims
  onnx::AttributeProto *keepdims_attr = node->add_attribute();
  keepdims_attr->set_name("keepdims");
  keepdims_attr->set_type(onnx::AttributeProto::INT);
  keepdims_attr->set_i(layer->keepdims);

  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

#endif // defined(cPROTO)
