#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/core/unsqueeze_onnx.h"

// ONNX export
Layer* build_unsqueeze_layer(onnx::NodeProto *node,
                             map<string, vector<float>> &map_init_values,
                             map<string, vector<int>> &map_init_dims,
                             map<string, Layer *> &output_node_map,
                             LOG_LEVEL log_level,
                             int dev,
                             int mem)
{
  log_string("Unsqueeze layer detected", log_level, LOG_LEVEL::DEBUG);
  string name = node->name();  // Get the name of the layer
  vector<int> unsqueeze_axes;
  for (int j = 0; j < node->attribute_size(); j++)
  { // Set the attributes
    onnx::AttributeProto attribute = node->attribute(j);
    string attr_name = attribute.name();
    if (!attr_name.compare("axes"))
    {
      // Read the axes to squeeze
      for (int h = 0; h < attribute.ints_size(); h++)
      {
        unsqueeze_axes.push_back(attribute.ints(h));
      }
    }
  }

  string parent_name = node->input(0);
  if (map_init_values.count(parent_name))
  {
      // This case is for detecting the pattern that applies the scale and bias of the
      // bath normalization with Mul and Add operators feeded by the batchnorm output
      // and an Unsqueeze operator with the scale or bias as initializer input
      log_string("The Unsqueeze layer " + name + " has an initializer as input. "
                 "As the output is a constant we will not create a layer for it. "
                 "It will be handled by its child nodes.", log_level, LOG_LEVEL::DEBUG);
      // Propagate initializer input to child node
      string out_name = node->output(0);
      map_init_values[out_name] = map_init_values[parent_name];
      map_init_dims[out_name] = map_init_dims[parent_name];
      return nullptr;
  }
  Layer *parent = output_node_map[parent_name];
  vector<int> parent_out_shape = parent->output->getShape();
  // Check if we are trying to unsqueeze the axis 0 with a recurrent parent node
  // 		- In ONNX, the output of a recurrent encoder operator (the hidden state) has the number of directions
  // 			(1:onedirectional, 2:bidirectional) in the axis 0, so in the case of onedirectional models this
  // 			dimension is squeezed. And in the case of connecting the parent recurrent node to another one,
  // 			a unsqueeze node is usually used to undo the previous squeeze operator. And to build the EDDl model
  //			we don't need to create this ops, so we skip them.
  for (int i = 0; i < unsqueeze_axes.size(); ++i)
  {
    if (unsqueeze_axes[i] == 0 && parent->isrecurrent)
    {
      log_string("Removing 0 axis from Unsqueeze operator. The parent node is recurrent.", log_level, LOG_LEVEL::DEBUG);
      unsqueeze_axes.erase(unsqueeze_axes.begin() + i); // We remove the axis to squeeze
    }
  }

  // Check if all the axes are valid
  bool valid_axes = true;
  for (int ax : unsqueeze_axes)
  {
    if (ax > parent_out_shape.size())
    {
      valid_axes = false;
      break;
    }
  }

  if (unsqueeze_axes.size() == 0)
  {
    log_string("Skiping unsqueeze operation. No axes to unsqueeze.", log_level, LOG_LEVEL::DEBUG);
    return output_node_map[parent_name];
  }
  else if (!valid_axes)
  {
    log_string("Skiping unsqueeze operation. The axes to unsqueeze are not valid", log_level, LOG_LEVEL::DEBUG);
    return output_node_map[parent_name];
  }
  else
  { // There are axes to unsqueeze
    // Sort the axes to unsqueeze
    std::sort(unsqueeze_axes.begin(), unsqueeze_axes.end());
    // Search for duplicates. DUPLICATES ARE NOT ALLOWED
    for (int i = 0; i < unsqueeze_axes.size() - 1; i++)
    {
      if (unsqueeze_axes[i] == unsqueeze_axes[i + 1])
      {
        unsqueeze_axes.erase(unsqueeze_axes.begin() + i);
        log_string("Removing duplicates axis in Unsqueeze operator", log_level, LOG_LEVEL::WARN);
        i--;
      }
    }
    // Insert the new dims
    vector<int> target_shape = parent_out_shape;
    for (int unsq_ax : unsqueeze_axes)
    {
      target_shape.insert(target_shape.begin() + unsq_ax, 1);
    }
    Layer *actual_layer = new LReshape(parent, target_shape, name, dev, mem);  // TODO: use unsqueeze layer
    log_string("Unsqueeze (with Reshape) layer created", log_level, LOG_LEVEL::DEBUG);
    return actual_layer;
  }
}

// ONNX export
void build_unsqueeze_node(LUnsqueeze *layer, onnx::GraphProto *graph)
{
  unsqueeze_node_builder(
      layer->name,
      layer->parent[0]->name,
      layer->name,
      {layer->axis+1},  // +1 to add batch dimension
      graph);
}

// ONNX export
void unsqueeze_node_builder(string node_name, string input, string output, vector<int> axes, onnx::GraphProto *graph)
{
  onnx::NodeProto *node_usq = graph->add_node();
  node_usq->set_op_type("Unsqueeze");
  node_usq->set_name(node_name);
  node_usq->add_input(input);
  onnx::AttributeProto *axes_attr = node_usq->add_attribute();
  axes_attr->set_name("axes");
  axes_attr->set_type(onnx::AttributeProto::INTS);
  for (int ax : axes)
    axes_attr->add_ints(ax);
  node_usq->add_output(output);
}

#endif // defined(cPROTO)
