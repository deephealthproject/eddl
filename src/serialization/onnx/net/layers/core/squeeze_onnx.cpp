#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/core/squeeze_onnx.h"

// ONNX import
Layer* build_squeeze_layer(onnx::NodeProto *node,
                           map<string, Layer *> &output_node_map,
                           LOG_LEVEL log_level,
                           int dev,
                           int mem)
{
  log_string("Squeeze layer detected", log_level, LOG_LEVEL::DEBUG);
  vector<int> squeeze_axes;
  for (int j = 0; j < node->attribute_size(); j++)
  { // Set the attributes
    onnx::AttributeProto attribute = node->attribute(j);
    string attr_name = attribute.name();
    if (!attr_name.compare("axes"))
    {
      // Read the axes to squeeze
      for (int h = 0; h < attribute.ints_size(); h++)
      {
        squeeze_axes.push_back(attribute.ints(h));
      }
    }
  }

  string parent_name;
  parent_name = node->input(0);
  Layer *parent = output_node_map[parent_name];
  vector<int> parent_out_shape = parent->output->getShape();
  // Check if we are trying to squeeze the axis 0 or 1 with a recurrent parent node
  //		- In ONNX, the output tensors of a recurrent operator have a dimension with the number of directions
  // 			of the layer (1:onedirectional, 2:bidirectional). In the case of a onedirectional layer the axis must
  // 			be squeezed. But for the creation of the EDDL model we don't need to do this operation, so we skip it.
  for (int i = 0; i < squeeze_axes.size(); ++i)
  {
    if ((squeeze_axes[i] == 0 || squeeze_axes[i] == 1) && parent->isrecurrent)
    {
      log_string("Removing axes " + to_string(squeeze_axes[i]) + " from Squeeze operator. Operation not needed because the parent node is recurrent.",
                 log_level,
                 LOG_LEVEL::DEBUG);
      squeeze_axes.erase(squeeze_axes.begin() + i); // We remove the axis to squeeze
    }
  }

  // Check if all the axes are valid
  bool valid_axes = true;
  for (int ax : squeeze_axes)
  {
    if (ax >= parent_out_shape.size())
    {
      valid_axes = false;
      break;
    }
  }

  if (squeeze_axes.size() == 0)
  {
    log_string("Skiping squeeze operation. No axes to squeeze.", log_level, LOG_LEVEL::DEBUG);
    return output_node_map[parent_name];
  }
  else if (!valid_axes)
  {
    log_string("Skiping squeeze operation. The axes to squeeze are not valid", log_level, LOG_LEVEL::DEBUG);
    return output_node_map[parent_name];
  }
  else
  { // There are axes to squeeze
    vector<int> target_shape;
    bool to_squeeze = false;
    for (int parent_ax = 0; parent_ax < parent_out_shape.size(); ++parent_ax)
    {
      to_squeeze = false;
      for (int target_ax : squeeze_axes)
      {
        if (parent_ax == target_ax)
        {
          if (parent_out_shape[parent_ax] == 1)
          {
            to_squeeze = true;
            break;
          }
          else
          {
            log_string("Trying to squeeze an axis with value different than one. Skiping the operator.", log_level, LOG_LEVEL::WARN);
            return output_node_map[parent_name];
          }
        }
      }
      if (!to_squeeze)
        target_shape.push_back(parent_out_shape[parent_ax]);
    }
    Layer *actual_layer = new LReshape(parent, target_shape, node->name(), dev, mem);  // TODO: Use Squeeze layer
    log_string("Squeeze (with Reshape) layer created", log_level, LOG_LEVEL::DEBUG);
    return actual_layer;
  }
}

void build_squeeze_node(LSqueeze *layer, onnx::GraphProto *graph)
{
  vector<int> axes;
  if (layer->axis == -1)
  {
    // Detect the axes to squeeze
    vector<int> input_shape = layer->input->getShape();
    for (int i = 1/*skip batch*/; i < input_shape.size(); ++i)
      if (input_shape[i] == 1)
        axes.push_back(i);
  }
  else
    axes = {layer->axis+1}; // +1 to add batch dimension

  squeeze_node_builder(
      layer->name,
      layer->parent[0]->name,
      layer->name,
      axes,
      graph);
}

// ONNX export
void squeeze_node_builder(string node_name, string input, string output, vector<int> axes, onnx::GraphProto *graph)
{
  onnx::NodeProto *node_sq = graph->add_node();
  node_sq->set_op_type("Squeeze");
  node_sq->set_name(node_name);
  node_sq->add_input(input);
  onnx::AttributeProto *axes_attr = node_sq->add_attribute();
  axes_attr->set_name("axes");
  axes_attr->set_type(onnx::AttributeProto::INTS);
  for (int ax : axes)
    axes_attr->add_ints(ax);
  node_sq->add_output(output);
}

#endif // defined(cPROTO)
