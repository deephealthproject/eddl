#if defined(cPROTO)
#include <limits.h>
#include "eddl/serialization/onnx/layers/core/select_onnx.h"
#include "eddl/serialization/onnx/utils_onnx.h"

// ONNX import
Layer* build_select_layer(onnx::NodeProto *node,
                          map<string, vector<float>> &map_init_values,
                          map<string, Layer *> &output_node_map,
                          int dev,
                          int mem)
{
  vector<int> starts_vec; // firs position from the axis to select
  vector<int> ends_vec;   // last position (exclusive) from the axis to select
  vector<int> axes_vec;   // axes indexes to apply the selection
  if (node->attribute_size() == 3) // Operator version 1
  {
    for (int j = 0; j < node->attribute_size(); j++)
    { // Set the attributes
      onnx::AttributeProto attribute = node->attribute(j);
      string attr_name = attribute.name();
      if (!attr_name.compare("axes"))
      {
        for (int h = 0; h < attribute.ints_size(); h++)
          axes_vec.push_back(attribute.ints(h));

        for (int &ax : axes_vec)
          if (--ax < 0)
            msg("EDDL can't import a Slice operator that performs a selection "
                "over the batch dimension", "ONNX::ImportNet");
      }
      else if (!attr_name.compare("ends"))
      {
        for (int h = 0; h < attribute.ints_size(); h++)
          ends_vec.push_back(attribute.ints(h));
      }
      else if (!attr_name.compare("starts"))
      {
        for (int h = 0; h < attribute.ints_size(); h++)
          starts_vec.push_back(attribute.ints(h));
      }
      else
        msg("Found an unexpected attribute (" + attr_name +") in Slice operator", "ONNX::ImportNet");
    }
  }
  else // Operator version 13, 11 and 10
  {
    // Get starts indexes
    string starts_node_name = node->input(1);
    starts_vec = vf2vi(map_init_values[starts_node_name]);

    // Get ends indexes
    string ends_node_name = node->input(2);
    ends_vec = vf2vi(map_init_values[ends_node_name]);

    // Get axes to apply indexes 
    if (node->input_size() > 3) // This input is optional
    {
      string axes_node_name = node->input(3);
      axes_vec = vf2vi(map_init_values[axes_node_name]);
      // Shift indexes to skip batch dimension (not supported by EDDL Select)
      //   Note: In LSelect the axis 0 means the first axis after the batch dimension
      for (int &ax : axes_vec)
        if (--ax < 0)
          msg("EDDL can't import a Slice operator that performs a selection "
              "over the batch dimension", "ONNX::ImportNet");
    }
  }

  // Prepare parameters to create the Select layer
  Layer *parent = output_node_map[node->input(0)];
  int n_dims = parent->output->getShape().size() - 1; // Dims without batch
  // If the axes to apply the selection are not provided we select from all the
  // dimensions (skiping the batch).
  if (axes_vec.empty())
    for (int i = 0; i < n_dims; ++i)
      axes_vec.push_back(i);

  vector<string> indices;
  auto starts_it = starts_vec.begin();
  auto ends_it = ends_vec.begin();
  auto axes_it = axes_vec.begin();
  for (int i = 0; i < n_dims; ++i)
  {
    if (*axes_it == i)
    {
      string ax_selector = to_string(*starts_it) + ":";
      if (*ends_it > numeric_limits<int>::min() && *ends_it < numeric_limits<int>::max())
          ax_selector += to_string(*ends_it);
      indices.push_back(ax_selector);
      starts_it++;
      ends_it++;
      axes_it++;
    }
    else 
    {
      indices.push_back(":");
    }
  }

  return new LSelect(parent, indices, node->name(), dev, mem);
}

// ONNX export
void build_select_node(LSelect *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Slice");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);

  // Initializer with "starts" indexes
  string starts_name = layer->name + "_starts";
  onnx::TensorProto *starts = graph->add_initializer();
  starts->set_name(starts_name);
  starts->set_data_type(onnx::TensorProto::INT64);
  starts->add_dims(layer->sd->idxs_range.size());

  // Initializer with "ends" indexes
  string ends_name = layer->name + "_ends";
  onnx::TensorProto *ends = graph->add_initializer();
  ends->set_name(ends_name);
  ends->set_data_type(onnx::TensorProto::INT64);
  ends->add_dims(layer->sd->idxs_range.size());

  // Initializer with the "axes" to apply the selections
  string axes_name = layer->name + "_axes";
  onnx::TensorProto *axes = graph->add_initializer();
  axes->set_name(axes_name);
  axes->set_data_type(onnx::TensorProto::INT64);
  axes->add_dims(layer->sd->idxs_range.size());

  // Fill "starts", "ends" and "axes" initializers
  int axis = 1; // Skip batch dimension
  for (vector<int> &dim_idxs : layer->sd->idxs_range)
  {
    starts->add_int64_data(dim_idxs[0]);
    ends->add_int64_data(dim_idxs[1] + 1); // Exclusive range
    axes->add_int64_data(axis++);
  }
  
  // Set the initializers as inputs of the Slice op
  node->add_input(starts_name);
  node->add_input(ends_name);
  node->add_input(axes_name);
}

#endif // defined(cPROTO)
