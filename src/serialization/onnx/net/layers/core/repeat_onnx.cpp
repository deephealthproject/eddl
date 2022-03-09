#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/core/repeat_onnx.h"

// ONNX import
Layer* build_repeat_layer(onnx::NodeProto *node,
                          map<string, onnx::NodeProto *> &constant_node_map, 
                          map<string, vector<float>> &map_init_values,
                          map<string, Layer *> &output_node_map,
                          LOG_LEVEL log_level,
                          int dev,
                          int mem)
{
  log_string("Repeat layer detected", log_level, LOG_LEVEL::DEBUG);
  string repeats_node_name = node->input(1);
  vector<int> repeats;
  if (constant_node_map.count(repeats_node_name))
  {
    onnx::NodeProto *repeats_node = constant_node_map[repeats_node_name];
    onnx::AttributeProto repeats_attribute = repeats_node->attribute(0);
    if (repeats_attribute.name().compare("value"))
    {
      // This means an error ocurred, but don't know how to proceed then.
      printf("An error ocurred when reading the \"repeats\" input of a Tile node\n");
    }
    onnx::TensorProto repeats_tensor = repeats_attribute.t();
    repeats = vf2vi(parseTensorValues(repeats_tensor));
  }
  else
  {
    repeats = vf2vi(map_init_values[repeats_node_name]);
  }

  string name = node->name();
  string parent_name = node->input(0);
  Layer *parent = output_node_map[parent_name];

  Layer *aux_layer;
  bool layer_created = false;
  for (int axis = 0; axis < repeats.size(); ++axis)
  {
    int r = repeats[axis];
    if (r == 1) continue;
    else if (axis == 0)
    {
      log_string("The Tile node " + name + " is trying to repeat the dimension 0. "
                 "We are going to skip it", log_level, LOG_LEVEL::DEBUG);
      return parent; // Skip the layer
    }
    else if (r > 1)
    {
      aux_layer = new LRepeat(parent, r, axis, name, dev, mem);
      parent = aux_layer;
      layer_created = true;
    }
    else
    {
      msg("The repeat value \"" + to_string(r) + "\" not valid! In layer " + name,
          "ONNX::ImportNet");
    }
  }

  if (!layer_created)
  {
    log_string("The Tile node " + name + " has no effect on the input", log_level, LOG_LEVEL::WARN);
    return parent; // Skip the layer
  }

  return aux_layer;
}

void build_tile_node(LRepeat *layer, onnx::GraphProto *graph)
{
  // Chech that all the repeats have the same value.
  // Note: The Tile operator only accepts one repeat value per axis
  unsigned int repeat_value = layer->rd->vrepeats[0];
  for (auto v : layer->rd->vrepeats)
      if (v != repeat_value)
          msg("Error exporting the Repeat layer " + layer->name +
              ". Repeat layers with different repeats per axis can't be exported to ONNX",
              "ONNX::ExportNet");

  vector<int> repeats;
  repeats.push_back(1); // Batch dimension
  for (int i = 1; i < layer->input->getShape().size(); ++i)
    if (layer->rd->axis == i)
      // Add the repeat value only for the target axis
      repeats.push_back(repeat_value);
    else
      repeats.push_back(1);

  tile_node_builder(layer->name,
                    layer->parent[0]->name,
                    layer->name,
                    repeats,
                    graph);
}

void tile_node_builder(string node_name, string input, string output, vector<int> repeats, onnx::GraphProto *graph)
{
  // Constant node input to the Tile node: repeats
  onnx::NodeProto *repeats_const_node = graph->add_node();
  repeats_const_node->add_output(node_name + "_repeats");
  repeats_const_node->set_op_type("Constant");
  onnx::AttributeProto *repeats_attr = repeats_const_node->add_attribute();
  repeats_attr->set_name("value");
  repeats_attr->set_type(onnx::AttributeProto::TENSOR);
  onnx::TensorProto *repeats_tensor = repeats_attr->mutable_t();
  repeats_tensor->set_name("const_tensor");
  repeats_tensor->set_data_type(onnx::TensorProto::INT64);
  repeats_tensor->add_dims(repeats.size());
  for (const int r : repeats)
    repeats_tensor->add_int64_data(r);

  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Tile");
  node->set_name(node_name);
  // Set the name of the input operator
  node->add_input(input);
  // Set the input with the repeats of the layer
  node->add_input(node_name + "_repeats");
  // Set the name of the output of the node to link with other nodes
  node->add_output(output);
}

#endif // defined(cPROTO)
