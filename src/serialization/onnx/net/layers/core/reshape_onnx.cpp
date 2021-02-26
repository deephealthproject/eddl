#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/core/reshape_onnx.h"

// ONNX import
Layer* build_reshape_layer(onnx::NodeProto *node,
                           map<string, onnx::NodeProto *> &constant_node_map, 
                           map<string, vector<float>> &map_init_values,
                           map<string, vector<int>> &map_init_dims,
                           map<string, vector<onnx::NodeProto *>> &input_node_map,
                           map<string, Layer *> &output_node_map,
                           int dev,
                           int mem)
{
  string shape_node_name = node->input(1);
  vector<int> shape;
  if (constant_node_map.count(shape_node_name))
  {
    onnx::NodeProto *shape_node = constant_node_map[shape_node_name];
    onnx::AttributeProto shape_attribute = shape_node->attribute(0);
    if (shape_attribute.name().compare("value"))
    {
      // This means an error ocurred, but don't know how to proceed then.
      printf("An error ocurred when reading the shape of reshape\n");
    }
    onnx::TensorProto shape_tensor = shape_attribute.t();
    shape = vf2vi(parseTensorValues(shape_tensor));
  }
  else
  {
    shape = vf2vi(map_init_values[shape_node_name]);
  }
  string name = node->name();
  string parent_name = node->input(0);
  if (output_node_map.count(parent_name))
  {
    shape[0] = 1; // Default batch size = 1
    Layer *parent = output_node_map[parent_name];
    return new LReshape(parent, shape, name, dev, mem);
  }
  else if (map_init_values.count(parent_name))
  { // This means it is a parameter and not a layer
    for (int i = 0; i < node->output_size(); i++)
    {
      map_init_values[node->output(i)] = map_init_values[parent_name];
      map_init_dims[node->output(i)] = shape;
    }
    return nullptr;
  }
  else
    cerr << "Uknown parent type for reshape" << endl;

  return nullptr;
}

Layer* build_flatten_layer(onnx::NodeProto *node,
                           map<string, Layer *> &output_node_map,
                           int dev,
                           int mem)
{
  string parent_name = node->input(0); // Get parent
  Layer *parent = output_node_map[parent_name];

  return new LReshape(parent, {1, -1}, node->name(), dev, mem);
}

// ONNX export
void build_reshape_node(LReshape *layer, onnx::GraphProto *graph)
{
  // Constant node input to the reshape node: shape
  onnx::NodeProto *shape_const_node = graph->add_node();
  shape_const_node->add_output(layer->name + "_target_shape");
  shape_const_node->set_op_type("Constant");
  onnx::AttributeProto *shape_attr = shape_const_node->add_attribute();
  shape_attr->set_name("value");
  shape_attr->set_type(onnx::AttributeProto::TENSOR);
  onnx::TensorProto *target_shape_tensor = shape_attr->mutable_t();
  target_shape_tensor->set_name("const_tensor");
  target_shape_tensor->set_data_type(onnx::TensorProto::INT64);
  target_shape_tensor->add_dims(layer->ls.size());
  // Set the target shape
  target_shape_tensor->add_int64_data(-1); // For batch_size
  for (int i = 1; i < layer->ls.size(); ++i)
    target_shape_tensor->add_int64_data(layer->ls[i]);

  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Reshape");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the input with the target shape of the op
  node->add_input(layer->name + "_target_shape");
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

#endif // defined(cPROTO)
