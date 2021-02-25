#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/core/select_onnx.h"
#include "eddl/serialization/onnx/utils_onnx.h"

// ONNX import
Layer* build_select_layer(onnx::NodeProto *node,
                          map<string, onnx::NodeProto *> &constant_node_map, 
                          map<string, Layer *> &output_node_map,
                          int dev,
                          int mem)
{

  // Get starts indexes
  string starts_node_name = node->input(1);
  onnx::NodeProto *starts_node = constant_node_map[starts_node_name];
  onnx::AttributeProto starts_attr = starts_node->attribute(0);
  onnx::TensorProto starts_tensor = starts_attr.t();
  vector<int> starts_vec = vf2vi(parseTensorValues(starts_tensor));

  // Get ends indexes
  string ends_node_name = node->input(2);
  onnx::NodeProto *ends_node = constant_node_map[ends_node_name];
  onnx::AttributeProto ends_attr = ends_node->attribute(0);
  onnx::TensorProto ends_tensor = ends_attr.t();
  vector<int> ends_vec = vf2vi(parseTensorValues(ends_tensor));

  // Get axes to apply indexes 
  vector<int> axes_vec;
  if (node->input_size() > 3) // This input is optional
  {
    string axes_node_name = node->input(3);
    onnx::NodeProto *axes_node = constant_node_map[axes_node_name];
    onnx::AttributeProto axes_attr = axes_node->attribute(0);
    onnx::TensorProto axes_tensor = axes_attr.t();
    axes_vec = vf2vi(parseTensorValues(axes_tensor));
    // Shift indexes to skip batch dimension (not supported by EDDL Select)
    //   Note: In LSelect the axis 0 means the first axis after the batch dimension
    for (int &ax : axes_vec)
      if (--ax < 0)
        msg("EDDL can't import a Slice operator that performs a selection "
            "over the batch dimension", "ONNX::ImportNet");
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
      string ax_selector = to_string(*starts_it) + ":" + to_string(*ends_it);
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

  cout << "indices: ";
  for (string &s : indices)
    cout << s << ", ";
  cout << endl;

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

  // Constant tensor with "starts" indexes
  onnx::NodeProto *starts_const = graph->add_node();
  string starts_tensor_name = layer->name + "_starts";
  starts_const->add_output(starts_tensor_name);
  starts_const->set_op_type("Constant");
  onnx::AttributeProto *starts_attr = starts_const->add_attribute();
  starts_attr->set_name("value");
  starts_attr->set_type(onnx::AttributeProto::TENSOR);
  onnx::TensorProto *starts_tensor = starts_attr->mutable_t();
  starts_tensor->set_name("const_tensor");
  starts_tensor->set_data_type(onnx::TensorProto::INT64);
  starts_tensor->add_dims(layer->sd->idxs_range.size());

  // Constant tensor with "ends" indexes
  onnx::NodeProto *ends_const = graph->add_node();
  string ends_tensor_name = layer->name + "_ends";
  ends_const->add_output(ends_tensor_name);
  ends_const->set_op_type("Constant");
  onnx::AttributeProto *ends_attr = ends_const->add_attribute();
  ends_attr->set_name("value");
  ends_attr->set_type(onnx::AttributeProto::TENSOR);
  onnx::TensorProto *ends_tensor = ends_attr->mutable_t();
  ends_tensor->set_name("const_tensor");
  ends_tensor->set_data_type(onnx::TensorProto::INT64);
  ends_tensor->add_dims(layer->sd->idxs_range.size());

  // Constant tensor with "axes" to apply the selections
  onnx::NodeProto *axes_const = graph->add_node();
  string axes_tensor_name = layer->name + "_axes";
  axes_const->add_output(axes_tensor_name);
  axes_const->set_op_type("Constant");
  onnx::AttributeProto *axes_attr = axes_const->add_attribute();
  axes_attr->set_name("value");
  axes_attr->set_type(onnx::AttributeProto::TENSOR);
  onnx::TensorProto *axes_tensor = axes_attr->mutable_t();
  axes_tensor->set_name("const_tensor");
  axes_tensor->set_data_type(onnx::TensorProto::INT64);
  axes_tensor->add_dims(layer->sd->idxs_range.size());

  // Fill "starts", "ends" and "axes" tensors
  int axis = 1; // Skip batch dimension
  for (vector<int> &dim_idxs : layer->sd->idxs_range)
  {
    starts_tensor->add_int64_data(dim_idxs[0]);
    ends_tensor->add_int64_data(dim_idxs[1] + 1); // Exclusive range
    axes_tensor->add_int64_data(axis++);
  }
  
  // Set the constant tensors as inputs of the Slice op
  node->add_input(starts_tensor_name);
  node->add_input(ends_tensor_name);
  node->add_input(axes_tensor_name);
}

#endif // defined(cPROTO)
