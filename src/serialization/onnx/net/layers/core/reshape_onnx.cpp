#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/core/reshape_onnx.h"

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
