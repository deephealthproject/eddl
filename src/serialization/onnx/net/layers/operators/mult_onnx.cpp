#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/operators/mult_onnx.h"
#include "eddl/layers/normalization/layer_normalization.h"
#include "eddl/serialization/onnx/utils_onnx.h"

// ONNX import
Layer* build_mul_layer(onnx::NodeProto *node,
                       map<string, vector<float>> &map_init_values,
                       map<string, vector<int>> &map_init_dims,
                       map<string, Layer *> &output_node_map,
                       int dev,
                       int mem)
{
  string first_operator_name = node->input(0);
  Layer *first_operator = output_node_map[first_operator_name];

  string second_operator_name = node->input(1);
  // Detect pattern for applying scale and bias of batchnorm using Mult
  // and Add operators
  if(map_init_dims.count(second_operator_name))
    if (LBatchNorm *l = dynamic_cast<LBatchNorm*>(first_operator))
    {
      // Set the scale value of the input batchnorm layer
      vector<float> *scale_weights = &(map_init_values[second_operator_name]);
      vector<int> scale_dims = map_init_dims[second_operator_name];
      Tensor *scale_tensor = new Tensor(scale_dims, nullptr, dev);
      COPY_FROM_VECTOR_PTR_TO_TENSOR(scale_weights, scale_tensor);
      Tensor::copy(scale_tensor, l->bn_g);
      delete scale_tensor;

      // Set the batchnorm layer as parent for the child nodes
      output_node_map[node->output(0)] = first_operator;
      return nullptr;
    }

  Layer *second_operator = output_node_map[second_operator_name];

  return new LMult(first_operator, second_operator, node->name(), dev, mem);
}

// ONNX export
void build_mul_node(LMult *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Mul");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

#endif // defined(cPROTO)
