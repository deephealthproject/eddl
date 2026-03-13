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
  vector<float> second_operator_scalars;
  string first_operator_name = node->input(0);
  string second_operator_name = node->input(1);
  Layer *first_operator = output_node_map[first_operator_name];
  Layer *second_operator = output_node_map[second_operator_name];


  if(map_init_values.count(second_operator_name))
  {
    second_operator_scalars = map_init_values[second_operator_name];
    if (second_operator_scalars.size() > 1) {

      // vector<int> shape_second_op = second_operator->getShape();
      vector<int> shape_first_op  = first_operator->getShape();
      vector<int> shape_second_op = map_init_dims[second_operator_name];
      
      Tensor *aux_first  = Tensor::zeros(shape_first_op);
      Tensor *aux_second = new Tensor(second_operator_scalars, shape_second_op);
      Tensor *constant;

      int first_op_ndim  = aux_first->ndim;
      int second_op_ndim = aux_second->ndim;

      // Assume the second operand as the one to be broadcasted
      if (second_op_ndim + 1 == first_op_ndim) {
        aux_second = aux_second->unsqueeze();

        // Find which is the different dimension
        int dim = -1;
        for (size_t i = 1; i < shape_first_op.size(); ++i) {
          if (shape_first_op[i] != shape_second_op[i]) {
            dim = i;
            break;
          }
        }
        if (dim < 0) 
          msg("Error: Could not find the different dim of the Mult layer " + node->name() + " operands", "ONNX:ImportNet");
        int times = shape_first_op[dim];
        constant = Tensor::repeat(aux_second, times, dim);
      }
      else if (second_op_ndim == first_op_ndim) {
        constant = new Tensor(second_operator_scalars, shape_first_op);
      }
      else {
        msg("Error: The second input operand of the Mult layer " + node->name() + " is not valid", "ONNX::ImportNet");
      }

      delete(aux_first);
      delete(aux_second);
      return new LMult(first_operator, constant, node->name(), dev, mem);
    }
    // Detect pattern for applying scale and bias of batchnorm using Mult
    // and Add operators
    else if (LBatchNorm *l = dynamic_cast<LBatchNorm*>(first_operator))
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
    else // Is a multiplication of a tensor by a scalar
    {
      vector<float> scalars = map_init_values[second_operator_name];
      if (scalars.size() == 1)
      {
        return new LMult(first_operator, scalars[0], node->name(), dev, mem);
      }
      else
      {
        msg("Error: The second input factor of the Mult layer " + node->name() + " is not valid", "ONNX::ImportNet");
        return nullptr;
      }
    }
  }

  vector<Layer *> operators = expand_broadcast({first_operator, second_operator});

  return new LMult(operators[0], operators[1], node->name(), dev, mem);
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

  if (!layer->binary)
  {
    string value_name(layer->name + "_value");
    node->add_input(value_name); // Add the value initializer as input
    // Create the value initializer
    onnx::TensorProto *mult_value = graph->add_initializer();
    mult_value->set_name(value_name);
    mult_value->set_data_type(onnx::TensorProto::FLOAT);
    mult_value->add_float_data(layer->val);
  }
}

#endif // defined(cPROTO)
