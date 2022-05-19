#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/auxiliar/constoftensor_onnx.h"

// ONNX import
Layer* build_constoftensor_layer(onnx::NodeProto *node,
                                 map<string, vector<float>> &map_init_values,
                                 map<string, Layer *> &output_node_map,
                                 int dev,
                                 int mem)
{
  Tensor *const_data;
  bool value_found = false;
  for (int j = 0; j < node->attribute_size(); j++)
  { // Set the attributes
    onnx::AttributeProto attribute = node->attribute(j);
    string attr_name = attribute.name();
    if (attr_name == "value")
    {
      value_found = true;
      // Create the constant data Tensor
      const onnx::TensorProto& data_tensor = attribute.t();

      // Get the values
      vector<float> data = parseTensorValues(data_tensor);

      // Get the tensor shape
      vector<int> shape;
      for (int i = 0; i < data_tensor.dims_size(); ++i) {
        int dim;
        if (data_tensor.dims(i) == 0) dim = 1; else dim = data_tensor.dims(i);
        shape.push_back(dim);
        //shape.push_back(data_tensor.dims(i));
      }

      // Create the tensor
      const_data = new Tensor(data, shape);

      // Check that the batch dimension is 1. We need it to be 1 because the LConstOfTensor layer will replicate the
      // tensor along the batch dimension. So we need to provide a tensor without the batch dimension.
      int batch_size = data_tensor.dims(0);
      if (batch_size != 1)
      {
        // We select the first sample from the batch dimension
        vector<string> slice_selector = {"0"};
        slice_selector.insert(slice_selector.end(), data_tensor.dims_size() - 1, ":");
        Tensor* aux = const_data->select(slice_selector);
        delete const_data;
        const_data = aux;
      }
      const_data->squeeze_(0); // Remove the batch dimension with value 1
    }
  }

  if (!value_found)
      msg("Error: In node " + node->name() + ", the attribute \"value\" was not found", "ONNX::ImportNet");

  LConstOfTensor *l = new LConstOfTensor(const_data, node->name(), dev, mem);
  delete const_data; // Free memory, the tensor has been cloned inside the layer constructor

  return l;
}

// ONNX export
void build_constant_node(LConstOfTensor *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto* node = graph->add_node();
  node->set_op_type("Constant");
  node->set_name(layer->name);

  // Note: This node doesn't have inputs

  // Create value attribute with the output tensor
  onnx::AttributeProto *value_attr = node->add_attribute();
  value_attr->set_name("value");
  value_attr->set_type(onnx::AttributeProto::TENSOR);
  onnx::TensorProto *value_tensor = value_attr->mutable_t();
  value_tensor->set_name("const_tensor");
  value_tensor->set_data_type(onnx::TensorProto::FLOAT);
  // Set tensor shape
  int batch_dim = layer->output->getShape()[0];
  cout << "=================================================================================================\n"
       << "Going to export a model with a constant tensor with batch_size = " << batch_dim << ".\n"
       << "Note that in case of importing this model with other library that batch_size will be fixed. \n"
       << "If you want to use other batch_size resize the model before exporting it.\n"
       << "=================================================================================================\n";

  // Set the shape of the constant data
  value_tensor->add_dims(batch_dim); // batch dimension
  for (int dim : layer->const_tensor->getShape())
      value_tensor->add_dims(dim);

  Tensor *aux_tensor_ptr = layer->const_tensor;
  if (!layer->const_tensor->isCPU())
  {
    // Create a copy in CPU
    Tensor *aux_copy = layer->const_tensor->clone();
    aux_copy->toCPU();
    aux_tensor_ptr = aux_copy;
  }
  // Set data values
  for (int b = 0; b < batch_dim; ++b) {
    for (int i = 0; i < aux_tensor_ptr->size; ++i)
      value_tensor->add_float_data(aux_tensor_ptr->ptr[i]);
  }
  if (!layer->const_tensor->isCPU())
    // Delete the CPU copy
    delete aux_tensor_ptr;

  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

#endif // defined(cPROTO)
