#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/onnx_helpers/export_helpers.h"

void build_identity_node(string node_name, string input, string output, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Identity");
  node->set_name(node_name);
  node->add_input(input);
  node->add_output(output);
}

void build_cast_node(string node_name, string input, string output, int cast_type, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Cast");
  node->set_name(node_name);
  node->add_input(input);
  node->add_output(output);
  /*
   * Attr "to". To select the type of the cast
   *
   * Available types to cast (from TensorProto class in "onnx.proto") :
   *   FLOAT = 1;   // float
   *   UINT8 = 2;   // uint8_t
   *   INT8 = 3;    // int8_t
   *   UINT16 = 4;  // uint16_t
   *   INT16 = 5;   // int16_t
   *   INT32 = 6;   // int32_t
   *   INT64 = 7;   // int64_t
   *   STRING = 8;  // string
   *   BOOL = 9;    // bool
   */
  onnx::AttributeProto *to_attr = node->add_attribute();
  to_attr->set_name("to");
  to_attr->set_type(onnx::AttributeProto::INT);
  to_attr->set_i(cast_type);
}

void build_gather_node(string node_name, string input, string output, LEmbedding *layer, onnx::GraphProto *graph)
{
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Gather");
  node->set_name(node_name);
  // Set the inputs: word indexes and embedding values (data)
  node->add_input(layer->name + "_data");
  node->add_input(input);
  node->add_output(output);

  // Create the initializer with the embedding data
  onnx::TensorProto *embed_data = graph->add_initializer();
  embed_data->set_name(layer->name + "_data");
  embed_data->set_data_type(onnx::TensorProto::FLOAT);
  vector<int> embed_data_dims{layer->vocsize, layer->dim};
  embed_data->mutable_dims()->Add(embed_data_dims.begin(), embed_data_dims.end());      // Set the shape of the weights
  embed_data->mutable_float_data()->Add(layer->E->ptr, layer->E->ptr + layer->E->size); // Set the data values
}

#endif // defined(cPROTO)
