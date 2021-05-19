#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/da/pad_onnx.h"
#include "eddl/serialization/onnx/utils_onnx.h"

// ONNX import
Layer* build_pad_layer(onnx::NodeProto *node,
                       map<string, vector<float>> &map_init_values,
                       map<string, Layer *> &output_node_map,
                       int dev,
                       int mem)
{
  vector<int> pads;
  string mode("constant");
  float constant = 0.0;
  bool op_version_11 = true; // To mark if the op version is >= 11

  for (int j = 0; j < node->attribute_size(); j++)
  { // Set the attributes
    const onnx::AttributeProto& attribute = node->attribute(j);
    const string& attr_name = attribute.name();
    if (attr_name == "mode")
      if (attribute.s() != "constant")
        msg("Error importing layer " + node->name() + ". Only \"constant\" mode is supported (passed \"" + attribute.s() + "\").", "ONNX::ImportNet");
    else if (attr_name == "pads")
    {
      op_version_11 = false;
      for (int i = 0; i < attribute.ints_size(); ++i)
        pads.push_back(attribute.ints(i));
    }
    else if (attr_name == "value")
      constant = attribute.f();
  }

  const string& parent_name = node->input(0);
  Layer *parent = output_node_map[parent_name];

  if (op_version_11) // We have to take the pad values from a node input
  {
    if (node->input_size() > 1)
    {
      const string& pads_name = node->input(1);
      int n_pad_values = (&(map_init_values[pads_name]))->size();
      float *pads_values = new float [n_pad_values];
      COPY_FROM_VECTOR_PTR_TO_FLOAT_PTR(&(map_init_values[pads_name]), pads_values);

      if (n_pad_values != 8)
        msg("Error importing layer " + node->name() + ". The expected number of padding values is 8, got " +
            to_string(n_pad_values) + ".", "ONNX::ImportNet");
      else
      {
        // pads_values = (batch_begin, ch_begin, h_begin, w_begin, batch_end, ch_end, h_end, w_end)
        // EDDL wants = (h_begin, w_end, h_end, w_begin) == (top, right, bottom, left)
        pads.push_back(pads_values[2]);
        pads.push_back(pads_values[7]);
        pads.push_back(pads_values[6]);
        pads.push_back(pads_values[3]);
      }

      delete [] pads_values;
    }
    if (node->input_size() > 2) // Constant value provided as input
    {
      const string& constant_name = node->input(2);
      int n_pad_values = (&(map_init_values[constant_name]))->size();
      float *constant_value = new float [n_pad_values];
      COPY_FROM_VECTOR_PTR_TO_FLOAT_PTR(&(map_init_values[constant_name]), constant_value);

      if (n_pad_values != 1)
        msg("Unexpected error importing layer " + node->name() + ". The number of constant values is not 1", "ONNX::ImportNet");
      else
        constant = constant_value[0];

      delete [] constant_value;
    }
  }

  return new LPad(parent, pads, constant, node->name(), DEV_CPU, 0);
}

// ONNX export
void build_pad_node(LPad *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Pad");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
    node->add_input(parentl->name);
  node->add_input(layer->name + "_pads");
  node->add_input(layer->name + "_constant_value");
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);

  // Padding mode attribute
  onnx::AttributeProto *mode_attr = node->add_attribute();
  mode_attr->set_name("mode");
  mode_attr->set_type(onnx::AttributeProto::STRING);
  mode_attr->set_s("constant");

  // "pads" input
  int n_pad_values = layer->padding.size();
  if (n_pad_values != 4)
    msg("Unexpected error exporting layer " + layer->name + ". The number of padding values is not 4, is " +
        to_string(n_pad_values) + ".", "ONNX::ExportNet");
  // Create a initializer to store the pads values
  onnx::TensorProto *roi = graph->add_initializer();
  roi->set_name(layer->name + "_pads");
  roi->set_data_type(onnx::TensorProto::INT64);
  roi->add_dims(4 + n_pad_values); // 4 for (batch_begin, ch_begin, batch_end, ch_end)
  // layer->padding = (top, right, bottom, left)
  // ONNX needs => (batch_begin, ch_begin, h_begin, w_begin, batch_end, ch_end, h_end, w_end)
  roi->add_int64_data(0); // batch_begin
  roi->add_int64_data(0); // ch_begin
  roi->add_int64_data(layer->padding[0]); // h_begin == top
  roi->add_int64_data(layer->padding[3]); // w_begin == left
  roi->add_int64_data(0); // batch_end
  roi->add_int64_data(0); // ch_end
  roi->add_int64_data(layer->padding[2]); // h_end == bottom
  roi->add_int64_data(layer->padding[1]); // w_end == right

  // Constant value for padding
  onnx::TensorProto *scales = graph->add_initializer();
  scales->set_name(layer->name + "_constant_value");
  scales->set_data_type(onnx::TensorProto::FLOAT);
  scales->add_dims(1);
  scales->add_float_data(layer->constant);
}

#endif // defined(cPROTO)
