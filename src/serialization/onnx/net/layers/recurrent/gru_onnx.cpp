#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/recurrent/gru_onnx.h"
#include "eddl/serialization/onnx/layers/core/squeeze_onnx.h"

void build_gru_node(LGRU *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("GRU");
  node->set_name(layer->name);
  // Set the input sequence of the GRU
  node->add_input(layer->parent[0]->name);
  node->add_input(layer->name + "_W");
  node->add_input(layer->name + "_R");
  node->add_input(layer->name + "_B");
  node->add_input(""); // Empty str to skip the sequence_lens input
  // Check if we have to copy states for a decoder GRU
  if (layer->parent.size() > 1 && layer->isdecoder)
  {
    string l_copyStates_name = layer->parent[1]->name;
    node->add_input(l_copyStates_name + "_h");
  }

  // Attr activation alpha (for GRU activation functions)
  // Not used in EDDL
  //onnx::AttributeProto* activation_alpha_attr = node->add_attribute();
  //activation_alpha_attr->set_name( "activation_alpha" );
  //activation_alpha_attr->set_type( onnx::AttributeProto::FLOATS );

  // Attr activation beta
  // Not used in EDDL
  //onnx::AttributeProto* activation_beta_attr = node->add_attribute();
  //activation_beta_attr->set_name( "activation_beta" );  // Not used in EDDL
  //activation_beta_attr->set_type( onnx::AttributeProto::FLOATS );

  // Attr activations
  onnx::AttributeProto *activations_attr = node->add_attribute();
  activations_attr->set_name("activations");
  activations_attr->set_type(onnx::AttributeProto::STRINGS);
  activations_attr->add_strings("Sigmoid"); // For gates z, r
  activations_attr->add_strings("Tanh");    // For gate n

  // Attr clip (cell clip threshold, [-threshold, +threshold])
  // Not used in EDDL
  //onnx::AttributeProto* hidden_size_attr = node->add_attribute();
  //hidden_size_attr->set_name( "clip" );
  //hidden_size_attr->set_type( onnx::AttributeProto::FLOAT );
  //hidden_size_attr->set_i( /*?*/ );

  // Attr direction
  onnx::AttributeProto *direction_attr = node->add_attribute();
  direction_attr->set_name("direction");
  direction_attr->set_type(onnx::AttributeProto::STRING);
  direction_attr->set_s("forward"); // Current implementation of GRU

  // Attr hidden size
  onnx::AttributeProto *hidden_size_attr = node->add_attribute();
  hidden_size_attr->set_name("hidden_size");
  hidden_size_attr->set_type(onnx::AttributeProto::INT);
  hidden_size_attr->set_i(layer->units);

  // Attr linear transformation before reset
  onnx::AttributeProto *linear_trans_attr = node->add_attribute();
  linear_trans_attr->set_name("linear_before_reset");
  linear_trans_attr->set_type(onnx::AttributeProto::INT);
  // We apply the linear transformation before the r gate.
  // See "linear_before_reset" attribute in  https://github.com/onnx/onnx/blob/master/docs/Operators.md#GRU
  linear_trans_attr->set_i(1);

  // Warn if the layer uses mask zeros. Not supported in ONNX
  if (layer->mask_zeros) {
    cout << "[ONNX::Export] Warning: The GRU layer " << layer->name << " has mask_zeros=true. "
         << "This attribute is not supported in ONNX, so the model exported will not have this attribute."
         << endl;
  }

  // W input (weights for all the layers W[zrn])
  onnx::TensorProto *w = graph->add_initializer();
  w->set_name(layer->name + "_W");
  w->set_data_type(onnx::TensorProto::FLOAT);
  vector<int> w_dims{1, 3 * layer->units, layer->input->shape[1]}; // w_dims shape[0] = 1 beacuse is only forward
  w->mutable_dims()->Add(w_dims.begin(), w_dims.end());            // Set the shape of the weights
  /*
   * The Weights are permuted before saving them (required by ONNX standad)
   */
  Tensor *Wz_x = layer->Wz_x->permute({1, 0});
  w->mutable_float_data()->Add(Wz_x->ptr, Wz_x->ptr + Wz_x->size); // z weights
  delete Wz_x;
  Tensor *Wr_x = layer->Wr_x->permute({1, 0});
  w->mutable_float_data()->Add(Wr_x->ptr, Wr_x->ptr + Wr_x->size); // r weights
  delete Wr_x;
  Tensor *Wn_x = layer->Wn_x->permute({1, 0});
  w->mutable_float_data()->Add(Wn_x->ptr, Wn_x->ptr + Wn_x->size); // n weights
  delete Wn_x;

  // R input (recurrent weights for all the layers W[zrh])
  onnx::TensorProto *r = graph->add_initializer();
  r->set_name(layer->name + "_R");
  r->set_data_type(onnx::TensorProto::FLOAT);
  vector<int> r_dims{1, 3 * layer->units, layer->units}; // r_dims shape[0] = 1 beacuse is only forward
  r->mutable_dims()->Add(r_dims.begin(), r_dims.end());  // Set the shape of the weights
  /*
   * The Weights are permuted before saving them (required by ONNX standad)
   */
  Tensor *Wz_hidden = layer->Uz_h->permute({1, 0});
  r->mutable_float_data()->Add(Wz_hidden->ptr, Wz_hidden->ptr + Wz_hidden->size); // z recurrent weights
  delete Wz_hidden;
  Tensor *Wr_hidden = layer->Ur_h->permute({1, 0});
  r->mutable_float_data()->Add(Wr_hidden->ptr, Wr_hidden->ptr + Wr_hidden->size); // r recurrent weights
  delete Wr_hidden;
  Tensor *Wn_hidden = layer->Un_h->permute({1, 0});
  r->mutable_float_data()->Add(Wn_hidden->ptr, Wn_hidden->ptr + Wn_hidden->size); // n recurrent weights
  delete Wn_hidden;

  // B input (biases for all the layers)
  onnx::TensorProto *b = graph->add_initializer();
  b->set_name(layer->name + "_B");
  b->set_data_type(onnx::TensorProto::FLOAT);
  vector<int> b_dims{1, 6 * layer->units};              // b_dims shape[0] = 1 for weights in one directions
  b->mutable_dims()->Add(b_dims.begin(), b_dims.end()); // Set the shape of the weights

  b->mutable_float_data()->Add(layer->bias_z_t->ptr, layer->bias_z_t->ptr + layer->bias_z_t->size); // z bias
  b->mutable_float_data()->Add(layer->bias_r_t->ptr, layer->bias_r_t->ptr + layer->bias_r_t->size); // r bias
  b->mutable_float_data()->Add(layer->bias_n_t->ptr, layer->bias_n_t->ptr + layer->bias_n_t->size); // n bias

  // Set recurrent forward biases to 0 for gates z and r
  for (int i = 0; i < 2 * layer->units; ++i)
    b->add_float_data(0.0);

  // The recurrent bias for n is set. Because we need it for applying the linear transformation before the
  // r gate. See "linear_before_reset" attribute in  https://github.com/onnx/onnx/blob/master/docs/Operators.md#GRU
  b->mutable_float_data()->Add(layer->bias_n_t_hidden->ptr, layer->bias_n_t_hidden->ptr + layer->bias_n_t_hidden->size); // n recurrent bias

  /* Set the outputs of the node to link with the other nodes
   *   - In ONNX the GRU operator can have up to 2 outputs:
   *       * Y -> [seq_len, num_directions, batch_size, hidden_size]
   *       * Y_h (optional) -> [num_directions, batch_size, hidden_size]
   *   - If the layer is encoder we select Y_h as output
   *   - If the layer is encoder but there are more stacked GRU, we select Y as output
   *   - If the layer is decoder we select Y as output
   *
   *   Note: To select the output of the GRU that the next layer in the graph takes as input
   *         we have to set that output name to the layer name (layer->name)
   */
  node->add_output(layer->name + "_Y");
  node->add_output(layer->name + "_Y_h");
  if (layer->isdecoder || layer->child[0]->isrecurrent /*To detect stacked GRU*/)
  {
    // Squeeze: [seq_length, num_directions, batch_size, hidden_size] -> [seq_length, batch_size, hidden_size]
    //   Note: The EDDL only supports one-directional GRU, so num_directions=1
    build_squeeze_node(
        layer->name + "_outputSqueeze", // node name
        layer->name + "_Y",             // input name
        layer->name,                    // Output name
        {1},                            // axes to squeeze
        graph);
  }
  else
  { // is encoder
    // Squeeze: [num_directions, batch_size, hidden_size] -> [batch_size, hidden_size]
    //   Note: The EDDL only supports one-directional GRU, so num_directions=1
    build_squeeze_node(
        layer->name + "_outputSqueeze", // node name
        layer->name + "_Y_h",           // input name
        layer->name,                    // Output name
        {0},                            // axes to squeeze
        graph);
  }
}

#endif // defined(cPROTO)
