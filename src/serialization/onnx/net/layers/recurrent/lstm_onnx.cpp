#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/recurrent/lstm_onnx.h"
#include "eddl/serialization/onnx/layers/core/squeeze_onnx.h"

void build_lstm_node(LLSTM *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("LSTM");
  node->set_name(layer->name);
  // Set the input sequence of the LSTM
  node->add_input(layer->parent[0]->name);
  node->add_input(layer->name + "_W");
  node->add_input(layer->name + "_R");
  node->add_input(layer->name + "_B");
  node->add_input(""); // Empty str to skip the sequence_lens input
  // Check if we have to copy states for a decoder LSTM
  if (layer->parent.size() > 1 && layer->isdecoder)
  {
    string l_copyStates_name = layer->parent[1]->name;
    node->add_input(l_copyStates_name + "_h");
    node->add_input(l_copyStates_name + "_c");
  }

  // Attr activation alpha (for LSTM activation functions)
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
  activations_attr->add_strings("Sigmoid"); // For gates i, f, o
  activations_attr->add_strings("Tanh");    // For gate c
  activations_attr->add_strings("Tanh");    // For gate h

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
  direction_attr->set_s("forward"); // Current implementation of LSTM

  // Attr hidden size
  onnx::AttributeProto *hidden_size_attr = node->add_attribute();
  hidden_size_attr->set_name("hidden_size");
  hidden_size_attr->set_type(onnx::AttributeProto::INT);
  hidden_size_attr->set_i(layer->units);

  // Attr input forget
  onnx::AttributeProto *input_forget_attr = node->add_attribute();
  input_forget_attr->set_name("input_forget");
  input_forget_attr->set_type(onnx::AttributeProto::INT);
  input_forget_attr->set_i(0); // To not couple the input and forget gates

  // Warn if the layer uses mask zeros. Not supported in ONNX
  if (layer->mask_zeros) {
    cout << "[ONNX::Export] Warning: The LSTM layer " << layer->name << " has mask_zeros=true. "
         << "This attribute is not supported in ONNX, so the model exported will not have this attribute."
         << endl;
  }

  // W input (weights for all the layers W[iofc])
  onnx::TensorProto *w = graph->add_initializer();
  w->set_name(layer->name + "_W");
  w->set_data_type(onnx::TensorProto::FLOAT);
  vector<int> w_dims{1, 4 * layer->units, layer->input->shape[1]}; // w_dims shape[0] = 1 beacuse is only forward
  w->mutable_dims()->Add(w_dims.begin(), w_dims.end());            // Set the shape of the weights
  /*
   * The Weights are permuted before saving them (required by ONNX standad)
   */
  Tensor *Wix = layer->Wix->permute({1, 0});
  w->mutable_float_data()->Add(Wix->ptr, Wix->ptr + Wix->size); // i weights
  delete Wix;
  Tensor *Wox = layer->Wox->permute({1, 0});
  w->mutable_float_data()->Add(Wox->ptr, Wox->ptr + Wox->size); // o weights
  delete Wox;
  Tensor *Wfx = layer->Wfx->permute({1, 0});
  w->mutable_float_data()->Add(Wfx->ptr, Wfx->ptr + Wfx->size); // f weights
  delete Wfx;
  Tensor *Wcx = layer->Wcx->permute({1, 0});
  w->mutable_float_data()->Add(Wcx->ptr, Wcx->ptr + Wcx->size); // c weights
  delete Wcx;

  // R input (recurrent weights for all the layers W[iofc])
  onnx::TensorProto *r = graph->add_initializer();
  r->set_name(layer->name + "_R");
  r->set_data_type(onnx::TensorProto::FLOAT);
  vector<int> r_dims{1, 4 * layer->units, layer->units}; // r_dims shape[0] = 1 beacuse is only forward
  r->mutable_dims()->Add(r_dims.begin(), r_dims.end());  // Set the shape of the weights
  /*
   * The Weights are permuted before saving them (required by ONNX standad)
   */
  Tensor *Wih = layer->Wih->permute({1, 0});
  r->mutable_float_data()->Add(Wih->ptr, Wih->ptr + Wih->size); // i recurrent weights
  delete Wih;
  Tensor *Woh = layer->Woh->permute({1, 0});
  r->mutable_float_data()->Add(Woh->ptr, Woh->ptr + Woh->size); // o recurrent weights
  delete Woh;
  Tensor *Wfh = layer->Wfh->permute({1, 0});
  r->mutable_float_data()->Add(Wfh->ptr, Wfh->ptr + Wfh->size); // f recurrent weights
  delete Wfh;
  Tensor *Wch = layer->Wch->permute({1, 0});
  r->mutable_float_data()->Add(Wch->ptr, Wch->ptr + Wch->size); // c recurrent weights
  delete Wch;

  // B input (biases for all the layers)
  onnx::TensorProto *b = graph->add_initializer();
  b->set_name(layer->name + "_B");
  b->set_data_type(onnx::TensorProto::FLOAT);
  vector<int> b_dims{1, 8 * layer->units};              // b_dims shape[0] = 1 for weights in one directions
  b->mutable_dims()->Add(b_dims.begin(), b_dims.end()); // Set the shape of the weights

  b->mutable_float_data()->Add(layer->inbias->ptr, layer->inbias->ptr + layer->inbias->size); // i bias
  b->mutable_float_data()->Add(layer->onbias->ptr, layer->onbias->ptr + layer->onbias->size); // o bias
  b->mutable_float_data()->Add(layer->fnbias->ptr, layer->fnbias->ptr + layer->fnbias->size); // f bias
  b->mutable_float_data()->Add(layer->cnbias->ptr, layer->cnbias->ptr + layer->cnbias->size); // c bias

  // Set recurrent forward biases to 0 (only one bias used, not one for x and another for h)
  for (int i = 0; i < 4 * layer->units; ++i)
    b->add_float_data(0);

  // Set backward biases to 0 (bidirectional LSTM not implemented)
  //for( int i = 0; i < 8*layer->units; ++i )
  //	b->add_float_data(0);

  /* Set the outputs of the node to link with the other nodes
   *   - In ONNX the LSTM operator can have up to 3 outputs:
   *       * Y -> [seq_len, num_directions, batch_size, hidden_size]
   *       * Y_h (optional) -> [num_directions, batch_size, hidden_size]
   *       * Y_c (optional) -> [num_directions, batch_size, hidden_size]
   *   - If the layer is encoder we select Y_h as output
   *   - If the layer is encoder but there are more stacked LSTM, we select Y as output
   *   - If the layer is decoder we select Y as output
   *
   *   Note: To select the output of the LSTM that the next layer in the graph takes as input
   *         we have to set that output name to the layer name (layer->name)
   */
  node->add_output(layer->name + "_Y");
  node->add_output(layer->name + "_Y_h");
  node->add_output(layer->name + "_Y_c");
  if (layer->isdecoder || layer->child[0]->isrecurrent /*To detect stacked LSTM*/)
  {
    // Squeeze: [seq_length, num_directions, batch_size, hidden_size] -> [seq_length, batch_size, hidden_size]
    //   Note: The EDDL only supports one-directional LSTM, so num_directions=1
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
    //   Note: The EDDL only supports one-directional LSTM, so num_directions=1
    build_squeeze_node(
        layer->name + "_outputSqueeze", // node name
        layer->name + "_Y_h",           // input name
        layer->name,                    // Output name
        {0},                            // axes to squeeze
        graph);
  }
}

#endif // defined(cPROTO)
