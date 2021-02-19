#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/recurrent/rnn_onnx.h"
#include "eddl/serialization/onnx/layers/core/squeeze_onnx.h"

void build_rnn_node(LRNN *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("RNN");
  node->set_name(layer->name);
  // Set the input sequence of the RNN
  node->add_input(layer->parent[0]->name);
  node->add_input(layer->name + "_W");
  node->add_input(layer->name + "_R");
  if (layer->use_bias) node->add_input(layer->name + "_B");
  else node->add_input("");
  node->add_input(""); // Empty str to skip the sequence_lens input
  // Check if we have to copy states for a decoder RNN
  if (layer->parent.size() > 1 && layer->isdecoder)
  {
    string l_copyStates_name = layer->parent[1]->name;
    node->add_input(l_copyStates_name + "_h");
  }

  // Attr activations
  float alpha, beta; // optional auxiliary parameters
  bool activation_with_params = false;
  onnx::AttributeProto *activations_attr = node->add_attribute();
  activations_attr->set_name("activations");
  activations_attr->set_type(onnx::AttributeProto::STRINGS);
  if (layer->activation == "relu")
    activations_attr->add_strings("Relu");
  else if (layer->activation == "sigmoid")
    activations_attr->add_strings("Sigmoid");
  else if (layer->activation == "hard_sigmoid") {
    activations_attr->add_strings("HardSigmoid");
    alpha = 0.2;
    beta = 0.5;
    activation_with_params = true;
  } else if (layer->activation == "tanh")
    activations_attr->add_strings("Tanh");
  else if (layer->activation == "none") {
    activations_attr->add_strings("Affine");
    // Achieve linear activation: alpha * x + beta -> where alpha = 1.0 and beta = 0.0
    alpha = 1.0;
    beta = 0.0;
    activation_with_params = true;
  } else
    msg("Activation not supported for RNN", "ONNX::ExportNet");

  if (activation_with_params) {
    // Auxiliary alpha attribute for the activation functions
    onnx::AttributeProto* activation_alpha_attr = node->add_attribute();
    activation_alpha_attr->set_name("activation_alpha");
    activation_alpha_attr->set_type(onnx::AttributeProto::FLOATS);
    activation_alpha_attr->add_floats(alpha);

    // Auxiliary beta attribute for the activation functions
    onnx::AttributeProto* activation_beta_attr = node->add_attribute();
    activation_beta_attr->set_name("activation_beta");
    activation_beta_attr->set_type(onnx::AttributeProto::FLOATS);
    activation_beta_attr->add_floats(beta);
  }

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
  direction_attr->set_s("forward"); // Current implementation of RNN

  // Attr hidden size
  onnx::AttributeProto *hidden_size_attr = node->add_attribute();
  hidden_size_attr->set_name("hidden_size");
  hidden_size_attr->set_type(onnx::AttributeProto::INT);
  hidden_size_attr->set_i(layer->units);

  // Weights for input
  onnx::TensorProto *w = graph->add_initializer();
  w->set_name(layer->name + "_W");
  w->set_data_type(onnx::TensorProto::FLOAT);
  vector<int> w_dims{1, layer->units, layer->input->shape[1]}; // w_dims shape[0] = 1 beacuse is only forward
  w->mutable_dims()->Add(w_dims.begin(), w_dims.end());        // Set the shape of the weights
  /*
   * The Weights are permuted before saving them (required by ONNX standad)
   */
  Tensor *Wx = layer->Wx->permute({1, 0});
  w->mutable_float_data()->Add(Wx->ptr, Wx->ptr + Wx->size);
  delete Wx;

  // Recurrent weights
  onnx::TensorProto *r = graph->add_initializer();
  r->set_name(layer->name + "_R");
  r->set_data_type(onnx::TensorProto::FLOAT);
  vector<int> r_dims{1, layer->units, layer->units};     // r_dims shape[0] = 1 beacuse is only forward
  r->mutable_dims()->Add(r_dims.begin(), r_dims.end());  // Set the shape of the weights
  /*
   * The Weights are permuted before saving them (required by ONNX standad)
   */
  Tensor *Wy = layer->Wy->permute({1, 0});
  r->mutable_float_data()->Add(Wy->ptr, Wy->ptr + Wy->size);
  delete Wy;

  // Bias
  if (layer->use_bias) {
    onnx::TensorProto *b = graph->add_initializer();
    b->set_name(layer->name + "_B");
    b->set_data_type(onnx::TensorProto::FLOAT);
    vector<int> b_dims{1, 2 * layer->units};              // b_dims shape[0] = 1 for weights in one directions
    b->mutable_dims()->Add(b_dims.begin(), b_dims.end()); // Set the shape of the weights
    b->mutable_float_data()->Add(layer->bias->ptr, layer->bias->ptr + layer->bias->size);
    // Set recurrent biases to 0
    for (int i = 0; i < layer->units; ++i)
      b->add_float_data(0.0);
  }

  /* Set the outputs of the node to link with the other nodes
   *   - In ONNX the LSTM operator can have up to 2 outputs:
   *       * Y -> [seq_len, num_directions, batch_size, hidden_size]
   *       * Y_h (optional) -> [num_directions, batch_size, hidden_size]
   *   - If the layer is encoder we select Y_h as output
   *   - If the layer is encoder but there are more stacked RNN, we select Y as output
   *   - If the layer is decoder we select Y as output
   *
   *   Note: To select the output of the RNN that the next layer in the graph takes as input
   *         we have to set that output name to the layer name (layer->name)
   */
  node->add_output(layer->name + "_Y");
  node->add_output(layer->name + "_Y_h");
  if (layer->isdecoder || layer->child[0]->isrecurrent /*To detect stacked RNN*/)
  {
    // Squeeze: [seq_length, num_directions, batch_size, hidden_size] -> [seq_length, batch_size, hidden_size]
    //   Note: The EDDL only supports one-directional RNN, so num_directions=1
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
    //   Note: The EDDL only supports one-directional RNN, so num_directions=1
    build_squeeze_node(
        layer->name + "_outputSqueeze", // node name
        layer->name + "_Y_h",           // input name
        layer->name,                    // Output name
        {0},                            // axes to squeeze
        graph);
  }
}

#endif // defined(cPROTO)
