#if defined(cPROTO)
#include "eddl/serialization/onnx/import_helpers.h"
#include "eddl/serialization/onnx/layers/recurrent/rnn_onnx.h"
#include "eddl/serialization/onnx/layers/core/squeeze_onnx.h"

// ONNX import
Layer* build_rnn_layer(onnx::NodeProto *node,
                       map<string, vector<float>> &map_init_values,
                       map<string, vector<int>> &map_init_dims,
                       map<string, vector<onnx::NodeProto *>> &input_node_map,
                       map<string, Layer *> &output_node_map,
                       vector<string> &inputs2remove,
                       LOG_LEVEL log_level,
                       int dev,
                       int mem)
{
  log_string("RNN layer detected", log_level, LOG_LEVEL::DEBUG);
  string name = node->name();     // Name of the layer
  vector<float> activation_alpha; // Values for configuring some activations with extra parameters
  vector<float> activation_beta;  // Values for configuring some activations with extra parameters
  vector<string> activations;     // Activation functions in order for each gate
  float clip = -1;                // Value for clipping
  string direction = "";          // Forward, backward or reverse (Forward by default)
  int hidden_size = -1;           // Number of neurons in the hidden layer
  bool use_bias = node->input_size() > 3;

  for (int j = 0; j < node->attribute_size(); j++)
  { // Set the attributes
    onnx::AttributeProto attribute = node->attribute(j);
    string attr_name = attribute.name();
    if (!attr_name.compare("activation_alpha"))
    { // Not used yet in eddl but implemented
      for (int h = 0; h < attribute.floats_size(); h++)
      {
        activation_alpha.push_back(attribute.floats(h));
      }
    }
    else if (!attr_name.compare("activation_beta"))
    { // Not used yet in eddl but implemented
      for (int h = 0; h < attribute.floats_size(); h++)
      {
        activation_beta.push_back(attribute.floats(h));
      }
    }
    else if (!attr_name.compare("activations"))
    {
      for (int h = 0; h < attribute.strings_size(); h++)
      {
        activations.push_back(attribute.strings(h));
      }
    }
    else if (!attr_name.compare("clip"))
    { // Not used yet in eddl but implemented
      clip = attribute.f();
    }
    else if (!attr_name.compare("direction"))
    {
      direction = attribute.s();
      if (direction.compare("forward")) 
      {
        msg("RNN layer " + name + " is not forward direction. EDDL only supports one-directional RNN", "ONNX::ImportNet");
      }
    }
    else if (!attr_name.compare("hidden_size"))
    {
      hidden_size = attribute.i();
    }
    //else if (!attr_name.compare("linear_before_reset")) {}
  }

  // Take forward activation function
  string activation;
  if (activations.size() > 0) {
    string forward_activation = activations[0];
    if (forward_activation == "Relu")
      activation = "relu";
    else if (forward_activation == "Sigmoid")
      activation = "sigmoid";
    else if (forward_activation == "HardSigmoid") {
      float epsilon = 1e-5;
      float alpha = 0.2;
      float beta = 0.5;
      if (activation_alpha.size() > 0) alpha = activation_alpha[0]; 
      if (activation_beta.size() > 0) beta = activation_beta[0]; 
      bool is_not_valid = abs(alpha - 0.2) > epsilon;
      is_not_valid |= abs(beta - 0.5) > epsilon;
      // Check that is equivalent to our hard sigmoid implementation
      if (is_not_valid) {
        msg("The HardSigmoid activation function with alpha != 0.2 or beta != 0.5 is not supported for RNN.",
            "ONNX::ImportNet");
      } else {
        activation = "hard_sigmoid";
      }
    } else if (forward_activation == "Tanh")
      activation = "tanh";
    else if (forward_activation == "Affine") {
      float alpha = 1.0;
      float beta = 0.0;
      if (activation_alpha.size() > 0) alpha = activation_alpha[0]; 
      if (activation_beta.size() > 0) beta = activation_beta[0]; 
      // Check that is equivalent to linear activation function
      if (alpha != 1.0 || beta != 0.0) {
        msg("The Affine activation function with alpha != 1.0 or beta != 0.0 is not supported for RNN.",
            "ONNX::ImportNet");
      } else {
        activation = "none";
      }
    } else
      msg("Activation function \"" + forward_activation + "\" is not supported for RNN.",
          "ONNX::ImportNet");
  } else {
    msg("RNN layer " + name + " doesn't provide an activation function.", 
        "ONNX::ImportNet");
  }

  if (hidden_size < 0)
    msg("RNN layer " + name + " doesn't have the number of neurons.", "ONNX::ImportNet");

  string parent_name = node->input(0); // Get parent
  Layer *parent = output_node_map[parent_name];
  vector<int> parent_shape = parent->output->shape;
  vector<Layer *> parents = {parent};

  /*
   * Check if the layer is Decoder by checking if there is not a recurrent layer after this one. To avoid
   * conflicts with the stacked RNN layers that are encoders.
   */
  bool is_decoder = node_is_decoder(node, input_node_map);

  if (is_decoder)
  {
    log_string("The layer " + name + " is decoder", log_level, LOG_LEVEL::DEBUG);
    // We have to create the copy states layer for the decoder
    Layer *parent_hstate = output_node_map[node->input(5)]; // 5: hidden state
    Layer *cps = new LCopyStates({parent_hstate}, "", dev, mem);
    parents.push_back(cps); // Add the layer to the parents for the RNN
  }

  string weights_gates = node->input(1); // Get weights and dims
  vector<float> *weights_g = &(map_init_values[weights_gates]);
  vector<int> dims_g = map_init_dims[weights_gates];
  int input_size = dims_g[2];

  // Load input weights with shape [hidden_size, input_size]. After load we transpose
  //    Note: EDDL input weights are of shape [input_size, hidden_size]
  vector<int> dims_input_gru = {dims_g[1], input_size};

  vector<float> *weights_x = new vector<float>;
  int w_size = input_size * hidden_size;
  weights_x->assign(weights_g->begin() , weights_g->begin() + w_size);

  string recurrence_weights_gates = node->input(2); // Get weights and dims
  vector<float> *recurrence_weights_g = &(map_init_values[recurrence_weights_gates]);
  vector<int> recurrence_dims_g = map_init_dims[recurrence_weights_gates];

  vector<int> dims_recurrent_gru = {recurrence_dims_g[2], recurrence_dims_g[2]};

  vector<float> *weights_h = new vector<float>;
  w_size = hidden_size * hidden_size;
  weights_h->assign(recurrence_weights_g->begin(), recurrence_weights_g->begin() + w_size);

  LRNN *rnn = new LRNN(parents, hidden_size, activation, use_bias, false, name, dev, mem);

  if (is_decoder)
  {
    // Set attribute for unrolling
    rnn->isdecoder = true;
    set_decoder(rnn->parent[0]);
    // We also have to remove the input layer that feeds the decoder from the input layers of the model
    // First we search the corresponding input layer for the decoder
    Layer *dec_linput = get_model_input_layer(rnn);
    if (dec_linput != nullptr)
      inputs2remove.push_back(dec_linput->name);
    else
      msg("Input layer for decoder " + name + " not found", "ONNX::ImportNet");
  }

  /*
   * The Weights are permuted before copying them to the RNN layer (mismatch between ONNX standad and EDDL implementation)
   */
  Tensor *weights_x_tensor = new Tensor(dims_input_gru, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(weights_x, weights_x_tensor);
  weights_x_tensor->permute_({1, 0});
  Tensor::copy(weights_x_tensor, rnn->Wx);
  delete weights_x_tensor;
  delete weights_x;

  Tensor *weights_h_tensor = new Tensor(dims_recurrent_gru, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(weights_h, weights_h_tensor);
  weights_h_tensor->permute_({1, 0});
  Tensor::copy(weights_h_tensor, rnn->Wy);
  delete weights_h_tensor;
  delete weights_h;

  if (use_bias) {
    string biases_name = node->input(3);
    vector<float> *biases = &(map_init_values[biases_name]);
    vector<int> bias_dims = {hidden_size};

    vector<float> *bias_x = new vector<float>;
    vector<float> *bias_h = new vector<float>;

    bias_x->assign(biases->begin() + hidden_size * 0, biases->begin() + hidden_size * 1);
    bias_h->assign(biases->begin() + hidden_size * 1, biases->begin() + hidden_size * 2);

    Tensor *bias_x_tensor = new Tensor(bias_dims, nullptr, dev);
    COPY_FROM_VECTOR_PTR_TO_TENSOR(bias_x, bias_x_tensor);
    Tensor::copy(bias_x_tensor, rnn->bias);
    delete bias_x_tensor;
    delete bias_x;

    // Add the recurrent bias values for gates z and r
    Tensor *bias_h_tensor = new Tensor(bias_dims, nullptr, dev);
    COPY_FROM_VECTOR_PTR_TO_TENSOR(bias_h, bias_h_tensor);
    Tensor::add(bias_h_tensor, rnn->bias, rnn->bias);
    delete bias_h_tensor;
    delete bias_h;
  }

  log_string("RNN layer created", log_level, LOG_LEVEL::DEBUG);
  return rnn;
}

// ONNX export
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
    squeeze_node_builder(
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
    squeeze_node_builder(
        layer->name + "_outputSqueeze", // node name
        layer->name + "_Y_h",           // input name
        layer->name,                    // Output name
        {0},                            // axes to squeeze
        graph);
  }
}

#endif // defined(cPROTO)
