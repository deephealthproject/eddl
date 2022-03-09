#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/recurrent/gru_onnx.h"
#include "eddl/serialization/onnx/layers/core/squeeze_onnx.h"
#include "eddl/serialization/onnx/import_helpers.h"

// ONNX import
Layer* build_gru_layer(onnx::NodeProto *node,
                       map<string, vector<float>> &map_init_values,
                       map<string, vector<int>> &map_init_dims,
                       map<string, vector<onnx::NodeProto *>> &input_node_map,
                       map<string, Layer *> &output_node_map,
                       LOG_LEVEL log_level,
                       int dev,
                       int mem)
{
  log_string("GRU layer detected", log_level, LOG_LEVEL::DEBUG);
  string name = node->name();     // Name of the layer
  vector<float> activation_alpha; // Values for configuring some activations with extra parameters
  vector<float> activation_beta;  // Values for configuring some activations with extra parameters
  vector<string> activations;     // Activation functions in order for each gate
  float clip = -1;                // Value for clipping
  string direction = "";          // Forward, backward or reverse (Forward by default)
  int hidden_size = -1;           // Number of neurons in the hidden layer

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
    { // Not used yet in eddl but implemented. We default to Sigmoid, TanH
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
        msg("GRU layer " + name + " is not forward direction. EDDL only supports one-directional GRU", "ONNX::ImportNet");
      }
    }
    else if (!attr_name.compare("hidden_size"))
    {
      hidden_size = attribute.i();
    }
    //else if (!attr_name.compare("linear_before_reset")) {}
  }

  if (hidden_size < 0)
    msg("The layer " + name + " (GRU) does not provide the hidden_size attribute.", "[ONNX::ImportNet]");

  string parent_name = node->input(0); // Get parent
  Layer *parent = output_node_map[parent_name];
  vector<int> parent_shape = parent->output->shape;
  vector<Layer *> parents = {parent};

  /*
   * Check if the layer is Decoder by checking if there is not a recurrent layer after this one. To avoid
   * conflicts with the stacked GRU layers that are encoders.
   */
  bool is_decoder = node_is_decoder(node, input_node_map);

  if (is_decoder && node->input_size() > 5)
  {
    log_string("The layer " + name + " is decoder", log_level, LOG_LEVEL::DEBUG);
    // We have to create the copy states layer for the decoder
    Layer *parent_hstate = output_node_map[node->input(5)]; // 5: hidden state
    Layer *cps = new LCopyStates({parent_hstate}, "", dev, mem);
    parents.push_back(cps); // Add the layer to the parents for the GRU
  }

  string weights_gates = node->input(1); // Get weights and dims
  vector<float> *weights_g = &(map_init_values[weights_gates]);
  vector<int> dims_g = map_init_dims[weights_gates];
  int input_size = dims_g[2];

  // Load input weights with shape [hidden_size, input_size]. After load we transpose
  //    Note: EDDL input weights are of shape [input_size, hidden_size]
  vector<int> dims_input_gru = {dims_g[1] / 3, input_size};

  vector<float> *weights_z_g = new vector<float>;
  vector<float> *weights_r_g = new vector<float>;
  vector<float> *weights_n_g = new vector<float>;
  int w_size = input_size * hidden_size;
  weights_z_g->assign(weights_g->begin() + w_size * 0, weights_g->begin() + w_size * 1);
  weights_r_g->assign(weights_g->begin() + w_size * 1, weights_g->begin() + w_size * 2);
  weights_n_g->assign(weights_g->begin() + w_size * 2, weights_g->begin() + w_size * 3);

  string recurrence_weights_gates = node->input(2); // Get weights and dims
  vector<float> *recurrence_weights_g = &(map_init_values[recurrence_weights_gates]);
  vector<int> recurrence_dims_g = map_init_dims[recurrence_weights_gates];

  vector<int> dims_recurrent_gru = {recurrence_dims_g[2], recurrence_dims_g[2]};

  vector<float> *recurrence_weights_z_g = new vector<float>;
  vector<float> *recurrence_weights_r_g = new vector<float>;
  vector<float> *recurrence_weights_n_g = new vector<float>;
  w_size = hidden_size * hidden_size;
  recurrence_weights_z_g->assign(recurrence_weights_g->begin() + w_size * 0, recurrence_weights_g->begin() + w_size * 1);
  recurrence_weights_r_g->assign(recurrence_weights_g->begin() + w_size * 1, recurrence_weights_g->begin() + w_size * 2);
  recurrence_weights_n_g->assign(recurrence_weights_g->begin() + w_size * 2, recurrence_weights_g->begin() + w_size * 3);

  LGRU *gru = new LGRU(parents, hidden_size, 0, 0, name, dev, mem);

  /*
   * The Weights are permuted before copying them to the GRU layer (mismatch between ONNX standad and EDDL implementation)
   */
  Tensor *weights_z_tensor = new Tensor(dims_input_gru, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(weights_z_g, weights_z_tensor);
  weights_z_tensor->permute_({1, 0});
  Tensor::copy(weights_z_tensor, gru->Wz_x);
  delete weights_z_tensor;
  delete weights_z_g;

  Tensor *weights_r_tensor = new Tensor(dims_input_gru, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(weights_r_g, weights_r_tensor);
  weights_r_tensor->permute_({1, 0});
  Tensor::copy(weights_r_tensor, gru->Wr_x);
  delete weights_r_tensor;
  delete weights_r_g;

  Tensor *weights_n_tensor = new Tensor(dims_input_gru, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(weights_n_g, weights_n_tensor);
  weights_n_tensor->permute_({1, 0});
  Tensor::copy(weights_n_tensor, gru->Wn_x);
  delete weights_n_tensor;
  delete weights_n_g;

  Tensor *recurrence_weights_z_tensor = new Tensor(dims_recurrent_gru, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(recurrence_weights_z_g, recurrence_weights_z_tensor);
  recurrence_weights_z_tensor->permute_({1, 0});
  Tensor::copy(recurrence_weights_z_tensor, gru->Uz_h);
  delete recurrence_weights_z_tensor;
  delete recurrence_weights_z_g;

  Tensor *recurrence_weights_r_tensor = new Tensor(dims_recurrent_gru, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(recurrence_weights_r_g, recurrence_weights_r_tensor);
  recurrence_weights_r_tensor->permute_({1, 0});
  Tensor::copy(recurrence_weights_r_tensor, gru->Ur_h);
  delete recurrence_weights_r_tensor;
  delete recurrence_weights_r_g;

  Tensor *recurrence_weights_n_tensor = new Tensor(dims_recurrent_gru, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(recurrence_weights_n_g, recurrence_weights_n_tensor);
  recurrence_weights_n_tensor->permute_({1, 0});
  Tensor::copy(recurrence_weights_n_tensor, gru->Un_h);
  delete recurrence_weights_n_tensor;
  delete recurrence_weights_n_g;

  /*
   * Set bias values
   */
  vector<int> bias_dims = {hidden_size};
  // Vectors to store the imported weights
  vector<float> *bias_z = new vector<float>;
  vector<float> *bias_r = new vector<float>;
  vector<float> *bias_n = new vector<float>;
  vector<float> *bias_recurrence_z = new vector<float>;
  vector<float> *bias_recurrence_r = new vector<float>;
  vector<float> *bias_recurrence_n = new vector<float>;

  if (node->input_size() > 3) { // Check that we have bias
    string biases_name = node->input(3);
    vector<float> *biases = &(map_init_values[biases_name]);
    // Forward bias (zrh)
    bias_z->assign(biases->begin() + hidden_size * 0, biases->begin() + hidden_size * 1);
    bias_r->assign(biases->begin() + hidden_size * 1, biases->begin() + hidden_size * 2);
    bias_n->assign(biases->begin() + hidden_size * 2, biases->begin() + hidden_size * 3);
    // Recurrent bias (zrh)
    bias_recurrence_z->assign(biases->begin() + hidden_size * 3, biases->begin() + hidden_size * 4);
    bias_recurrence_r->assign(biases->begin() + hidden_size * 4, biases->begin() + hidden_size * 5);
    bias_recurrence_n->assign(biases->begin() + hidden_size * 5, biases->begin() + hidden_size * 6);
  } else {
    // Set bias values to 0.0
    //   Note: In EDDL we don't have use_bias option for GRU so to achieve the same
    //         result we set the bias values to 0.0
    vector<float> zero_bias(hidden_size, 0.0);
    bias_z->assign(zero_bias.begin(), zero_bias.begin() + hidden_size);
    bias_r->assign(zero_bias.begin(), zero_bias.begin() + hidden_size);
    bias_n->assign(zero_bias.begin(), zero_bias.begin() + hidden_size);
    bias_recurrence_z->assign(zero_bias.begin(), zero_bias.begin() + hidden_size);
    bias_recurrence_r->assign(zero_bias.begin(), zero_bias.begin() + hidden_size);
    bias_recurrence_n->assign(zero_bias.begin(), zero_bias.begin() + hidden_size);
  }

  Tensor *bias_z_tensor = new Tensor(bias_dims, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(bias_z, bias_z_tensor);
  Tensor::copy(bias_z_tensor, gru->bias_z_t);
  delete bias_z_tensor;
  delete bias_z;

  Tensor *bias_r_tensor = new Tensor(bias_dims, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(bias_r, bias_r_tensor);
  Tensor::copy(bias_r_tensor, gru->bias_r_t);
  delete bias_r_tensor;
  delete bias_r;

  Tensor *bias_n_tensor = new Tensor(bias_dims, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(bias_n, bias_n_tensor);
  Tensor::copy(bias_n_tensor, gru->bias_n_t);
  delete bias_n_tensor;
  delete bias_n;

  // Add the recurrent bias values for gates z and r
  Tensor *bias_recurrence_z_tensor = new Tensor(bias_dims, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(bias_recurrence_z, bias_recurrence_z_tensor);
  Tensor::add(bias_recurrence_z_tensor, gru->bias_z_t, gru->bias_z_t);
  delete bias_recurrence_z_tensor;
  delete bias_recurrence_z;

  Tensor *bias_recurrence_r_tensor = new Tensor(bias_dims, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(bias_recurrence_r, bias_recurrence_r_tensor);
  Tensor::add(bias_recurrence_r_tensor, gru->bias_r_t, gru->bias_r_t);
  delete bias_recurrence_r_tensor;
  delete bias_recurrence_r;

  // The recurrent bias for h goes to its own tensor beacuse we need it for applying the linear transformation
  // before the r gate. See "linear_before_reset" attribute in  https://github.com/onnx/onnx/blob/master/docs/Operators.md#GRU
  Tensor *bias_recurrence_n_tensor = new Tensor(bias_dims, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(bias_recurrence_n, bias_recurrence_n_tensor);
  Tensor::copy(bias_recurrence_n_tensor, gru->bias_n_t_hidden);
  delete bias_recurrence_n_tensor;
  delete bias_recurrence_n;

  log_string("GRU layer created", log_level, LOG_LEVEL::DEBUG);
  return gru;
}

// ONNX export
void build_gru_node(LGRU *layer, onnx::GraphProto *graph, bool gradients)
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
  if (!gradients) {
    Tensor *Wz_x = layer->Wz_x->permute({1, 0});
    w->mutable_float_data()->Add(Wz_x->ptr, Wz_x->ptr + Wz_x->size); // z weights
    delete Wz_x;
    Tensor *Wr_x = layer->Wr_x->permute({1, 0});
    w->mutable_float_data()->Add(Wr_x->ptr, Wr_x->ptr + Wr_x->size); // r weights
    delete Wr_x;
    Tensor *Wn_x = layer->Wn_x->permute({1, 0});
    w->mutable_float_data()->Add(Wn_x->ptr, Wn_x->ptr + Wn_x->size); // n weights
    delete Wn_x;
  } else {
    Tensor *Wz_x = layer->acc_gWz_x->permute({1, 0});
    w->mutable_float_data()->Add(Wz_x->ptr, Wz_x->ptr + Wz_x->size); // z weights
    delete Wz_x;
    Tensor *Wr_x = layer->acc_gWr_x->permute({1, 0});
    w->mutable_float_data()->Add(Wr_x->ptr, Wr_x->ptr + Wr_x->size); // r weights
    delete Wr_x;
    Tensor *Wn_x = layer->acc_gWn_x->permute({1, 0});
    w->mutable_float_data()->Add(Wn_x->ptr, Wn_x->ptr + Wn_x->size); // n weights
    delete Wn_x;
  }

  // R input (recurrent weights for all the layers W[zrh])
  onnx::TensorProto *r = graph->add_initializer();
  r->set_name(layer->name + "_R");
  r->set_data_type(onnx::TensorProto::FLOAT);
  vector<int> r_dims{1, 3 * layer->units, layer->units}; // r_dims shape[0] = 1 beacuse is only forward
  r->mutable_dims()->Add(r_dims.begin(), r_dims.end());  // Set the shape of the weights
  /*
   * The Weights are permuted before saving them (required by ONNX standad)
   */
  if (!gradients) {
    Tensor *Wz_hidden = layer->Uz_h->permute({1, 0});
    r->mutable_float_data()->Add(Wz_hidden->ptr, Wz_hidden->ptr + Wz_hidden->size); // z recurrent weights
    delete Wz_hidden;
    Tensor *Wr_hidden = layer->Ur_h->permute({1, 0});
    r->mutable_float_data()->Add(Wr_hidden->ptr, Wr_hidden->ptr + Wr_hidden->size); // r recurrent weights
    delete Wr_hidden;
    Tensor *Wn_hidden = layer->Un_h->permute({1, 0});
    r->mutable_float_data()->Add(Wn_hidden->ptr, Wn_hidden->ptr + Wn_hidden->size); // n recurrent weights
    delete Wn_hidden;
  } else {
    Tensor *Wz_hidden = layer->acc_gUz_h->permute({1, 0});
    r->mutable_float_data()->Add(Wz_hidden->ptr, Wz_hidden->ptr + Wz_hidden->size); // z recurrent weights
    delete Wz_hidden;
    Tensor *Wr_hidden = layer->acc_gUr_h->permute({1, 0});
    r->mutable_float_data()->Add(Wr_hidden->ptr, Wr_hidden->ptr + Wr_hidden->size); // r recurrent weights
    delete Wr_hidden;
    Tensor *Wn_hidden = layer->acc_gUn_h->permute({1, 0});
    r->mutable_float_data()->Add(Wn_hidden->ptr, Wn_hidden->ptr + Wn_hidden->size); // n recurrent weights
    delete Wn_hidden;
  }

  // B input (biases for all the layers)
  onnx::TensorProto *b = graph->add_initializer();
  b->set_name(layer->name + "_B");
  b->set_data_type(onnx::TensorProto::FLOAT);
  vector<int> b_dims{1, 6 * layer->units};              // b_dims shape[0] = 1 for weights in one directions
  b->mutable_dims()->Add(b_dims.begin(), b_dims.end()); // Set the shape of the weights

  if (!gradients) {
    b->mutable_float_data()->Add(layer->bias_z_t->ptr, layer->bias_z_t->ptr + layer->bias_z_t->size); // z bias
    b->mutable_float_data()->Add(layer->bias_r_t->ptr, layer->bias_r_t->ptr + layer->bias_r_t->size); // r bias
    b->mutable_float_data()->Add(layer->bias_n_t->ptr, layer->bias_n_t->ptr + layer->bias_n_t->size); // n bias
  } else {
    b->mutable_float_data()->Add(layer->acc_g_bias_z_t->ptr, layer->acc_g_bias_z_t->ptr + layer->acc_g_bias_z_t->size); // z bias
    b->mutable_float_data()->Add(layer->acc_g_bias_r_t->ptr, layer->acc_g_bias_r_t->ptr + layer->acc_g_bias_r_t->size); // r bias
    b->mutable_float_data()->Add(layer->acc_g_bias_n_t->ptr, layer->acc_g_bias_n_t->ptr + layer->acc_g_bias_n_t->size); // n bias
  }

  // Set recurrent forward biases to 0 for gates z and r
  for (int i = 0; i < 2 * layer->units; ++i)
    b->add_float_data(0.0);

  // The recurrent bias for n is set. Because we need it for applying the linear transformation before the
  // r gate. See "linear_before_reset" attribute in  https://github.com/onnx/onnx/blob/master/docs/Operators.md#GRU
  if (!gradients)
    b->mutable_float_data()->Add(layer->bias_n_t_hidden->ptr, layer->bias_n_t_hidden->ptr + layer->bias_n_t_hidden->size); // n recurrent bias
  else
    b->mutable_float_data()->Add(layer->acc_g_bias_n_t_hidden->ptr, layer->acc_g_bias_n_t_hidden->ptr + layer->acc_g_bias_n_t_hidden->size); // n recurrent bias

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
    //   Note: The EDDL only supports one-directional GRU, so num_directions=1
    squeeze_node_builder(
        layer->name + "_outputSqueeze", // node name
        layer->name + "_Y_h",           // input name
        layer->name,                    // Output name
        {0},                            // axes to squeeze
        graph);
  }
}

/*
 * DISTRIBUTED TRAINING
 */

vector<Tensor *> get_gru_tensors(onnx::NodeProto &node,
                                 map<string, vector<float>> &map_init_values,
                                 map<string, vector<int>> &map_init_dims)
{
  vector<Tensor *> gru_tensors;
  int hidden_size = -1;           // Number of neurons in the hidden layer

  for (int j = 0; j < node.attribute_size(); j++)
  { // Set the attributes
    onnx::AttributeProto attribute = node.attribute(j);
    string attr_name = attribute.name();
    if (!attr_name.compare("hidden_size")) {
      hidden_size = attribute.i();
      break;
    }
  }

  if (hidden_size < 0)
    msg("The layer " + node.name() + " (GRU) does not provide the hidden_size attribute.", "[ONNX::ImportNet]");

  string weights_gates = node.input(1); // Get weights and dims
  vector<float> *weights_g = &(map_init_values[weights_gates]);
  vector<int> dims_g = map_init_dims[weights_gates];
  int input_size = dims_g[2];

  // Load input weights with shape [hidden_size, input_size]. After load we transpose
  //    Note: EDDL input weights are of shape [input_size, hidden_size]
  vector<int> dims_input_gru = {dims_g[1] / 3, input_size};

  vector<float> *weights_z_g = new vector<float>;
  vector<float> *weights_r_g = new vector<float>;
  vector<float> *weights_n_g = new vector<float>;
  int w_size = input_size * hidden_size;
  weights_z_g->assign(weights_g->begin() + w_size * 0, weights_g->begin() + w_size * 1);
  weights_r_g->assign(weights_g->begin() + w_size * 1, weights_g->begin() + w_size * 2);
  weights_n_g->assign(weights_g->begin() + w_size * 2, weights_g->begin() + w_size * 3);

  string recurrence_weights_gates = node.input(2); // Get weights and dims
  vector<float> *recurrence_weights_g = &(map_init_values[recurrence_weights_gates]);
  vector<int> recurrence_dims_g = map_init_dims[recurrence_weights_gates];

  vector<int> dims_recurrent_gru = {recurrence_dims_g[2], recurrence_dims_g[2]};

  vector<float> *recurrence_weights_z_g = new vector<float>;
  vector<float> *recurrence_weights_r_g = new vector<float>;
  vector<float> *recurrence_weights_n_g = new vector<float>;
  w_size = hidden_size * hidden_size;
  recurrence_weights_z_g->assign(recurrence_weights_g->begin() + w_size * 0, recurrence_weights_g->begin() + w_size * 1);
  recurrence_weights_r_g->assign(recurrence_weights_g->begin() + w_size * 1, recurrence_weights_g->begin() + w_size * 2);
  recurrence_weights_n_g->assign(recurrence_weights_g->begin() + w_size * 2, recurrence_weights_g->begin() + w_size * 3);

  /*
   * The Weights are permuted before copying them to the GRU layer (mismatch between ONNX standad and EDDL implementation)
   */
  int dev = DEV_CPU;
  Tensor *weights_z_tensor = new Tensor(dims_input_gru, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(weights_z_g, weights_z_tensor);
  weights_z_tensor->permute_({1, 0});
  delete weights_z_g;

  Tensor *weights_r_tensor = new Tensor(dims_input_gru, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(weights_r_g, weights_r_tensor);
  weights_r_tensor->permute_({1, 0});
  delete weights_r_g;

  Tensor *weights_n_tensor = new Tensor(dims_input_gru, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(weights_n_g, weights_n_tensor);
  weights_n_tensor->permute_({1, 0});
  delete weights_n_g;

  Tensor *recurrence_weights_z_tensor = new Tensor(dims_recurrent_gru, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(recurrence_weights_z_g, recurrence_weights_z_tensor);
  recurrence_weights_z_tensor->permute_({1, 0});
  delete recurrence_weights_z_g;

  Tensor *recurrence_weights_r_tensor = new Tensor(dims_recurrent_gru, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(recurrence_weights_r_g, recurrence_weights_r_tensor);
  recurrence_weights_r_tensor->permute_({1, 0});
  delete recurrence_weights_r_g;

  Tensor *recurrence_weights_n_tensor = new Tensor(dims_recurrent_gru, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(recurrence_weights_n_g, recurrence_weights_n_tensor);
  recurrence_weights_n_tensor->permute_({1, 0});
  delete recurrence_weights_n_g;

  gru_tensors.push_back(weights_z_tensor);
  gru_tensors.push_back(weights_r_tensor);
  gru_tensors.push_back(weights_n_tensor);
  gru_tensors.push_back(recurrence_weights_z_tensor);
  gru_tensors.push_back(recurrence_weights_r_tensor);
  gru_tensors.push_back(recurrence_weights_n_tensor);

  /*
   * Set bias values
   */
  vector<int> bias_dims = {hidden_size};
  // Vectors to store the imported weights
  vector<float> *bias_z = new vector<float>;
  vector<float> *bias_r = new vector<float>;
  vector<float> *bias_n = new vector<float>;
  vector<float> *bias_recurrence_z = new vector<float>;
  vector<float> *bias_recurrence_r = new vector<float>;
  vector<float> *bias_recurrence_n = new vector<float>;

  if (node.input_size() > 3) { // Check that we have bias
    string biases_name = node.input(3);
    vector<float> *biases = &(map_init_values[biases_name]);
    // Forward bias (zrh)
    bias_z->assign(biases->begin() + hidden_size * 0, biases->begin() + hidden_size * 1);
    bias_r->assign(biases->begin() + hidden_size * 1, biases->begin() + hidden_size * 2);
    bias_n->assign(biases->begin() + hidden_size * 2, biases->begin() + hidden_size * 3);
    // Recurrent bias (zrh)
    bias_recurrence_z->assign(biases->begin() + hidden_size * 3, biases->begin() + hidden_size * 4);
    bias_recurrence_r->assign(biases->begin() + hidden_size * 4, biases->begin() + hidden_size * 5);
    bias_recurrence_n->assign(biases->begin() + hidden_size * 5, biases->begin() + hidden_size * 6);
  } else {
    // Set bias values to 0.0
    //   Note: In EDDL we don't have use_bias option for GRU so to achieve the same
    //         result we set the bias values to 0.0
    vector<float> zero_bias(hidden_size, 0.0);
    bias_z->assign(zero_bias.begin(), zero_bias.begin() + hidden_size);
    bias_r->assign(zero_bias.begin(), zero_bias.begin() + hidden_size);
    bias_n->assign(zero_bias.begin(), zero_bias.begin() + hidden_size);
    bias_recurrence_z->assign(zero_bias.begin(), zero_bias.begin() + hidden_size);
    bias_recurrence_r->assign(zero_bias.begin(), zero_bias.begin() + hidden_size);
    bias_recurrence_n->assign(zero_bias.begin(), zero_bias.begin() + hidden_size);
  }

  Tensor *bias_z_tensor = new Tensor(bias_dims, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(bias_z, bias_z_tensor);
  delete bias_z;

  Tensor *bias_r_tensor = new Tensor(bias_dims, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(bias_r, bias_r_tensor);
  delete bias_r;

  Tensor *bias_n_tensor = new Tensor(bias_dims, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(bias_n, bias_n_tensor);
  delete bias_n;

  gru_tensors.push_back(bias_z_tensor);
  gru_tensors.push_back(bias_r_tensor);
  gru_tensors.push_back(bias_n_tensor);

  // Add the recurrent bias values for gates z and r
  /* Not needed for importing gradients
  Tensor *bias_recurrence_z_tensor = new Tensor(bias_dims, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(bias_recurrence_z, bias_recurrence_z_tensor);
  Tensor::add(bias_recurrence_z_tensor, gru->bias_z_t, gru->bias_z_t);
  delete bias_recurrence_z_tensor;
  delete bias_recurrence_z;

  Tensor *bias_recurrence_r_tensor = new Tensor(bias_dims, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(bias_recurrence_r, bias_recurrence_r_tensor);
  Tensor::add(bias_recurrence_r_tensor, gru->bias_r_t, gru->bias_r_t);
  delete bias_recurrence_r_tensor;
  delete bias_recurrence_r;
  */

  // The recurrent bias for h goes to its own tensor beacuse we need it for applying the linear transformation
  // before the r gate. See "linear_before_reset" attribute in  https://github.com/onnx/onnx/blob/master/docs/Operators.md#GRU
  Tensor *bias_recurrence_n_tensor = new Tensor(bias_dims, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(bias_recurrence_n, bias_recurrence_n_tensor);
  delete bias_recurrence_n;
  gru_tensors.push_back(bias_recurrence_n_tensor);

  return gru_tensors;
}

#endif // defined(cPROTO)
