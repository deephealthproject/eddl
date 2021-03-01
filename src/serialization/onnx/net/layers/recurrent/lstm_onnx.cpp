#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/recurrent/lstm_onnx.h"
#include "eddl/serialization/onnx/layers/core/squeeze_onnx.h"
#include "eddl/serialization/onnx/import_helpers.h"

// ONNX import
Layer* build_lstm_layer(onnx::NodeProto *node,
                         map<string, vector<float>> &map_init_values,
                         map<string, vector<int>> &map_init_dims,
                         map<string, vector<onnx::NodeProto *>> &input_node_map,
                         map<string, Layer *> &output_node_map,
                         vector<string> &inputs2remove,
                         LOG_LEVEL log_level,
                         int dev,
                         int mem)
{
  log_string("LSTM layer detected", log_level, LOG_LEVEL::DEBUG);
  string name = node->name();     // Name of the layer
  vector<float> activation_alpha; // Values for configuring some activations with extra parameters
  vector<float> activation_beta;  // Values for configuring some activations with extra parameters
  vector<string> activations;     // Activation functions in order for each gate
  float clip = -1;                // Value for clipping
  string direction = "";          // Forward, backward or reverse (Forward by default)
  int hidden_size = -1;           // Number of neurons in the hidden layer
  int input_forget = 0;           // If 1, couple the input and forget gates

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
    { // Not used yet in eddl but implemented. We default to Sigmoid, Sigmoid, Sigmoid, TanH
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
        msg("LSTM layer " + name + " is not forward direction. EDDL only supports one-directional LSTM",
            "ONNX::ImportNet");
      }
    }
    else if (!attr_name.compare("hidden_size"))
    {
      hidden_size = attribute.i();
    }
    else if (!attr_name.compare("input_forget"))
    { // Not used yet in eddl but we read it
      input_forget = attribute.i();
    }
  }

  string parent_name = node->input(0); // Get parent
  Layer *parent = output_node_map[parent_name];
  vector<int> parent_shape = parent->output->shape;
  vector<Layer *> parents = {parent};

  /*
   * Check if the layer is Decoder by checking if there is not a recurrent layer after this one. To avoid
   * conflicts with the stacked LSTM layers that are encoders.
   */
  bool is_decoder = node_is_decoder(node, input_node_map);

  if (is_decoder)
  {
    log_string("The layer " + name + " is decoder", log_level, LOG_LEVEL::DEBUG);
    // We have to create the copy states layer for the decoder
    Layer *parent_hstate = output_node_map[node->input(5)]; // 5: hidden state
    Layer *cps = new LCopyStates({parent_hstate}, "", dev, mem);
    parents.push_back(cps); // Add the layer to the parents for the LSTM
  }

  if (hidden_size < 0)
  {
    cerr << "Model contains a LSTM without the number of neurons" << endl;
  }

  string weights_gates = node->input(1); // Get weights and dims
  vector<float> *weights_g = &(map_init_values[weights_gates]);
  vector<int> dims_g = map_init_dims[weights_gates];
  int input_size = dims_g[2];

  // Load input weights with shape [hidden_size, input_size]. After load we transpose
  //    Note: EDDL input weights are of shape [input_size, hidden_size]
  vector<int> dims_input_lstm = {dims_g[1] / 4, dims_g[2]};

  vector<float> *weights_input_g = new vector<float>;
  vector<float> *weights_output_g = new vector<float>;
  vector<float> *weights_forget_g = new vector<float>;
  vector<float> *weights_cell_g = new vector<float>;
  int w_size = input_size * hidden_size;
  weights_input_g->assign(weights_g->begin() + w_size * 0, weights_g->begin() + w_size * 1);
  weights_output_g->assign(weights_g->begin() + w_size * 1, weights_g->begin() + w_size * 2);
  weights_forget_g->assign(weights_g->begin() + w_size * 2, weights_g->begin() + w_size * 3);
  weights_cell_g->assign(weights_g->begin() + w_size * 3, weights_g->begin() + w_size * 4);

  string recurrence_weights_gates = node->input(2); // Get weights and dims
  vector<float> *recurrence_weights_g = &(map_init_values[recurrence_weights_gates]);
  vector<int> recurrence_dims_g = map_init_dims[recurrence_weights_gates];

  vector<int> dims_recurrent_lstm = {recurrence_dims_g[2], recurrence_dims_g[2]};

  vector<float> *recurrence_weights_input_g = new vector<float>;
  vector<float> *recurrence_weights_output_g = new vector<float>;
  vector<float> *recurrence_weights_forget_g = new vector<float>;
  vector<float> *recurrence_weights_cell_g = new vector<float>;
  w_size = hidden_size * hidden_size;
  recurrence_weights_input_g->assign(recurrence_weights_g->begin() + w_size * 0, recurrence_weights_g->begin() + w_size * 1);
  recurrence_weights_output_g->assign(recurrence_weights_g->begin() + w_size * 1, recurrence_weights_g->begin() + w_size * 2);
  recurrence_weights_forget_g->assign(recurrence_weights_g->begin() + w_size * 2, recurrence_weights_g->begin() + w_size * 3);
  recurrence_weights_cell_g->assign(recurrence_weights_g->begin() + w_size * 3, recurrence_weights_g->begin() + w_size * 4);

  LLSTM *lstm = new LLSTM(parents, hidden_size, 0, 0, name, dev, mem);

  if (is_decoder)
  {
    // Set attribute for unrolling
    lstm->isdecoder = true;
    set_decoder(lstm->parent[0]);
    // We also have to remove the input layer that feeds the decoder from the input layers of the model
    // First we search the corresponding input layer for the decoder
    Layer *dec_linput = get_model_input_layer(lstm);
    if (dec_linput != nullptr)
      inputs2remove.push_back(dec_linput->name);
    else
      msg("Input layer for decoder " + name + " not found", "ONNX::ImportNet");
  }

  /*
   * The Weights are permuted before copying them to the LSTM layer (mismatch between ONNX standad and EDDL implementation)
   */
  Tensor *weights_input_tensor = new Tensor(dims_input_lstm, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(weights_input_g, weights_input_tensor);
  weights_input_tensor->permute_({1, 0});
  Tensor::copy(weights_input_tensor, lstm->Wix);
  delete weights_input_tensor;
  delete weights_input_g;

  Tensor *weights_output_tensor = new Tensor(dims_input_lstm, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(weights_output_g, weights_output_tensor);
  weights_output_tensor->permute_({1, 0});
  Tensor::copy(weights_output_tensor, lstm->Wox);
  delete weights_output_tensor;
  delete weights_output_g;

  Tensor *weights_forget_tensor = new Tensor(dims_input_lstm, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(weights_forget_g, weights_forget_tensor);
  weights_forget_tensor->permute_({1, 0});
  Tensor::copy(weights_forget_tensor, lstm->Wfx);
  delete weights_forget_tensor;
  delete weights_forget_g;

  Tensor *weights_cell_tensor = new Tensor(dims_input_lstm, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(weights_cell_g, weights_cell_tensor);
  weights_cell_tensor->permute_({1, 0});
  Tensor::copy(weights_cell_tensor, lstm->Wcx);
  delete weights_cell_tensor;
  delete weights_cell_g;

  Tensor *recurrence_weights_input_tensor = new Tensor(dims_recurrent_lstm, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(recurrence_weights_input_g, recurrence_weights_input_tensor);
  recurrence_weights_input_tensor->permute_({1, 0});
  Tensor::copy(recurrence_weights_input_tensor, lstm->Wih);
  delete recurrence_weights_input_tensor;
  delete recurrence_weights_input_g;

  Tensor *recurrence_weights_output_tensor = new Tensor(dims_recurrent_lstm, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(recurrence_weights_output_g, recurrence_weights_output_tensor);
  recurrence_weights_output_tensor->permute_({1, 0});
  Tensor::copy(recurrence_weights_output_tensor, lstm->Woh);
  delete recurrence_weights_output_tensor;
  delete recurrence_weights_output_g;

  Tensor *recurrence_weights_forget_tensor = new Tensor(dims_recurrent_lstm, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(recurrence_weights_forget_g, recurrence_weights_forget_tensor);
  recurrence_weights_forget_tensor->permute_({1, 0});
  Tensor::copy(recurrence_weights_forget_tensor, lstm->Wfh);
  delete recurrence_weights_forget_tensor;
  delete recurrence_weights_forget_g;

  Tensor *recurrence_weights_cell_tensor = new Tensor(dims_recurrent_lstm, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(recurrence_weights_cell_g, recurrence_weights_cell_tensor);
  recurrence_weights_cell_tensor->permute_({1, 0});
  Tensor::copy(recurrence_weights_cell_tensor, lstm->Wch);
  delete recurrence_weights_cell_tensor;
  delete recurrence_weights_cell_g;

  /*
   * Set bias values
   */
  vector<int> bias_dims = {hidden_size};
  // Vectors to store the imported weights
  vector<float> *bias_input = new vector<float>;
  vector<float> *bias_output = new vector<float>;
  vector<float> *bias_forget = new vector<float>;
  vector<float> *bias_cell = new vector<float>;
  vector<float> *bias_recurrence_input = new vector<float>;
  vector<float> *bias_recurrence_output = new vector<float>;
  vector<float> *bias_recurrence_forget = new vector<float>;
  vector<float> *bias_recurrence_cell = new vector<float>;

  if (node->input_size() > 3) {
    string biases_name = node->input(3); //Get weights and dims
    vector<float> *biases = &(map_init_values[biases_name]);

    bias_input->assign(biases->begin() + hidden_size * 0, biases->begin() + hidden_size * 1);
    bias_output->assign(biases->begin() + hidden_size * 1, biases->begin() + hidden_size * 2);
    bias_forget->assign(biases->begin() + hidden_size * 2, biases->begin() + hidden_size * 3);
    bias_cell->assign(biases->begin() + hidden_size * 3, biases->begin() + hidden_size * 4);
    bias_recurrence_input->assign(biases->begin() + hidden_size * 4, biases->begin() + hidden_size * 5);
    bias_recurrence_output->assign(biases->begin() + hidden_size * 5, biases->begin() + hidden_size * 6);
    bias_recurrence_forget->assign(biases->begin() + hidden_size * 6, biases->begin() + hidden_size * 7);
    bias_recurrence_cell->assign(biases->begin() + hidden_size * 7, biases->begin() + hidden_size * 8);
  } else {
    // Set bias values to 0.0
    //   Note: In EDDL we don't have use_bias option for LSTM so to achieve the same
    //         result we set the bias values to 0.0
    vector<float> zero_bias(hidden_size, 0.0);
    bias_input->assign(zero_bias.begin(), zero_bias.begin() + hidden_size);
    bias_output->assign(zero_bias.begin(), zero_bias.begin() + hidden_size);
    bias_forget->assign(zero_bias.begin(), zero_bias.begin() + hidden_size);
    bias_cell->assign(zero_bias.begin(), zero_bias.begin() + hidden_size);
    bias_recurrence_input->assign(zero_bias.begin(), zero_bias.begin() + hidden_size);
    bias_recurrence_output->assign(zero_bias.begin(), zero_bias.begin() + hidden_size);
    bias_recurrence_forget->assign(zero_bias.begin(), zero_bias.begin() + hidden_size);
    bias_recurrence_cell->assign(zero_bias.begin(), zero_bias.begin() + hidden_size);
  }


  Tensor *bias_input_tensor = new Tensor(bias_dims, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(bias_input, bias_input_tensor);
  Tensor::copy(bias_input_tensor, lstm->inbias);
  delete bias_input_tensor;
  delete bias_input;

  Tensor *bias_output_tensor = new Tensor(bias_dims, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(bias_output, bias_output_tensor);
  Tensor::copy(bias_output_tensor, lstm->onbias);
  delete bias_output_tensor;
  delete bias_output;

  Tensor *bias_forget_tensor = new Tensor(bias_dims, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(bias_forget, bias_forget_tensor);
  Tensor::copy(bias_forget_tensor, lstm->fnbias);
  delete bias_forget_tensor;
  delete bias_forget;

  Tensor *bias_cell_tensor = new Tensor(bias_dims, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(bias_cell, bias_cell_tensor);
  Tensor::copy(bias_cell_tensor, lstm->cnbias);
  delete bias_cell_tensor;
  delete bias_cell;

  // Add the recurrent bias values
  Tensor *bias_recurrence_input_tensor = new Tensor(bias_dims, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(bias_recurrence_input, bias_recurrence_input_tensor);
  Tensor::add(bias_recurrence_input_tensor, lstm->inbias, lstm->inbias);
  delete bias_recurrence_input_tensor;
  delete bias_recurrence_input;

  Tensor *bias_recurrence_output_tensor = new Tensor(bias_dims, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(bias_recurrence_output, bias_recurrence_output_tensor);
  Tensor::add(bias_recurrence_output_tensor, lstm->onbias, lstm->onbias);
  delete bias_recurrence_output_tensor;
  delete bias_recurrence_output;

  Tensor *bias_recurrence_forget_tensor = new Tensor(bias_dims, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(bias_recurrence_forget, bias_recurrence_forget_tensor);
  Tensor::add(bias_recurrence_forget_tensor, lstm->fnbias, lstm->fnbias);
  delete bias_recurrence_forget_tensor;
  delete bias_recurrence_forget;

  Tensor *bias_recurrence_cell_tensor = new Tensor(bias_dims, nullptr, dev);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(bias_recurrence_cell, bias_recurrence_cell_tensor);
  Tensor::add(bias_recurrence_cell_tensor, lstm->cnbias, lstm->cnbias);
  delete bias_recurrence_cell_tensor;
  delete bias_recurrence_cell;

  log_string("LSTM layer created", log_level, LOG_LEVEL::DEBUG);
  return lstm;
}

// ONNX export
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
    //   Note: The EDDL only supports one-directional LSTM, so num_directions=1
    squeeze_node_builder(
        layer->name + "_outputSqueeze", // node name
        layer->name + "_Y_h",           // input name
        layer->name,                    // Output name
        {0},                            // axes to squeeze
        graph);
  }
}

#endif // defined(cPROTO)
