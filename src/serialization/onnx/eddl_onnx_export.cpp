#include <cstdio>
#include <fstream>
#include "eddl/layers/core/layer_core.h"
#include "eddl/layers/layer.h"
#include "eddl/layers/operators/layer_operators.h"
#include "eddl/layers/recurrent/layer_recurrent.h"
#include "eddl/layers/reductions/layer_reductions.h"
#include "eddl/layers/da/layer_da.h"
#include "eddl/net/net.h"
#include "eddl/optimizers/optim.h"
#include "eddl/serialization/onnx/eddl_onnx.h"

using namespace std;

#if defined(cPROTO)
#include "onnx.pb.h"
#endif

// Builds the onnx model from the net
#if defined(cPROTO)
onnx::ModelProto build_onnx_model(Net *net, bool gradients);

// Builds the graph of the ModelProto from the net
void set_graph(onnx::ModelProto *model, Net *net, bool gradients);

// Builds a node in the onnx graph from the layer of eddl
void build_node_from_layer(Layer *layer, onnx::GraphProto *graph, bool gradients, bool is_recurrent);

// Fixes the input shape for recurrent models
void prepare_recurrent_input(string input_name, string output_name, vector<int> input_shape, onnx::GraphProto *graph);

// Fixes the output shape for recurrent models
void prepare_recurrent_output(string input_name, string output_name, vector<int> output_shape, onnx::GraphProto *graph);

// Synchronize the weights of the snets
void sync_params(Net *net);

// Synchronize the accumulated gradients of the snets
void sync_acc_gradients(Net *net);

// Node builders
//----------------------------------------------------------------------------------------

// OPSET: 11, 1
void build_conv_node(LConv *layer, onnx::GraphProto *graph, bool gradients);

// OPSET: 11, 1 (same operator than Conv2D)
void build_conv1D_node(LConv1D *layer, onnx::GraphProto *graph, bool gradients);

// OPSET: 13, 11
void build_gemm_node(LDense *layer, onnx::GraphProto *graph, bool gradients);

// MatMul OPSET: 13, 9, 1 - Add OPSET: 13, 7
void build_dense_with_matmul_node(LDense *layer, onnx::GraphProto *graph, bool gradients);

// OPSET: 12, 11, 10, 8, 1
void build_maxpool_node(LMaxPool *layer, onnx::GraphProto *graph);

// OPSET: 12, 11, 10, 8, 1
void build_maxpool1D_node(LMaxPool1D *layer, onnx::GraphProto *graph);

// OPSET: 11, 10, 7, 1
void build_averagepool_node(LAveragePool *layer, onnx::GraphProto *graph);

// OPSET: 13, 5
void build_reshape_node(LReshape *layer, onnx::GraphProto *graph);

// OPSET: 13, 1
void build_permute_node(LPermute *layer, onnx::GraphProto *graph);

// OPSET: 14, 13, 6
void build_relu_node(LActivation *layer, onnx::GraphProto *graph);

// OPSET: 13, 6
void build_sigmoid_node(LActivation *layer, onnx::GraphProto *graph);

// OPSET: 6
void build_hard_sigmoid_node(LActivation *layer, onnx::GraphProto *graph);

// OPSET: 13, 6
void build_tanh_node(LActivation *layer, onnx::GraphProto *graph);

// Not in ONNX: Custom operator
void build_exponential_node(LActivation *layer, onnx::GraphProto *graph);

// Not in ONNX: Custom operator
void build_linear_node(LActivation *layer, onnx::GraphProto *graph);

// OPSET: 6
void build_leaky_relu_node(LActivation *layer, onnx::GraphProto *graph);

// OPSET: 10
void build_thresholded_relu_node(LActivation *layer, onnx::GraphProto *graph);

// OPSET: 6
void build_elu_node(LActivation *layer, onnx::GraphProto *graph);

// OPSET: 6
void build_selu_node(LActivation *layer, onnx::GraphProto *graph);

// OPSET: 13, 11, 1
void build_softmax_node(LActivation *layer, onnx::GraphProto *graph);

// OPSET: 1
void build_softsign_node(LActivation *layer, onnx::GraphProto *graph);

// OPSET: 1
void build_softplus_node(LActivation *layer, onnx::GraphProto *graph);

// OPSET: 13, 11, 4, 1
void build_concat_node(LConcat *layer, onnx::GraphProto *graph);

// OPSET: 13, 6
void build_abs_node(LAbs *layer, onnx::GraphProto *graph);

// OPSET: 13, 7
void build_add_node(LAdd *layer, onnx::GraphProto *graph);

// OPSET: 13, 7
void build_div_node(LDiv *layer, onnx::GraphProto *graph);

// OPSET: 13, 6
void build_exp_node(LExp *layer, onnx::GraphProto *graph);

// OPSET: 13, 6
void build_log_node(LLog *layer, onnx::GraphProto *graph);

// OPSET: 13, 7
void build_mul_node(LMult *layer, onnx::GraphProto *graph);

// OPSET: 13, 12, 7
// TODO: Implement layer LPow
//void build_pow_node( LPow *layer, onnx::GraphProto *graph );

// OPSET: 13, 6
void build_sqrt_node(LSqrt *layer, onnx::GraphProto *graph);

// OPSET: 13, 7
void build_sub_node(LDiff *layer, onnx::GraphProto *graph);

// OPSET: 13, 11, 1
void build_rmean_node(LRMean *layer, onnx::GraphProto *graph);

// OPSET: 11, 1
void build_rsum_node(LRSum *layer, onnx::GraphProto *graph);

// OPSET: 13, 12, 11, 1
void build_rmax_node(LRMax *layer, onnx::GraphProto *graph);

// OPSET: 13, 12, 11, 1
void build_rmin_node(LRMin *layer, onnx::GraphProto *graph);

// OPSET: 13, 12, 11, 1
void build_rargmax_node(LRArgmax *layer, onnx::GraphProto *graph);

// OPSET: 9
void build_batchnorm_node(LBatchNorm *layer, onnx::GraphProto *graph);

// OPSET: 10, 7
void build_dropout_node(LDropout *layer, onnx::GraphProto *graph);

// OPSET: 9
void build_upsample_node(LUpSampling *layer, onnx::GraphProto *graph);

// OPSET: 11, 1
void build_squeeze_node(string node_name, string input, string output, vector<int> axes, onnx::GraphProto *graph);

// OPSET: 11, 1
void build_unsqueeze_node(string node_name, string input, string output, vector<int> axes, onnx::GraphProto *graph);

// OPSET: 7, 1
void build_lstm_node(LLSTM *layer, onnx::GraphProto *graph);

// OPSET: 7, 3, 1
void build_gru_node(LGRU *layer, onnx::GraphProto *graph);

// OPSET: 7, 1
void build_rnn_node(LRNN *layer, onnx::GraphProto *graph);

// OPSET: 13
void build_resize_node(LScale *layer, onnx::GraphProto *graph);

// Implemented with Gather Op for OPSET: 13, 11, 1
void build_embedding_node(LEmbedding *layer, onnx::GraphProto *graph);

// OPSET: 13, 1
void build_identity_node(string node_name, string input, string output, onnx::GraphProto *graph);

// OPSET: 13, 9, 6
void build_cast_node(string node_name, string input, string output, int cast_type, onnx::GraphProto *graph);

// OPSET: 13, 11, 1
void build_gather_node(string node_name, string input, string output, LEmbedding *layer, onnx::GraphProto *graph);

// Not an ONNX operator. Built from unsqueeze and identity operators.
void handle_copy_states(LCopyStates *layer, onnx::GraphProto *graph);

#endif

#ifdef cPROTO

void sync_params(Net *net)
{
  for (int j = 0; j < net->layers.size(); j++)
  {
    for (int k = 0; k < net->layers[j]->params.size(); k++)
    {
      net->layers[j]->params[k]->fill_(0.0);
      for (int i = 0; i < net->snets.size(); i++)
      {
        Tensor::inc(net->snets[i]->layers[j]->params[k], net->layers[j]->params[k]);
      }
      net->layers[j]->params[k]->div_(net->snets.size());
    }
  }
}

void sync_acc_gradients(Net *net)
{
  for (int j = 0; j < net->layers.size(); j++)
  {
    for (int k = 0; k < net->layers[j]->acc_gradients.size(); k++)
    {
      net->layers[j]->acc_gradients[k]->fill_(0.0);
      for (int i = 0; i < net->snets.size(); i++)
      {
        Tensor::inc(net->snets[i]->layers[j]->acc_gradients[k], net->layers[j]->acc_gradients[k]);
      }
      net->layers[j]->acc_gradients[k]->div_(net->snets.size());
    }
  }
}

void save_net_to_onnx_file(Net *net, string path)
{
  // Check if the folder exists
  string folder = path.substr(0, path.find_last_of("\\/"));
  if (folder != path && !pathExists(folder))
  {
    msg("The file could not be saved. Check if the directory exists or if you have permissions to write in it.", "ONNX::ExportNet");
  }

  if (net->snets[0]->dev != DEV_CPU)
    sync_params(net);
  bool export_gradients = false; // We always store weights to file
  onnx::ModelProto model = build_onnx_model(net, export_gradients);
  // Create the file stream and save the serialization of the onnx model in it
  fstream ofs(path, ios::out | ios::binary);
  if (!model.SerializeToOstream(&ofs))
  { // The serialization is automated by the protobuf library
    cerr << "Failed to write the model in onnx." << endl;
  }
  ofs.close();
}

size_t serialize_net_to_onnx_pointer(Net *net, void *&serialized_model, bool gradients)
{
  if (net->snets[0]->dev != DEV_CPU)
  {
    sync_params(net);
    if (gradients)
      sync_acc_gradients(net);
  }
  onnx::ModelProto model = build_onnx_model(net, gradients);
  // Serialization of the model to an array of bytes
  size_t size = model.ByteSizeLong(); // Get the size of the serialized model
  serialized_model = new char[size];
  memset(serialized_model, 0, size);
  if (!model.SerializeToArray(serialized_model, size))
  {
    cerr << "Failed to serialize the model in onnx into the buffer." << endl;
  }
  return size;
}

string *serialize_net_to_onnx_string(Net *net, bool gradients)
{
  if (net->snets[0]->dev != DEV_CPU)
  {
    sync_params(net);
    if (gradients)
      sync_acc_gradients(net);
  }
  onnx::ModelProto model = build_onnx_model(net, gradients);
  // Serialization of the model to an array of bytes
  string *model_string = new string();
  if (!model.SerializeToString(model_string))
  {
    cerr << "Failed to serialize the model in onnx into a string ." << endl;
  }
  return model_string;
}

onnx::ModelProto build_onnx_model(Net *net, bool gradients)
{
  string producer_name("EDDL");
  string producer_version("0.1"); // ????????????????
  // Create the empty Model in onnx
  onnx::ModelProto model;
  model.set_ir_version(onnx::Version::IR_VERSION);
  model.set_producer_name(producer_name);
  model.set_producer_version(producer_version);

  // Builds all the graph of the model
  set_graph(&model, net, gradients);

  // Return the finished model
  return model;
}

void set_graph(onnx::ModelProto *model, Net *net, bool gradients)
{
  // Add a new empty graph to the model
  onnx::GraphProto *graph = model->mutable_graph();
  graph->set_name("Computational Graph");
  onnx::OperatorSetIdProto *opset = model->add_opset_import();
  opset->set_version(11);

  // Check whether the model is encoder, decoder or both.
  bool is_encoder = false;
  bool is_decoder = false;
  for (int i = 0; i < net->vfts.size(); i++)
  {
    if (net->vfts[i]->isdecoder)
    {
      is_decoder = true;
      break;
    }
    else if (net->vfts[i]->isrecurrent)
      is_encoder = true;
  }
  bool is_recurrent = is_encoder || is_decoder;

  /*
   * We get all the input layers from the layers vector of the model
   * instead of taking them from net->lin. Beacause for the case of
   * a recurrent net with decoder the input layer that is connected
   * to the decoder is not added in the lin vector of the model.
   * With this way we ensure that we are taking all the input layers
   * of the model.
   */
  vector<Layer *> model_inputs = {};
  for (Layer *aux_layer : net->layers)
    if (LInput *t = dynamic_cast<LInput *>(aux_layer))
      model_inputs.push_back(aux_layer);

  // Set the inputs shapes of the graph
  for (Layer *input : model_inputs)
  {
    onnx::ValueInfoProto *input_info = graph->add_input();
    input_info->set_name(input->name);
    onnx::TypeProto *input_type = input_info->mutable_type();
    onnx::TypeProto::Tensor *input_type_tensor = input_type->mutable_tensor_type();
    input_type_tensor->set_elem_type(onnx::TensorProto::FLOAT);
    onnx::TensorShapeProto *input_type_tensor_shape = input_type_tensor->mutable_shape();
    onnx::TensorShapeProto::Dimension *input_type_tensor_dim;
    vector<int> input_shape = input->input->getShape();

    if (is_encoder)
    {
      // Set variable batch size
      input_type_tensor_dim = input_type_tensor_shape->add_dim();
      input_type_tensor_dim->set_dim_param("batch");
      // Set variable sequence lenght
      input_type_tensor_dim = input_type_tensor_shape->add_dim();
      input_type_tensor_dim->set_dim_param("sequence");
      for (int i = 1 /*skip batch*/; i < input_shape.size(); ++i)
      {
        input_type_tensor_dim = input_type_tensor_shape->add_dim();
        input_type_tensor_dim->set_dim_value(input_shape[i]);
      }
      // Fix input shape to add the seq_len dimension
      vector<int>::iterator it = input_shape.begin();
      input_shape.insert(it + 1, 1); // Insert seq_len=1 afer batch_size
      prepare_recurrent_input(input->name + "orig", input->name, input_shape, graph);
      input_info->set_name(input->name + "orig");
    }
    else
    {
      // Set the first dimension to a variable named "batch", to avoid setting a fixed batch size
      input_type_tensor_dim = input_type_tensor_shape->add_dim();
      input_type_tensor_dim->set_dim_param("batch");
      // Set the rest of the dimensions
      for (int i = 1 /*skip batch*/; i < input_shape.size(); ++i)
      {
        input_type_tensor_dim = input_type_tensor_shape->add_dim();
        input_type_tensor_dim->set_dim_value(input_shape[i]);
      }
    }
  }

  // Set the outputs shapes of the graph
  for (Layer *aux_output : net->lout)
  {
    // Create the required ONNX output objects
    onnx::ValueInfoProto *output_info = graph->add_output();
    output_info->set_name(aux_output->name);
    onnx::TypeProto *output_type = output_info->mutable_type();
    onnx::TypeProto::Tensor *output_type_tensor = output_type->mutable_tensor_type();
    output_type_tensor->set_elem_type(onnx::TensorProto::FLOAT);
    // Create the output_shape vectors
    vector<int> output_shape = aux_output->output->getShape(); // Get shape from output tensor
    onnx::TensorShapeProto *output_type_tensor_shape = output_type_tensor->mutable_shape();
    onnx::TensorShapeProto::Dimension *output_type_tensor_dim;
    // Set the first dimension to a variable named "batch", to avoid setting a fixed batch size
    output_type_tensor_dim = output_type_tensor_shape->add_dim();
    output_type_tensor_dim->set_dim_param("batch");
    if (is_decoder)
    {
      output_type_tensor_dim = output_type_tensor_shape->add_dim();
      output_type_tensor_dim->set_dim_param("sequence");

      // Fix output shape to add the seq_len dimension
      vector<int> output_shape = aux_output->output->getShape();
      vector<int>::iterator it = output_shape.begin();
      output_shape.insert(it + 1, 1); // Insert seq_len=1 afer batch_size
      prepare_recurrent_output(aux_output->name, aux_output->name + "_output", output_shape, graph);
      output_info->set_name(aux_output->name + "_output");
    }
    // Set the rest of the dimensions
    for (int i = 1 /*skip batch*/; i < output_shape.size(); ++i)
    {
      output_type_tensor_dim = output_type_tensor_shape->add_dim();
      output_type_tensor_dim->set_dim_value(output_shape[i]);
    }
  }

  // Computational graph
  for (Layer *aux_layer : net->layers)
  {
    // Builds a node of the graph from the layer in EDDL
    build_node_from_layer(aux_layer, graph, gradients, is_recurrent);
  }
}

void build_node_from_layer(Layer *layer, onnx::GraphProto *graph, bool gradients, bool is_recurrent)
{
  // Check the class of the layer to call the corresponding function to
  // build the node
  if (LInput *t = dynamic_cast<LInput *>(layer))
  {
    return; // Skip the input layers
  }
  else if (LConv *t = dynamic_cast<LConv *>(layer))
  {
    build_conv_node((LConv *)(LinLayer *)layer, graph, gradients);
  }
  else if (LConv1D *t = dynamic_cast<LConv1D *>(layer))
  {
    build_conv1D_node((LConv1D *)(LinLayer *)layer, graph, gradients);
  }
  else if (LDense *t = dynamic_cast<LDense *>(layer))
  {
    if (is_recurrent)
      build_dense_with_matmul_node((LDense *)(LinLayer *)layer, graph, gradients);
    else
      build_gemm_node((LDense *)(LinLayer *)layer, graph, gradients);
  }
  else if (LMaxPool *t = dynamic_cast<LMaxPool *>(layer))
  {
    build_maxpool_node((LMaxPool *)(LinLayer *)layer, graph);
  }
  else if (LMaxPool1D *t = dynamic_cast<LMaxPool1D *>(layer))
  {
    build_maxpool1D_node((LMaxPool1D *)(LinLayer *)layer, graph);
  }
  else if (LAveragePool *t = dynamic_cast<LAveragePool *>(layer))
  {
    build_averagepool_node((LAveragePool *)(LinLayer *)layer, graph);
  }
  else if (LReshape *t = dynamic_cast<LReshape *>(layer))
  {
    build_reshape_node((LReshape *)(LinLayer *)layer, graph);
  }
  else if (LPermute *t = dynamic_cast<LPermute *>(layer))
  {
    build_permute_node((LPermute *)(OperatorLayer *)layer, graph);
  }
  else if (LUpSampling *t = dynamic_cast<LUpSampling *>(layer))
  {
    build_upsample_node((LUpSampling *)(LinLayer *)layer, graph);
  }
  else if (LActivation *t = dynamic_cast<LActivation *>(layer))
  {
    // Check the type of activation layer
    if (!((LActivation *)(layer))->act.compare("relu"))
    {
      build_relu_node((LActivation *)(LinLayer *)layer, graph);
    }
    else if (!((LActivation *)(layer))->act.compare("sigmoid"))
    {
      build_sigmoid_node((LActivation *)(LinLayer *)layer, graph);
    }
    else if (!((LActivation *)(layer))->act.compare("hard_sigmoid"))
    {
      build_hard_sigmoid_node((LActivation *)(LinLayer *)layer, graph);
    }
    else if (!((LActivation *)(layer))->act.compare("tanh"))
    {
      build_tanh_node((LActivation *)(LinLayer *)layer, graph);
    }
    else if (!((LActivation *)(layer))->act.compare("exp"))
    {
      build_exponential_node((LActivation *)(LinLayer *)layer, graph);
    }
    else if (!((LActivation *)(layer))->act.compare("linear"))
    {
      build_linear_node((LActivation *)(LinLayer *)layer, graph);
    }
    else if (!((LActivation *)(layer))->act.compare("leaky_relu"))
    {
      build_leaky_relu_node((LActivation *)(LinLayer *)layer, graph);
    }
    else if (!((LActivation *)(layer))->act.compare("thresholded_relu"))
    {
      build_thresholded_relu_node((LActivation *)(LinLayer *)layer, graph);
    }
    else if (!((LActivation *)(layer))->act.compare("elu"))
    {
      build_elu_node((LActivation *)(LinLayer *)layer, graph);
    }
    else if (!((LActivation *)(layer))->act.compare("selu"))
    {
      build_selu_node((LActivation *)(LinLayer *)layer, graph);
    }
    else if (!((LActivation *)(layer))->act.compare("softmax") || !((LActivation *)(layer))->act.compare("full_softmax"))
    {
      build_softmax_node((LActivation *)(LinLayer *)layer, graph);
    }
    else if (!((LActivation *)(layer))->act.compare("softsign"))
    {
      build_softsign_node((LActivation *)(LinLayer *)layer, graph);
    }
    else if (!((LActivation *)(layer))->act.compare("softplus"))
    {
      build_softplus_node((LActivation *)(LinLayer *)layer, graph);
    }
    else
    {
      cout << "The activation layer " << layer->name << "has no valid type to export." << endl;
      return;
    }
  }
  else if (LConcat *t = dynamic_cast<LConcat *>(layer))
  {
    build_concat_node((LConcat *)(MLayer *)layer, graph);
  }
  else if (LAbs *t = dynamic_cast<LAbs *>(layer))
  {
    build_abs_node((LAbs *)(OperatorLayer *)layer, graph);
  }
  else if (LAdd *t = dynamic_cast<LAdd *>(layer))
  {
    build_add_node((LAdd *)(MLayer *)layer, graph);
  }
  else if (LDiv *t = dynamic_cast<LDiv *>(layer))
  {
    build_div_node((LDiv *)(OperatorLayer *)layer, graph);
  }
  else if (LExp *t = dynamic_cast<LExp *>(layer))
  {
    build_exp_node((LExp *)(OperatorLayer *)layer, graph);
  }
  else if (LLog *t = dynamic_cast<LLog *>(layer))
  {
    build_log_node((LLog *)(OperatorLayer *)layer, graph);
  }
  else if (LMult *t = dynamic_cast<LMult *>(layer))
  {
    build_mul_node((LMult *)(OperatorLayer *)layer, graph);
    //} else if (LPow *t = dynamic_cast<LPow *>(layer)) {
    //  build_pow_node((LPow *)(OperatorLayer *)layer, graph);
  }
  else if (LSqrt *t = dynamic_cast<LSqrt *>(layer))
  {
    build_sqrt_node((LSqrt *)(OperatorLayer *)layer, graph);
  }
  else if (LDiff *t = dynamic_cast<LDiff *>(layer))
  {
    build_sub_node((LDiff *)(OperatorLayer *)layer, graph);
  }
  else if (LRMean *t = dynamic_cast<LRMean *>(layer))
  {
    build_rmean_node((LRMean *)(ReductionLayer *)layer, graph);
  }
  else if (LRSum *t = dynamic_cast<LRSum *>(layer))
  {
    build_rsum_node((LRSum *)(ReductionLayer *)layer, graph);
  }
  else if (LRMax *t = dynamic_cast<LRMax *>(layer))
  {
    build_rmax_node((LRMax *)(ReductionLayer *)layer, graph);
  }
  else if (LRMin *t = dynamic_cast<LRMin *>(layer))
  {
    build_rmin_node((LRMin *)(ReductionLayer *)layer, graph);
  }
  else if (LRArgmax *t = dynamic_cast<LRArgmax *>(layer))
  {
    build_rargmax_node((LRArgmax *)(ReductionLayer2 *)layer, graph);
  }
  else if (LBatchNorm *t = dynamic_cast<LBatchNorm *>(layer))
  {
    build_batchnorm_node((LBatchNorm *)(LinLayer *)layer, graph);
  }
  else if (LDropout *t = dynamic_cast<LDropout *>(layer))
  {
    build_dropout_node((LDropout *)(LinLayer *)layer, graph);
  }
  else if (LLSTM *t = dynamic_cast<LLSTM *>(layer))
  {
    build_lstm_node((LLSTM *)(MLayer *)layer, graph);
  }
  else if (LGRU *t = dynamic_cast<LGRU *>(layer))
  {
    build_gru_node((LGRU *)(MLayer *)layer, graph);
  }
  else if (LRNN *t = dynamic_cast<LRNN *>(layer))
  {
    build_rnn_node((LRNN *)(MLayer *)layer, graph);
  }
  else if (LCopyStates *t = dynamic_cast<LCopyStates *>(layer))
  {
    handle_copy_states((LCopyStates *)(MLayer *)layer, graph);
  }
  else if (LEmbedding *t = dynamic_cast<LEmbedding *>(layer))
  {
    build_embedding_node((LEmbedding *)(LinLayer *)layer, graph);
  }
  else if (LScale *t = dynamic_cast<LScale *>(layer))
  {
    build_resize_node((LScale *)(LDataAugmentation *)(LinLayer *)layer, graph);
  }
  else
  {
    cout << "The layer " << layer->name << "has no OpType in Onnx." << endl;
    return;
  }
}

// Node builders
//----------------------------------------------------------------------------------------

void build_conv_node(LConv *layer, onnx::GraphProto *graph, bool gradients)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Conv");
  node->set_name(layer->name);
  // Set the inputs of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the input params names of the conv op
  node->add_input(layer->name + "_W");
  if (layer->cd->use_bias)
    node->add_input(layer->name + "_b");
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);

  ////////////////////////// Attributes of the Conv operation //////////////////////////////////
  // Attr dilations
  onnx::AttributeProto *conv_dilations = node->add_attribute();
  conv_dilations->set_name("dilations");
  conv_dilations->set_type(onnx::AttributeProto::INTS);
  vector<int> vdilations{1, 1};
  for (int i : vdilations)
  {
    conv_dilations->add_ints(i);
  }

  //Attr group
  onnx::AttributeProto *conv_group = node->add_attribute();
  conv_group->set_name("group");
  conv_group->set_type(onnx::AttributeProto::INT);
  conv_group->set_i(1);

  // Attr kernel_shape
  onnx::AttributeProto *conv_kernel_shape = node->add_attribute();
  conv_kernel_shape->set_name("kernel_shape");
  conv_kernel_shape->set_type(onnx::AttributeProto::INTS);
  conv_kernel_shape->add_ints(layer->cd->kr);
  conv_kernel_shape->add_ints(layer->cd->kc);

  // Attr pads
  onnx::AttributeProto *conv_pads = node->add_attribute();
  conv_pads->set_name("pads");
  conv_pads->set_type(onnx::AttributeProto::INTS);
  conv_pads->add_ints(layer->cd->padrt);
  conv_pads->add_ints(layer->cd->padcl);
  conv_pads->add_ints(layer->cd->padrb);
  conv_pads->add_ints(layer->cd->padcr);

  // Attr strides
  onnx::AttributeProto *conv_strides = node->add_attribute();
  conv_strides->set_name("strides");
  conv_strides->set_type(onnx::AttributeProto::INTS);
  conv_strides->add_ints(layer->cd->sr);
  conv_strides->add_ints(layer->cd->sc);

  // Check if we are exporting weights or accumulated gradients
  if (!gradients)
  {
    // Weights input
    onnx::TensorProto *conv_w = graph->add_initializer();
    conv_w->set_name(layer->name + "_W");
    conv_w->set_data_type(onnx::TensorProto::FLOAT);
    conv_w->mutable_dims()->Add(layer->cd->K->shape.begin(), layer->cd->K->shape.end());          // Set the shape of the weights
    conv_w->mutable_float_data()->Add(layer->cd->K->ptr, layer->cd->K->ptr + layer->cd->K->size); // Set the weights values
    //conv_w->mutable_raw_data()->assign( reinterpret_cast<const char*>(layer->cd->K->ptr), sizeof(float) * layer->cd->K->size );

    // Bias input
    if (layer->cd->use_bias)
    {
      onnx::TensorProto *conv_b = graph->add_initializer();
      conv_b->set_name(layer->name + "_b");
      conv_b->set_data_type(onnx::TensorProto::FLOAT);
      conv_b->mutable_dims()->Add(layer->cd->bias->shape.begin(), layer->cd->bias->shape.end());             // Set the shape of the bias
      conv_b->mutable_float_data()->Add(layer->cd->bias->ptr, layer->cd->bias->ptr + layer->cd->bias->size); // Set the bias values
      //conv_b->mutable_raw_data()->assign( reinterpret_cast<const char*>(layer->cd->bias->ptr), sizeof(float) * layer->cd->bias->size );
    }
  }
  else
  {
    // Accumulated gradients (Weights) input
    onnx::TensorProto *conv_w = graph->add_initializer();
    conv_w->set_name(layer->name + "_W");
    conv_w->set_data_type(onnx::TensorProto::FLOAT);
    conv_w->mutable_dims()->Add(layer->cd->acc_gK->shape.begin(), layer->cd->acc_gK->shape.end());               // Set the accumulated gradiens shape (weights)
    conv_w->mutable_float_data()->Add(layer->cd->acc_gK->ptr, layer->cd->acc_gK->ptr + layer->cd->acc_gK->size); // Set the accumulated gradients values (weights)
    //conv_w->mutable_raw_data()->assign( reinterpret_cast<const char*>(layer->cd->acc_gK->ptr), sizeof(float) * layer->cd->acc_gK->size );

    // Accumulated gradients (bias) input
    if (layer->cd->use_bias)
    {
      onnx::TensorProto *conv_b = graph->add_initializer();
      conv_b->set_name(layer->name + "_b");
      conv_b->set_data_type(onnx::TensorProto::FLOAT);
      conv_b->mutable_dims()->Add(layer->cd->acc_gbias->shape.begin(), layer->cd->acc_gbias->shape.end());                  // Set the accumulated gradients shape (bias)
      conv_b->mutable_float_data()->Add(layer->cd->acc_gbias->ptr, layer->cd->acc_gbias->ptr + layer->cd->acc_gbias->size); // Set the accumulated gradients values (bias)
      //conv_b->mutable_raw_data()->assign( reinterpret_cast<const char*>(layer->cd->acc_gbias->ptr), sizeof(float) * layer->cd->acc_gbias->size );
    }
  }
}

void build_conv1D_node(LConv1D *layer, onnx::GraphProto *graph, bool gradients)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Conv");
  node->set_name(layer->name);
  // Set the inputs of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the input params names of the conv op
  node->add_input(layer->name + "_W");
  if (layer->cd->use_bias)
    node->add_input(layer->name + "_b");
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);

  ////////////////////////// Attributes of the Conv operation //////////////////////////////////
  // Attr dilations
  onnx::AttributeProto *conv_dilations = node->add_attribute();
  conv_dilations->set_name("dilations");
  conv_dilations->set_type(onnx::AttributeProto::INTS);
  conv_dilations->add_ints(1);

  //Attr group
  onnx::AttributeProto *conv_group = node->add_attribute();
  conv_group->set_name("group");
  conv_group->set_type(onnx::AttributeProto::INT);
  conv_group->set_i(1);

  // Attr kernel_shape
  onnx::AttributeProto *conv_kernel_shape = node->add_attribute();
  conv_kernel_shape->set_name("kernel_shape");
  conv_kernel_shape->set_type(onnx::AttributeProto::INTS);
  conv_kernel_shape->add_ints(layer->cd->kr);

  // Attr pads
  onnx::AttributeProto *conv_pads = node->add_attribute();
  conv_pads->set_name("pads");
  conv_pads->set_type(onnx::AttributeProto::INTS);
  conv_pads->add_ints(layer->cd->padrt);
  conv_pads->add_ints(layer->cd->padrb);

  // Attr strides
  onnx::AttributeProto *conv_strides = node->add_attribute();
  conv_strides->set_name("strides");
  conv_strides->set_type(onnx::AttributeProto::INTS);
  conv_strides->add_ints(layer->cd->sr);

  // Check if we are exporting weights or accumulated gradients
  if (!gradients)
  {
    // Weights input
    onnx::TensorProto *conv_w = graph->add_initializer();
    conv_w->set_name(layer->name + "_W");
    conv_w->set_data_type(onnx::TensorProto::FLOAT);
    conv_w->mutable_dims()->Add(layer->cd->K->shape.begin(), --layer->cd->K->shape.end());        // Set the shape of the weights
    conv_w->mutable_float_data()->Add(layer->cd->K->ptr, layer->cd->K->ptr + layer->cd->K->size); // Set the weights values
    //conv_w->mutable_raw_data()->assign( reinterpret_cast<const char*>(layer->cd->K->ptr), sizeof(float) * layer->cd->K->size );

    // Bias input
    if (layer->cd->use_bias)
    {
      onnx::TensorProto *conv_b = graph->add_initializer();
      conv_b->set_name(layer->name + "_b");
      conv_b->set_data_type(onnx::TensorProto::FLOAT);
      conv_b->mutable_dims()->Add(layer->cd->bias->shape.begin(), layer->cd->bias->shape.end());             // Set the shape of the bias
      conv_b->mutable_float_data()->Add(layer->cd->bias->ptr, layer->cd->bias->ptr + layer->cd->bias->size); // Set the bias values
      //conv_b->mutable_raw_data()->assign( reinterpret_cast<const char*>(layer->cd->bias->ptr), sizeof(float) * layer->cd->bias->size );
    }
  }
  else
  {
    // Accumulated gradients (Weights) input
    onnx::TensorProto *conv_w = graph->add_initializer();
    conv_w->set_name(layer->name + "_W");
    conv_w->set_data_type(onnx::TensorProto::FLOAT);
    conv_w->mutable_dims()->Add(layer->cd->acc_gK->shape.begin(), --layer->cd->acc_gK->shape.end());             // Set the accumulated gradiens shape (weights)
    conv_w->mutable_float_data()->Add(layer->cd->acc_gK->ptr, layer->cd->acc_gK->ptr + layer->cd->acc_gK->size); // Set the accumulated gradients values (weights)
    //conv_w->mutable_raw_data()->assign( reinterpret_cast<const char*>(layer->cd->acc_gK->ptr), sizeof(float) * layer->cd->acc_gK->size );

    // Accumulated gradients (bias) input
    if (layer->cd->use_bias)
    {
      onnx::TensorProto *conv_b = graph->add_initializer();
      conv_b->set_name(layer->name + "_b");
      conv_b->set_data_type(onnx::TensorProto::FLOAT);
      conv_b->mutable_dims()->Add(layer->cd->acc_gbias->shape.begin(), layer->cd->acc_gbias->shape.end());                  // Set the accumulated gradients shape (bias)
      conv_b->mutable_float_data()->Add(layer->cd->acc_gbias->ptr, layer->cd->acc_gbias->ptr + layer->cd->acc_gbias->size); // Set the accumulated gradients values (bias)
      //conv_b->mutable_raw_data()->assign( reinterpret_cast<const char*>(layer->cd->acc_gbias->ptr), sizeof(float) * layer->cd->acc_gbias->size );
    }
  }
}

void build_gemm_node(LDense *layer, onnx::GraphProto *graph, bool gradients)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Gemm");
  node->set_name(layer->name);
  // Set the inputs of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the input params names of the Gemm(Dense) op
  node->add_input(layer->name + "_W");
  if (layer->use_bias)
    node->add_input(layer->name + "_b");
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);

  // Attr alpha
  onnx::AttributeProto *dense_alpha = node->add_attribute();
  dense_alpha->set_name("alpha");
  dense_alpha->set_type(onnx::AttributeProto::FLOAT);
  dense_alpha->set_f(1);

  // Attr beta
  onnx::AttributeProto *dense_beta = node->add_attribute();
  dense_beta->set_name("beta");
  dense_beta->set_type(onnx::AttributeProto::FLOAT);
  dense_beta->set_f(layer->use_bias);

  // Attr transA
  onnx::AttributeProto *dense_transA = node->add_attribute();
  dense_transA->set_name("transA");
  dense_transA->set_type(onnx::AttributeProto::INT);
  dense_transA->set_i(0);

  // Attr transB
  onnx::AttributeProto *dense_transB = node->add_attribute();
  dense_transB->set_name("transB");
  dense_transB->set_type(onnx::AttributeProto::INT);
  dense_transB->set_i(0);

  // Check if we are exporting weights or accumulated gradients
  if (!gradients)
  {
    // Weights input
    onnx::TensorProto *weight = graph->add_initializer();
    weight->set_name(layer->name + "_W");
    weight->set_data_type(onnx::TensorProto::FLOAT);
    weight->mutable_dims()->Add(layer->W->shape.begin(), layer->W->shape.end());      // Set the shape of the weights
    weight->mutable_float_data()->Add(layer->W->ptr, layer->W->ptr + layer->W->size); // Set the weights values
    //weight->mutable_raw_data()->assign( reinterpret_cast<const char*>(layer->W->ptr), sizeof(float) * layer->W->size );
    if (layer->use_bias)
    {
      // Bias input
      onnx::TensorProto *bias = graph->add_initializer();
      bias->set_name(layer->name + "_b");
      bias->set_data_type(onnx::TensorProto::FLOAT);
      bias->mutable_dims()->Add(layer->bias->shape.begin(), layer->bias->shape.end());         // Set the bias shape
      bias->mutable_float_data()->Add(layer->bias->ptr, layer->bias->ptr + layer->bias->size); // Set the bias values
      //bias->mutable_raw_data()->assign( reinterpret_cast<const char*>(layer->bias->ptr), sizeof(float) * layer->bias->size );
    }
  }
  else
  {
    // Accumulated gradients (Weights) input
    onnx::TensorProto *weight = graph->add_initializer();
    weight->set_name(layer->name + "_W");
    weight->set_data_type(onnx::TensorProto::FLOAT);
    weight->mutable_dims()->Add(layer->acc_gW->shape.begin(), layer->acc_gW->shape.end());           // Set the accumulated gradients shape (weights)
    weight->mutable_float_data()->Add(layer->acc_gW->ptr, layer->acc_gW->ptr + layer->acc_gW->size); // Set the accumulated gradients values (weights)
    //weight->mutable_raw_data()->assign( reinterpret_cast<const char*>(layer->acc_gW->ptr), sizeof(float) * layer->acc_gW->size );

    // Check if we are using bias
    if (layer->use_bias)
    {
      // Accumulated gradients (bias) input
      onnx::TensorProto *bias = graph->add_initializer();
      bias->set_name(layer->name + "_b");
      bias->set_data_type(onnx::TensorProto::FLOAT);
      bias->mutable_dims()->Add(layer->acc_gbias->shape.begin(), layer->acc_gbias->shape.end());              // Set the accumulated gradients shape (bias)
      bias->mutable_float_data()->Add(layer->acc_gbias->ptr, layer->acc_gbias->ptr + layer->acc_gbias->size); // Set the accumulated gradients values (bias)
      //bias->mutable_raw_data()->assign( reinterpret_cast<const char*>(layer->acc_gbias->ptr), sizeof(float) * layer->acc_gbias->size );
    }
  }
}

void build_dense_with_matmul_node(LDense *layer, onnx::GraphProto *graph, bool gradients)
{
  /*
     * Build a dense layer by composing a MatMul with an Add
     */

  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("MatMul");
  node->set_name(layer->name + "_MatMul");
  // Set the inputs of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the input param name of the Weight matrix
  node->add_input(layer->name + "_W");
  // Set the name of the output of the node to link with other nodes
  if (layer->use_bias)
    // Output name to link with the Add operator for the bias
    node->add_output(layer->name + "_MatMul");
  else
    // Set the proper output name to connect with the next layer of the model
    node->add_output(layer->name);

  // Set weights for the MatMul
  onnx::TensorProto *weight = graph->add_initializer();
  weight->set_name(layer->name + "_W");
  weight->set_data_type(onnx::TensorProto::FLOAT);
  weight->mutable_dims()->Add(layer->W->shape.begin(), layer->W->shape.end());      // Set the shape of the weights
  weight->mutable_float_data()->Add(layer->W->ptr, layer->W->ptr + layer->W->size); // Set the weights values

  // Create the Add node in case of using bias in the Dense layer
  if (layer->use_bias)
  {
    // Add an empty node to the graph
    onnx::NodeProto *node_bias = graph->add_node();
    node_bias->set_op_type("Add");
    node_bias->set_name(layer->name + "_Add");
    // Take the input from the previous MatMul
    node_bias->add_input(layer->name + "_MatMul");
    // Set the input param name of the Bias matrix
    node_bias->add_input(layer->name + "_b");
    // Set the name of the output of the node to link with other nodes
    node_bias->add_output(layer->name);
    // Set weights for Add (Dense bias)
    onnx::TensorProto *bias = graph->add_initializer();
    bias->set_name(layer->name + "_b");
    bias->set_data_type(onnx::TensorProto::FLOAT);
    bias->mutable_dims()->Add(layer->bias->shape.begin(), layer->bias->shape.end());         // Set the bias shape
    bias->mutable_float_data()->Add(layer->bias->ptr, layer->bias->ptr + layer->bias->size); // Set the bias values
  }
}

void build_maxpool_node(LMaxPool *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("MaxPool");
  node->set_name(layer->name);
  // Set the inputs of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);

  // Attr kernel_shape
  onnx::AttributeProto *max_pool_ks = node->add_attribute();
  max_pool_ks->set_name("kernel_shape");
  max_pool_ks->set_type(onnx::AttributeProto::INTS);
  for (int i : layer->pd->ksize)
  {
    max_pool_ks->add_ints(i);
  }

  // Attr pads
  onnx::AttributeProto *max_pool_pads = node->add_attribute();
  max_pool_pads->set_name("pads");
  max_pool_pads->set_type(onnx::AttributeProto::INTS);
  max_pool_pads->add_ints(layer->pd->padrt);
  max_pool_pads->add_ints(layer->pd->padcl);
  max_pool_pads->add_ints(layer->pd->padrb);
  max_pool_pads->add_ints(layer->pd->padcr);

  // Attr strides
  onnx::AttributeProto *max_pool_strides = node->add_attribute();
  max_pool_strides->set_name("strides");
  max_pool_strides->set_type(onnx::AttributeProto::INTS);
  max_pool_strides->add_ints(layer->pd->sr);
  max_pool_strides->add_ints(layer->pd->sc);
}

void build_maxpool1D_node(LMaxPool1D *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("MaxPool");
  node->set_name(layer->name);
  // Set the inputs of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);

  // Attr kernel_shape
  onnx::AttributeProto *max_pool_ks = node->add_attribute();
  max_pool_ks->set_name("kernel_shape");
  max_pool_ks->set_type(onnx::AttributeProto::INTS);
  max_pool_ks->add_ints(layer->pd->kr);

  // Attr pads
  onnx::AttributeProto *max_pool_pads = node->add_attribute();
  max_pool_pads->set_name("pads");
  max_pool_pads->set_type(onnx::AttributeProto::INTS);
  max_pool_pads->add_ints(layer->pd->padrt);
  max_pool_pads->add_ints(layer->pd->padrb);

  // Attr strides
  onnx::AttributeProto *max_pool_strides = node->add_attribute();
  max_pool_strides->set_name("strides");
  max_pool_strides->set_type(onnx::AttributeProto::INTS);
  max_pool_strides->add_ints(layer->pd->sr);
}

void build_averagepool_node(LAveragePool *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("AveragePool");
  node->set_name(layer->name);
  // Set the inputs of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);

  // Attr kernel_shape
  onnx::AttributeProto *max_pool_ks = node->add_attribute();
  max_pool_ks->set_name("kernel_shape");
  max_pool_ks->set_type(onnx::AttributeProto::INTS);
  for (int i : layer->pd->ksize)
  {
    max_pool_ks->add_ints(i);
  }

  // Attr pads
  onnx::AttributeProto *max_pool_pads = node->add_attribute();
  max_pool_pads->set_name("pads");
  max_pool_pads->set_type(onnx::AttributeProto::INTS);
  max_pool_pads->add_ints(layer->pd->padrt);
  max_pool_pads->add_ints(layer->pd->padcl);
  max_pool_pads->add_ints(layer->pd->padrb);
  max_pool_pads->add_ints(layer->pd->padcr);

  // Attr strides
  onnx::AttributeProto *max_pool_strides = node->add_attribute();
  max_pool_strides->set_name("strides");
  max_pool_strides->set_type(onnx::AttributeProto::INTS);
  max_pool_strides->add_ints(layer->pd->sr);
  max_pool_strides->add_ints(layer->pd->sc);
}

void build_reshape_node(LReshape *layer, onnx::GraphProto *graph)
{
  // Constant node input to the reshape node: shape
  onnx::NodeProto *shape_const_node = graph->add_node();
  shape_const_node->add_output(layer->name + "_target_shape");
  shape_const_node->set_op_type("Constant");
  onnx::AttributeProto *shape_attr = shape_const_node->add_attribute();
  shape_attr->set_name("value");
  shape_attr->set_type(onnx::AttributeProto::TENSOR);
  onnx::TensorProto *target_shape_tensor = shape_attr->mutable_t();
  target_shape_tensor->set_name("const_tensor");
  target_shape_tensor->set_data_type(onnx::TensorProto::INT64);
  target_shape_tensor->add_dims(layer->ls.size());
  // Set the target shape
  target_shape_tensor->add_int64_data(-1); // For batch_size
  for (int i = 1; i < layer->ls.size(); ++i)
    target_shape_tensor->add_int64_data(layer->ls[i]);

  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Reshape");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the input with the target shape of the op
  node->add_input(layer->name + "_target_shape");
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

void build_permute_node(LPermute *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Transpose");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);

  // Attr perm
  onnx::AttributeProto *perm_attr = node->add_attribute();
  perm_attr->set_name("perm");
  perm_attr->set_type(onnx::AttributeProto::INTS);
  perm_attr->add_ints(0); // Set the batch size position. It must not be permuted in EDDL
  for (int i : layer->sd->dims)
  {
    perm_attr->add_ints(i + 1); // Add 1 to fix the batch dim adition
  }
}

void build_relu_node(LActivation *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Relu");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

void build_sigmoid_node(LActivation *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Sigmoid");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

void build_hard_sigmoid_node(LActivation *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("HardSigmoid");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

void build_tanh_node(LActivation *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Tanh");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

void build_exponential_node(LActivation *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Exponential"); // **Custom operator**
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

void build_linear_node(LActivation *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Linear"); // **Custom operator**
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);

  // Attr alpha
  onnx::AttributeProto *alpha_attr = node->add_attribute();
  alpha_attr->set_name("alpha");
  alpha_attr->set_type(onnx::AttributeProto::FLOAT);
  alpha_attr->set_f(layer->params[0]);
}

void build_leaky_relu_node(LActivation *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("LeakyRelu");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);

  // Attr alpha
  onnx::AttributeProto *alpha_attr = node->add_attribute();
  alpha_attr->set_name("alpha");
  alpha_attr->set_type(onnx::AttributeProto::FLOAT);
  alpha_attr->set_f(layer->params[0]);
}

void build_thresholded_relu_node(LActivation *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("ThresholdedRelu");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);

  // Attr alpha
  onnx::AttributeProto *alpha_attr = node->add_attribute();
  alpha_attr->set_name("alpha");
  alpha_attr->set_type(onnx::AttributeProto::FLOAT);
  alpha_attr->set_f(layer->params[0]);
}

void build_elu_node(LActivation *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Elu");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);

  // Attr alpha
  onnx::AttributeProto *alpha_attr = node->add_attribute();
  alpha_attr->set_name("alpha");
  alpha_attr->set_type(onnx::AttributeProto::FLOAT);
  alpha_attr->set_f(layer->params[0]);
}

void build_selu_node(LActivation *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Selu");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);

  // Attr alpha
  onnx::AttributeProto *alpha_attr = node->add_attribute();
  alpha_attr->set_name("alpha");
  alpha_attr->set_type(onnx::AttributeProto::FLOAT);
  alpha_attr->set_f(layer->params[0]);

  // Attr gamma
  onnx::AttributeProto *gamma_attr = node->add_attribute();
  gamma_attr->set_name("gamma");
  gamma_attr->set_type(onnx::AttributeProto::FLOAT);
  gamma_attr->set_f(layer->params[1]);
}

void build_softmax_node(LActivation *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Softmax");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Attr axis
  onnx::AttributeProto *axis_attr = node->add_attribute();
  axis_attr->set_name("axis");
  axis_attr->set_type(onnx::AttributeProto::INT);
  axis_attr->set_i((int)layer->params[0]);
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

void build_softsign_node(LActivation *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Softsign");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

void build_softplus_node(LActivation *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Softplus");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

void build_concat_node(LConcat *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Concat");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);

  // Attr axis
  onnx::AttributeProto *concat_axis = node->add_attribute();
  concat_axis->set_name("axis");
  concat_axis->set_type(onnx::AttributeProto::INT);
  concat_axis->set_i(1);
}

void build_add_node(LAdd *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Add");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

void build_abs_node(LAbs *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Abs");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

void build_div_node(LDiv *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Div");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

void build_exp_node(LExp *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Exp");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

void build_log_node(LLog *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Log");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

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
}

/*
  void build_pow_node(LPow *layer, onnx::GraphProto *graph)
  {
    // Add an empty node to the graph
    onnx::NodeProto* node = graph->add_node();
    node->set_op_type("Pow");
    node->set_name(layer->name);
    // Set the inputs names of the node from the parents of the layer
    for (Layer* parentl : layer->parent)
    {
      node->add_input(parentl->name);
    }
    // Set the name of the output of the node to link with other nodes
    node->add_output(layer->name);
  }
*/

void build_sqrt_node(LSqrt *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Sqrt");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

void build_sub_node(LDiff *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Sub");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

void build_rmean_node(LRMean *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("ReduceMean");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }

  // Attr axes
  onnx::AttributeProto *axes_attr = node->add_attribute();
  axes_attr->set_name("axes");
  axes_attr->set_type(onnx::AttributeProto::INTS);
  for (int ax : layer->axis)
    axes_attr->add_ints(ax + 1);

  // Attr keepdims
  onnx::AttributeProto *keepdims_attr = node->add_attribute();
  keepdims_attr->set_name("keepdims");
  keepdims_attr->set_type(onnx::AttributeProto::INT);
  keepdims_attr->set_i((int)layer->keepdims);

  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

void build_rsum_node(LRSum *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("ReduceSum");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }

  // Attr axes
  onnx::AttributeProto *axes_attr = node->add_attribute();
  axes_attr->set_name("axes");
  axes_attr->set_type(onnx::AttributeProto::INTS);
  for (int ax : layer->axis)
    axes_attr->add_ints(ax + 1);

  // Attr keepdims
  onnx::AttributeProto *keepdims_attr = node->add_attribute();
  keepdims_attr->set_name("keepdims");
  keepdims_attr->set_type(onnx::AttributeProto::INT);
  keepdims_attr->set_i(layer->keepdims);

  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

void build_rmax_node(LRMax *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("ReduceMax");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }

  // Attr axes
  onnx::AttributeProto *axes_attr = node->add_attribute();
  axes_attr->set_name("axes");
  axes_attr->set_type(onnx::AttributeProto::INTS);
  for (int ax : layer->axis)
    axes_attr->add_ints(ax + 1);

  // Attr keepdims
  onnx::AttributeProto *keepdims_attr = node->add_attribute();
  keepdims_attr->set_name("keepdims");
  keepdims_attr->set_type(onnx::AttributeProto::INT);
  keepdims_attr->set_i(layer->keepdims);

  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

void build_rmin_node(LRMin *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("ReduceMin");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }

  // Attr axes
  onnx::AttributeProto *axes_attr = node->add_attribute();
  axes_attr->set_name("axes");
  axes_attr->set_type(onnx::AttributeProto::INTS);
  for (int ax : layer->axis)
    axes_attr->add_ints(ax + 1);

  // Attr keepdims
  onnx::AttributeProto *keepdims_attr = node->add_attribute();
  keepdims_attr->set_name("keepdims");
  keepdims_attr->set_type(onnx::AttributeProto::INT);
  keepdims_attr->set_i(layer->keepdims);

  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

void build_rargmax_node(LRArgmax *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("ArgMax");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }

  // Attr axis
  onnx::AttributeProto *axis_attr = node->add_attribute();
  axis_attr->set_name("axis");
  axis_attr->set_type(onnx::AttributeProto::INT);
  axis_attr->set_i(layer->axis[0] + 1);

  // Attr keepdims
  onnx::AttributeProto *keepdims_attr = node->add_attribute();
  keepdims_attr->set_name("keepdims");
  keepdims_attr->set_type(onnx::AttributeProto::INT);
  keepdims_attr->set_i((int)layer->keepdims);

  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

void build_batchnorm_node(LBatchNorm *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("BatchNormalization");
  node->set_name(layer->name);
  // Set the inputs of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  node->add_input(layer->name + "_scale");
  node->add_input(layer->name + "_bias");
  node->add_input(layer->name + "_mean");
  node->add_input(layer->name + "_variance");
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);

  // Attr epsilon
  onnx::AttributeProto *epsilon_attr = node->add_attribute();
  epsilon_attr->set_name("epsilon");
  epsilon_attr->set_type(onnx::AttributeProto::FLOAT);
  epsilon_attr->set_f(layer->epsilon);

  // Attr momentum
  onnx::AttributeProto *momentum_attr = node->add_attribute();
  momentum_attr->set_name("momentum");
  momentum_attr->set_type(onnx::AttributeProto::FLOAT);
  momentum_attr->set_f(layer->momentum);

  int n_features = layer->input->getShape()[1];

  // Scale input
  onnx::TensorProto *scale = graph->add_initializer();
  scale->set_name(layer->name + "_scale");
  scale->set_data_type(onnx::TensorProto::FLOAT);
  scale->add_dims(n_features);

  // Bias input
  onnx::TensorProto *bias = graph->add_initializer();
  bias->set_name(layer->name + "_bias");
  bias->set_data_type(onnx::TensorProto::FLOAT);
  bias->add_dims(n_features);

  // Check if the layer has trainable parameters
  if (layer->affine)
  {
    for (int i = 0; i < n_features; ++i)
    {
      scale->add_float_data(layer->bn_g->ptr[i]);
      bias->add_float_data(layer->bn_b->ptr[i]);
    }
  }
  else
  {
    for (int i = 0; i < n_features; ++i)
    {
      // Set the scale values to 1 (1 is the default value in case of not having trainable parameters)
      scale->add_float_data(1);
      // Set the bias values to 0 (0 is the default value in case of not having trainable parameters)
      bias->add_float_data(0);
    }
  }

  // Mean input
  onnx::TensorProto *mean = graph->add_initializer();
  mean->set_name(layer->name + "_mean");
  mean->set_data_type(onnx::TensorProto::FLOAT);
  mean->add_dims(n_features);
  mean->mutable_float_data()->Add(layer->mean->ptr, layer->mean->ptr + layer->mean->size); // Set the mean values

  // variance input
  onnx::TensorProto *variance = graph->add_initializer();
  variance->set_name(layer->name + "_variance");
  variance->set_data_type(onnx::TensorProto::FLOAT);
  variance->add_dims(n_features);
  variance->mutable_float_data()->Add(layer->variance->ptr, layer->variance->ptr + layer->variance->size); // Set the mean values
}

void build_dropout_node(LDropout *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Dropout");
  node->set_name(layer->name);
  // Set the inputs of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);

  // Attr ratio
  onnx::AttributeProto *momentum_attr = node->add_attribute();
  momentum_attr->set_name("ratio");
  momentum_attr->set_type(onnx::AttributeProto::FLOAT);
  momentum_attr->set_f(layer->df);
}

void build_upsample_node(LUpSampling *layer, onnx::GraphProto *graph)
{
  // Scales input for the upsample node
  onnx::TensorProto *scales = graph->add_initializer();
  scales->set_name(layer->name + "_scales");
  scales->set_data_type(onnx::TensorProto::FLOAT);
  scales->add_dims(2 + layer->size.size()); // (batch_size, channels, height, width)
  // Add the scale factor for the first two dimensions
  for (int i = 0; i < 2; ++i)
  {
    scales->add_float_data(1);
  }
  for (int i = 0; i < layer->size.size(); ++i)
  {
    scales->add_float_data(layer->size[i]);
  }

  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Upsample");
  node->set_name(layer->name);
  // Set the inputs of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the input with the scale values
  node->add_input(layer->name + "_scales");
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);

  // Attr mode
  onnx::AttributeProto *mode_attr = node->add_attribute();
  mode_attr->set_name("mode");
  mode_attr->set_type(onnx::AttributeProto::STRING);
  mode_attr->set_s(layer->interpolation);
}

void build_squeeze_node(string node_name, string input, string output, vector<int> axes, onnx::GraphProto *graph)
{
  onnx::NodeProto *node_sq = graph->add_node();
  node_sq->set_op_type("Squeeze");
  node_sq->set_name(node_name);
  node_sq->add_input(input);
  onnx::AttributeProto *axes_attr = node_sq->add_attribute();
  axes_attr->set_name("axes");
  axes_attr->set_type(onnx::AttributeProto::INTS);
  for (int ax : axes)
    axes_attr->add_ints(ax);
  node_sq->add_output(output);
}

void build_unsqueeze_node(string node_name, string input, string output, vector<int> axes, onnx::GraphProto *graph)
{
  onnx::NodeProto *node_usq = graph->add_node();
  node_usq->set_op_type("Unsqueeze");
  node_usq->set_name(node_name);
  node_usq->add_input(input);
  onnx::AttributeProto *axes_attr = node_usq->add_attribute();
  axes_attr->set_name("axes");
  axes_attr->set_type(onnx::AttributeProto::INTS);
  for (int ax : axes)
    axes_attr->add_ints(ax);
  node_usq->add_output(output);
}

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


void build_resize_node(LScale *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Resize");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  node->add_input(layer->name + "_roi");
  node->add_input(layer->name + "_scales");
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);

  // coordinate_transformation_mode attr
  onnx::AttributeProto *trans_mode_attr = node->add_attribute();
  trans_mode_attr->set_name("coordinate_transformation_mode");
  trans_mode_attr->set_type(onnx::AttributeProto::STRING);
  trans_mode_attr->set_s("asymmetric");

  // coordinate_transformation_mode attr
  onnx::AttributeProto *mode_attr = node->add_attribute();
  mode_attr->set_name("mode");
  mode_attr->set_type(onnx::AttributeProto::STRING);
  mode_attr->set_s("nearest");

  // roi input
  onnx::TensorProto *roi = graph->add_initializer();
  roi->set_name(layer->name + "_roi");
  roi->set_data_type(onnx::TensorProto::FLOAT);
  roi->add_dims(8);
  // Set roi to : [0,0,0,0,1,1,1,1] (To select the full input tensor)
  int parent_dims = layer->parent[0]->output->getShape().size();
  for (int i = 0; i < parent_dims; ++i)
    roi->add_float_data(0);
  for (int i = 0; i < parent_dims; ++i)
    roi->add_float_data(1);

  // scales input
  onnx::TensorProto *scales = graph->add_initializer();
  scales->set_name(layer->name + "_scales");
  scales->set_data_type(onnx::TensorProto::FLOAT);
  scales->add_dims(4);
  scales->add_float_data(1);                                                 // Batch
  scales->add_float_data(1);                                                 // Channels
  scales->add_float_data(layer->new_shape[0] / layer->input->getShape()[2]); // H
  scales->add_float_data(layer->new_shape[1] / layer->input->getShape()[3]); // H
}

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

void build_embedding_node(LEmbedding *layer, onnx::GraphProto *graph)
{
  /*
   * To create the embedding operation in ONNX we have to use the following steps:
   *     1. Squeeze the last dim if the input shape is [batch, seq_len, 1]
   *     2. Create a Cast op to int type for the indexes of the words
   *     3. Create a Gather op to select the embeddings from the indexes from the Cast
   */

  // 1. Create the Squeeze node for dim 2
  string cast_node_input;
  if (layer->length == 1)
  {
    string squeeze_node_name = layer->name + "_squeeze";
    string squeeze_node_input = layer->parent[0]->name;
    string squeeze_node_output = layer->name + "_squeeze";
    build_squeeze_node(
        squeeze_node_name,
        squeeze_node_input,
        squeeze_node_output,
        {2},
        graph);
    cast_node_input = squeeze_node_output;
  }
  else
  {
    msg("The input of the embedding layer must have length 1 in order to export it", "ONNX::ExportNet");
  }

  // 2. Create the Cast op
  string cast_node_name = layer->name + "_indexes_cast";
  string cast_node_output = layer->name + "_cast";
  build_cast_node(
      cast_node_name,
      cast_node_input,
      cast_node_output,
      6, // cast type to int32
      graph);

  // 3. Creathe the Gahter op
  build_gather_node(
      layer->name,      // node name
      cast_node_output, // node input from cast node
      layer->name,      // node output name
      layer,
      graph);
}

// End: Node builders
//----------------------------------------------------------------------------------------

void handle_copy_states(LCopyStates *layer, onnx::GraphProto *graph)
{
  string parent_name = layer->parent[0]->name;
  string child_name = layer->child[0]->name;

  // Set the node to copy the hidden (h) state
  string node_name = parent_name + "_to_" + child_name + "_CopyState_h";
  string input_name = parent_name;
  string output_name = layer->name + "_h";
  /*
   * Add an Unsqueeze layer to reshape the h state to the desired shape for LSTM.
   *
   *   Note: The h state coming from the previous LSTM has been squeezed, so we
   *         have to unsqueeze it to get the desired shape for the decoder LSTM
   */
  build_unsqueeze_node(
      layer->name + "_h_unsqueeze", // node name
      input_name,                   // input name
      output_name,                  // Output name
      {0},                          // axes to squeeze
      graph);

  // Set the node to copy the cell (c) state in case of LSTM
  if (LLSTM *l = dynamic_cast<LLSTM *>(layer->parent[0]))
  {
    node_name = parent_name + "_to_" + child_name + "_CopyState_c";
    input_name = parent_name + "_Y_c";
    output_name = layer->name + "_c";
    build_identity_node(node_name, input_name, output_name, graph);
  }
}

void prepare_recurrent_input(string input_name, string output_name, vector<int> input_shape, onnx::GraphProto *graph)
{
  /*
   * This functions takes a graph of a recurrent net and adds a transpose operator
   * to fix the input shape from (batch_size, seq_len, in_shape) to (seq_len, batch_size, in_shape)
   */
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Transpose");
  node->set_name(input_name + "_transpose");
  // Set the inputs names of the node from the parents of the layer
  node->add_input(input_name);
  // Set the name of the output of the node to link with other nodes
  node->add_output(output_name);

  // Attr perm
  onnx::AttributeProto *alpha_attr = node->add_attribute();
  alpha_attr->set_name("perm");
  alpha_attr->set_type(onnx::AttributeProto::INTS);
  // Permute batch_size and seq_len
  alpha_attr->add_ints(1);
  alpha_attr->add_ints(0);
  // Add the rest of dimensions
  for (int i = 2 /*Skip batch and seq*/; i < input_shape.size(); ++i)
  {
    alpha_attr->add_ints(i);
  }
}

void prepare_recurrent_output(string input_name, string output_name, vector<int> output_shape, onnx::GraphProto *graph)
{
  /*
   * This functions takes a graph of a recurrent net and adds a transpose operator
   * to fix the output shape from (seq_len, batch_size, out_shape) to (batch_size, seq_len, out_shape)
   */
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Transpose");
  node->set_name(input_name + "_transpose");
  // Set the inputs name of the node from the parents of the layer
  node->add_input(input_name);
  // Set the name of the output of the node to link to the output of the model
  node->add_output(output_name);

  // Attr perm
  onnx::AttributeProto *alpha_attr = node->add_attribute();
  alpha_attr->set_name("perm");
  alpha_attr->set_type(onnx::AttributeProto::INTS);
  // Permute batch_size and seq_len
  alpha_attr->add_ints(1);
  alpha_attr->add_ints(0);
  // Add the rest of dimensions
  for (int i = 2 /*Skip batch and seq*/; i < output_shape.size(); ++i)
  {
    alpha_attr->add_ints(i);
  }
}

// End: Exporting Module
//----------------------------------------------------------------------------------------

#else

void save_net_to_onnx_file(Net *net, string path)
{
  cerr << "Not compiled for ONNX. Missing Protobuf" << endl;
}

size_t serialize_net_to_onnx_pointer(Net *net, void *&serialized_model, bool gradients)
{
  cerr << "Not compiled for ONNX. Missing Protobuf. Returning -1" << endl;
  return -1;
}

std::string *serialize_net_to_onnx_string(Net *net, bool gradients)
{
  cerr << "Not compiled for ONNX. Missing Protobuf. Returning nullptr" << endl;
  return nullptr;
}

#endif //cPROTO
