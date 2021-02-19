#if defined(cPROTO)
#include "eddl/serialization/onnx/export_helpers.h"

#include "eddl/serialization/onnx/layers/core/dense_onnx.h"
#include "eddl/serialization/onnx/layers/core/reshape_onnx.h"
#include "eddl/serialization/onnx/layers/core/permute_onnx.h"
#include "eddl/serialization/onnx/layers/core/activation_onnx.h"
#include "eddl/serialization/onnx/layers/core/drop_onnx.h"
#include "eddl/serialization/onnx/layers/core/squeeze_onnx.h"
#include "eddl/serialization/onnx/layers/core/unsqueeze_onnx.h"
#include "eddl/serialization/onnx/layers/core/embedding_onnx.h"
#include "eddl/serialization/onnx/layers/merge/concat_onnx.h"
#include "eddl/serialization/onnx/layers/merge/add_onnx.h"
#include "eddl/serialization/onnx/layers/operators/abs_onnx.h"
#include "eddl/serialization/onnx/layers/operators/diff_onnx.h"
#include "eddl/serialization/onnx/layers/operators/div_onnx.h"
#include "eddl/serialization/onnx/layers/operators/exp_onnx.h"
#include "eddl/serialization/onnx/layers/operators/log_onnx.h"
#include "eddl/serialization/onnx/layers/operators/mult_onnx.h"
#include "eddl/serialization/onnx/layers/operators/sqrt_onnx.h"
#include "eddl/serialization/onnx/layers/reductions/mean_onnx.h"
#include "eddl/serialization/onnx/layers/reductions/sum_onnx.h"
#include "eddl/serialization/onnx/layers/reductions/max_onnx.h"
#include "eddl/serialization/onnx/layers/reductions/min_onnx.h"
#include "eddl/serialization/onnx/layers/reductions/argmax_onnx.h"
#include "eddl/serialization/onnx/layers/normalization/batchnorm_onnx.h"
#include "eddl/serialization/onnx/layers/conv/conv_onnx.h"
#include "eddl/serialization/onnx/layers/conv/conv1D_onnx.h"
#include "eddl/serialization/onnx/layers/conv/upsampling_onnx.h"
#include "eddl/serialization/onnx/layers/pool/maxpool_onnx.h"
#include "eddl/serialization/onnx/layers/pool/maxpool1D_onnx.h"
#include "eddl/serialization/onnx/layers/pool/avgpool_onnx.h"
#include "eddl/serialization/onnx/layers/recurrent/lstm_onnx.h"
#include "eddl/serialization/onnx/layers/recurrent/gru_onnx.h"
#include "eddl/serialization/onnx/layers/recurrent/rnn_onnx.h"
#include "eddl/serialization/onnx/layers/recurrent/cps_onnx.h"
#include "eddl/serialization/onnx/layers/da/scale_onnx.h"

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
  if (LInput *l = dynamic_cast<LInput *>(layer))
    return; // Skip the input layers
  else if (LConv *l = dynamic_cast<LConv *>(layer))
    build_conv_node(l, graph, gradients);
  else if (LConv1D *l = dynamic_cast<LConv1D *>(layer))
    build_conv1D_node(l, graph, gradients);
  else if (LDense *l = dynamic_cast<LDense *>(layer))
    if (is_recurrent)
      build_dense_with_matmul_node(l, graph, gradients);
    else
      build_gemm_node(l, graph, gradients);
  else if (LMaxPool *l = dynamic_cast<LMaxPool *>(layer))
    build_maxpool_node(l, graph);
  else if (LMaxPool1D *l = dynamic_cast<LMaxPool1D *>(layer))
    build_maxpool1D_node(l, graph);
  else if (LAveragePool *l = dynamic_cast<LAveragePool *>(layer))
    build_averagepool_node(l, graph);
  else if (LReshape *l = dynamic_cast<LReshape *>(layer))
    build_reshape_node(l, graph);
  else if (LPermute *l = dynamic_cast<LPermute *>(layer))
    build_permute_node(l, graph);
  else if (LUpSampling *l = dynamic_cast<LUpSampling *>(layer))
    build_upsample_node(l, graph);
  else if (LActivation *l = dynamic_cast<LActivation *>(layer))
    // Check the type of activation layer
    if (!l->act.compare("relu"))
      build_relu_node((LActivation *)(LinLayer *)layer, graph);
    else if (!l->act.compare("sigmoid"))
      build_sigmoid_node((LActivation *)(LinLayer *)layer, graph);
    else if (!l->act.compare("hard_sigmoid"))
      build_hard_sigmoid_node((LActivation *)(LinLayer *)layer, graph);
    else if (!l->act.compare("tanh"))
      build_tanh_node((LActivation *)(LinLayer *)layer, graph);
    else if (!l->act.compare("exp"))
      build_exponential_node((LActivation *)(LinLayer *)layer, graph);
    else if (!l->act.compare("linear"))
      build_linear_node((LActivation *)(LinLayer *)layer, graph);
    else if (!l->act.compare("leaky_relu"))
      build_leaky_relu_node(l, graph);
    else if (!l->act.compare("thresholded_relu"))
      build_thresholded_relu_node(l, graph);
    else if (!l->act.compare("elu"))
      build_elu_node(l, graph);
    else if (!l->act.compare("selu"))
      build_selu_node(l, graph);
    else if (!l->act.compare("softmax") || !l->act.compare("full_softmax"))
      build_softmax_node(l, graph);
    else if (!l->act.compare("softsign"))
      build_softsign_node(l, graph);
    else if (!l->act.compare("softplus"))
      build_softplus_node(l, graph);
    else 
    {
      cout << "The activation layer " << layer->name << "has no valid type to export." << endl;
      return;
    }
  else if (LConcat *l = dynamic_cast<LConcat *>(layer))
    build_concat_node(l, graph);
  else if (LAbs *l = dynamic_cast<LAbs *>(layer))
    build_abs_node(l, graph);
  else if (LAdd *l = dynamic_cast<LAdd *>(layer))
    build_add_node(l, graph);
  else if (LDiv *l = dynamic_cast<LDiv *>(layer))
    build_div_node(l, graph);
  else if (LExp *l = dynamic_cast<LExp *>(layer))
    build_exp_node(l, graph);
  else if (LLog *l = dynamic_cast<LLog *>(layer))
    build_log_node(l, graph);
  else if (LMult *l = dynamic_cast<LMult *>(layer))
    build_mul_node(l, graph);
  //else if (LPow *l = dynamic_cast<LPow *>(layer))
  //  build_pow_node(l, graph);
  else if (LSqrt *l = dynamic_cast<LSqrt *>(layer))
    build_sqrt_node(l, graph);
  else if (LDiff *l = dynamic_cast<LDiff *>(layer))
    build_sub_node(l, graph);
  else if (LRMean *l = dynamic_cast<LRMean *>(layer))
    build_rmean_node(l, graph);
  else if (LRSum *l = dynamic_cast<LRSum *>(layer))
    build_rsum_node(l, graph);
  else if (LRMax *l = dynamic_cast<LRMax *>(layer))
    build_rmax_node(l, graph);
  else if (LRMin *l = dynamic_cast<LRMin *>(layer))
    build_rmin_node(l, graph);
  else if (LRArgmax *l = dynamic_cast<LRArgmax *>(layer))
    build_rargmax_node(l, graph);
  else if (LBatchNorm *l = dynamic_cast<LBatchNorm *>(layer))
    build_batchnorm_node(l, graph);
  else if (LDropout *l = dynamic_cast<LDropout *>(layer))
    build_dropout_node(l, graph);
  else if (LLSTM *l = dynamic_cast<LLSTM *>(layer))
    build_lstm_node(l, graph);
  else if (LGRU *l = dynamic_cast<LGRU *>(layer))
    build_gru_node(l, graph);
  else if (LRNN *l = dynamic_cast<LRNN *>(layer))
    build_rnn_node(l, graph);
  else if (LCopyStates *l = dynamic_cast<LCopyStates *>(layer))
    handle_copy_states(l, graph);
  else if (LEmbedding *l = dynamic_cast<LEmbedding *>(layer))
    build_embedding_node(l, graph);
  else if (LScale *l = dynamic_cast<LScale *>(layer))
    build_resize_node(l, graph);
  else
  {
    cout << "The layer " << layer->name << "has no OpType in Onnx." << endl;
    return;
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

#endif // defined(cPROTO)
