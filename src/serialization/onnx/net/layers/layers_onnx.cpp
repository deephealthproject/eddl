#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/layers_onnx.h"
#include "eddl/serialization/onnx/layers/core/dense_onnx.h"
#include "eddl/serialization/onnx/layers/core/drop_onnx.h"
#include "eddl/serialization/onnx/layers/core/reshape_onnx.h"
#include "eddl/serialization/onnx/layers/core/activation_onnx.h"
#include "eddl/serialization/onnx/layers/core/squeeze_onnx.h"
#include "eddl/serialization/onnx/layers/core/unsqueeze_onnx.h"
#include "eddl/serialization/onnx/layers/core/permute_onnx.h"
#include "eddl/serialization/onnx/layers/core/embedding_onnx.h"
#include "eddl/serialization/onnx/layers/core/select_onnx.h"
#include "eddl/serialization/onnx/layers/core/split_onnx.h"
#include "eddl/serialization/onnx/layers/core/resize_onnx.h"
#include "eddl/serialization/onnx/layers/core/repeat_onnx.h"
#include "eddl/serialization/onnx/layers/conv/conv_onnx.h"
#include "eddl/serialization/onnx/layers/conv/conv1D_onnx.h"
#include "eddl/serialization/onnx/layers/conv/conv3D_onnx.h"
#include "eddl/serialization/onnx/layers/conv/convT_onnx.h"
#include "eddl/serialization/onnx/layers/conv/convT3D_onnx.h"
#include "eddl/serialization/onnx/layers/conv/upsampling2D_onnx.h"
#include "eddl/serialization/onnx/layers/conv/upsampling3D_onnx.h"
#include "eddl/serialization/onnx/layers/pool/avgpool_onnx.h"
#include "eddl/serialization/onnx/layers/pool/avgpool1D_onnx.h"
#include "eddl/serialization/onnx/layers/pool/avgpool3D_onnx.h"
#include "eddl/serialization/onnx/layers/pool/maxpool_onnx.h"
#include "eddl/serialization/onnx/layers/pool/maxpool1D_onnx.h"
#include "eddl/serialization/onnx/layers/pool/maxpool3D_onnx.h"
#include "eddl/serialization/onnx/layers/normalization/batchnorm_onnx.h"
#include "eddl/serialization/onnx/layers/merge/concat_onnx.h"
#include "eddl/serialization/onnx/layers/merge/add_onnx.h"
#include "eddl/serialization/onnx/layers/merge/matmul_onnx.h"
#include "eddl/serialization/onnx/layers/operators/abs_onnx.h"
#include "eddl/serialization/onnx/layers/operators/div_onnx.h"
#include "eddl/serialization/onnx/layers/operators/exp_onnx.h"
#include "eddl/serialization/onnx/layers/operators/log_onnx.h"
#include "eddl/serialization/onnx/layers/operators/mult_onnx.h"
#include "eddl/serialization/onnx/layers/operators/sqrt_onnx.h"
#include "eddl/serialization/onnx/layers/operators/diff_onnx.h"
#include "eddl/serialization/onnx/layers/operators/clamp_onnx.h"
#include "eddl/serialization/onnx/layers/operators/sum_onnx.h"
#include "eddl/serialization/onnx/layers/operators/pow_onnx.h"
#include "eddl/serialization/onnx/layers/reductions/max_onnx.h"
#include "eddl/serialization/onnx/layers/reductions/min_onnx.h"
#include "eddl/serialization/onnx/layers/reductions/mean_onnx.h"
#include "eddl/serialization/onnx/layers/reductions/rsum_onnx.h"
#include "eddl/serialization/onnx/layers/reductions/argmax_onnx.h"
#include "eddl/serialization/onnx/layers/recurrent/lstm_onnx.h"
#include "eddl/serialization/onnx/layers/recurrent/gru_onnx.h"
#include "eddl/serialization/onnx/layers/recurrent/rnn_onnx.h"
#include "eddl/serialization/onnx/layers/recurrent/cps_onnx.h"
#include "eddl/serialization/onnx/layers/da/scale_onnx.h"
#include "eddl/serialization/onnx/layers/da/pad_onnx.h"
#include "eddl/serialization/onnx/layers/onnx_nodes/onnx_node_conversion.h"
#include "eddl/serialization/onnx/layers/auxiliar/expand_onnx.h"
#include "eddl/serialization/onnx/layers/auxiliar/constoftensor_onnx.h"

/*
 * ONNX IMPORT
 */

// Creates a map where the key is the onnx name for the layer type and the 
// value is the constant value in the enumeration for onnx layer type.
map<string, ONNX_LAYERS> create_enum_map()
{
  map<string, ONNX_LAYERS> map_layers;
  map_layers["BatchNormalization"] = ONNX_LAYERS::BATCHNORM;
  map_layers["Conv"] = ONNX_LAYERS::CONV;
  map_layers["ConvTranspose"] = ONNX_LAYERS::CONVTRANSPOSE;
  map_layers["Gemm"] = ONNX_LAYERS::DENSE;
  map_layers["Dropout"] = ONNX_LAYERS::DROP;
  map_layers["Reshape"] = ONNX_LAYERS::RESHAPE;
  map_layers["Flatten"] = ONNX_LAYERS::FLATTEN;
  map_layers["Transpose"] = ONNX_LAYERS::TRANSPOSE;
  map_layers["Squeeze"] = ONNX_LAYERS::SQUEEZE;
  map_layers["Unsqueeze"] = ONNX_LAYERS::UNSQUEEZE;
  map_layers["Upsample"] = ONNX_LAYERS::UPSAMPLING;
  map_layers["Softmax"] = ONNX_LAYERS::SOFTMAX;
  map_layers["MaxPool"] = ONNX_LAYERS::MAXPOOL;
  map_layers["AveragePool"] = ONNX_LAYERS::AVGPOOL;
  map_layers["GlobalMaxPool"] = ONNX_LAYERS::GLOBMAXPOOL;
  map_layers["GlobalAveragePool"] = ONNX_LAYERS::GLOBAVGPOOL;
  // Activation layers
  map_layers["Relu"] = ONNX_LAYERS::RELU;
  map_layers["Sigmoid"] = ONNX_LAYERS::SIGMOID;
  map_layers["HardSigmoid"] = ONNX_LAYERS::HARD_SIGMOID;
  map_layers["Tanh"] = ONNX_LAYERS::TANH;
  map_layers["Linear"] = ONNX_LAYERS::LINEAR;
  map_layers["Exponential"] = ONNX_LAYERS::EXPONENTIAL;
  map_layers["LeakyRelu"] = ONNX_LAYERS::LEAKY_RELU;
  map_layers["ThresholdedRelu"] = ONNX_LAYERS::THRESHOLDED_RELU;
  map_layers["Elu"] = ONNX_LAYERS::ELU;
  map_layers["Selu"] = ONNX_LAYERS::SELU;
  map_layers["Softsign"] = ONNX_LAYERS::SOFTSIGN;
  map_layers["Softplus"] = ONNX_LAYERS::SOFTPLUS;
  // Merge Layers
  map_layers["Concat"] = ONNX_LAYERS::CONCAT;
  map_layers["Add"] = ONNX_LAYERS::ADD;
  map_layers["MatMul"] = ONNX_LAYERS::MAT_MUL;

  map_layers["LSTM"] = ONNX_LAYERS::LSTM;
  map_layers["GRU"] = ONNX_LAYERS::GRU;
  map_layers["RNN"] = ONNX_LAYERS::RNN;
  map_layers["Identity"] = ONNX_LAYERS::IDENTITY;
  map_layers["Gather"] = ONNX_LAYERS::GATHER;
  map_layers["Cast"] = ONNX_LAYERS::CAST;
  map_layers["Abs"] = ONNX_LAYERS::ABS;
  map_layers["Sum"] = ONNX_LAYERS::SUM;
  map_layers["Div"] = ONNX_LAYERS::DIV;
  map_layers["Exp"] = ONNX_LAYERS::EXP;
  map_layers["Log"] = ONNX_LAYERS::LOG;
  map_layers["Pow"] = ONNX_LAYERS::POW;
  map_layers["Mul"] = ONNX_LAYERS::MUL;
  map_layers["Clip"] = ONNX_LAYERS::CLIP;
  map_layers["Sqrt"] = ONNX_LAYERS::SQRT;
  map_layers["Sub"] = ONNX_LAYERS::SUB;
  map_layers["ReduceMax"] = ONNX_LAYERS::RMAX;
  map_layers["ReduceMin"] = ONNX_LAYERS::RMIN;
  map_layers["ReduceMean"] = ONNX_LAYERS::RMEAN;
  map_layers["ReduceSum"] = ONNX_LAYERS::RSUM;
  map_layers["ArgMax"] = ONNX_LAYERS::ARGMAX;
  map_layers["Resize"] = ONNX_LAYERS::RESIZE;
  map_layers["Pad"] = ONNX_LAYERS::PAD;
  map_layers["Slice"] = ONNX_LAYERS::SLICE;
  map_layers["Split"] = ONNX_LAYERS::SPLIT;
  map_layers["Expand"] = ONNX_LAYERS::EXPAND;
  map_layers["Constant"] = ONNX_LAYERS::CONSTANT;
  map_layers["Tile"] = ONNX_LAYERS::REPEAT;

  return map_layers;
}

ONNX_LAYERS get_layer_type(string layer_type_name, map<string, ONNX_LAYERS> &map_layers)
{
  if (map_layers.count(layer_type_name))
    return map_layers[layer_type_name];
  else
    return ONNX_LAYERS::NOT_SUPPORTED;
}

Layer* build_layer_from_node(onnx::NodeProto *node,
                             map<string, ONNX_LAYERS> &map_layers,
                             map<string, vector<float>> &map_init_values,
                             map<string, vector<int>> &map_init_dims,
                             map<string, vector<onnx::NodeProto *>> &input_node_map,
                             map<string, Layer *> &output_node_map,
                             map<string, onnx::NodeProto *> &constant_node_map,
                             vector<string> &inputs2remove,
                             bool recurrent_net,
                             LOG_LEVEL log_level,
                             int dev,
                             int mem)
{
  string name = node->name();
  string layer_type_name = node->op_type();
  log_string("Node " + name + " has operation type = " + layer_type_name, log_level, LOG_LEVEL::DEBUG);
  ONNX_LAYERS layer_type = get_layer_type(layer_type_name, map_layers);

  Layer *new_layer = nullptr;
  switch (layer_type)
  {
    case ONNX_LAYERS::BATCHNORM:
      new_layer = build_batchnorm_layer(node, map_init_values, map_init_dims, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::CONV:
      new_layer = build_conv_layer(node, map_init_values, map_init_dims, output_node_map, log_level, dev, mem);
      break;
    case ONNX_LAYERS::CONVTRANSPOSE:
      new_layer = build_convT_layer(node, map_init_values, map_init_dims, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::DENSE:
      new_layer = build_dense_layer(node, map_init_values, map_init_dims, output_node_map, log_level, dev, mem);
      break;
    case ONNX_LAYERS::UPSAMPLING:
      new_layer = build_upsampling_layer(node, map_init_values, map_init_dims, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::DROP:
      new_layer = build_dropout_layer(node, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::MAXPOOL:
      new_layer = build_maxpool_layer(node, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::GLOBMAXPOOL:
      new_layer = build_globalmaxpool_layer(node, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::AVGPOOL:
      new_layer = build_averagepool_layer(node, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::GLOBAVGPOOL:
      new_layer = build_globalaveragegpool_layer(node, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::RESHAPE:
      new_layer = build_reshape_layer(node, constant_node_map, map_init_values, map_init_dims, input_node_map, output_node_map, log_level, dev, mem);
      break;
    case ONNX_LAYERS::FLATTEN:
      new_layer = build_flatten_layer(node, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::RELU:
      new_layer = build_relu_layer(node, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::SIGMOID:
      new_layer = build_sigmoid_layer(node, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::HARD_SIGMOID:
      new_layer = build_hard_sigmoid_layer(node, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::TANH:
      new_layer = build_tanh_layer(node, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::EXPONENTIAL:
      new_layer = build_exponential_layer(node, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::LINEAR:
      new_layer = build_linear_layer(node, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::LEAKY_RELU:
      new_layer = build_leaky_relu_layer(node, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::THRESHOLDED_RELU:
      new_layer = build_thresholded_relu_layer(node, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::ELU:
      new_layer = build_elu_layer(node, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::SELU:
      new_layer = build_selu_layer(node, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::SOFTSIGN:
      new_layer = build_softsign_layer(node, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::SOFTPLUS:
      new_layer = build_softplus_layer(node, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::SOFTMAX:
      new_layer = build_softmax_layer(node, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::CONCAT:
      new_layer = build_concat_layer(node, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::ADD:
      new_layer = build_add_layer(node, map_init_values, map_init_dims, output_node_map, log_level, dev, mem);
      break;
    case ONNX_LAYERS::ABS:
      new_layer = build_abs_layer(node, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::SUM:
      new_layer = build_sum_layer(node, map_init_values, map_init_dims, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::DIV:
      new_layer = build_div_layer(node, map_init_values, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::EXP:
      new_layer = build_exp_layer(node, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::LOG:
      new_layer = build_log_layer(node, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::MUL:
      new_layer = build_mul_layer(node, map_init_values, map_init_dims, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::CLIP:
      new_layer = build_clamp_layer(node, map_init_values, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::POW:
      new_layer = build_pow_layer(node, map_init_values, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::SQRT:
      new_layer = build_sqrt_layer(node, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::SUB:
      new_layer = build_diff_layer(node, map_init_values, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::RMAX:
      new_layer = build_rmax_layer(node, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::RMIN:
      new_layer = build_rmin_layer(node, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::RMEAN:
      new_layer = build_rmean_layer(node, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::RSUM:
      new_layer = build_rsum_layer(node, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::ARGMAX:
      new_layer = build_rargmax_layer(node, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::MAT_MUL:
      new_layer = build_matmul_layer(node, map_init_values, map_init_dims, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::LSTM:
      new_layer = build_lstm_layer(node, map_init_values, map_init_dims, input_node_map, output_node_map, inputs2remove, log_level, dev, mem);
      break;
    case ONNX_LAYERS::GRU:
      new_layer = build_gru_layer(node, map_init_values, map_init_dims, input_node_map, output_node_map, inputs2remove, log_level, dev, mem);
      break;
    case ONNX_LAYERS::RNN:
      new_layer = build_rnn_layer(node, map_init_values, map_init_dims, input_node_map, output_node_map, inputs2remove, log_level, dev, mem);
      break;
    case ONNX_LAYERS::IDENTITY:
      new_layer = handle_identity_node(node, output_node_map, log_level, dev, mem);
      break;
    case ONNX_LAYERS::CAST:
      new_layer = handle_cast_node(node, output_node_map, log_level, dev, mem);
      break;
    case ONNX_LAYERS::GATHER:
      new_layer = handle_gather_node(node, map_init_values, map_init_dims, output_node_map, log_level, dev, mem);
      break;
    case ONNX_LAYERS::SQUEEZE:
      new_layer = build_squeeze_layer(node, output_node_map, log_level, dev, mem);
      break;
    case ONNX_LAYERS::UNSQUEEZE:
      new_layer = build_unsqueeze_layer(node, map_init_values, map_init_dims, output_node_map, log_level, dev, mem);
      break;
    case ONNX_LAYERS::TRANSPOSE:
      new_layer = build_permute_layer(node, output_node_map, recurrent_net, log_level, dev, mem);
      break;
    case ONNX_LAYERS::RESIZE:
      new_layer = build_resize_layer(node, map_init_values, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::PAD:
      new_layer = build_pad_layer(node, map_init_values, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::SLICE:
      new_layer = build_select_layer(node, map_init_values, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::SPLIT:
      new_layer = build_split_layer(node, map_init_values, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::EXPAND:
      new_layer = build_expand_layer(node, map_init_values, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::CONSTANT:
      new_layer = build_constoftensor_layer(node, map_init_values, output_node_map, dev, mem);
      break;
    case ONNX_LAYERS::REPEAT:
      new_layer = build_repeat_layer(node, constant_node_map, map_init_values, output_node_map, log_level, dev, mem);
      break;
    default: {
        std::cerr << "==================================================================" << std::endl;
        std::cerr << "[ONNX IMPORTING ERROR]: " << "The onnx node '" << layer_type_name << "' is not supported yet" << std::endl;
        std::cerr << "Potential fixes:" << std::endl;
        std::cerr << "\t- You can try to use 'ONNX Simplifier' to simplify the model in order to remove the redundant operators with their constant outputs." << std::endl;
        std::cerr << "\t- Documentation: https://deephealthproject.github.io/eddl/model/onnx.html#simplifying-a-onnx-model" << std::endl;
        std::cerr << "\t- ONNX Simplifier: https://github.com/daquexian/onnx-simplifier" << std::endl;
        std::cerr << "==================================================================" << std::endl;
    }
  }

  return new_layer;
}

/*
 * ONNX EXPORT
 */

// Builds a node in the onnx graph from the layer of eddl
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
  else if (LConv3D *l = dynamic_cast<LConv3D *>(layer))
    build_conv3D_node(l, graph, gradients);
  else if (LConvT2D *l = dynamic_cast<LConvT2D *>(layer))
    build_convT_node(l, graph, gradients);
  else if (LConvT3D *l = dynamic_cast<LConvT3D *>(layer))
    build_convT3D_node(l, graph, gradients);
  else if (LDense *l = dynamic_cast<LDense *>(layer))
    if (is_recurrent)
      build_dense_with_matmul_node(l, graph, gradients);
    else
      build_gemm_node(l, graph, gradients);
  else if (LMaxPool *l = dynamic_cast<LMaxPool *>(layer))
    build_maxpool_node(l, graph);
  else if (LMaxPool1D *l = dynamic_cast<LMaxPool1D *>(layer))
    build_maxpool1D_node(l, graph);
  else if (LMaxPool3D *l = dynamic_cast<LMaxPool3D *>(layer))
    build_maxpool3D_node(l, graph);
  else if (LAveragePool *l = dynamic_cast<LAveragePool *>(layer))
    build_averagepool_node(l, graph);
  else if (LAveragePool1D *l = dynamic_cast<LAveragePool1D *>(layer))
    build_averagepool1D_node(l, graph);
  else if (LAveragePool3D *l = dynamic_cast<LAveragePool3D *>(layer))
    build_averagepool3D_node(l, graph);
  else if (LReshape *l = dynamic_cast<LReshape *>(layer))
    build_reshape_node(l, graph);
  else if (LSqueeze *l = dynamic_cast<LSqueeze *>(layer))
    build_squeeze_node(l, graph);
  else if (LUnsqueeze *l = dynamic_cast<LUnsqueeze *>(layer))
    build_unsqueeze_node(l, graph);
  else if (LPermute *l = dynamic_cast<LPermute *>(layer))
    build_permute_node(l, graph);
  else if (LUpSampling *l = dynamic_cast<LUpSampling *>(layer))
    build_upsample_node(l, graph);
  else if (LUpSampling3D *l = dynamic_cast<LUpSampling3D *>(layer))
    build_resize_node_from_upsampling3D(l, graph);
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
      cerr << "[ONNX EXPORTING ERROR]: The activation layer " << layer->name << " has no valid type to export." << endl;
      return;
    }
  else if (LConcat *l = dynamic_cast<LConcat *>(layer))
    build_concat_node(l, graph);
  else if (LAbs *l = dynamic_cast<LAbs *>(layer))
    build_abs_node(l, graph);
  else if (LSum *l = dynamic_cast<LSum *>(layer))
    build_sum_node(l, graph);
  else if (LAdd *l = dynamic_cast<LAdd *>(layer))
    build_add_node(l, graph);
  else if (LDiv *l = dynamic_cast<LDiv *>(layer))
    build_div_node(l, graph);
  else if (LExp *l = dynamic_cast<LExp *>(layer))
    build_exp_node(l, graph);
  else if (LLog *l = dynamic_cast<LLog *>(layer))
    build_log_node(l, graph);
  else if (LPow *l = dynamic_cast<LPow *>(layer))
    build_pow_node(l, graph);
  else if (LMult *l = dynamic_cast<LMult *>(layer))
    build_mul_node(l, graph);
  else if (LClamp *l = dynamic_cast<LClamp *>(layer))
    build_clip_node(l, graph);
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
  else if (LResize *l = dynamic_cast<LResize *>(layer))
    build_resize_node(l, graph);
  else if (LScale *l = dynamic_cast<LScale *>(layer))
    build_resize_node_from_scale(l, graph);
  else if (LPad *l = dynamic_cast<LPad *>(layer))
    build_pad_node(l, graph);
  else if (LSelect *l = dynamic_cast<LSelect *>(layer))
    build_select_node(l, graph);
  else if (LExpand *l = dynamic_cast<LExpand *>(layer))
    build_expand_node(l, graph);
  else if (LConstOfTensor *l = dynamic_cast<LConstOfTensor *>(layer))
    build_constant_node(l, graph);
  else if (LRepeat *l = dynamic_cast<LRepeat *>(layer))
    build_tile_node(l, graph);
  else
  {
    cerr << "[ONNX EXPORTING ERROR]: The layer " << layer->name << " has no OpType in Onnx." << endl;
    return;
  }
}

/*
 * DISTRIBUTED TRAINING
 */

void update_layer_weights(Layer *layer, vector<Tensor *> weights)
{
  if (weights.size() == 0)
  {
    cerr << "[ONNX::WARNING] Trying to update the weights of the layer \""
         << layer->name << "\" with an empty list of tensors." << endl;
    return;
  }

  if (LConv *l = dynamic_cast<LConv *>(layer))
    update_conv_weights(l, weights);
  else if (LDense *l = dynamic_cast<LDense *>(layer))
    update_dense_weights(l, weights);
  else
    cerr << "The layer " << l->name << " has no support for setting weights" << endl;
}

void apply_grads_to_layer(Layer *layer, vector<Tensor *> grads)
{
  if (grads.size() == 0)
  {
    cerr << "[ONNX::WARNING] Trying to apply gradients to the layer \""
         << layer->name << "\" with an empty list of tensors." << endl;
    return;
  }

  if (LConv *l = dynamic_cast<LConv *>(layer))
    apply_grads_to_conv(l, grads);
  else if (LDense *l = dynamic_cast<LDense *>(layer))
    apply_grads_to_dense(l, grads);
  else
    cerr << "The layer " << l->name << " has no support for applying gradients" << endl;
}

#endif // defined(cPROTO)
