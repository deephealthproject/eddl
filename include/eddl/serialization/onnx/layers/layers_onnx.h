#if defined(cPROTO)
#ifndef EDDL_LAYERS_ONNX_H
#define EDDL_LAYERS_ONNX_H
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/serialization/onnx/utils_onnx.h"
#include "eddl/layers/layer.h"

/*
 * ONNX IMPORT
 */

enum ONNX_LAYERS {
  NOT_SUPPORTED,    // To handle not supported nodes
  BATCHNORM,        // OPSET: 9
  CONV,             // OPSET: 11, 1
  CONVTRANSPOSE,    // OPSET: 11, 1
  DENSE,            // OPSET: 13, 11
  DROP,             // OPSET: 10, 7
  RESHAPE,          // OPSET: 13, 5
  SQUEEZE,          // OPSET: 11, 1
  UNSQUEEZE,        // OPSET: 11, 1
  FLATTEN,          // OPSET: 13, 11, 9, 1
  TRANSPOSE,        // OPSET: 13, 1
  UPSAMPLING,       // OPSET: 9 (Op deprecated in ONNX)
  MAXPOOL,          // OPSET: 12, 11, 10, 8, 1
  AVGPOOL,          // OPSET: 11, 10, 7, 1
  GLOBAVGPOOL,      // OPSET: 1
  GLOBMAXPOOL,      // OPSET: 1
  RELU,             // OPSET: 14, 13, 6
  SOFTMAX,          // OPSET: 11, 1
  SIGMOID,          // OPSET: 13, 6
  HARD_SIGMOID,     // OPSET: 6
  TANH,             // OPSET: 13, 6
  LINEAR,           // Not in ONNX: Custom operator
  EXPONENTIAL,      // Not in ONNX: Custom operator
  LEAKY_RELU,       // OPSET: 6
  THRESHOLDED_RELU, // OPSET: 10
  ELU,              // OPSET: 6
  SELU,             // OPSET: 6
  SOFTPLUS,         // OPSET: 1
  SOFTSIGN,         // OPSET: 1
  CONCAT,           // OPSET: 13, 11, 4, 1
  ADD,              // OPSET: 13, 7
  MAT_MUL,          // OPSET: 13, 9, 1
  LSTM,             // OPSET: 7, 1
  GRU,              // OPSET: 7, 3, 1
  RNN,              // OPSET: 7, 1
  IDENTITY,         // We skip this layer when found
  GATHER,           // OPSET: 13, 11, 1
  CAST,             // We skip this layer when found
  ABS,              // OPSET: 13, 6
  SUM,              // OPSET: 13, 8, 6, 1
  DIV,              // OPSET: 13, 7
  EXP,              // OPSET: 13, 6
  LOG,              // OPSET: 13, 6
  POW,              // OPSET: 13, 12, 7
  MUL,              // OPSET: 13, 7
  CLIP,             // OPSET: 13, 12, 11
  SQRT,             // OPSET: 13, 6
  SUB,              // OPSET: 13, 7
  RMAX,             // OPSET: 13, 12, 11, 1
  RMIN,             // OPSET: 13, 12, 11, 1
  RMEAN,            // OPSET: 13, 11, 1
  RSUM,             // OPSET: 11, 1
  ARGMAX,           // OPSET: 13, 12, 11, 1
  RESIZE,           // OPSET: 11
  PAD,              // OPSET: 13, 11, 2, 1
  SLICE,            // OPSET: 13, 11, 10
  SPLIT,            // OPSET: 13, 11, 2
  EXPAND,           // OPSET: 13, 8
  MULTITHRESHOLD,   // Not in ONNX: Custom
  TOPK,             // OPSET: ????
  CONSTANT,         // OPSET: 13, 12, 11, 9, 1
  REPEAT,           // OPSET: 13, 6
  LRN               // Skiped with LBypass
};

map<string, ONNX_LAYERS> create_enum_map();

ONNX_LAYERS get_layer_type(string layer_type_name, map<string, ONNX_LAYERS> &map_layers);

Layer* build_layer_from_node(onnx::NodeProto *node,
                             map<string, ONNX_LAYERS> &map_layers,
                             map<string, vector<float>> &map_init_values,
                             map<string, vector<int>> &map_init_dims,
                             map<string, vector<onnx::NodeProto *>> &input_node_map,
                             map<string, Layer *> &output_node_map,
                             map<string, onnx::NodeProto *> &constant_node_map,
                             bool recurrent_net,
                             LOG_LEVEL log_level,
                             int dev,
                             int mem);

/*
 * ONNX EXPORT
 */

void build_node_from_layer(Layer *layer, onnx::GraphProto *graph, bool gradients, bool is_recurrent, int seq_len = 0);

/*
 * DISTRIBUTED TRAINING
 */

map<string, vector<Tensor *>> get_tensors_from_onnx_nodes(vector<onnx::NodeProto> &nodes,
                                                          map<string, vector<float>> &map_init_values,
                                                          map<string, vector<int>> &map_init_dims);
#endif // EDDL_LAYERS_ONNX_H
#endif // cPROTO
