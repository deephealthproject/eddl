#if defined(cPROTO)
#ifndef EDDL_IMPORT_HELPERS_H
#define EDDL_IMPORT_HELPERS_H
#include <queue>

#include "eddl/net/net.h"
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/serialization/onnx/utils_onnx.h"

enum ONNX_LAYERS {
  NOT_SUPPORTED,    // To handle not supported nodes
  BATCHNORM,        // OPSET: 9
  CONV,             // OPSET: 11, 1
  DENSE,            // OPSET: 13, 11
  DROP,             // OPSET: 10, 7
  RESHAPE,          // OPSET: 13, 5
  SQUEEZE,          // OPSET: 11, 1
  UNSQUEEZE,        // OPSET: 11, 1
  FLATTEN,          // OPSET: 13, 11, 9, 1
  TRANSPOSE,        // OPSET: 13, 1
  UPSAMPLING,       // OPSET: 9 (Op deprecated in ONNX)
  MAXPOOL,          // OPSET: 12, 11, 10, 8, 1
  AVGPOOL,          // OPSET: 11, 10, 7, 1 - TODO: testing
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
  DIV,              // OPSET: 13, 7
  EXP,              // OPSET: 13, 6
  LOG,              // OPSET: 13, 6
  MUL,              // OPSET: 13, 7
  SQRT,             // OPSET: 13, 6
  SUB,              // OPSET: 13, 7
  RMAX,             // OPSET: 13, 12, 11, 1
  RMIN,             // OPSET: 13, 12, 11, 1
  RMEAN,            // OPSET: 13, 11, 1
  RSUM,             // OPSET: 11, 1
  ARGMAX,           // OPSET: 13, 12, 11, 1
  RESIZE            // OPSET: 13
  // POW,            // OPSET: 13, 12, 7 (TODO: Implement LPow)
};

Net *build_net_onnx(onnx::ModelProto model, vector<int> input_shape, int mem,
                    LOG_LEVEL log_level);

vector<onnx::TensorProto> get_initializers(onnx::GraphProto graph);

map<string, vector<onnx::NodeProto *>> initialize_input_node_map(vector<onnx::NodeProto> &nodes);

map<string, onnx::NodeProto *> initialize_constant_nodes(vector<onnx::NodeProto> &nodes);

queue<onnx::NodeProto *> process_inputs(vector<Layer *> *inputs,
                                        vector<onnx::ValueInfoProto> *inputs_onnx,
                                        map<string, vector<onnx::NodeProto *>> *input_node_map,
                                        map<string, Layer *> *output_node_map);

map<string, ONNX_LAYERS> create_enum_map();

void get_initializers_maps(vector<onnx::TensorProto> tensors, 
                           map<string, vector<float>> &values_map, 
                           map<string, vector<int>> &dims_map);

vector<int> parse_IO_tensor(onnx::TypeProto::Tensor tensor, bool recurrent_net);

vector<Layer *> parse_IO_tensors(vector<onnx::ValueInfoProto> io_onnx, 
                                 vector<int> input_shape, 
                                 int mem, 
                                 bool recurrent_net);

vector<onnx::ValueInfoProto> get_inputs(onnx::GraphProto graph);

vector<string> get_outputs(onnx::GraphProto graph);

vector<onnx::NodeProto> get_graph_nodes(onnx::GraphProto graph);

Layer *get_model_input_layer(Layer *l);

bool node_is_recurrent(onnx::NodeProto *node, map<string, ONNX_LAYERS> &map_layers);

void set_decoder(Layer *l);

bool node_is_decoder(onnx::NodeProto *node, map<string, vector<onnx::NodeProto *>> &input_node_map);

bool check_recurrent_nodes(vector<onnx::NodeProto> nodes);

void share_weights(Net *net);

map<string, vector<Tensor *>> get_tensors_from_onnx(onnx::ModelProto model);

void log_model_metadata(onnx::ModelProto& model, LOG_LEVEL log_level);

void queue_constant_nodes(vector<onnx::NodeProto> &nodes,
                          map<string, vector<float>> &map_init_values,
                          map<string, onnx::NodeProto *> &constant_node_map,
                          queue<onnx::NodeProto *> &nodeQueue,
                          LOG_LEVEL log_level);

ONNX_LAYERS get_layer_type(string layer_type_name, map<string, ONNX_LAYERS> &map_layers);

Net *build_net_onnx(onnx::ModelProto model, vector<int> input_shape, int mem, LOG_LEVEL log_level);

#endif // EDDL_IMPORT_HELPERS_H
#endif // cPROTO
