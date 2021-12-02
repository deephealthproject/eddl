#if defined(cPROTO)
#ifndef EDDL_IMPORT_HELPERS_H
#define EDDL_IMPORT_HELPERS_H
#include <queue>

#include "eddl/net/net.h"
#include "eddl/serialization/onnx/onnx.pb.h"
#include "eddl/serialization/onnx/utils_onnx.h"
#include "eddl/serialization/onnx/layers/layers_onnx.h"

Net *build_net_onnx(onnx::ModelProto model, vector<int> input_shape, int mem,
                    LOG_LEVEL log_level);

vector<onnx::TensorProto> get_initializers(onnx::GraphProto graph);

map<string, vector<onnx::NodeProto *>> initialize_input_node_map(vector<onnx::NodeProto> &nodes);

map<string, onnx::NodeProto *> initialize_constant_nodes(vector<onnx::NodeProto> &nodes,
                                                         map<string, vector<onnx::NodeProto *>> &input_node_map);

queue<onnx::NodeProto *> process_inputs(vector<Layer *> *inputs,
                                        vector<onnx::ValueInfoProto> *inputs_onnx,
                                        map<string, vector<onnx::NodeProto *>> *input_node_map,
                                        map<string, Layer *> *output_node_map);

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
                          map<string, vector<onnx::NodeProto *>> &input_node_map,
                          map<string, onnx::NodeProto *> &constant_node_map,
                          queue<onnx::NodeProto *> &nodeQueue,
                          LOG_LEVEL log_level);

void process_node_queue(queue<onnx::NodeProto *> &nodeQueue,
                        map<string, vector<float>> &map_init_values,
                        map<string, vector<int>> &map_init_dims,
                        map<string, vector<onnx::NodeProto *>> &input_node_map,
                        map<string, Layer *> &output_node_map,
                        map<string, onnx::NodeProto *> &constant_node_map,
                        vector<string> &inputs2remove,
                        bool recurrent_net,
                        int mem,
                        LOG_LEVEL log_level);

Net *build_net_onnx(onnx::ModelProto model, vector<int> input_shape, int mem, LOG_LEVEL log_level);

void set_weights_from_model_proto(Net *net, onnx::ModelProto model_proto);

void apply_grads_from_model_proto(Net *net, onnx::ModelProto model_proto);

#endif // EDDL_IMPORT_HELPERS_H
#endif // cPROTO
