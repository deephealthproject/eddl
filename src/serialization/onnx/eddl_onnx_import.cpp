#include "eddl/layers/operators/layer_operators.h"
#include "eddl/serialization/onnx/eddl_onnx.h"
#include <queue>
#include <fstream>
#include "eddl/layers/core/layer_core.h"
#include "eddl/layers/conv/layer_conv.h"
#include "eddl/layers/normalization/layer_normalization.h"
#include "eddl/layers/pool/layer_pool.h"
#include "eddl/layers/recurrent/layer_recurrent.h"
#include "eddl/layers/reductions/layer_reductions.h"
#include "eddl/layers/da/layer_da.h"
#include "eddl/tensor/tensor.h"
#include "eddl/utils.h"
#include <map>
#include <set>
#include <algorithm>

// #define NEW_FROM_VECTOR_PTR(v) (copy((v)->begin(), (v)->end(), new float[(v)->size()]) - (v)->size()) -- this generates memory leaks
#define COPY_FROM_VECTOR_PTR_TO_TENSOR(v, t) (copy((v)->begin(), (v)->end(), t->ptr))
#define COPY_FROM_VECTOR_PTR_TO_FLOAT_PTR(v, ptr) (copy((v)->begin(), (v)->end(), ptr))

std::vector<int> vf2vi(const std::vector<float> &vf)
{
  std::vector<int> vi;
  vi.reserve(vf.size());
  for (const auto &x : vf)
  {
    vi.emplace_back(static_cast<int>(x));
  }
  return vi;
}

#if defined(cPROTO)
#include "onnx.pb.h"
#endif

#if defined(cPROTO)
Net *build_net_onnx(onnx::ModelProto model, vector<int> input_shape, int mem, int log_level);
#endif

#if defined(cPROTO)
map<string, vector<Tensor *>> get_tensors_from_onnx(onnx::ModelProto model);
#endif

using namespace std;

#ifdef cPROTO
enum ONNX_LAYERS
{
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
  MAT_MUL,          // OPSET: 13, 9, 1 (Only for MatMul+Add Dense layer)
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
  //POW,            // OPSET: 13, 12, 7 (TODO: Implement LPow)
};

void log_string(string log, int actual_log_level, int string_log_level)
{
  if (actual_log_level <= string_log_level)
  {
    cout << "[ONNX::LOG] " << log << endl;
  }
}

int verbose = 0;
// Gets the initializers from the onnx layer graph
vector<onnx::TensorProto> get_initializers(onnx::GraphProto graph)
{
  vector<onnx::TensorProto> initializers;
  for (int i = 0; i < graph.initializer_size(); i++)
    initializers.push_back(graph.initializer(i));
  return initializers;
}

// Creates a map containing the name of the node as a key, and the value is a vector containing the nodes that have this node as a input.
map<string, vector<onnx::NodeProto *>> initialize_input_node_map(vector<onnx::NodeProto> &nodes)
{
  map<string, vector<onnx::NodeProto *>> input_node_map;
  for (int j = 0; j < nodes.size(); j++)
  {
    onnx::NodeProto node = nodes[j];
    for (int i = 0; i < node.input_size(); i++)
    {
      string input_name = node.input(i);
      if (input_node_map.count(input_name) == 0)
      {
        vector<onnx::NodeProto *> v;
        input_node_map[input_name] = v;
      }
      vector<onnx::NodeProto *> v = input_node_map[input_name];
      v.push_back(&(nodes[j]));
      input_node_map[input_name] = v;
    }
  }
  return input_node_map;
}

// Creates a map with the name of the constant node as key and the node container from onnx as value.
map<string, onnx::NodeProto *> initialize_constant_nodes(vector<onnx::NodeProto> &nodes)
{
  map<string, onnx::NodeProto *> constant_node_map;
  for (int i = 0; i < nodes.size(); i++)
  {
    if (!nodes[i].op_type().compare("Constant"))
    {
      for (int j = 0; j < nodes[i].output_size(); j++)
      {
        constant_node_map[nodes[i].output(j)] = &(nodes[i]);
      }
    }
  }
  return constant_node_map;
}

// Initializes the queue that we are going to explore to construct the eddl graph of the net. The first values are the input layers.
queue<onnx::NodeProto *> process_inputs(vector<Layer *> *inputs, vector<onnx::ValueInfoProto> *inputs_onnx, map<string, vector<onnx::NodeProto *>> *input_node_map, map<string, Layer *> *output_node_map)
{
  queue<onnx::NodeProto *> nodeQueue;
  for (int i = 0; i < inputs->size(); i++)
  {
    onnx::ValueInfoProto nameContainer = (*inputs_onnx)[i];
    string name = nameContainer.name();
    (*output_node_map)[name] = (*inputs)[i];
    vector<onnx::NodeProto *> v = (*input_node_map)[name];
    for (onnx::NodeProto *layer : (*input_node_map)[name])
    {
      nodeQueue.push(layer);
    }
  }
  return nodeQueue;
}

// Creates a map where the key is the onnx name for the layer type and the value is the constant value in the enumeration for onnx layer type.
map<string, ONNX_LAYERS> create_enum_map()
{
  map<string, ONNX_LAYERS> map_layers;
  map_layers["BatchNormalization"] = ONNX_LAYERS::BATCHNORM;
  map_layers["Conv"] = ONNX_LAYERS::CONV;
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
  map_layers["Div"] = ONNX_LAYERS::DIV;
  map_layers["Exp"] = ONNX_LAYERS::EXP;
  map_layers["Log"] = ONNX_LAYERS::LOG;
  map_layers["Mul"] = ONNX_LAYERS::MUL;
  //map_layers["Pow"] = ONNX_LAYERS::POW;
  map_layers["Sqrt"] = ONNX_LAYERS::SQRT;
  map_layers["Sub"] = ONNX_LAYERS::SUB;
  map_layers["ReduceMax"] = ONNX_LAYERS::RMAX;
  map_layers["ReduceMin"] = ONNX_LAYERS::RMIN;
  map_layers["ReduceMean"] = ONNX_LAYERS::RMEAN;
  map_layers["ReduceSum"] = ONNX_LAYERS::RSUM;
  map_layers["ArgMax"] = ONNX_LAYERS::ARGMAX;
  map_layers["Resize"] = ONNX_LAYERS::RESIZE;

  return map_layers;
}

// Converts a raw onnx value tensor and writes it to a vector of that value type.
template <class T>
bool TryConvertingTensorRawValues(const onnx::TensorProto &onnx_tensor, vector<T> &field)
{
  if (!onnx_tensor.has_raw_data())
  {
    return false;
  }
  size_t raw_size = onnx_tensor.raw_data().size();
  if (raw_size % sizeof(T) != 0)
  {
    return false;
  }

  size_t num_elements = raw_size / sizeof(T);
  const void *src_ptr = static_cast<const void *>(onnx_tensor.raw_data().data());
  field.resize(num_elements, 0);
  void *target_ptr = static_cast<void *>(field.data());
  memcpy(target_ptr, src_ptr, raw_size);
  return true;
}

// Parses the values of the onnx tensor to a c++ vector of that type
vector<float> parseTensorValues(onnx::TensorProto t)
{
  int data_type = t.data_type(); // Only works for non raw data for now
  vector<float> values;
  switch (data_type)
  {
  case onnx::TensorProto::UNDEFINED:
    // TODO: Make this
    break;
  case onnx::TensorProto::FLOAT:
    if (t.has_raw_data())
    {
      TryConvertingTensorRawValues(t, values);
    }
    else
    {
      for (int i = 0; i < t.float_data_size(); i++)
      {
        values.push_back(t.float_data(i));
      }
    }
    break;
  case onnx::TensorProto::UINT8:
    for (int i = 0; i < t.int32_data_size(); i++)
    {
      values.push_back(t.int32_data(i));
    }
    break;
  case onnx::TensorProto::INT8:
    for (int i = 0; i < t.int32_data_size(); i++)
    {
      values.push_back(t.int32_data(i));
    }
    break;
  case onnx::TensorProto::UINT16:
    for (int i = 0; i < t.int32_data_size(); i++)
    {
      values.push_back(t.int32_data(i));
    }
    break;
  case onnx::TensorProto::INT16:
    for (int i = 0; i < t.int32_data_size(); i++)
    {
      values.push_back(t.int32_data(i));
    }
    break;
  case onnx::TensorProto::INT32:
    for (int i = 0; i < t.int32_data_size(); i++)
    {
      values.push_back(t.int32_data(i));
    }
    break;
  case onnx::TensorProto::INT64:
    if (t.has_raw_data())
    {
      vector<int64_t> aux_values; // Vector to read the int64 values
      TryConvertingTensorRawValues(t, aux_values);
      for (float i : aux_values)
      { // Cast to float
        values.push_back(i);
      }
    }
    else
    {
      for (int i = 0; i < t.int64_data_size(); i++)
      {
        values.push_back(t.int64_data(i));
      }
    }
    break;
  case onnx::TensorProto::STRING: //TODO: Make this
    break;
  case onnx::TensorProto::BOOL:
    for (int i = 0; i < t.int32_data_size(); i++)
    {
      values.push_back(t.int32_data(i));
    }
    break;
  case onnx::TensorProto::FLOAT16:
    break;
  case onnx::TensorProto::DOUBLE:
    for (int i = 0; i < t.double_data_size(); i++)
    {
      values.push_back(t.double_data(i));
    }
    break;
  case onnx::TensorProto::UINT32:
    for (int i = 0; i < t.uint64_data_size(); i++)
    {
      values.push_back(t.uint64_data(i));
    }
    break;
  case onnx::TensorProto::UINT64:
    for (int i = 0; i < t.uint64_data_size(); i++)
    {
      values.push_back(t.uint64_data(i));
    }
    break;
  case onnx::TensorProto::COMPLEX64:
    //TODO: Make this
    break;
  case onnx::TensorProto::COMPLEX128:
    //TODO: Make this
    break;
  case onnx::TensorProto::BFLOAT16:
    //TODO: Make this
    break;

  default:
    cerr << "Vector type not recognized" << endl;
    break;
  }
  return values;
}

// Creates two maps. Both have the name of the initializer node as key. The values are a vector containing the weights and a vector containing the shape of the vector, respectively.
void get_initializers_maps(vector<onnx::TensorProto> tensors, map<string, vector<float>> &values_map, map<string, vector<int>> &dims_map)
{

  for (onnx::TensorProto tensor : tensors)
  {
    vector<int> dims;
    for (int i = 0; i < tensor.dims_size(); i++)
    {
      dims.push_back(tensor.dims(i));
    }
    string tensorName = tensor.name();
    vector<float> values = parseTensorValues(tensor);
    values_map[tensorName] = values;
    dims_map[tensorName] = dims;
  }
  return;
}

// Parses one TensorProto pointer (Input or output) to eddl Tensor pointer
vector<int> parse_IO_tensor(onnx::TypeProto::Tensor tensor, bool recurrent_net)
{
  onnx::TensorShapeProto tensorShape = tensor.shape();
  vector<int> shape;
  shape.push_back(1); // Set batch to 1
  int start_index = 1;
  if (recurrent_net)
    start_index = 2; // Avoid sequence dim

  for (int i = start_index; i < tensorShape.dim_size(); i++)
  {
    shape.push_back(tensorShape.dim(i).dim_value());
  }

  return shape;
}

// Converts one vector of TensorProto pointers (Input or output)
// to one vector of eddl Tensor pointers.
vector<Layer *> parse_IO_tensors(vector<onnx::ValueInfoProto> io_onnx, vector<int> input_shape, int mem, bool recurrent_net)
{
  vector<Layer *> io;
  onnx::TypeProto::Tensor tensor;
  int dev = DEV_CPU;
  bool reshape_input = !input_shape.empty();

  if (reshape_input) 
  {
    if (io_onnx.size() > 1)
      msg("The imported model has more than 1 input layer and the shape provided to reshape the model can only be applied"
           "if the model has only one input layer.", "ONNX::ImportNet");
    else
    {
      tensor = io_onnx[0].type().tensor_type();
      string name = io_onnx[0].name();
      input_shape.insert(input_shape.begin(), 1); // Add the batch dimension
      io.push_back(new LInput(new Tensor(input_shape), name, dev, mem));
    }
  } 
  else 
  {
    for (onnx::ValueInfoProto infoProto : io_onnx)
    {
      tensor = infoProto.type().tensor_type();
      string name = infoProto.name();
      io.push_back(new LInput(new Tensor(parse_IO_tensor(tensor, recurrent_net)), name, dev, mem));
    }
  }

  return io;
}

// Returns a vector with the input names of the net
vector<onnx::ValueInfoProto> get_inputs(onnx::GraphProto graph)
{
  set<string> input_names;
  set<string> initializer_names; // We make the substraction of both sets to find the true inputs

  for (int i = 0; i < graph.input_size(); i++)
  { // Construct set of input names
    input_names.insert(graph.input(i).name());
  }

  vector<onnx::TensorProto> initializers = get_initializers(graph);
  for (int i = 0; i < initializers.size(); i++)
  { // Construct set of initializer names
    if (initializers[i].has_name())
      initializer_names.insert(initializers[i].name());
  }

  vector<string> true_inputs(100);
  std::set_difference(input_names.begin(), input_names.end(), initializer_names.begin(), initializer_names.end(), true_inputs.begin());
  true_inputs.shrink_to_fit();

  vector<onnx::ValueInfoProto> returnVector; // This is for returning the tensor, but we need the names
  for (int i = 0; i < graph.input_size(); i++)
  {
    onnx::ValueInfoProto auxInfoProto = graph.input(i);
    if (count(true_inputs.begin(), true_inputs.end(), auxInfoProto.name()))
    {                                       // If the name is a true input
      returnVector.push_back(auxInfoProto); // Push it to input vector
    }
  }
  return returnVector;
}

// Returns a vector containing the output names of the net
vector<string> get_outputs(onnx::GraphProto graph)
{
  vector<string> output_names;

  for (int i = 0; i < graph.output_size(); i++)
  { // Construct set of output names
    output_names.push_back(graph.output(i).name());
  }
  return output_names;
}

// Returns a vector containing all nodes of the graph in onnx containers.
vector<onnx::NodeProto> get_graph_nodes(onnx::GraphProto graph)
{
  vector<onnx::NodeProto> nodes;
  for (int i = 0; i < graph.node_size(); i++)
  {
    onnx::NodeProto node = graph.node(i);
    nodes.push_back(node);
  }

  return nodes;
}

// Imports a net stored in a onnx file
Net *import_net_from_onnx_file(std::string path, int mem, int log_level)
{
  // Check if the path exists
  if (!pathExists(path))
  {
    msg("The specified path does not exist: " + path, "ONNX::ImportNet");
  }

  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  onnx::ModelProto model;
  {
    // Read the existing net.
    fstream input(path, ios::in | ios::binary);
    if (!model.ParseFromIstream(&input))
    {
      cerr << "Failed to parse model." << endl;
      //return;
    }
    input.close();
  }
  return build_net_onnx(model, {}, mem, log_level);
}

// Imports a net stored in a onnx file
Net *import_net_from_onnx_file(std::string path, vector<int> input_shape, int mem, int log_level)
{
  // Check if the path exists
  if (!pathExists(path))
  {
    msg("The specified path does not exist: " + path, "ONNX::ImportNet");
  }

  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  onnx::ModelProto model;
  {
    // Read the existing net.
    fstream input(path, ios::in | ios::binary);
    if (!model.ParseFromIstream(&input))
    {
      cerr << "Failed to parse model." << endl;
      //return;
    }
    input.close();
  }
  return build_net_onnx(model, input_shape, mem, log_level);
}

// Imports a net from a pointer passed as argument
Net *import_net_from_onnx_pointer(void *serialized_model, size_t size, int mem)
{
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  onnx::ModelProto model;
  {
    if (!model.ParseFromArray(serialized_model, size))
    {
      cerr << "Failed to parse model." << endl;
    }
    else if (verbose >= 2)
      cout << "Model parsed succesfuly" << endl;
  }
  return build_net_onnx(model, {}, mem, LOG_LEVEL::INFO);
}

// Imports a net from a c++ string passed as argument.
Net *import_net_from_onnx_string(string *model_string, int mem)
{
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  onnx::ModelProto model;
  {
    if (!model.ParseFromString(*model_string))
    {
      cerr << "Failed to parse model." << endl;
    }
    else if (verbose >= 2)
      cout << "Model parsed succesfuly" << endl;
  }
  return build_net_onnx(model, {}, mem, LOG_LEVEL::INFO);
}

Layer *get_model_input_layer(Layer *l)
{
  if (LInput *lin = dynamic_cast<LInput *>(l))
    return lin;

  for (Layer *parent : l->parent)
  {
    Layer *auxl = get_model_input_layer(parent);
    if (auxl != nullptr)
      return auxl;
  }

  return nullptr;
}

bool node_is_recurrent(onnx::NodeProto *node, map<string, ONNX_LAYERS> &map_layers)
{
  ONNX_LAYERS layer_type = map_layers[node->op_type()];
  if (layer_type == ONNX_LAYERS::LSTM || 
      layer_type == ONNX_LAYERS::GRU  ||
      layer_type == ONNX_LAYERS::RNN)
    return true;

  return false;
}

void set_decoder(Layer *l)
{
  l->isdecoder = true;
  for (int i = 0; i < l->parent.size(); i++)
    set_decoder(l->parent[i]);
}

bool node_is_decoder(onnx::NodeProto *node, map<string, vector<onnx::NodeProto *>> &input_node_map)
{
  /* Steps to check if the node is decoder:
   *    1. The child nodes take the input from the output Y of the node. So the node is generating a sequence.
   *
   *    2. The childs are not recurrent. Because in the case of an encoder with stacked recurrent layers, 
   *       the output of the layers is also the sequence (except the last one for encoding). And in that case
   *       we don't want to create a decoder layer, we just connect the stacked recurrent layers.
   */
  bool is_decoder = false;

  // 1. Check that the childs are getting the output from Y (the sequence with the output for each timestep)
  string sequence_output_name = node->output(0);
  vector<onnx::NodeProto *> childs = input_node_map[sequence_output_name];
  if (childs.size() > 0)
  {
    is_decoder = true;
  }
  else
  {
    return false;
  }

  // 2. Check that childs are not recurrent
  queue<onnx::NodeProto *> forward_nodes_queue;
  for (onnx::NodeProto *node : input_node_map[sequence_output_name])
    forward_nodes_queue.push(node);
  map<string, ONNX_LAYERS> map_layers = create_enum_map();
  onnx::NodeProto *child;
  while (!forward_nodes_queue.empty())
  {
    child = forward_nodes_queue.front();
    // Add childs of each output of the current child node
    for (int i = 0; i < child->output_size(); ++i)
    {
      for (onnx::NodeProto *node : input_node_map[child->output(i)])
      {
        forward_nodes_queue.push(node);
      }
    }
    // Check if the child is a recurrent operator
    if (node_is_recurrent(child, map_layers))
    {
      is_decoder = false;
      break;
    }
    // Remove the current child from the queue
    forward_nodes_queue.pop();
  }
  return is_decoder;
}

bool check_recurrent_nodes(vector<onnx::NodeProto> nodes)
{
  map<string, ONNX_LAYERS> map_layers = create_enum_map();
  for (int i = 0; i < nodes.size(); i++)
  { // Check if any node is recurrent
    if (node_is_recurrent(&nodes[i], map_layers))
      return true;
  }
  return false;
}

// Builds a eddl Net from an instance of the onnx container for model
Net *build_net_onnx(onnx::ModelProto model, vector<int> input_shape, int mem, int log_level)
{

  long long int ir_version = model.ir_version();
  // We omit the OperatorSetIdProto, since it doesn't do anything for EDDL
  log_string("Ir_version = " + to_string(ir_version), log_level, LOG_LEVEL::DEBUG);
  for (int i = 0; i < model.opset_import_size(); i++)
  {
    log_string("Operator domain  = " + model.opset_import(i).domain(), log_level, LOG_LEVEL::DEBUG);
    log_string("Operator version  = " + to_string(model.opset_import(i).version()), log_level, LOG_LEVEL::DEBUG);
  }
  log_string("Producer_name: " + model.producer_name(), log_level, LOG_LEVEL::DEBUG);
  log_string("Producer_version: " + model.producer_version(), log_level, LOG_LEVEL::DEBUG);
  log_string("Domain: " + model.domain(), log_level, LOG_LEVEL::DEBUG);
  log_string("Model_version: " + to_string(model.model_version()), log_level, LOG_LEVEL::DEBUG);
  int counter = 0;
  onnx::GraphProto graph = model.graph(); // Get the graph of the model.

  // Model needs input in the constructor, so we start with that.
  vector<onnx::ValueInfoProto> inputs_onnx = get_inputs(graph); // Get the inputs
  vector<onnx::NodeProto> nodes = get_graph_nodes(graph);
  bool recurrent_net = check_recurrent_nodes(nodes);
  if (recurrent_net)
    log_string("The net is recurrent", log_level, LOG_LEVEL::DEBUG);

  vector<Layer *> inputs = parse_IO_tensors(inputs_onnx, input_shape, mem, recurrent_net); // Parse ONNX inputs to EDDL inputs

  vector<onnx::TensorProto> initializers = get_initializers(graph); // Retrieves the initializers from the graph.
  // The weights for the layers can be found in the initializers.

  map<string, vector<float>> map_init_values; // Key: Input Name - Value: Weights
  map<string, vector<int>> map_init_dims;     // Key: Input Name - Value: Dims
  // Initialize the maps
  get_initializers_maps(initializers, map_init_values, map_init_dims);

  /*
   * The methodology is the following:
   * We create three maps:
   * map <string input, vector<onnx::NodeProto *> > input_node_map: The input will point towards the nodes that have this input
   * map <string output, Layer* parent > output_node_map: To know from which (parent) node comes each input (The input is the output of the parent node)
   * map <string input/output, bool> input_active_map: The outputs will be put inside a bool, where we will see if it is active or not.
   * 
   * The algorithm is the following:
   * 	1-We check the inputs of each node.
   *    For each input we insert the input string as a key and the Node(s) that use it as a value in the input_node_map.
   *    If that input is already on use, we will append the node to the existing vector
   * 	2-We check the outputs of each node. //NOT REQUIRED
   *    For each output we insert the output string as a key and the Node that generates it as a value in the outputs_map //NOT REQUIRED
   * 	3-When we add an input/output to the map, we also add it to the input_active_map as key, and the value will be false by default. If it is already there, we do nothing. //NOT REQUIRED
   * 	4-Once we have constructed these maps, we create an empty queue of NodeProto
   * 	5-For the input nodes in the graph, we create the EDDL layer and add the nodes that use its output(s) to the queue
   * 	6-While the queue is not empty:
   *    For each node:
   *    6.1-Check if all its inputs (not the ones in 'initializers') exist in output_node_map
   *    6.2-If they are not  --> continue
   *    6.3-Else:
   *        Create the EDDL layer
   *        Add the nodes that use its outputs to the queue
   * To create each EDDL layer:
   * 	 1-Get its parent(s) by accessing to output_node_map using this node's input(s) as key(s)
   * 	 2-Get its weights from 'initializers'
   * 	 3-Create layer
   * 
   *   We need another map for storing the constant nodes, who are always active
   *   We design this map as map<string, onnx::NodeProto> and called constant_node_map
   */

  map<string, Layer *> output_node_map;

  // 1 2 and 3: Initialize maps
  map<string, vector<onnx::NodeProto *>> input_node_map = initialize_input_node_map(nodes);

  // 4 and 5: Create queue of NodeProto
  map<string, onnx::NodeProto *> constant_node_map = initialize_constant_nodes(nodes);

  queue<onnx::NodeProto *> nodeQueue = process_inputs(&inputs, &inputs_onnx, &input_node_map, &output_node_map);

  // Check if any node only has initializers and constant nodes as parameters, so we can process it right away
  for (int i = 0; i < nodes.size(); i++)
  {
    onnx::NodeProto *node = &nodes[i];
    bool avaliable = true;
    for (int j = 0; j < node->input_size(); j++)
    {
      string input = node->input(j);
      if (map_init_values.count(input))
      {
        continue;
      }
      if (constant_node_map.count(input))
      {
        continue;
      }
      avaliable = false;
      break;
    }
    if (avaliable)
    {
      log_string("Node " + node->name() + " is avaliable, since only has initializers and constant nodes as parameters.", log_level, LOG_LEVEL::DEBUG);
      if (node->op_type() == "Constant")
        continue;
      nodeQueue.push(node);
    }
  }

  /*
   * In the case of models with recurrent decoders, we have to track the input layers of the decoder layers
   * and avoid adding them to the input layers of the model
   */
  vector<string> inputs2remove = {};

  map<string, ONNX_LAYERS> map_layers = create_enum_map();
  // 6 - While the queue is not empty:
  while (!nodeQueue.empty())
  {
    counter = 0;
    onnx::NodeProto *node = nodeQueue.front();
    log_string("Next node: " + node->name(), log_level, LOG_LEVEL::DEBUG);

    // Look for inputs with empty ("") names that some libraries create, and delete them
    auto *inputs_list = node->mutable_input();
    for (auto i = inputs_list->begin(); i != inputs_list->end(); i++)
      if ((*i).empty())
        i = --inputs_list->erase(i);

    // 6.1: Check all inputs are avaliable
    bool avaliable = true;

    for (int i = 0; i < node->input_size(); i++)
    {
      string input = node->input(i);
      if (map_init_values.count(input))
      {
        continue;
      }
      if (output_node_map.count(input))
      {
        continue;
      }
      if (constant_node_map.count(input))
      {
        continue;
      }
      avaliable = false;
      log_string("Node " + node->name() + " is not avaliable yet. Missing input: " + input, log_level, LOG_LEVEL::DEBUG);
      break;
    }
    string output_name = node->output(0);
    if (output_node_map.count(output_name))
    {
      nodeQueue.pop();
      log_string("Node " + node->name() + " was already created.", log_level, LOG_LEVEL::DEBUG);
      continue; // This means this node was already created
    }

    // 6.2
    if (!avaliable)
    {
      nodeQueue.pop();
      continue;
    }

    // 6.3
    // Lets assume the maximum quantity of layer inputs a layer can have is 2
    // vector<Layer *> parents; //Not required because inputs are ordered.
    vector<float> weights;
    vector<int> dims;
    // We have to know which layer to create. For it, I suggest
    // a map <String-Enumeration> for creating a switch, where
    // we call the constructor of that layer
    string layer_type_name = node->op_type();
    log_string("Node " + node->name() + " has operation type = " + layer_type_name, log_level, LOG_LEVEL::DEBUG);
    ONNX_LAYERS layer_type = map_layers[layer_type_name];
    string name = node->name();
    int dev = DEV_CPU;
    Layer *actual_layer;

    switch (layer_type)
    { // Every case should create the corresponding layer and asign it to "actual_layer" variable
    case ONNX_LAYERS::BATCHNORM:
    {
      double epsilon = 1e-05; // Default value
      double momentum = 0.9;  // Default value
      for (int j = 0; j < node->attribute_size(); j++)
      { // Set the attributes
        onnx::AttributeProto attribute = node->attribute(j);
        string attr_name = attribute.name();
        if (!attr_name.compare("epsilon"))
          epsilon = attribute.f();
        if (!attr_name.compare("momentum"))
          momentum = attribute.f();
      }

      string parent_name = node->input(0); // Get parent
      Layer *parent = output_node_map[parent_name];
      vector<int> parent_shape = parent->output->shape;

      string scale_name = node->input(1); // Scale parameter
      vector<float> *scale_weights = &(map_init_values[scale_name]);
      vector<int> scale_dims = map_init_dims[scale_name];

      string bias_name = node->input(2); // Bias parameter
      vector<float> *bias_weights = &(map_init_values[bias_name]);
      vector<int> bias_dims = map_init_dims[bias_name];

      string mean_name = node->input(3); // Get weights and dims
      vector<float> *mean_weights = &(map_init_values[mean_name]);
      vector<int> mean_dims = map_init_dims[mean_name];

      string variance_name = node->input(4); // Get weights and dims
      vector<float> *variance_weights = &(map_init_values[variance_name]);
      vector<int> variance_dims = map_init_dims[variance_name];

      string name = node->name();

      bool affine = true; // The ONNX operator description does not have an "affine" attribute. We have to assume that this will be allways true.

      actual_layer = new LBatchNorm(parent, momentum, epsilon, affine, name, dev, mem);

      Tensor *scale_tensor = new Tensor(scale_dims, nullptr, dev);
      COPY_FROM_VECTOR_PTR_TO_TENSOR(scale_weights, scale_tensor);
      Tensor::copy(scale_tensor, ((LBatchNorm *)(actual_layer))->bn_g);
      delete scale_tensor;

      Tensor *bias_tensor = new Tensor(bias_dims, nullptr, dev);
      COPY_FROM_VECTOR_PTR_TO_TENSOR(bias_weights, bias_tensor);
      Tensor::copy(bias_tensor, ((LBatchNorm *)(actual_layer))->bn_b);
      delete bias_tensor;

      Tensor *mean_tensor = new Tensor(mean_dims, nullptr, dev);
      COPY_FROM_VECTOR_PTR_TO_TENSOR(mean_weights, mean_tensor);
      Tensor::copy(mean_tensor, ((LBatchNorm *)(actual_layer))->mean);
      delete mean_tensor;

      Tensor *variance_tensor = new Tensor(variance_dims, nullptr, dev);
      COPY_FROM_VECTOR_PTR_TO_TENSOR(variance_weights, variance_tensor);
      Tensor::copy(variance_tensor, ((LBatchNorm *)(actual_layer))->variance);
      delete variance_tensor;
    }
    break;

    case ONNX_LAYERS::CONV:
    {
      int filters;
      vector<int> kernel_shape;
      vector<int> strides;
      vector<int> pads = {};
      string auto_pad_option = "custom";
      vector<float> *bias;
      bool use_bias = node->input_size() > 2;
      bool conv1d = false;
      int groups = 1;
      vector<int> dilation_rate = {1, 1};

      for (int j = 0; j < node->attribute_size(); j++)
      { // Set the attributes
        onnx::AttributeProto attribute = node->attribute(j);
        string attr_name = attribute.name();
        if (!attr_name.compare("auto_pad"))
        {
          if (!attribute.s().compare("NOTSET"))
            auto_pad_option = "custom";
          else if (!attribute.s().compare("VALID"))
            auto_pad_option = "valid";
          else if (!attribute.s().compare("SAME_UPPER"))
            auto_pad_option = "same";
        }
        //else if (!attr_name.compare("dilations")) { It isn't implemented in eddl
        //}
        //else if (!attr_name.compare("group")) { It isn't implemented in eddl
        //}
        else if (!attr_name.compare("kernel_shape"))
        {
          for (int h = 0; h < attribute.ints_size(); h++)
          {
            kernel_shape.push_back(attribute.ints(h));
          }
          if (attribute.ints_size() == 1)
          { // If is conv1D, we make the equivalent in conv2D
            conv1d = true;
          }
        }
        else if (!attr_name.compare("pads"))
        {
          for (int h = 0; h < attribute.ints_size(); h++)
          {
            pads.push_back(attribute.ints(h));
          }
          if (attribute.ints_size() == 4)
            swap(pads[1], pads[2]);
        }
        else if (!attr_name.compare("strides"))
        {
          for (int h = 0; h < attribute.ints_size(); h++)
          {
            strides.push_back(attribute.ints(h));
          }
          if (attribute.ints_size() == 1)
          { // If is conv1D, we make the equivalent in conv2D
            conv1d = true;
          }
        }
      }

      string parent_name = node->input(0); // Get parent
      Layer *parent = output_node_map[parent_name];
      vector<int> parent_shape = parent->output->shape;

      string weights_name = node->input(1); // Get weights and dims
      vector<float> *weights = &(map_init_values[weights_name]);
      vector<int> dims = map_init_dims[weights_name];

      if (parent_shape.size() == 3)
      {
        conv1d = true;
      }

      if (conv1d)
      {
        strides.push_back(1);
        kernel_shape.push_back(1);
        dims.push_back(1);
        pads.push_back(0);
        pads.push_back(0);
      }

      filters = dims[0];
      string name = node->name();
      ConvolDescriptor *cd = new ConvolDescriptor(filters, 
                                                  kernel_shape, 
                                                  strides, 
                                                  auto_pad_option,
                                                  pads, 
                                                  groups, 
                                                  dilation_rate, 
                                                  use_bias, 
                                                  mem);

      if (conv1d)
        actual_layer = new LConv1D(parent, cd, name, dev, mem);
      else
        actual_layer = new LConv(parent, cd, name, dev, mem);

      if (use_bias)
      {
        string bias_name = node->input(2);
        bias = &(map_init_values[bias_name]);
        vector<int> bias_shape;
        bias_shape.push_back(bias->size());
        Tensor *bias_tensor = new Tensor(bias_shape, nullptr, dev);
        COPY_FROM_VECTOR_PTR_TO_TENSOR(bias, bias_tensor);
        Tensor::copy(bias_tensor, cd->bias);
        delete bias_tensor;
      }
      Tensor *weights_tensor = new Tensor(dims, nullptr, dev);
      COPY_FROM_VECTOR_PTR_TO_TENSOR(weights, weights_tensor);
      Tensor::copy(weights_tensor, cd->K);
      delete weights_tensor;
    }
    break;

    case ONNX_LAYERS::DENSE:
    {
      log_string("Dense detected", log_level, LOG_LEVEL::DEBUG);
      int ndim;
      bool use_bias = false;
      float alpha;
      float beta;
      int transA = 0;
      int transB = 0;
      vector<int> bias_dims;
      vector<float> *bias;
      for (int j = 0; j < node->attribute_size(); j++)
      {
        onnx::AttributeProto attribute = node->attribute(j);
        string attr_name = attribute.name();
        if (!attr_name.compare("alpha"))
        {
          alpha = attribute.f();
        }
        else if (!attr_name.compare("beta"))
        {
          beta = attribute.f();
        }
        else if (!attr_name.compare("transA"))
        {
          transA = attribute.i();
        }
        else if (!attr_name.compare("transB"))
        {
          transB = attribute.i();
        }
      }

      string parent_name;
      Layer *parent;
      string weights_name;
      string bias_name;
      vector<float> *weights;
      vector<int> dims;

      for (int i = 0; i < 2; i++)
      {
        string input = node->input(i);
        if (!map_init_values.count(input))
        { // parent
          parent_name = node->input(0);
          parent = output_node_map[input];
        }
        else
        { // weights
          weights_name = node->input(i);
          weights = &(map_init_values[input]);
          dims = map_init_dims[input];
          ndim = dims.size();
        }
      }
      use_bias = node->input_size() > 2;
      int neuronas = 0;
      if (transB)
      {
        neuronas = dims[0];
      }
      else
        neuronas = dims[1];
      string name = node->name();
      Tensor *input_size = parent->output;
      LDense *dense = new LDense(parent, neuronas, use_bias, name, dev, mem);

      Tensor *weights_tensor = new Tensor(dims, nullptr, dev);
      COPY_FROM_VECTOR_PTR_TO_TENSOR(weights, weights_tensor);

      if (transB)
        weights_tensor->permute_({1, 0});
      Tensor::copy(weights_tensor, dense->W);
      delete weights_tensor;
      if (use_bias)
      {
        bias_name = node->input(2);
        bias = &(map_init_values[bias_name]);
        bias_dims = map_init_dims[bias_name];
        Tensor *bias_tensor = new Tensor(bias_dims, nullptr, dev);
        COPY_FROM_VECTOR_PTR_TO_TENSOR(bias, bias_tensor);
        Tensor::copy(bias_tensor, dense->bias);
        delete bias_tensor;
      }
      actual_layer = dense;
    }
    break;

    case ONNX_LAYERS::UPSAMPLING:
    {
      string interpolation_mode;
      float batch_scale;
      float channel_scale;
      float height_scale;
      float width_scale;
      for (int j = 0; j < node->attribute_size(); j++)
      { // Set the attributes
        onnx::AttributeProto attribute = node->attribute(j);
        string attr_name = attribute.name();
        if (!attr_name.compare("mode"))
          interpolation_mode = attribute.s();
      }

      string parent_name = node->input(0); // Get parent
      Layer *parent = output_node_map[parent_name];
      vector<int> parent_shape = parent->output->shape;

      string scales_name = node->input(1); // Get scales and dims
      vector<float> *scales = &(map_init_values[scales_name]);
      vector<int> scales_dims = map_init_dims[scales_name];

      if (scales_dims[0] != 4)
      {
        cerr << "Dimensions of upsampling layer in onnx are wrong" << endl;
      }
      batch_scale = scales->at(0);
      channel_scale = scales->at(1);
      height_scale = scales->at(2);
      width_scale = scales->at(3);

      string name = node->name();
      vector<int> size_vector;
      size_vector.push_back((int)height_scale);
      size_vector.push_back((int)width_scale);
      actual_layer = new LUpSampling(parent, size_vector, interpolation_mode, name, dev, mem);
    }
    break;

    case ONNX_LAYERS::DROP:
    {
      float ratio = 0.5;
      for (int j = 0; j < node->attribute_size(); j++)
      { // Set the attributes
        onnx::AttributeProto attribute = node->attribute(j);
        string attr_name = attribute.name();
        if (!attr_name.compare("ratio"))
          ratio = attribute.f();
      }

      string parent_name = node->input(0); // Get parent
      Layer *parent = output_node_map[parent_name];
      vector<int> parent_shape = parent->output->shape;

      string name = node->name();
      actual_layer = new LDropout(parent, ratio, true, name, dev, mem);
    }
    break;

    case ONNX_LAYERS::AVGPOOL:
    {
      int filters;
      vector<int> kernel_shape;
      vector<int> strides;
      vector<int> pads(4, 0); // Default value. 4 zeros
      bool explicit_padding = false;
      int ceil_mode = 0;
      int count_include_pad = 0;
      vector<int> dilations;
      int storage_order = 0;

      for (int j = 0; j < node->attribute_size(); j++)
      { // Set the attributes
        onnx::AttributeProto attribute = node->attribute(j);
        string attr_name = attribute.name();
        if (!attr_name.compare("auto_pad"))
        {
          if (!attribute.s().compare("NOTSET"))
            continue;
          // if(!attribute.s().compare("VALID")) explicit_padding=false;
        }
        //else if (!attr_name.compare("ceil_mode")) { Not in EDDL
        //}
        //else if (!attr_name.compare("count_include_pad")) { Not in EDDL
        //}
        else if (!attr_name.compare("kernel_shape"))
        {
          for (int h = 0; h < attribute.ints_size(); h++)
          {
            kernel_shape.push_back(attribute.ints(h));
          }
        }
        else if (!attr_name.compare("pads"))
        {
          explicit_padding = true;
          for (int h = 0; h < 4; h++)
          {
            pads[h] = attribute.ints(h);
          }
        }
        else if (!attr_name.compare("strides"))
        {
          for (int h = 0; h < attribute.ints_size(); h++)
          {
            strides.push_back(attribute.ints(h));
          }
        }
      }

      string parent_name = node->input(0); // Get parent
      Layer *parent = output_node_map[parent_name];
      vector<int> parent_shape = parent->output->shape;

      string name = node->name();
      actual_layer = new LAveragePool(parent, new PoolDescriptor(kernel_shape, strides, pads), name, dev, mem);
    }
    break;

    case ONNX_LAYERS::GLOBAVGPOOL:
    {
      string parent_name = node->input(0); // Get parent
      Layer *parent = output_node_map[parent_name];
      vector<int> parent_shape = parent->output->shape;

      int h = parent_shape[2];
      int w = parent_shape[3];

      actual_layer = new LAveragePool(parent, {h, w}, {1, 1}, "none", node->name(), dev, mem);
    }
    break;
    case ONNX_LAYERS::RESHAPE:
    {
      string shape_node_name = node->input(1);
      vector<int> shape;
      if (constant_node_map.count(shape_node_name))
      {
        onnx::NodeProto *shape_node = constant_node_map[shape_node_name];
        onnx::AttributeProto shape_attribute = shape_node->attribute(0);
        if (shape_attribute.name().compare("value"))
        {
          // This means an error ocurred, but don't know how to proceed then.
          printf("An error ocurred when reading the shape of reshape\n");
        }
        onnx::TensorProto shape_tensor = shape_attribute.t();
        shape = vf2vi(parseTensorValues(shape_tensor));
      }
      else
      {
        shape = vf2vi(map_init_values[shape_node_name]);
      }
      string name = node->name();
      string parent_name = node->input(0);
      if (output_node_map.count(parent_name))
      {
        shape[0] = 1; // Default batch size = 1
        Layer *parent = output_node_map[parent_name];
        actual_layer = new LReshape(parent, shape, name, dev, mem);
      }
      else if (map_init_values.count(parent_name))
      { // This means it is a parameter and not a layer
        for (int i = 0; i < node->output_size(); i++)
        {
          map_init_values[node->output(i)] = map_init_values[parent_name];
          map_init_dims[node->output(i)] = shape;
          vector<onnx::NodeProto *> child_nodes = input_node_map[node->output(i)];
          for (onnx::NodeProto *child : child_nodes)
          {
            nodeQueue.push(child);
          }
        }
        nodeQueue.pop();
        continue; // We need to do the update of the queue here because we are not creating a true layer
      }
      else
        cerr << "Uknown parent type for reshape" << endl;
    }
    break;

    case ONNX_LAYERS::FLATTEN:
    {
      string parent_name = node->input(0); // Get parent
      Layer *parent = output_node_map[parent_name];

      actual_layer = new LReshape(parent, {1, -1}, name, dev, mem);
    }
    break;

    case ONNX_LAYERS::RELU:
    {
      string parent_name = node->input(0);
      Layer *parent = output_node_map[parent_name];

      string name = node->name();
      vector<float> param;
      actual_layer = new LActivation(parent, "relu", param, name, dev, mem);
    }
    break;

    case ONNX_LAYERS::SIGMOID:
    {
      string parent_name = node->input(0);
      Layer *parent = output_node_map[parent_name];

      string name = node->name();
      vector<float> param;
      actual_layer = new LActivation(parent, "sigmoid", param, name, dev, mem);
    }
    break;

    case ONNX_LAYERS::HARD_SIGMOID:
    {
      string parent_name = node->input(0);
      Layer *parent = output_node_map[parent_name];

      string name = node->name();
      vector<float> param;
      actual_layer = new LActivation(parent, "hard_sigmoid", param, name, dev, mem);
    }
    break;

    case ONNX_LAYERS::TANH:
    {
      string parent_name = node->input(0);
      Layer *parent = output_node_map[parent_name];

      string name = node->name();
      vector<float> param;
      actual_layer = new LActivation(parent, "tanh", param, name, dev, mem);
    }
    break;

    case ONNX_LAYERS::EXPONENTIAL:
    {
      string parent_name = node->input(0);
      Layer *parent = output_node_map[parent_name];

      string name = node->name();
      vector<float> param;
      actual_layer = new LActivation(parent, "exp", param, name, dev, mem);
    }
    break;

    case ONNX_LAYERS::LINEAR:
    {
      float alpha;
      for (int j = 0; j < node->attribute_size(); j++)
      { // Set the attributes
        onnx::AttributeProto attribute = node->attribute(j);
        string attr_name = attribute.name();
        if (!attr_name.compare("alpha"))
          alpha = attribute.f();
      }
      string parent_name = node->input(0);
      Layer *parent = output_node_map[parent_name];

      string name = node->name();
      vector<float> param;
      param.push_back(alpha);
      actual_layer = new LActivation(parent, "linear", param, name, dev, mem);
    }
    break;

    case ONNX_LAYERS::LEAKY_RELU:
    {
      float alpha;
      for (int j = 0; j < node->attribute_size(); j++)
      { // Set the attributes
        onnx::AttributeProto attribute = node->attribute(j);
        string attr_name = attribute.name();
        if (!attr_name.compare("alpha"))
          alpha = attribute.f();
      }
      string parent_name = node->input(0);
      Layer *parent = output_node_map[parent_name];

      string name = node->name();
      vector<float> param;
      param.push_back(alpha);
      actual_layer = new LActivation(parent, "leaky_relu", param, name, dev, mem);
    }
    break;

    case ONNX_LAYERS::THRESHOLDED_RELU:
    {
      float alpha;
      for (int j = 0; j < node->attribute_size(); j++)
      { // Set the attributes
        onnx::AttributeProto attribute = node->attribute(j);
        string attr_name = attribute.name();
        if (!attr_name.compare("alpha"))
          alpha = attribute.f();
      }
      string parent_name = node->input(0);
      Layer *parent = output_node_map[parent_name];

      string name = node->name();
      vector<float> param;
      param.push_back(alpha);
      actual_layer = new LActivation(parent, "thresholded_relu", param, name, dev, mem);
    }
    break;

    case ONNX_LAYERS::ELU:
    {
      float alpha;
      for (int j = 0; j < node->attribute_size(); j++)
      { // Set the attributes
        onnx::AttributeProto attribute = node->attribute(j);
        string attr_name = attribute.name();
        if (!attr_name.compare("alpha"))
          alpha = attribute.f();
      }
      string parent_name = node->input(0);
      Layer *parent = output_node_map[parent_name];

      string name = node->name();
      vector<float> param;
      param.push_back(alpha);
      actual_layer = new LActivation(parent, "elu", param, name, dev, mem);
    }
    break;

    case ONNX_LAYERS::SELU:
    {
      float alpha = 1.67326;
      float gamma = 1.0507;
      for (int j = 0; j < node->attribute_size(); j++)
      { // Set the attributes
        onnx::AttributeProto attribute = node->attribute(j);
        string attr_name = attribute.name();
        if (!attr_name.compare("alpha"))
          alpha = attribute.f();
        if (!attr_name.compare("gamma"))
          gamma = attribute.f();
      }
      string parent_name = node->input(0);
      Layer *parent = output_node_map[parent_name];

      string name = node->name();
      vector<float> param;
      param.push_back(alpha);
      param.push_back(gamma);
      actual_layer = new LActivation(parent, "selu", param, name, dev, mem);
    }
    break;

    case ONNX_LAYERS::SOFTSIGN:
    {
      string parent_name = node->input(0);
      Layer *parent = output_node_map[parent_name];

      string name = node->name();
      vector<float> param;
      actual_layer = new LActivation(parent, "softsign", param, name, dev, mem);
    }
    break;

    case ONNX_LAYERS::SOFTPLUS:
    {
      string parent_name = node->input(0);
      Layer *parent = output_node_map[parent_name];

      string name = node->name();
      vector<float> param;
      actual_layer = new LActivation(parent, "softplus", param, name, dev, mem);
    }
    break;

    case ONNX_LAYERS::SOFTMAX:
    {
      string parent_name = node->input(0);
      Layer *parent = output_node_map[parent_name];
      int axis = 1;

      for (int j = 0; j < node->attribute_size(); j++)
      {
        onnx::AttributeProto attribute = node->attribute(j);
        string attr_name = attribute.name();
        if (!attr_name.compare("axis"))
        {
          axis = attribute.i(); // No use for it on eddl because it is not configurable
        }
        else
          printf("Error with softmax attributes\n");
      }

      string name = node->name();
      int parent_dims = parent->output->getShape().size();
      if (axis < 0)                        // Check if the target axis is a negative index
        axis = parent_dims + axis;         // Get the target axis index
      if (axis < 0 || axis >= parent_dims) // Check for invalid axis index
        msg("The target axis for Softmax is not valid: axis = " + to_string(axis), "ONNX::ImportNet");

      vector<float> param = {(float)axis};
      actual_layer = new LActivation(parent, "softmax", param, name, dev, mem);
    }
    break;

    case ONNX_LAYERS::CONCAT:
    {
      int axis = 1;
      for (int j = 0; j < node->attribute_size(); j++)
      {
        onnx::AttributeProto attribute = node->attribute(j);
        string attr_name = attribute.name();
        if (!attr_name.compare("axis"))
        {
          axis = attribute.i();
        }
        else
          printf("Error with concat attributes. Attribute name is: %s\n", attr_name.c_str());
      }
      vector<Layer *> parents;
      string parent_name;
      for (int j = 0; j < node->input_size(); j++)
      {
        parent_name = node->input(j);
        parents.push_back(output_node_map[parent_name]);
      }
      string name = node->name();
      actual_layer = new LConcat(parents, axis, name, dev, mem);
    }
    break;

    case ONNX_LAYERS::ADD:
    {
      log_string("Add detected", log_level, LOG_LEVEL::DEBUG);
      vector<Layer *> parents;
      string parent_name;
      bool parameter_input = false;
      int index_parameter = -1; // Possible values 0 and 1, we won't expect parameters in an add with more than two parents
      for (int j = 0; j < node->input_size(); j++)
      {
        parent_name = node->input(j);
        if (output_node_map.count(parent_name))
          parents.push_back(output_node_map[parent_name]);
        else if (map_init_values.count(parent_name))
          parameter_input = true;
        index_parameter = j;
      }
      if (parameter_input)
      {
        LConv *conv;
        LDense *dense;
        if ((conv = dynamic_cast<LConv *>(parents[0])))
        {
          ConvolDescriptor *cd = conv->cd;
          string bias_name = node->input(index_parameter);
          vector<float> *bias = &(map_init_values[bias_name]);
          vector<int> bias_shape;
          bias_shape.push_back(bias->size());
          Tensor *bias_tensor = new Tensor(bias_shape, nullptr, dev);
          COPY_FROM_VECTOR_PTR_TO_TENSOR(bias, bias_tensor);
          if (!cd->use_bias)
          {
            cd->use_bias = true; // We need to enable the bias
            Tensor::copy(bias_tensor, cd->bias);
          }
          else
          {
            Tensor *auxiliar_tensor = Tensor::add(cd->bias, bias_tensor);
            Tensor::copy(auxiliar_tensor, cd->bias);
            delete auxiliar_tensor;
          }
          delete bias_tensor;
          actual_layer = conv;
          break;
        }
        else if ((dense = dynamic_cast<LDense *>(parents[0])))
        {
          log_string("Detected a Dense layer as the parent of the Add node.", log_level, LOG_LEVEL::DEBUG);
          string bias_name = node->input(index_parameter);
          vector<float> *bias = &(map_init_values[bias_name]);
          vector<int> bias_dims = map_init_dims[bias_name];
          if (!dense->use_bias)
          {
            log_string("Setting the bias values of the parent Dense to the Add parameters.", log_level, LOG_LEVEL::DEBUG);
            dense->use_bias = true;
            dense->bias = new Tensor(bias_dims, nullptr, dev);
            COPY_FROM_VECTOR_PTR_TO_TENSOR(bias, dense->bias);
            dense->params.push_back(dense->bias);
            dense->gbias = new Tensor(bias_dims, dev);
            dense->gradients.push_back(dense->gbias);
          }
          else
          { // If dense already has a bias, we sum it in top of the bias
            log_string("The parent Dense already has a bias. Adding the parameters of the Add operator to the parent bias.", log_level, LOG_LEVEL::DEBUG);
            Tensor *add_to_bias = new Tensor(bias_dims, nullptr, dev);
            COPY_FROM_VECTOR_PTR_TO_TENSOR(bias, add_to_bias);
            Tensor::add(add_to_bias, dense->bias, dense->bias);
            delete add_to_bias;
          }
          actual_layer = dense;
          break;
        }
        else
          cerr << "Error, add with a parameter input where the other input is not a dense or a convolutional layer" << endl;
      }
      string name = node->name();
      actual_layer = new LAdd(parents, name, dev, mem);
      log_string("Add layer created", log_level, LOG_LEVEL::DEBUG);
    }
    break;

    case ONNX_LAYERS::ABS:
    {
      string parent_name = node->input(0);
      Layer *parent = output_node_map[parent_name];

      string name = node->name();
      actual_layer = new LAbs(parent, name, dev, mem);
    }
    break;

    case ONNX_LAYERS::DIV:
    {
      string first_operator_name = node->input(0);
      Layer *first_operator = output_node_map[first_operator_name];

      string second_operator_name = node->input(1);
      Layer *second_operator = output_node_map[second_operator_name];

      string name = node->name();
      actual_layer = new LDiv(first_operator, second_operator, name, dev, mem);
    }
    break;

    case ONNX_LAYERS::EXP:
    {
      string parent_name = node->input(0);
      Layer *parent = output_node_map[parent_name];

      string name = node->name();
      actual_layer = new LExp(parent, name, dev, mem);
    }
    break;

    case ONNX_LAYERS::LOG:
    {
      string parent_name = node->input(0);
      Layer *parent = output_node_map[parent_name];

      string name = node->name();
      actual_layer = new LLog(parent, name, dev, mem);
    }
    break;

    case ONNX_LAYERS::MUL:
    {
      string first_operator_name = node->input(0);
      Layer *first_operator = output_node_map[first_operator_name];

      string second_operator_name = node->input(1);
      Layer *second_operator = output_node_map[second_operator_name];

      string name = node->name();
      actual_layer = new LMult(first_operator, second_operator, name, dev, mem);
    }
    break;

      /*
    case ONNX_LAYERS::POW:
    {
      string first_operator_name = node->input(0);
      Layer *first_operator = output_node_map[first_operator_name];

      string second_operator_name = node->input(1);
      Layer *second_operator = output_node_map[second_operator_name];

      string name = node->name();
      actual_layer = new LPow(first_operator, second_operator, name, dev, mem);
    }
    break;
    */

    case ONNX_LAYERS::SQRT:
    {
      string parent_name = node->input(0);
      Layer *parent = output_node_map[parent_name];

      string name = node->name();
      actual_layer = new LSqrt(parent, name, dev, mem);
    }
    break;

    case ONNX_LAYERS::SUB:
    {
      string first_operator_name = node->input(0);
      Layer *first_operator = output_node_map[first_operator_name];

      string second_operator_name = node->input(1);
      Layer *second_operator = output_node_map[second_operator_name];

      string name = node->name();
      actual_layer = new LDiff(first_operator, second_operator, name, dev, mem);
    }
    break;

    case ONNX_LAYERS::RMAX:
    {
      vector<int> axes;
      bool keepdims = 1;
      for (int j = 0; j < node->attribute_size(); j++)
      {
        onnx::AttributeProto attribute = node->attribute(j);
        string attr_name = attribute.name();
        if (!attr_name.compare("axes"))
        {
          for (int h = 0; h < attribute.ints_size(); h++)
          {
            axes.push_back(attribute.ints(h));
          }
        }
        else if (!attr_name.compare("keepdims"))
        {
          keepdims = attribute.i();
        }
        else
          printf("Error with ReduceMax attributes. Attribute name is: %s\n", attr_name.c_str());
      }

      string parent_name = node->input(0);
      Layer *parent = output_node_map[parent_name];

      // Prepare the axes for EDDL. Because in EDDL you can't reduce the batch axis (0).
      for (int i = 0; i < axes.size(); ++i)
      {
        if (axes[i] > 0)
          axes[i]--;
        else if (axes[i] == 0)
          msg("You can't reduce the batch axis in Reduce Max layer.", "ONNX::ImportNet");
        else
        {
          // From negative to positive axis value
          int parent_out_rank = parent->getShape().size();
          axes[i] += parent_out_rank;

          axes[i]--;
        }
      }

      string name = node->name();
      actual_layer = new LRMax(parent, axes, keepdims, name, dev, mem);
    }
    break;

    case ONNX_LAYERS::RMIN:
    {
      vector<int> axes;
      bool keepdims = 1;
      for (int j = 0; j < node->attribute_size(); j++)
      {
        onnx::AttributeProto attribute = node->attribute(j);
        string attr_name = attribute.name();
        if (!attr_name.compare("axes"))
        {
          for (int h = 0; h < attribute.ints_size(); h++)
          {
            axes.push_back(attribute.ints(h));
          }
        }
        else if (!attr_name.compare("keepdims"))
        {
          keepdims = attribute.i();
        }
        else
          printf("Error with ReduceMin attributes. Attribute name is: %s\n", attr_name.c_str());
      }

      string parent_name = node->input(0);
      Layer *parent = output_node_map[parent_name];

      // Prepare the axes for EDDL. Because in EDDL you can't reduce the batch axis (0).
      for (int i = 0; i < axes.size(); ++i)
      {
        if (axes[i] > 0)
          axes[i]--;
        else if (axes[i] == 0)
          msg("You can't reduce the batch axis in Reduce Min layer.", "ONNX::ImportNet");
        else
        {
          // From negative to positive axis value
          int parent_out_rank = parent->getShape().size();
          axes[i] += parent_out_rank;

          axes[i]--;
        }
      }

      string name = node->name();
      actual_layer = new LRMin(parent, axes, keepdims, name, dev, mem);
    }
    break;

    case ONNX_LAYERS::RMEAN:
    {
      vector<int> axes;
      bool keepdims = 1;
      for (int j = 0; j < node->attribute_size(); j++)
      {
        onnx::AttributeProto attribute = node->attribute(j);
        string attr_name = attribute.name();
        if (!attr_name.compare("axes"))
        {
          for (int h = 0; h < attribute.ints_size(); h++)
          {
            axes.push_back(attribute.ints(h));
          }
        }
        else if (!attr_name.compare("keepdims"))
        {
          keepdims = attribute.i();
        }
        else
          printf("Error with ReduceMean attributes. Attribute name is: %s\n", attr_name.c_str());
      }

      string parent_name = node->input(0);
      Layer *parent = output_node_map[parent_name];

      // Prepare the axes for EDDL. Because in EDDL you can't reduce the batch axis (0).
      for (int i = 0; i < axes.size(); ++i)
      {
        if (axes[i] > 0)
          axes[i]--;
        else if (axes[i] == 0)
          msg("You can't reduce the batch axis in Reduce Mean layer.", "ONNX::ImportNet");
        else
        {
          // From negative to positive axis value
          int parent_out_rank = parent->getShape().size();
          axes[i] += parent_out_rank;

          axes[i]--;
        }
      }

      string name = node->name();
      actual_layer = new LRMean(parent, axes, keepdims, name, dev, mem);
    }
    break;

    case ONNX_LAYERS::RSUM:
    {
      vector<int> axes;
      bool keepdims = 1;
      for (int j = 0; j < node->attribute_size(); j++)
      {
        onnx::AttributeProto attribute = node->attribute(j);
        string attr_name = attribute.name();
        if (!attr_name.compare("keepdims"))
        {
          keepdims = attribute.i();
        }
        else if (!attr_name.compare("axes"))
        {
          for (int h = 0; h < attribute.ints_size(); h++)
          {
            axes.push_back(attribute.ints(h));
          }
        }
        else
          printf("Error with ReduceSum attributes. Attribute name is: %s\n", attr_name.c_str());
      }

      string parent_name = node->input(0);
      Layer *parent = output_node_map[parent_name];

      // Prepare the axes for EDDL. Because in EDDL you can't reduce the batch axis (0).
      for (int i = 0; i < axes.size(); ++i)
      {
        if (axes[i] > 0)
          axes[i]--;
        else if (axes[i] == 0)
          msg("You can't reduce the batch axis in Reduce Sum layer.", "ONNX::ImportNet");
        else
        {
          // From negative to positive axis value
          int parent_out_rank = parent->getShape().size();
          axes[i] += parent_out_rank;

          axes[i]--;
        }
      }

      string name = node->name();
      actual_layer = new LRSum(parent, axes, keepdims, name, dev, mem);
    }
    break;

    case ONNX_LAYERS::ARGMAX:
    {
      int axis = 1;
      bool keepdims = 1;
      for (int j = 0; j < node->attribute_size(); j++)
      {
        onnx::AttributeProto attribute = node->attribute(j);
        string attr_name = attribute.name();
        if (!attr_name.compare("axis"))
        {
          axis = attribute.i();
        }
        else if (!attr_name.compare("keepdims"))
        {
          keepdims = attribute.i();
        }
        //else if (!attr_name.compare("select_last_index")) {  Not implemented in EDDL
        //}
        else
          printf("Error with Argmax attributes. Attribute name is: %s\n", attr_name.c_str());
      }

      string parent_name = node->input(0);
      Layer *parent = output_node_map[parent_name];

      // Prepare the axis for EDDL. Because in EDDL you can't reduce the batch axis (0).
      if (axis > 0)
        axis--;
      else if (axis == 0)
        msg("You can't select the batch axis in Arg Max layer.", "ONNX::ImportNet");
      else
      {
        // From negative to positive axis value
        int parent_out_rank = parent->getShape().size();
        axis = parent_out_rank + axis;

        axis--;
      }

      string name = node->name();
      actual_layer = new LRArgmax(parent, {axis}, keepdims, name, dev, mem);
    }
    break;

    case ONNX_LAYERS::MAT_MUL:
    {
      vector<Layer *> parents;
      string parent_name;
      bool dense_detected = false;
      int index_parameter = -1;
      for (int j = 0; j < node->input_size(); j++)
      {
        parent_name = node->input(j);
        if (map_init_values.count(parent_name))
        {
          // Dense detected
          if (dense_detected)
          {
            cerr << "MAT_MUL with two parameters" << endl;
          }
          dense_detected = true;
          index_parameter = j;
        }
        else
          parents.push_back(output_node_map[parent_name]);
      }
      if (dense_detected)
      {
        string weights_name = node->input(index_parameter);
        vector<float> *weights = &(map_init_values[weights_name]);
        vector<int> dims = map_init_dims[weights_name];
        int ndim = dims.size();
        int neuronas = dims[1];
        Layer *parent = parents[1 - index_parameter];
        bool use_bias = false;
        LDense *dense = new LDense(parent, neuronas, use_bias, name, dev, mem);
        Tensor *weights_tensor = new Tensor(dims, nullptr, dev);
        COPY_FROM_VECTOR_PTR_TO_TENSOR(weights, weights_tensor);
        Tensor::copy(weights_tensor, dense->W);
        delete weights_tensor;
        actual_layer = dense;
        break;
      }
      string name = node->name();
      actual_layer = new LMatMul(parents, name, dev, mem);
    }
    break;

    case ONNX_LAYERS::MAXPOOL:
    {
      int filters;
      vector<int> kernel_shape;
      vector<int> strides;
      vector<int> pads(4, 0); // Default value. 4 zeros
      bool explicit_padding = false;
      int ceil_mode = 0;
      vector<int> dilations;
      int storage_order = 0;
      bool pool1d = false;

      for (int j = 0; j < node->attribute_size(); j++)
      { // Set the attributes
        onnx::AttributeProto attribute = node->attribute(j);
        string attr_name = attribute.name();
        if (!attr_name.compare("auto_pad"))
        { // We dont know if it is implemented
          if (!attribute.s().compare("NOTSET"))
            continue;
          // if(!attribute.s().compare("VALID")) explicit_padding=false;
        }
        //else if (!attr_name.compare("ceil_mode")) {
        //}
        //else if (!attr_name.compare("dilations")) {
        //}
        else if (!attr_name.compare("kernel_shape"))
        {
          for (int h = 0; h < attribute.ints_size(); h++)
          {
            kernel_shape.push_back(attribute.ints(h));
          }
          if (attribute.ints_size() == 1)
            pool1d = true;
        }
        else if (!attr_name.compare("pads"))
        {
          explicit_padding = true;
          for (int h = 0; h < attribute.ints_size(); h++)
          {
            pads[h] = attribute.ints(h);
          }
        }
        //else if (!attr_name.compare("storage_order")) {
        //}
        else if (!attr_name.compare("strides"))
        {
          for (int h = 0; h < attribute.ints_size(); h++)
          {
            strides.push_back(attribute.ints(h));
          }
          if (attribute.ints_size() == 1)
            pool1d = true;
        }
      }

      string parent_name = node->input(0); // Get parent
      Layer *parent = output_node_map[parent_name];
      vector<int> parent_shape = parent->output->shape;

      string name = node->name();

      if (parent_shape.size() == 3)
        pool1d = true;

      if (pool1d)
      {
        strides.push_back(1);
        kernel_shape.push_back(1);
        actual_layer = new LMaxPool1D(parent, new PoolDescriptor(kernel_shape, strides, pads), name, dev, mem);
      }
      else
      {
        actual_layer = new LMaxPool(parent, new PoolDescriptor(kernel_shape, strides, pads), name, dev, mem);
      }
    }
    break;

    case ONNX_LAYERS::GLOBMAXPOOL:
    {
      string parent_name = node->input(0); // Get parent
      Layer *parent = output_node_map[parent_name];
      vector<int> parent_shape = parent->output->shape;

      int h = parent_shape[2];
      int w = parent_shape[3];

      actual_layer = new LMaxPool(parent, {h, w}, {1, 1}, "none", "gpool", dev, mem);
    }
    break;

    case ONNX_LAYERS::LSTM:
    {
      log_string("LSTM layer detected", log_level, LOG_LEVEL::DEBUG);
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

      actual_layer = lstm;
      log_string("LSTM layer created", log_level, LOG_LEVEL::DEBUG);
    }
    break;

    case ONNX_LAYERS::GRU:
    {
      log_string("GRU layer detected", log_level, LOG_LEVEL::DEBUG);
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
        msg("GRU layer " + name + " doesn't have the number of neurons.", "ONNX::ImportNet");

      string parent_name = node->input(0); // Get parent
      Layer *parent = output_node_map[parent_name];
      vector<int> parent_shape = parent->output->shape;
      vector<Layer *> parents = {parent};

      /*
       * Check if the layer is Decoder by checking if there is not a recurrent layer after this one. To avoid
       * conflicts with the stacked GRU layers that are encoders.
       */
      bool is_decoder = node_is_decoder(node, input_node_map);

      if (is_decoder)
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

      if (is_decoder)
      {
        // Set attribute for unrolling
        gru->isdecoder = true;
        set_decoder(gru->parent[0]);
        // We also have to remove the input layer that feeds the decoder from the input layers of the model
        // First we search the corresponding input layer for the decoder
        Layer *dec_linput = get_model_input_layer(gru);
        if (dec_linput != nullptr)
          inputs2remove.push_back(dec_linput->name);
        else
          msg("Input layer for decoder " + name + " not found", "ONNX::ImportNet");
      }

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

      actual_layer = gru;
      log_string("GRU layer created", log_level, LOG_LEVEL::DEBUG);
    }
    break;

    case ONNX_LAYERS::RNN:
    {
      log_string("RNN layer detected", log_level, LOG_LEVEL::DEBUG);
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

      actual_layer = rnn;
      log_string("RNN layer created", log_level, LOG_LEVEL::DEBUG);
    }
    break;


    case ONNX_LAYERS::IDENTITY:
    {
      log_string("Identity layer detected", log_level, LOG_LEVEL::DEBUG);
      string parent_name;
      parent_name = node->input(0);
      actual_layer = output_node_map[parent_name];
    }
    break;

    case ONNX_LAYERS::CAST:
    {
      log_string("Cast layer detected", log_level, LOG_LEVEL::DEBUG);
      string parent_name;
      parent_name = node->input(0);
      actual_layer = output_node_map[parent_name];
    }
    break;

    case ONNX_LAYERS::GATHER:
    {
      log_string("Gather layer detected", log_level, LOG_LEVEL::DEBUG);
      int axis = 0; // Default value is 0
      for (int j = 0; j < node->attribute_size(); j++)
      { // Set the attributes
        onnx::AttributeProto attribute = node->attribute(j);
        string attr_name = attribute.name();
        if (!attr_name.compare("axis"))
        {
          axis = attribute.i();
        }
      }

      string weights_name = node->input(0); // Get weights and dims
      vector<float> *weights = &(map_init_values[weights_name]);
      vector<int> dims = map_init_dims[weights_name];

      string parent_name = node->input(1); // Get parent
      Layer *parent = output_node_map[parent_name];
      vector<int> parent_shape = parent->output->shape;

      LEmbedding *embedding = new LEmbedding(parent, dims[0], 1 /*parent_shape[1]*/, dims[1], 0, name, dev, mem);
      Tensor *weights_tensor = new Tensor(dims, nullptr, dev);
      COPY_FROM_VECTOR_PTR_TO_TENSOR(weights, weights_tensor);
      Tensor::copy(weights_tensor, embedding->E);

      delete weights_tensor;
      actual_layer = embedding;
    }
    break;

    case ONNX_LAYERS::SQUEEZE:
    {
      log_string("Squeeze layer detected", log_level, LOG_LEVEL::DEBUG);
      vector<int> squeeze_axes;
      for (int j = 0; j < node->attribute_size(); j++)
      { // Set the attributes
        onnx::AttributeProto attribute = node->attribute(j);
        string attr_name = attribute.name();
        if (!attr_name.compare("axes"))
        {
          // Read the axes to squeeze
          for (int h = 0; h < attribute.ints_size(); h++)
          {
            squeeze_axes.push_back(attribute.ints(h));
          }
        }
      }

      string parent_name;
      parent_name = node->input(0);
      Layer *parent = output_node_map[parent_name];
      vector<int> parent_out_shape = parent->output->getShape();
      // Check if we are trying to squeeze the axis 0 or 1 with a recurrent parent node
      //		- In ONNX, the output tensors of a recurrent operator have a dimension with the number of directions
      // 			of the layer (1:onedirectional, 2:bidirectional). In the case of a onedirectional layer the axis must
      // 			be squeezed. But for the creation of the EDDL model we don't need to do this operation, so we skip it.
      for (int i = 0; i < squeeze_axes.size(); ++i)
      {
        if ((squeeze_axes[i] == 0 || squeeze_axes[i] == 1) && parent->isrecurrent)
        {
          log_string("Removing axes " + to_string(squeeze_axes[i]) + " from Squeeze operator. Operation not needed because the parent node is recurrent.",
                     log_level,
                     LOG_LEVEL::DEBUG);
          squeeze_axes.erase(squeeze_axes.begin() + i); // We remove the axis to squeeze
        }
      }

      // Check if all the axes are valid
      bool valid_axes = true;
      for (int ax : squeeze_axes)
      {
        if (ax >= parent_out_shape.size())
        {
          valid_axes = false;
          break;
        }
      }

      if (squeeze_axes.size() == 0)
      {
        log_string("Skiping squeeze operation. No axes to squeeze.", log_level, LOG_LEVEL::DEBUG);
        actual_layer = output_node_map[parent_name];
        break;
      }
      else if (!valid_axes)
      {
        log_string("Skiping squeeze operation. The axes to squeeze are not valid", log_level, LOG_LEVEL::DEBUG);
        actual_layer = output_node_map[parent_name];
        break;
      }
      else
      { // There are axes to squeeze
        vector<int> target_shape;
        bool to_squeeze = false;
        for (int parent_ax = 0; parent_ax < parent_out_shape.size(); ++parent_ax)
        {
          to_squeeze = false;
          for (int target_ax : squeeze_axes)
          {
            if (parent_ax == target_ax)
            {
              if (parent_out_shape[parent_ax] == 1)
              {
                to_squeeze = true;
                break;
              }
              else
              {
                log_string("Trying to squeeze an axis with value different than one. Skiping the operator.", log_level, LOG_LEVEL::WARN);
                actual_layer = output_node_map[parent_name];
                break;
              }
            }
          }
          if (!to_squeeze)
            target_shape.push_back(parent_out_shape[parent_ax]);
        }
        actual_layer = new LReshape(parent, target_shape, name, dev, mem);
        log_string("Squeeze (with Reshape) layer created", log_level, LOG_LEVEL::DEBUG);
      }
    }
    break;

    case ONNX_LAYERS::UNSQUEEZE:
    {
      log_string("Unsqueeze layer detected", log_level, LOG_LEVEL::DEBUG);
      vector<int> unsqueeze_axes;
      for (int j = 0; j < node->attribute_size(); j++)
      { // Set the attributes
        onnx::AttributeProto attribute = node->attribute(j);
        string attr_name = attribute.name();
        if (!attr_name.compare("axes"))
        {
          // Read the axes to squeeze
          for (int h = 0; h < attribute.ints_size(); h++)
          {
            unsqueeze_axes.push_back(attribute.ints(h));
          }
        }
      }

      string parent_name;
      parent_name = node->input(0);
      Layer *parent = output_node_map[parent_name];
      vector<int> parent_out_shape = parent->output->getShape();
      // Check if we are trying to unsqueeze the axis 0 with a recurrent parent node
      // 		- In ONNX, the output of a recurrent encoder operator (the hidden state) has the number of directions
      // 			(1:onedirectional, 2:bidirectional) in the axis 0, so in the case of onedirectional models this
      // 			dimension is squeezed. And in the case of connecting the parent recurrent node to another one,
      // 			a unsqueeze node is usually used to undo the previous squeeze operator. And to build the EDDl model
      //			we don't need to create this ops, so we skip them.
      for (int i = 0; i < unsqueeze_axes.size(); ++i)
      {
        if (unsqueeze_axes[i] == 0 && parent->isrecurrent)
        {
          log_string("Removing 0 axis from Unsqueeze operator. The parent node is recurrent.", log_level, LOG_LEVEL::DEBUG);
          unsqueeze_axes.erase(unsqueeze_axes.begin() + i); // We remove the axis to squeeze
        }
      }

      // Check if all the axes are valid
      bool valid_axes = true;
      for (int ax : unsqueeze_axes)
      {
        if (ax > parent_out_shape.size())
        {
          valid_axes = false;
          break;
        }
      }

      if (unsqueeze_axes.size() == 0)
      {
        log_string("Skiping unsqueeze operation. No axes to unsqueeze.", log_level, LOG_LEVEL::DEBUG);
        actual_layer = output_node_map[parent_name];
        break;
      }
      else if (!valid_axes)
      {
        log_string("Skiping unsqueeze operation. The axes to unsqueeze are not valid", log_level, LOG_LEVEL::DEBUG);
        actual_layer = output_node_map[parent_name];
        break;
      }
      else
      { // There are axes to unsqueeze
        // Sort the axes to unsqueeze
        std::sort(unsqueeze_axes.begin(), unsqueeze_axes.end());
        // Search for duplicates. DUPLICATES ARE NOT ALLOWED
        for (int i = 0; i < unsqueeze_axes.size() - 1; i++)
        {
          if (unsqueeze_axes[i] == unsqueeze_axes[i + 1])
          {
            unsqueeze_axes.erase(unsqueeze_axes.begin() + i);
            log_string("Removing duplicates axis in Unsqueeze operator", log_level, LOG_LEVEL::WARN);
            i--;
          }
        }
        // Insert the new dims
        vector<int> target_shape = parent_out_shape;
        for (int unsq_ax : unsqueeze_axes)
        {
          target_shape.insert(target_shape.begin() + unsq_ax, 1);
        }
        actual_layer = new LReshape(parent, target_shape, name, dev, mem);
        log_string("Unsqueeze (with Reshape) layer created", log_level, LOG_LEVEL::DEBUG);
      }
    }
    break;

    case ONNX_LAYERS::TRANSPOSE:
    {
      log_string("Transpose layer detected", log_level, LOG_LEVEL::DEBUG);
      vector<int> perm;
      for (int j = 0; j < node->attribute_size(); j++)
      { // Set the attributes
        onnx::AttributeProto attribute = node->attribute(j);
        string attr_name = attribute.name();
        if (!attr_name.compare("perm"))
        {
          for (int h = 0; h < attribute.ints_size(); h++)
          {
            perm.push_back(attribute.ints(h));
          }
        }
      }
      log_string("perm vector created", log_level, LOG_LEVEL::DEBUG);

      string parent_name;
      parent_name = node->input(0);
      Layer *parent = output_node_map[parent_name];

      if (recurrent_net)
      {
        if (perm.size() > 1)
        {
          if (perm[0] != 0 || perm[1] != 1)
          {
            log_string("Transpose layers in recurrent nets can not swap batch or sequence dimensions. Skiping Transpose layer...", log_level, LOG_LEVEL::DEBUG);
            actual_layer = parent;
            break;
          }
        }
        else
        {
          log_string("WARNING: Transpose layer with permute indices size of " + to_string(perm.size()) + ". Skiping Transpose layer...", log_level, LOG_LEVEL::WARN);
          actual_layer = parent;
          break;
        }
      }

      // EDDL models have to be batch first in shape
      if (perm[0] != 0)
      {
        msg("The perm vector of the operator " + name + " is not valid (perm[0] != 0). EDDL tensors are batch first.", "ONNX::ImportNet");
      }
      else
      {
        // Remove batch dimension to create the Permute layer
        perm.erase(perm.begin());
        // Fix the perm vector after removing batch dim
        for (int i = 0; i < perm.size(); ++i)
          perm[i]--;
      }

      actual_layer = new LPermute(parent, perm, name, dev, mem);
      log_string("Permute layer created", log_level, LOG_LEVEL::DEBUG);
    }
    break;

    case ONNX_LAYERS::RESIZE:
    {
      bool reshape_out = true;
      string da_mode("nearest");
      float constant = 0.0;

      for (int j = 0; j < node->attribute_size(); j++)
      { // Set the attributes
        onnx::AttributeProto attribute = node->attribute(j);
        string attr_name = attribute.name();
        if (!attr_name.compare("coordinate_transformation_mode"))
        {
          if (attribute.s().compare("asymmetric"))
          {
            msg("In Resize operator, the coordinate transformation mode \"" + attribute.s() + "\" is not supported. It must be \"asymmetric\".", "ONNX::ImportNet");
          }
        }
        if (!attr_name.compare("mode"))
        {
          if (attribute.s().compare("nearest"))
          {
            // ONNX only supports: "nearest", "linear" and "cubic".
            msg("In Resize operator, the mode \"" + attribute.s() + "\" is not supported. It must be \"nearest\".", "ONNX::ImportNet");
          }
        }
      }

      string parent_name = node->input(0);
      Layer *parent = output_node_map[parent_name];
      vector<int> new_shape = parent->getShape();

      string weights_name = node->input(2);
      float *dim_scales = new float [(&(map_init_values[weights_name]))->size()];
      COPY_FROM_VECTOR_PTR_TO_FLOAT_PTR(&(map_init_values[weights_name]), dim_scales);

      // Compute new shape by scaling the parent output shape
      for (int i = 0; i < new_shape.size(); ++i)
      {
        new_shape[i] = new_shape[i] * dim_scales[i];
      }

      delete [] dim_scales;

      actual_layer = new LScale(parent, {new_shape[2], new_shape[3]}, reshape_out, getWrappingMode(da_mode), constant, name, DEV_CPU, 0);
    }
    break;

    default:
      log_string("FATAL: LAYER NOT RECOGNIZED WITH TYPE " + layer_type_name, log_level, LOG_LEVEL::ERROR);
      nodeQueue.pop();
      continue;
      break;
    }

    for (int i = 0; i < node->output_size(); i++)
    {
      output_node_map[node->output(i)] = actual_layer;
      vector<onnx::NodeProto *> child_nodes = input_node_map[node->output(i)];
      for (onnx::NodeProto *child : child_nodes)
      {
        nodeQueue.push(child);
      }
    }
    nodeQueue.pop();
  }
  vector<Layer *> input_layers;
  bool valid_input;
  for (Layer *layer : inputs)
  {
    valid_input = true;
    // Check if we have to avoid setting the current input layer as an input for the model
    for (string lname : inputs2remove)
      if (lname.compare(layer->name) == 0)
      {
        log_string("The input layer " + lname + " is not going to be a required input for the model. The EDDL will handle the input data for this layer.",
                   log_level,
                   LOG_LEVEL::DEBUG);
        valid_input = false;
      }

    if (valid_input)
      input_layers.push_back(layer);
  }

  vector<string> output_names = get_outputs(graph);
  vector<Layer *> output_layers;
  for (int i = 0; i < output_names.size(); i++)
  {
    output_layers.push_back(output_node_map[output_names[i]]);
  }

  Net *imported_net = new Net(input_layers, output_layers);

  log_string("Finished importing net from ONNX", log_level, LOG_LEVEL::DEBUG);
  return imported_net;
}

// Shares the weights to the snets on device
void share_weights(Net *net)
{
  for (int j = 0; j < net->layers.size(); j++)
  {
    for (int k = 0; k < net->layers[j]->params.size(); k++)
    {
      for (int i = 0; i < net->snets.size(); i++)
      {
        Tensor::copy(net->layers[j]->params[k], net->snets[i]->layers[j]->params[k]);
      }
    }
  }
}

// Sets the weights of a input Net to the ones stored in the onnx net inside the pointer
void set_weights_from_onnx_pointer(Net *net, void *ptr_model, size_t model_size)
{
  onnx::ModelProto model;
  {
    if (!model.ParseFromArray(ptr_model, model_size))
      cerr << "Failed to parse model." << endl;
    else if (verbose >= 2)
      cout << "Model parsed succesfuly" << endl;
  }

  map<string, vector<Tensor *>> tensors = get_tensors_from_onnx(model);
  LConv *conv;
  LDense *dense;
  for (Layer *l : net->layers)
  {
    if (!tensors.count(l->name))
    {
      //cout << "Layer with name " << l->name << " is not trainable " << endl;
      continue;
    }
    vector<Tensor *> layer_tensors = tensors[l->name];
    if ((conv = dynamic_cast<LConv *>(l)))
    {
      if (layer_tensors.size() > 1)
        conv->update_weights(layer_tensors[0], layer_tensors[1]);
      else
      {
        cerr << "EDDL has not implemented convolutional without bias " << endl;
        //conv.update_weights(layer_tensors[0]);
      }
    }
    else if ((dense = dynamic_cast<LDense *>(l)))
    {
      if (layer_tensors.size() > 1)
        dense->update_weights(layer_tensors[0], layer_tensors[1]);
      else
        dense->update_weights(layer_tensors[0]);
    }
    else
      cerr << "not implemented layer type" << endl;
  }

  // copy the new weights to devices
  share_weights(net);

  // erase the map we used to free the memory
  map<string, vector<Tensor *>>::iterator it;
  vector<Tensor *> delete_tensors;
  for (it = tensors.begin(); it != tensors.end(); ++it)
  {
    delete_tensors = it->second;
    for (int i = 0; i < delete_tensors.size(); ++i)
    {
      delete delete_tensors[i];
    }
  }
}

// Sets the weights of a input Net to the ones stored in the onnx net inside the c++ string
void set_weights_from_onnx(Net *net, std::string *model_string)
{
  onnx::ModelProto model;
  {
    if (!model.ParseFromString(*model_string))
    {
      cerr << "Failed to parse model." << endl;
    }
    else if (verbose >= 2)
      cout << "Model parsed succesfuly" << endl;
  }

  map<string, vector<Tensor *>> tensors = get_tensors_from_onnx(model);
  LConv *conv;
  LDense *dense;
  for (Layer *l : net->layers)
  {
    if (!tensors.count(l->name))
    {
      //cout << "Layer with name " << l->name << " is not trainable " << endl;
      continue;
    }
    vector<Tensor *> layer_tensors = tensors[l->name];
    if ((conv = dynamic_cast<LConv *>(l)))
    {
      if (layer_tensors.size() > 1)
        conv->update_weights(layer_tensors[0], layer_tensors[1]);
      else
      {
        cerr << "EDDL has not implemented convolutional without bias " << endl;
        //conv.update_weights(layer_tensors[0]);
      }
    }
    else if ((dense = dynamic_cast<LDense *>(l)))
    {
      if (layer_tensors.size() > 1)
        dense->update_weights(layer_tensors[0], layer_tensors[1]);
      else
        dense->update_weights(layer_tensors[0]);
    }
    else
      cerr << "not implemented layer type" << endl;
  }

  // copy the new weights to devices
  share_weights(net);

  // erase the map we used to free the memory
  map<string, vector<Tensor *>>::iterator it;
  vector<Tensor *> delete_tensors;
  for (it = tensors.begin(); it != tensors.end(); ++it)
  {
    delete_tensors = it->second;
    for (int i = 0; i < delete_tensors.size(); ++i)
    {
      delete delete_tensors[i];
    }
  }
}

// Accumulates the gradients stored in the pointer to the input net
void apply_grads_from_onnx_pointer(Net *net, void *ptr_onnx, size_t count)
{
  onnx::ModelProto model;
  {
    if (!model.ParseFromArray(ptr_onnx, count))
    {
      cerr << "Failed to parse model." << endl;
    }
    else if (verbose >= 2)
      cout << "Model parsed succesfuly" << endl;
  }

  map<string, vector<Tensor *>> tensors = get_tensors_from_onnx(model);
  LConv *conv;
  LDense *dense;
  for (Layer *l : net->layers)
  {
    if (!tensors.count(l->name))
    {
      continue;
    }
    vector<Tensor *> layer_tensors = tensors[l->name];
    if ((conv = dynamic_cast<LConv *>(l)))
    {
      if (layer_tensors.size() > 1)
      {
        conv->accumulate_accumulated_gradients(layer_tensors[0], layer_tensors[1]);
      }
      else
      {
        cerr << "EDDL has not implemented convolutional without bias." << endl;
        //conv.update_weights(layer_tensors[0]);
      }
    }
    else if ((dense = dynamic_cast<LDense *>(l)))
    {
      if (layer_tensors.size() > 1)
      {
        dense->accumulate_accumulated_gradients(layer_tensors[0], layer_tensors[1]);
      }
      else
      {
        dense->accumulate_accumulated_gradients(layer_tensors[0]);
      }
    }
    else
      cerr << "not implemented layer type" << endl;
  }
  // erase the map we used to free the memory
  map<string, vector<Tensor *>>::iterator it;
  vector<Tensor *> delete_tensors;
  for (it = tensors.begin(); it != tensors.end(); ++it)
  {
    delete_tensors = it->second;
    for (int i = 0; i < delete_tensors.size(); ++i)
    {
      delete delete_tensors[i];
    }
  }

  // copy the new weights to devices
  share_weights(net);
}

// Accumulates the gradients stored in the c++ string to the input net
void apply_grads_from_onnx(Net *net, std::string *model_string)
{
  onnx::ModelProto model;
  {
    if (!model.ParseFromString(*model_string))
    {
      cerr << "Failed to parse model." << endl;
    }
    else if (verbose >= 2)
      cout << "Model parsed succesfuly" << endl;
  }

  map<string, vector<Tensor *>> tensors = get_tensors_from_onnx(model);
  LConv *conv;
  LDense *dense;
  for (Layer *l : net->layers)
  {
    if (!tensors.count(l->name))
      continue;
    vector<Tensor *> layer_tensors = tensors[l->name];
    if ((conv = dynamic_cast<LConv *>(l)))
    {
      if (layer_tensors.size() > 1)
        conv->accumulate_accumulated_gradients(layer_tensors[0], layer_tensors[1]);
      else
      {
        cerr << "EDDL has not implemented convolutional without bias " << endl;
        //conv.update_weights(layer_tensors[0]);
      }
    }
    else if ((dense = dynamic_cast<LDense *>(l)))
    {
      if (layer_tensors.size() > 1)
        dense->accumulate_accumulated_gradients(layer_tensors[0], layer_tensors[1]);
      else
        dense->accumulate_accumulated_gradients(layer_tensors[0]);
    }
    else
      cerr << "not implemented layer type" << endl;
  }
  // erase the map we used to free the memory
  map<string, vector<Tensor *>>::iterator it;
  vector<Tensor *> delete_tensors;
  for (it = tensors.begin(); it != tensors.end(); ++it)
  {
    delete_tensors = it->second;
    for (int i = 0; i < delete_tensors.size(); ++i)
    {
      delete delete_tensors[i];
    }
  }

  // copy the new weights to devices
  share_weights(net);
}

// Returns a map containing the name of the layer as key and a tensor with the values of the model as value
map<string, vector<Tensor *>> get_tensors_from_onnx(onnx::ModelProto model)
{
  map<string, vector<Tensor *>> tensors;

  onnx::GraphProto graph = model.graph(); // Get the graph of the model.

  vector<onnx::TensorProto> initializers = get_initializers(graph); // Retrieves the initializers from the graph.
  // The weights for the layers can be found in the initializers.
  map<string, vector<float>> map_init_values;
  map<string, vector<int>> map_init_dims;
  get_initializers_maps(initializers, map_init_values, map_init_dims); // Creates 2 maps
  //  Key: Input Name . Value: Weights
  //  Key: Input Name . Value: Dims
  vector<onnx::NodeProto> nodes = get_graph_nodes(graph);

  map<string, ONNX_LAYERS> map_layers = create_enum_map();
  int dev = DEV_CPU;

  for (onnx::NodeProto node : nodes)
  {
    string layer_type_name = node.op_type();
    ONNX_LAYERS layer_type = map_layers[layer_type_name];
    string name = node.name();

    switch (layer_type)
    {
    case ONNX_LAYERS::CONV:
    {
      vector<Tensor *> conv_tensors;

      string weights_name = node.input(1); // Get weights and dims
      vector<float> *weights = &(map_init_values[weights_name]);
      vector<int> dims = map_init_dims[weights_name];

      Tensor * temp = new Tensor(dims, nullptr, dev);
      COPY_FROM_VECTOR_PTR_TO_TENSOR(weights, temp);
      conv_tensors.push_back(temp);

      if (node.input_size() > 2)
      { // This means we also have a bias
        string bias_name = node.input(2);
        vector<float> *bias = &(map_init_values[bias_name]);
        vector<int> bias_shape;
        bias_shape.push_back(bias->size());
        temp = new Tensor(bias_shape, nullptr, dev);
        COPY_FROM_VECTOR_PTR_TO_TENSOR(bias, temp);
        conv_tensors.push_back(temp);
      }

      tensors[name] = conv_tensors;
      break;
    }

    case ONNX_LAYERS::DENSE:
    {
      vector<Tensor *> dense_tensors;

      string weights_name = node.input(1); // Get weights and dims
      vector<float> *weights = &(map_init_values[weights_name]);
      vector<int> dims = map_init_dims[weights_name];

      Tensor * temp = new Tensor(dims, nullptr, dev);
      COPY_FROM_VECTOR_PTR_TO_TENSOR(weights, temp);
      dense_tensors.push_back(temp);

      if (node.input_size() > 2)
      {
        string bias_name = node.input(2);
        vector<float> *bias = &(map_init_values[bias_name]);
        vector<int> bias_dims = map_init_dims[bias_name];
        temp = new Tensor(bias_dims, nullptr, dev);
        COPY_FROM_VECTOR_PTR_TO_TENSOR(bias, temp);
        dense_tensors.push_back(temp);
      }

      tensors[name] = dense_tensors;
      break;
    }

    default:
      //cout << "The layer with type " << layer_type_name << " has no trainable parameters " << endl;
      continue;
      break;
    }
  }

  return tensors;
}
#else

Net *import_net_from_onnx_file(std::string path)
{
  cerr << "Not compiled for ONNX. Missing protobuf. Returning nullptr" << endl;
  return nullptr;
}

Net *import_net_from_onnx_pointer(void *serialized_model, size_t model_size)
{
  cerr << "Not compiled for ONNX. Missing protobuf. Returning nullptr" << endl;
  return nullptr;
}

Net *import_net_from_onnx_string(std::string *model_string)
{
  cerr << "Not compiled for ONNX. Missing protobuf. Returning nullptr" << endl;
  return nullptr;
}

#endif // cPROTO
