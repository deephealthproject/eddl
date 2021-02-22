#if defined(cPROTO)
#include "eddl/serialization/onnx/import_helpers.h"
#include "eddl/layers/core/layer_core.h"

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
queue<onnx::NodeProto *> process_inputs(vector<Layer *> *inputs, 
                                        vector<onnx::ValueInfoProto> *inputs_onnx, 
                                        map<string, vector<onnx::NodeProto *>> *input_node_map, 
                                        map<string, Layer *> *output_node_map)
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

// Creates a map where the key is the onnx name for the layer type and the 
// value is the constant value in the enumeration for onnx layer type.
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

// Parses the values of the onnx tensor to a c++ vector of that type
vector<float> parseTensorValues(onnx::TensorProto t)
{
  int data_type = t.data_type(); // Only works for non raw data for now
  vector<float> values;
  switch (data_type)
  {
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
      for (float i : aux_values) // Cast to float
        values.push_back(i);
    }
    else
    {
      for (int i = 0; i < t.int64_data_size(); i++)
      {
        values.push_back(t.int64_data(i));
      }
    }
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
  // TODO
  //case onnx::TensorProto::STRING:
  //  break;
  //case onnx::TensorProto::UNDEFINED:
  //  break;
  //case onnx::TensorProto::COMPLEX64:
  //  break;
  //case onnx::TensorProto::COMPLEX128:
  //  break;
  //case onnx::TensorProto::BFLOAT16:
  //  break;
  default:
    cerr << "Vector type not recognized" << endl;
    break;
  }
  return values;
}

// Creates two maps. Both have the name of the initializer node as key. 
// The values are a vector containing the weights and a vector containing 
// the shape of the vector, respectively.
void get_initializers_maps(vector<onnx::TensorProto> tensors, 
                           map<string, vector<float>> &values_map, 
                           map<string, vector<int>> &dims_map)
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

  if (!input_shape.empty()) // Check for custom input shape
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
  // Construct set of input names
  set<string> input_names;
  for (int i = 0; i < graph.input_size(); i++)
  { 
    input_names.insert(graph.input(i).name());
  }

  // Construct set of initializer names
  vector<onnx::TensorProto> initializers = get_initializers(graph);
  set<string> initializer_names;
  for (int i = 0; i < initializers.size(); i++)
    if (initializers[i].has_name())
      initializer_names.insert(initializers[i].name());

  // We make the substraction of both sets to find the true inputs
  vector<string> true_inputs;
  std::set_difference(input_names.begin(), input_names.end(), 
                      initializer_names.begin(), initializer_names.end(), 
                      std::inserter(true_inputs, true_inputs.begin()));

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
    output_names.push_back(graph.output(i).name());

  return output_names;
}

// Returns a vector containing all nodes of the graph in onnx containers.
vector<onnx::NodeProto> get_graph_nodes(onnx::GraphProto graph)
{
  vector<onnx::NodeProto> nodes;
  for (int i = 0; i < graph.node_size(); i++)
    nodes.push_back(graph.node(i));

  return nodes;
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
    is_decoder = true;
  else
    return false;

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

// Returns a map containing the name of the layer as key and a tensor with the values of the model as value
map<string, vector<Tensor *>> get_tensors_from_onnx(onnx::ModelProto model)
{
  // TODO: This method should be removed after changing the distibuted functions of ONNX by
  // a more generic ones (using params vector)
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

#endif // defined(cPROTO)
