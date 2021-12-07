#if defined(cPROTO)
#include "eddl/serialization/onnx/import_helpers.h"
#include "eddl/serialization/onnx/layers/layers_onnx.h"
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
      input_node_map[input_name].push_back(&(nodes[j]));
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
queue<onnx::NodeProto *> process_inputs(vector<Layer *> &inputs, 
                                        vector<onnx::ValueInfoProto> &inputs_onnx, 
                                        map<string, vector<onnx::NodeProto *>> &input_node_map, 
                                        map<string, Layer *> &output_node_map)
{
  queue<onnx::NodeProto *> nodeQueue;
  for (int i = 0; i < inputs.size(); i++)
  {
    onnx::ValueInfoProto nameContainer = inputs_onnx[i];
    string name = nameContainer.name();
    output_node_map[name] = inputs[i];
    vector<onnx::NodeProto *> v = input_node_map[name];
    for (onnx::NodeProto *layer : input_node_map[name])
    {
      nodeQueue.push(layer);
    }
  }
  return nodeQueue;
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

// Shows basic metadata of the model if the log level is DEBUG or lower
void log_model_metadata(onnx::ModelProto& model, LOG_LEVEL log_level)
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
}

void queue_constant_nodes(vector<onnx::NodeProto> &nodes,
                          map<string, vector<float>> &map_init_values,
                          map<string, onnx::NodeProto *> &constant_node_map,
                          queue<onnx::NodeProto *> &nodeQueue,
                          LOG_LEVEL log_level)
{
  for (int i = 0; i < nodes.size(); i++)
  {
    onnx::NodeProto *node = &nodes[i];
    bool avaliable = true;
    for (int j = 0; j < node->input_size(); j++)
    {
      string input = node->input(j);
      if (map_init_values.count(input))
        continue;
      if (constant_node_map.count(input))
        continue;
      avaliable = false;
      break;
    }
    if (avaliable)
    {
      log_string("Node " + node->name() + " is avaliable, since only has initializers and constant nodes as parameters.",
                 log_level,
                 LOG_LEVEL::DEBUG);
      if (node->op_type() == "Constant")
        continue;
      nodeQueue.push(node);
    }
  }
}

void process_node_queue(queue<onnx::NodeProto *> &nodeQueue,
                        map<string, vector<float>> &map_init_values,
                        map<string, vector<int>> &map_init_dims,
                        map<string, vector<onnx::NodeProto *>> &input_node_map,
                        map<string, Layer *> &output_node_map,
                        map<string, onnx::NodeProto *> &constant_node_map,
                        vector<string> &inputs2remove,
                        bool recurrent_net,
                        int mem,
                        LOG_LEVEL log_level)
{
  map<string, ONNX_LAYERS> map_layers = create_enum_map();

  // 6 - While the queue is not empty:
  //   Note: Check build_net_onnx() for full algorithm description
  while (!nodeQueue.empty())
  {
    printf("entra\n");
    onnx::NodeProto *node = nodeQueue.front();
    log_string("Next node: " + node->name(), log_level, LOG_LEVEL::DEBUG);

    // Look for inputs with empty ("") names that some libraries create, and delete them
    auto *inputs_list = node->mutable_input();
    for (auto i = inputs_list->begin(); i != inputs_list->end(); i++)
      if (i->empty())
        i = --inputs_list->erase(i); // -- to compensate the iterator skip made by erase()

    // 6.1: Check all inputs are avaliable
    bool avaliable = true;

    for (int i = 0; i < node->input_size(); i++)
    {
      string input_name = node->input(i);
      if (map_init_values.count(input_name))
        continue;
      if (output_node_map.count(input_name))
        continue;
      if (constant_node_map.count(input_name))
        continue;
      avaliable = false;
      log_string("Node " + node->name() + " is not avaliable yet. Missing input: " + input_name, log_level, LOG_LEVEL::DEBUG);
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

    // 6.3: Create the layer from the onnx node
    // Every case should create the corresponding layer and asign it to "actual_layer" variable
    int dev = DEV_CPU;
    Layer *actual_layer = build_layer_from_node(node,
                                                map_layers,
                                                map_init_values,
                                                map_init_dims,
                                                input_node_map,
                                                output_node_map,
                                                constant_node_map,
                                                inputs2remove,
                                                recurrent_net,
                                                log_level,
                                                dev,
                                                mem);

    // 6.3: Add the nodes that use the outputs of the created layer to the queue
    for (int i = 0; i < node->output_size(); i++)
    {
      if (actual_layer != nullptr)
          output_node_map[node->output(i)] = actual_layer;
      vector<onnx::NodeProto *> child_nodes = input_node_map[node->output(i)];
      for (onnx::NodeProto *child : child_nodes)
        nodeQueue.push(child);
    }

    nodeQueue.pop();  // Pop the node we just processed
  }
}

// Builds a eddl Net from an instance of the onnx container for model
Net *build_net_onnx(onnx::ModelProto model, vector<int> input_shape, int mem, LOG_LEVEL log_level)
{
  log_model_metadata(model, log_level);
  onnx::GraphProto graph = model.graph(); // Get the graph of the model.

  vector<onnx::ValueInfoProto> inputs_onnx = get_inputs(graph); // Get input nodes data
  vector<onnx::NodeProto> nodes = get_graph_nodes(graph); // Get the nodes (layers) of the model
  bool recurrent_net = check_recurrent_nodes(nodes);
  if (recurrent_net)
    log_string("The net is recurrent", log_level, LOG_LEVEL::DEBUG);

  /*
   * The methodology is the following:
   * We create three maps:
   * map <string input, vector<onnx::NodeProto *> > input_node_map: The input will point towards the nodes that have this input
   * map <string output, Layer* parent > output_node_map: To know from which (parent) node comes each input (The input is the output of the parent node)
   * map <string input/output, bool> input_active_map: The outputs will be put inside a bool, where we will see if it is active or not.
   * 
   * The algorithm is the following:
   *   1 - We check the inputs of each node.
   *       For each input we insert the input string as a key and the Node(s) that use it as a value in the input_node_map.
   *       If that input is already on use, we will append the node to the existing vector
   *   2 - We check the outputs of each node. //NOT REQUIRED
   *   For each output we insert the output string as a key and the Node that generates it as a value in the outputs_map //NOT REQUIRED
   *   3 - When we add an input/output to the map, we also add it to the input_active_map as key, and the value will be false by default. If it is already there, we do nothing. //NOT REQUIRED
   *   4 - Once we have constructed these maps, we create an empty queue of NodeProto
   *   5 - For the input nodes in the graph, we create the EDDL layer and add the nodes that use its output(s) to the queue
   *   6 - For each node (while the queue is not empty):
   *     6.1 - Check if all its inputs nodes (not the ones in 'initializers') exist in output_node_map
   *     6.2 - If they are not:  
   *       continue
   *     6.3 - Else:
   *       Create the EDDL layer
   *       Add the nodes that use its outputs to the queue
   *
   * To create each EDDL layer:
   *   1 - Get its parent(s) by accessing to output_node_map using this node's input(s) as key(s)
   *   2 - Get its weights from 'initializers'
   *   3 - Create layer
   * 
   *   We need another map for storing the constant nodes, who are always active
   *   We design this map as map<string, onnx::NodeProto> and called constant_node_map
   */

  // Parse ONNX inputs to EDDL inputs layers
  vector<Layer *> inputs = parse_IO_tensors(inputs_onnx, input_shape, mem, recurrent_net);

  // Get the initializers that store the layers weights and params
  vector<onnx::TensorProto> initializers = get_initializers(graph);

  // Create the main dictionaries to handle model parameters
  map<string, vector<float>> map_init_values; // Key: Input Name - Value: Weights
  map<string, vector<int>> map_init_dims;     // Key: Input Name - Value: Dims
  get_initializers_maps(initializers, map_init_values, map_init_dims); // Fill the maps

  // 1, 2 and 3: Initialize maps
  map<string, vector<onnx::NodeProto *>> input_node_map = initialize_input_node_map(nodes);

  // 4 and 5: Create queue of NodeProto
  map<string, onnx::NodeProto *> constant_node_map = initialize_constant_nodes(nodes);

  map<string, Layer *> output_node_map;
  queue<onnx::NodeProto *> nodeQueue = process_inputs(inputs, inputs_onnx, input_node_map, output_node_map);

  // Check if any node only has initializers and constant nodes as parameters, so we can process it right away
  queue_constant_nodes(nodes, map_init_values, constant_node_map, nodeQueue, log_level);

  /*
   * In the case of models with recurrent decoders, we have to track the input layers of the decoder layers
   * and avoid adding them to the input layers of the model
   */
  vector<string> inputs2remove = {};

  // 6: Process the node queue and create the model layers 
  process_node_queue(nodeQueue,
                     map_init_values,
                     map_init_dims,
                     input_node_map,
                     output_node_map,
                     constant_node_map,
                     inputs2remove,
                     recurrent_net,
                     mem,
                     log_level);

  // Get input layers of the model
  vector<Layer *> input_layers;
  for (Layer *layer : inputs)
  {
    bool valid_input = true;
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

  // Get output layers of the model
  vector<string> output_names = get_outputs(graph);
  vector<Layer *> output_layers;
  for (int i = 0; i < output_names.size(); i++)
    output_layers.push_back(output_node_map[output_names[i]]);

  Net *imported_net = new Net(input_layers, output_layers);

  log_string("Finished importing net from ONNX", log_level, LOG_LEVEL::DEBUG);
  return imported_net;
}

#endif // defined(cPROTO)
