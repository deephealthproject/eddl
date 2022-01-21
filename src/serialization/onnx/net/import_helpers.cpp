#if defined(cPROTO)
#include "eddl/serialization/onnx/import_helpers.h"
#include "eddl/serialization/onnx/layers/layers_onnx.h"
#include "eddl/layers/core/layer_core.h"
#include <queue>

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

vector<INPUT_TYPE> check_recurrent_inputs(vector<onnx::ValueInfoProto> inputs_onnx, map<string, vector<onnx::NodeProto *>> &input_node_map)
{
    vector<INPUT_TYPE> inputs_types(inputs_onnx.size(), INPUT_TYPE::NORMAL);
    map<string, ONNX_LAYERS> map_layers = create_enum_map();

    int input_index = 0;
    for (onnx::ValueInfoProto &in : inputs_onnx)
    {
      map<string, bool> visited;
      queue<onnx::NodeProto*> queue;
      for (onnx::NodeProto *child : input_node_map[in.name()])
        queue.push(child);

      while (!queue.empty()) {
        // Get the next node in the queue
        onnx::NodeProto *current = queue.front(); queue.pop();
        const string node_name = current->name();

        // Check if we already visited this node
        if (visited.find(node_name) != visited.end())
          continue;
        visited[node_name] = true; // Mark as visited

        if (node_is_recurrent(current, map_layers))
        {
          if (node_is_decoder(current, input_node_map))
            inputs_types[input_index] = INPUT_TYPE::SEQUENCE_DECODER;
          else
            inputs_types[input_index] = INPUT_TYPE::SEQUENCE_ENCODER;
          break;
        }

        // Try to detect the Unsqueeze + Tile operators to repeat the tensor to create a sequence
        // If we detect it, we have to remove this operators and conect the parent of the Unsqueeze
        // to the childs of the Tile
        if (current->op_type() == "Unsqueeze")
        {
          bool unsq_tile_detected = false;
          // Check if it is adding a sequence dimension on position 0
          for (int attr_idx = 0; attr_idx < current->attribute_size(); ++attr_idx)
          {
            onnx::AttributeProto attribute = current->attribute(attr_idx);
            if (attribute.name() == "axes")
            {
              vector<int> axes;
              for (int h = 0; h < attribute.ints_size(); h++)
                axes.push_back(attribute.ints(h));

              // Check if the Unsqueeze is adding a dimension before the batch
              if (std::find(axes.begin(), axes.end(), 0) != axes.end())
                // Check if one of the child nodes is a Tile
                for (int o = 0; o < current->output_size(); ++o)
                  for (onnx::NodeProto *child : input_node_map[current->output(o)])
                    if (child->op_type() == "Tile")
                      unsq_tile_detected = true;
            }
          }
          if (unsq_tile_detected)
            break; // The current processed input is not recurrent
        }

        // Push the current node childs to the queue
        for (int o = 0; o < current->output_size(); ++o)
          for (onnx::NodeProto *child : input_node_map[current->output(o)])
            queue.push(child);
      }
      input_index++;
    }

    return inputs_types;
}

// Creates a map with the name of the constant node as key and the node container from onnx as value.
map<string, onnx::NodeProto *> initialize_constant_nodes(vector<onnx::NodeProto> &nodes,
                                                         map<string, vector<onnx::NodeProto *>> &input_node_map)
{
  map<string, onnx::NodeProto *> constant_node_map;
  for (auto& node : nodes)
  {
    if (node.op_type() == "Constant")
    {
      // There are some constant nodes that will be processed directly from the child node (e.g. the Reshape layer)
      // because they are a parameter of the child layer, not an input.
      // In other case, those constant nodes will be used to create a ConstOfTensor layer.
      bool skip_node = false;
      for (const auto& child : input_node_map[node.output(0)])
        if (child->op_type() != "Reshape" &&
            child->op_type() != "Tile")
        {
          skip_node = true; // The constant data will be accessed directly from the ConstOfTensor layer constructor
          break;
        }

      if (skip_node)
        continue; // Dont add the node to the constant_node_map

      for (int j = 0; j < node.output_size(); j++)
        constant_node_map[node.output(j)] = &node;
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
vector<int> parse_IO_tensor(onnx::TypeProto::Tensor tensor, INPUT_TYPE input_type)
{
  onnx::TensorShapeProto tensorShape = tensor.shape();
  vector<int> shape;
  shape.push_back(1); // Set batch to 1
  int start_index = 1;
  if (input_type == INPUT_TYPE::SEQUENCE_DECODER || input_type == INPUT_TYPE::SEQUENCE_ENCODER)
    start_index = 2; // Avoid sequence dim

  for (int i = start_index; i < tensorShape.dim_size(); i++)
  {
    shape.push_back(tensorShape.dim(i).dim_value());
  }

  return shape;
}

// Converts one vector of TensorProto pointers (Input or output)
// to one vector of eddl Tensor pointers.
vector<Layer *> parse_IO_tensors(vector<onnx::ValueInfoProto> io_onnx, vector<int> input_shape, int mem, vector<INPUT_TYPE> inputs_types)
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
    int input_index = 0;
    for (onnx::ValueInfoProto infoProto : io_onnx)
    {
      tensor = infoProto.type().tensor_type();
      string name = infoProto.name();
      vector<int> input_shape = parse_IO_tensor(tensor, inputs_types[input_index]);
      io.push_back(new LInput(new Tensor(input_shape), name, dev, mem));
      input_index++;
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
  l->isdecoder=true;

  int p=l->child.size();
  for(int i=0;i<p;i++)
    set_decoder(l->child[i]);
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
  onnx::GraphProto graph = model.graph(); // Get the graph of the model

  // The weights for the layers can be found in the initializers
  vector<onnx::TensorProto> initializers = get_initializers(graph);
  map<string, vector<float>> map_init_values; // Key: Layer weights name - Value: Weights
  map<string, vector<int>> map_init_dims;     // Key: Layer weights name - Value: Shape of the weights
  get_initializers_maps(initializers, map_init_values, map_init_dims); // Creates 2 maps

  vector<onnx::NodeProto> nodes = get_graph_nodes(graph); // Nodes == model layers

  return get_tensors_from_onnx_nodes(nodes, map_init_values, map_init_dims);
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
                          map<string, vector<onnx::NodeProto *>> &input_node_map,
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
      if (node->op_type() == "Constant")
      {
        bool skip_node = false;
        for (const auto& child : input_node_map[node->output(0)])
          if (child->op_type() == "Reshape" ||
              child->op_type() == "Tile")
            skip_node = true; // The constant data will be accessed directly from the child layer constructor
        if (skip_node)
        {
          log_string("The constant node \"" + node->name() + "\" is a parameter, going to skip the node in the queue.",
                     log_level,
                     LOG_LEVEL::DEBUG);
          continue; // Don't add the node to the nodeQueue
        }
      }
      log_string("Node \"" + node->name() + "\" is avaliable, since only has initializers and constant nodes as parameters.",
                 log_level,
                 LOG_LEVEL::DEBUG);
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
                        bool recurrent_net,
                        int mem,
                        LOG_LEVEL log_level)
{
  map<string, ONNX_LAYERS> map_layers = create_enum_map();

  // 6 - While the queue is not empty:
  //   Note: Check build_net_onnx() for full algorithm description
  while (!nodeQueue.empty())
  {
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

  // Get the initializers that store the layers weights and params
  vector<onnx::TensorProto> initializers = get_initializers(graph);

  // Create the main dictionaries to handle model parameters
  map<string, vector<float>> map_init_values; // Key: Input Name - Value: Weights
  map<string, vector<int>> map_init_dims;     // Key: Input Name - Value: Dims
  get_initializers_maps(initializers, map_init_values, map_init_dims); // Fill the maps

  // 1, 2 and 3: Initialize maps
  map<string, vector<onnx::NodeProto *>> input_node_map = initialize_input_node_map(nodes);

  // 4 and 5: Create queue of NodeProto
  map<string, onnx::NodeProto *> constant_node_map = initialize_constant_nodes(nodes, input_node_map);

  // Parse ONNX inputs to EDDL inputs layers
  vector<INPUT_TYPE> inputs_types = check_recurrent_inputs(inputs_onnx, input_node_map);
  vector<Layer *> inputs = parse_IO_tensors(inputs_onnx, input_shape, mem, inputs_types);

  map<string, Layer *> output_node_map;
  queue<onnx::NodeProto *> nodeQueue = process_inputs(inputs, inputs_onnx, input_node_map, output_node_map);

  // Check if any node only has initializers and constant nodes as parameters, so we can process it right away
  queue_constant_nodes(nodes, map_init_values, input_node_map, constant_node_map, nodeQueue, log_level);

  // 6: Process the node queue and create the model layers 
  process_node_queue(nodeQueue,
                     map_init_values,
                     map_init_dims,
                     input_node_map,
                     output_node_map,
                     constant_node_map,
                     recurrent_net,
                     mem,
                     log_level);

  // Get input layers of the model
  vector<Layer *> model_input_layers;
  int aux_idx = 0;
  for (Layer *layer : inputs)
  {
    // Check if we have to avoid setting the current input layer as an input for the model
    // Note: The EDDL handles the input data for the decoders inputs
    if (inputs_types[aux_idx] == INPUT_TYPE::SEQUENCE_DECODER)
    {
        set_decoder(layer);
        log_string("The input layer " + layer->name + " is not going to be a required input for the model because it is the input of the decoder. "
                   "The EDDL will handle the input data for this layer.",
                   log_level,
                   LOG_LEVEL::DEBUG);
    }
    else
      model_input_layers.push_back(layer);
    aux_idx++;
  }

  // Get output layers of the model
  vector<string> output_names = get_outputs(graph);
  vector<Layer *> output_layers;
  for (int i = 0; i < output_names.size(); i++)
    output_layers.push_back(output_node_map[output_names[i]]);

  Net *imported_net = new Net(model_input_layers, output_layers);

  log_string("Finished importing net from ONNX", log_level, LOG_LEVEL::DEBUG);
  return imported_net;
}

void set_weights_from_model_proto(Net *net, onnx::ModelProto model_proto)
{
  map<string, vector<Tensor *>> tensors = get_tensors_from_onnx(model_proto);
  for (Layer *l : net->layers)
  {
    // Check if we have tensors with weights for the current layer
    if (!tensors.count(l->name))
      continue;

    // Get the layer weights
    vector<Tensor *> new_weights = tensors[l->name];
    if (new_weights.size() == 0)
    {
      cerr << "[ONNX::WARNING] Trying to update the weights of the layer \""
           << l->name << "\" with an empty list of tensors." << endl;
      continue;
    }

    // Apply the new weights
    l->update_weights(new_weights);
  }

  // Copy the new weights to devices
  share_weights(net);

  // Erase the map we used to free the memory
  map<string, vector<Tensor *>>::iterator it;
  vector<Tensor *> delete_tensors;
  for (it = tensors.begin(); it != tensors.end(); ++it)
  {
    delete_tensors = it->second;
    for (int i = 0; i < delete_tensors.size(); ++i)
      delete delete_tensors[i];
  }
}

void apply_grads_from_model_proto(Net *net, onnx::ModelProto model_proto)
{
  map<string, vector<Tensor *>> tensors = get_tensors_from_onnx(model_proto);
  for (Layer *l : net->layers)
  {
    // Check if we have tensors with gradients for the current layer
    if (!tensors.count(l->name))
      continue;

    // Get the layer gradients
    vector<Tensor *> acc_grads = tensors[l->name];
    if (acc_grads.size() == 0)
    {
      cerr << "[ONNX::WARNING] Trying to apply gradients to the layer \""
           << l->name << "\" with an empty list of tensors." << endl;
      continue;
    }

    // Apply the gradients
    l->accumulate_accumulated_gradients(acc_grads);
  }

  // Erase the map we used to free the memory
  map<string, vector<Tensor *>>::iterator it;
  vector<Tensor *> delete_tensors;
  for (it = tensors.begin(); it != tensors.end(); ++it)
  {
    delete_tensors = it->second;
    for (int i = 0; i < delete_tensors.size(); ++i)
      delete delete_tensors[i];
  }

  // Copy the new weights to devices
  share_weights(net);
}

#endif // defined(cPROTO)
