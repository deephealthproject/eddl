#if defined(cPROTO)
#include "eddl/serialization/onnx/export_helpers.h"
#include "eddl/serialization/onnx/layers/layers_onnx.h"
#include "eddl/layers/core/layer_core.h"
#include <queue>

onnx::ModelProto build_onnx_model(Net *net, bool gradients, int seq_len)
{
  string producer_name("EDDL");
  string producer_version("1.0");
  // Create the empty Model in onnx
  onnx::ModelProto model;
  model.set_ir_version(onnx::Version::IR_VERSION);
  model.set_producer_name(producer_name);
  model.set_producer_version(producer_version);

  // Builds all the graph of the model
  set_graph(&model, net, gradients, seq_len);

  // Return the finished model
  return model;
}

void set_graph(onnx::ModelProto *model, Net *net, bool gradients, int seq_len)
{
  // Add a new empty graph to the model
  onnx::GraphProto *graph = model->mutable_graph();
  graph->set_name("Computational Graph");
  onnx::OperatorSetIdProto *opset = model->add_opset_import();
  opset->set_version(11);

  // key: input layer name, val: INPUT_TYPE
  map<string, INPUT_TYPE> inputs_types = check_inputs_for_enc_or_dec(net);
  bool is_recurrent = false;

  // Set the inputs shapes of the graph
  for (auto const& item : inputs_types)
  {
    Layer *input = net->getLayer(item.first);
    onnx::ValueInfoProto *input_info = graph->add_input();
    input_info->set_name(input->name);
    onnx::TypeProto *input_type = input_info->mutable_type();
    onnx::TypeProto::Tensor *input_type_tensor = input_type->mutable_tensor_type();
    input_type_tensor->set_elem_type(onnx::TensorProto::FLOAT);
    onnx::TensorShapeProto *input_type_tensor_shape = input_type_tensor->mutable_shape();
    onnx::TensorShapeProto::Dimension *input_type_tensor_dim;
    vector<int> input_shape = input->input->getShape();

    if (item.second == INPUT_TYPE::SEQUENCE_ENCODER || item.second == INPUT_TYPE::SEQUENCE_DECODER)
    {
      // Set variable batch size
      input_type_tensor_dim = input_type_tensor_shape->add_dim();
      input_type_tensor_dim->set_dim_param("batch");
      // Set variable sequence lenght
      input_type_tensor_dim = input_type_tensor_shape->add_dim();
      if (seq_len > 0)
        input_type_tensor_dim->set_dim_value(seq_len);
      else
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

      is_recurrent = true;
    }
    else  // INPUT_TYPE::NORMAL
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

  // Computational graph
  for (Layer *aux_layer : net->layers)
  {
    // Builds a node of the graph from the layer in EDDL
    build_node_from_layer(aux_layer, graph, gradients, is_recurrent, seq_len);
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
    if (aux_output->isdecoder)
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
}

INPUT_TYPE get_input_type(LInput *l) {
    // If it is a decoder input it should be marked as decoder
    if (l->isdecoder)
        return INPUT_TYPE::SEQUENCE_DECODER;

    // Now check if it is INPUT_TYPE::ENCODER or INPUT_TYPE::NORMAL
    //
    // Use BFS to find one of this cases:
    //   1. recurrent but not decoder layer: Set the input type as ENCODER
    //
    //   2. decoder layer: Set the input type as NORMAL
    //
    //   3. No recurrent or decoder layer: Set the input type as Normal
    map<string, bool> visited;
    queue<Layer*> queue;
    for (Layer *child : l->child)
        queue.push(child);

    while (!queue.empty()) {
        // Get the next layer in the queue
        Layer *current = queue.front(); queue.pop();

        // Check if we already visited this layer
        if (visited.find(current->name) != visited.end())
            continue;
        visited[current->name] = true; // Mark as visited

        if (current->isrecurrent && !current->isdecoder)
            return INPUT_TYPE::SEQUENCE_ENCODER; // Case 1

        if (current->isdecoder)
            return INPUT_TYPE::NORMAL; // Case 2

        // Push the current layer childs to the queue
        for (Layer *child : current->child)
            queue.push(child);
    }

    return INPUT_TYPE::NORMAL; // Case 3
}

map<string, INPUT_TYPE> check_inputs_for_enc_or_dec(Net *net) {
  /*
   * We get all the input layers from the layers vector of the model
   * instead of taking them from net->lin. Because for the case of
   * a recurrent net with decoder the input layer that is connected
   * to the decoder is not added in the lin vector of the model.
   * With this way we ensure that we are taking all the input layers
   * of the model.
   */
  map<string, INPUT_TYPE> inputs_types;
  for (Layer *aux_layer : net->layers)
    if (LInput *in_layer = dynamic_cast<LInput *>(aux_layer))
        inputs_types[in_layer->name] = get_input_type(in_layer);

  return inputs_types;
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
