#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/core/activation_onnx.h"

/*
 * ONNX IMPORT
 */

Layer* build_relu_layer(onnx::NodeProto *node,
                        map<string, Layer *> &output_node_map,
                        int dev,
                        int mem)
{
  string parent_name = node->input(0);
  Layer *parent = output_node_map[parent_name];

  return new LActivation(parent, "relu", {}, node->name(), dev, mem);
}

Layer* build_sigmoid_layer(onnx::NodeProto *node,
                           map<string, Layer *> &output_node_map,
                           int dev,
                           int mem)
{
  string parent_name = node->input(0);
  Layer *parent = output_node_map[parent_name];

  return new LActivation(parent, "sigmoid", {}, node->name(), dev, mem);
}

Layer* build_hard_sigmoid_layer(onnx::NodeProto *node,
                               map<string, Layer *> &output_node_map,
                               int dev,
                               int mem)
{
  string parent_name = node->input(0);
  Layer *parent = output_node_map[parent_name];

  return new LActivation(parent, "hard_sigmoid", {}, node->name(), dev, mem);
}

Layer* build_tanh_layer(onnx::NodeProto *node,
                        map<string, Layer *> &output_node_map,
                        int dev,
                        int mem)
{
  string parent_name = node->input(0);
  Layer *parent = output_node_map[parent_name];

  return new LActivation(parent, "tanh", {}, node->name(), dev, mem);
}

Layer* build_exponential_layer(onnx::NodeProto *node,
                               map<string, Layer *> &output_node_map,
                               int dev,
                               int mem)
{
  string parent_name = node->input(0);
  Layer *parent = output_node_map[parent_name];

  return new LActivation(parent, "exp", {}, node->name(), dev, mem);
}

Layer* build_linear_layer(onnx::NodeProto *node,
                          map<string, Layer *> &output_node_map,
                          int dev,
                          int mem)
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

  return new LActivation(parent, "linear", {alpha}, node->name(), dev, mem);
}

Layer* build_leaky_relu_layer(onnx::NodeProto *node,
                              map<string, Layer *> &output_node_map,
                              int dev,
                              int mem)
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

  return new LActivation(parent, "leaky_relu", {alpha}, node->name(), dev, mem);
}

Layer* build_thresholded_relu_layer(onnx::NodeProto *node,
                                    map<string, Layer *> &output_node_map,
                                    int dev,
                                    int mem)
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

      return new LActivation(parent, "thresholded_relu", {alpha}, node->name(), dev, mem);
}

Layer* build_elu_layer(onnx::NodeProto *node,
                       map<string, Layer *> &output_node_map,
                       int dev,
                       int mem)
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

  return new LActivation(parent, "elu", {alpha}, node->name(), dev, mem);
}

Layer* build_selu_layer(onnx::NodeProto *node,
                       map<string, Layer *> &output_node_map,
                       int dev,
                       int mem)
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

  return new LActivation(parent, "selu", {alpha, gamma}, node->name(), dev, mem);
}

Layer* build_softsign_layer(onnx::NodeProto *node,
                            map<string, Layer *> &output_node_map,
                            int dev,
                            int mem)
{
  string parent_name = node->input(0);
  Layer *parent = output_node_map[parent_name];

  return new LActivation(parent, "softsign", {}, node->name(), dev, mem);
}

Layer* build_softplus_layer(onnx::NodeProto *node,
                            map<string, Layer *> &output_node_map,
                            int dev,
                            int mem)
{
  string parent_name = node->input(0);
  Layer *parent = output_node_map[parent_name];

  return new LActivation(parent, "softplus", {}, node->name(), dev, mem);
}

Layer* build_softmax_layer(onnx::NodeProto *node,
                           map<string, Layer *> &output_node_map,
                           int dev,
                           int mem)
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

  int parent_dims = parent->output->getShape().size();

  if (axis < 0)                        // Check if the target axis is a negative index
    axis = parent_dims + axis;         // Get the target axis index
  if (axis < 0 || axis >= parent_dims) // Check for invalid axis index
    msg("The target axis for Softmax is not valid: axis = " + to_string(axis), "ONNX::ImportNet");

  if (axis == 0  &&  parent_dims == 2)
    axis = 1; // let us correct the problem of axis = 0 when importing a model in ONNX format where input shape does not contain the batch_size

  if (axis == 0) { // Second check for invalid axis index
    std::cerr << __FILE__ << "(" << __LINE__ << ")  axis = " << axis << "  shape.size = " << parent_dims << endl;
    msg("The target axis for Softmax is not valid: axis = " + to_string(axis), "ONNX::ImportNet");
  }

  return new LActivation(parent, "softmax", {static_cast<float>(axis)}, node->name(), dev, mem);
}

/*
 * ONNX EXPORT
 */

void build_relu_node(LActivation *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Relu");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

void build_sigmoid_node(LActivation *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Sigmoid");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

void build_hard_sigmoid_node(LActivation *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("HardSigmoid");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

void build_tanh_node(LActivation *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Tanh");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

void build_exponential_node(LActivation *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Exponential"); // **Custom operator**
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

void build_linear_node(LActivation *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Linear"); // **Custom operator**
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);

  // Attr alpha
  onnx::AttributeProto *alpha_attr = node->add_attribute();
  alpha_attr->set_name("alpha");
  alpha_attr->set_type(onnx::AttributeProto::FLOAT);
  alpha_attr->set_f(layer->params[0]);
}

void build_leaky_relu_node(LActivation *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("LeakyRelu");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);

  // Attr alpha
  onnx::AttributeProto *alpha_attr = node->add_attribute();
  alpha_attr->set_name("alpha");
  alpha_attr->set_type(onnx::AttributeProto::FLOAT);
  alpha_attr->set_f(layer->params[0]);
}

void build_thresholded_relu_node(LActivation *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("ThresholdedRelu");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);

  // Attr alpha
  onnx::AttributeProto *alpha_attr = node->add_attribute();
  alpha_attr->set_name("alpha");
  alpha_attr->set_type(onnx::AttributeProto::FLOAT);
  alpha_attr->set_f(layer->params[0]);
}

void build_elu_node(LActivation *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Elu");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);

  // Attr alpha
  onnx::AttributeProto *alpha_attr = node->add_attribute();
  alpha_attr->set_name("alpha");
  alpha_attr->set_type(onnx::AttributeProto::FLOAT);
  alpha_attr->set_f(layer->params[0]);
}

void build_selu_node(LActivation *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Selu");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);

  // Attr alpha
  onnx::AttributeProto *alpha_attr = node->add_attribute();
  alpha_attr->set_name("alpha");
  alpha_attr->set_type(onnx::AttributeProto::FLOAT);
  alpha_attr->set_f(layer->params[0]);

  // Attr gamma
  onnx::AttributeProto *gamma_attr = node->add_attribute();
  gamma_attr->set_name("gamma");
  gamma_attr->set_type(onnx::AttributeProto::FLOAT);
  gamma_attr->set_f(layer->params[1]);
}

void build_softmax_node(LActivation *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Softmax");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Attr axis
  onnx::AttributeProto *axis_attr = node->add_attribute();
  axis_attr->set_name("axis");
  axis_attr->set_type(onnx::AttributeProto::INT);
  axis_attr->set_i((int)layer->params[0]);
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

void build_softsign_node(LActivation *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Softsign");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

void build_softplus_node(LActivation *layer, onnx::GraphProto *graph)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Softplus");
  node->set_name(layer->name);
  // Set the inputs names of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);
}

#endif // defined(cPROTO)
