#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/conv/convT_onnx.h"
#include "eddl/serialization/onnx/utils_onnx.h"

// ONNX import
Layer* build_convT_layer(onnx::NodeProto *node,
                        map<string, vector<float>> &map_init_values,
                        map<string, vector<int>> &map_init_dims,
                        map<string, Layer *> &output_node_map,
                        int dev,
                        int mem)
{
  int filters;
  vector<int> kernel_shape;
  vector<int> strides;
  vector<int> pads = {};
  string auto_pad_option = "custom";
  vector<float> *bias;
  bool use_bias = node->input_size() > 2;
  int conv_dim = 2; // Number of dimension of the convolution (1, 2 or 3)
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
    else if (!attr_name.compare("group"))
    {
      groups = attribute.i();
    }
    else if (!attr_name.compare("kernel_shape"))
    {
      for (int h = 0; h < attribute.ints_size(); h++)
        kernel_shape.push_back(attribute.ints(h));

      // Deduce conv dimension
      if (attribute.ints_size() == 1)
        conv_dim = 1;
      else if (attribute.ints_size() == 3)
        conv_dim = 3;
    }
    else if (!attr_name.compare("pads"))
    {
      for (int h = 0; h < attribute.ints_size(); h++)
        pads.push_back(attribute.ints(h));

      // Reorder padding from [x1_begin, x2_begin,..., x1_end, x2_end,...] to [x1_begin, x1_end, x2_begin, x2_end,...]
      if (attribute.ints_size() == 4) // ConvT2D
        swap(pads[1], pads[2]);
      else if (attribute.ints_size() == 6) // ConvT3D
      {
        swap(pads[1], pads[3]);
        swap(pads[2], pads[3]);
        swap(pads[3], pads[4]);
      }
    }
    else if (!attr_name.compare("strides"))
    {
      for (int h = 0; h < attribute.ints_size(); h++)
        strides.push_back(attribute.ints(h));

      // Deduce conv dimension
      if (attribute.ints_size() == 1)
        conv_dim = 1;
      else if (attribute.ints_size() == 3)
        conv_dim = 3;
    }
  }

  string name = node->name(); // Layer name
  string parent_name = node->input(0); // Get parent
  Layer *parent = output_node_map[parent_name];
  vector<int> parent_shape = parent->output->shape;

  string weights_name = node->input(1); // Get weights and dims
  vector<float> *weights = &(map_init_values[weights_name]);
  vector<int> dims = map_init_dims[weights_name];
  filters = dims[0];

  // Deduce conv dimension from layer input
  if (parent_shape.size() == 3)
    conv_dim = 1;
  else if (parent_shape.size() == 4)
    conv_dim = 2;
  else if (parent_shape.size() == 5)
    conv_dim = 3;

  Layer *actual_layer;
  if (conv_dim < 3) // Handle ConvT1D and ConvT2D (they use the same ConvolDescriptor)
  {
    if (conv_dim == 1)
    {
      // Prepare args to create a equivalent ConvT2D from a ConvT1D
      strides.push_back(1);
      kernel_shape.push_back(1);
      dims.push_back(1);
      pads.push_back(0);
      pads.push_back(0);
    }

    ConvolDescriptorT2D *cd = new ConvolDescriptorT2D(filters,
                                                      kernel_shape,
                                                      strides,
                                                      auto_pad_option,
                                                      pads,
                                                      groups,
                                                      dilation_rate,
                                                      use_bias,
                                                      mem);

    if (conv_dim == 1)
    {
      //actual_layer = new LConvT1D(parent, cd, name, dev, mem);
      actual_layer = nullptr;
      msg("Error: ConvT1D is not supported.", "[ONNX::ImportNet]");
    }
    else
      actual_layer = new LConvT2D(parent, cd, name, dev, mem);

    if (use_bias)
    {
      string bias_name = node->input(2);
      bias = &(map_init_values[bias_name]);
      Tensor *bias_tensor = new Tensor({(int)bias->size()}, nullptr, dev);
      COPY_FROM_VECTOR_PTR_TO_TENSOR(bias, bias_tensor);
      Tensor::copy(bias_tensor, cd->bias);
      delete bias_tensor;
    }
    Tensor *weights_tensor = new Tensor(dims, nullptr, dev);
    COPY_FROM_VECTOR_PTR_TO_TENSOR(weights, weights_tensor);
    Tensor::copy(weights_tensor, cd->K);
    delete weights_tensor;
  }
  else // Conv3D
  {
    ConvolDescriptorT3D *cd = new ConvolDescriptorT3D(filters,
                                                      kernel_shape,
                                                      strides,
                                                      auto_pad_option,
                                                      pads,
                                                      groups,
                                                      dilation_rate,
                                                      use_bias,
                                                      mem);

    actual_layer = new LConvT3D(parent, cd, name, dev, mem);

    if (use_bias)
    {
      string bias_name = node->input(2);
      bias = &(map_init_values[bias_name]);
      Tensor *bias_tensor = new Tensor({(int)bias->size()}, nullptr, dev);
      COPY_FROM_VECTOR_PTR_TO_TENSOR(bias, bias_tensor);
      Tensor::copy(bias_tensor, cd->bias);
      delete bias_tensor;
    }
    Tensor *weights_tensor = new Tensor(dims, nullptr, dev);
    COPY_FROM_VECTOR_PTR_TO_TENSOR(weights, weights_tensor);
    Tensor::copy(weights_tensor, cd->K);
    delete weights_tensor;
  }

  return actual_layer;
}

// ONNX export
void build_convT_node(LConvT2D *layer, onnx::GraphProto *graph, bool gradients)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("ConvTranspose");
  node->set_name(layer->name);
  // Set the inputs of the node from the parents of the layer
  for (Layer *parentl : layer->parent)
  {
    node->add_input(parentl->name);
  }
  // Set the input params names of the conv op
  node->add_input(layer->name + "_W");
  if (layer->cd->use_bias)
    node->add_input(layer->name + "_b");
  // Set the name of the output of the node to link with other nodes
  node->add_output(layer->name);

  ////////////////////////// Attributes of the ConvTranspose operation //////////////////////////////////
  // Attr dilations
  onnx::AttributeProto *conv_dilations = node->add_attribute();
  conv_dilations->set_name("dilations");
  conv_dilations->set_type(onnx::AttributeProto::INTS);
  vector<int> vdilations{1, 1};
  for (int i : vdilations)
  {
    conv_dilations->add_ints(i);
  }

  //Attr group
  onnx::AttributeProto *conv_group = node->add_attribute();
  conv_group->set_name("group");
  conv_group->set_type(onnx::AttributeProto::INT);
  conv_group->set_i(layer->cd->groups);

  // Attr kernel_shape
  onnx::AttributeProto *conv_kernel_shape = node->add_attribute();
  conv_kernel_shape->set_name("kernel_shape");
  conv_kernel_shape->set_type(onnx::AttributeProto::INTS);
  conv_kernel_shape->add_ints(layer->cd->kr);
  conv_kernel_shape->add_ints(layer->cd->kc);

  // Attr pads
  onnx::AttributeProto *conv_pads = node->add_attribute();
  conv_pads->set_name("pads");
  conv_pads->set_type(onnx::AttributeProto::INTS);
  conv_pads->add_ints(layer->cd->padrt);
  conv_pads->add_ints(layer->cd->padcl);
  conv_pads->add_ints(layer->cd->padrb);
  conv_pads->add_ints(layer->cd->padcr);

  // Attr strides
  onnx::AttributeProto *conv_strides = node->add_attribute();
  conv_strides->set_name("strides");
  conv_strides->set_type(onnx::AttributeProto::INTS);
  conv_strides->add_ints(layer->cd->sr);
  conv_strides->add_ints(layer->cd->sc);

  // Check if we are exporting weights or accumulated gradients
  if (!gradients)
  {
    // Weights input
    onnx::TensorProto *conv_w = graph->add_initializer();
    conv_w->set_name(layer->name + "_W");
    conv_w->set_data_type(onnx::TensorProto::FLOAT);
    conv_w->mutable_dims()->Add(layer->cd->K->shape.begin(), layer->cd->K->shape.end());          // Set the shape of the weights
    conv_w->mutable_float_data()->Add(layer->cd->K->ptr, layer->cd->K->ptr + layer->cd->K->size); // Set the weights values
    //conv_w->mutable_raw_data()->assign( reinterpret_cast<const char*>(layer->cd->K->ptr), sizeof(float) * layer->cd->K->size );

    // Bias input
    if (layer->cd->use_bias)
    {
      onnx::TensorProto *conv_b = graph->add_initializer();
      conv_b->set_name(layer->name + "_b");
      conv_b->set_data_type(onnx::TensorProto::FLOAT);
      conv_b->mutable_dims()->Add(layer->cd->bias->shape.begin(), layer->cd->bias->shape.end());             // Set the shape of the bias
      conv_b->mutable_float_data()->Add(layer->cd->bias->ptr, layer->cd->bias->ptr + layer->cd->bias->size); // Set the bias values
      //conv_b->mutable_raw_data()->assign( reinterpret_cast<const char*>(layer->cd->bias->ptr), sizeof(float) * layer->cd->bias->size );
    }
  }
  else
  {
    // Accumulated gradients (Weights) input
    onnx::TensorProto *conv_w = graph->add_initializer();
    conv_w->set_name(layer->name + "_W");
    conv_w->set_data_type(onnx::TensorProto::FLOAT);
    conv_w->mutable_dims()->Add(layer->cd->acc_gK->shape.begin(), layer->cd->acc_gK->shape.end());               // Set the accumulated gradiens shape (weights)
    conv_w->mutable_float_data()->Add(layer->cd->acc_gK->ptr, layer->cd->acc_gK->ptr + layer->cd->acc_gK->size); // Set the accumulated gradients values (weights)
    //conv_w->mutable_raw_data()->assign( reinterpret_cast<const char*>(layer->cd->acc_gK->ptr), sizeof(float) * layer->cd->acc_gK->size );

    // Accumulated gradients (bias) input
    if (layer->cd->use_bias)
    {
      onnx::TensorProto *conv_b = graph->add_initializer();
      conv_b->set_name(layer->name + "_b");
      conv_b->set_data_type(onnx::TensorProto::FLOAT);
      conv_b->mutable_dims()->Add(layer->cd->acc_gbias->shape.begin(), layer->cd->acc_gbias->shape.end());                  // Set the accumulated gradients shape (bias)
      conv_b->mutable_float_data()->Add(layer->cd->acc_gbias->ptr, layer->cd->acc_gbias->ptr + layer->cd->acc_gbias->size); // Set the accumulated gradients values (bias)
      //conv_b->mutable_raw_data()->assign( reinterpret_cast<const char*>(layer->cd->acc_gbias->ptr), sizeof(float) * layer->cd->acc_gbias->size );
    }
  }
}

/*
 * DISTRIBUTED TRAINING
 */

vector<Tensor *> get_convT_tensors(onnx::NodeProto &node,
                                   map<string, vector<float>> &map_init_values,
                                   map<string, vector<int>> &map_init_dims)
{
  vector<Tensor *> conv_tensors;

  string weights_name = node.input(1); // Get weights and dims
  vector<float> *weights = &(map_init_values[weights_name]);
  vector<int> dims = map_init_dims[weights_name];

  if (dims.size() == 3)
      msg("Error: ConvT1D is not supported.", "[ONNX::get_convT_tensors]");

  Tensor * temp = new Tensor(dims, nullptr, DEV_CPU);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(weights, temp);
  conv_tensors.push_back(temp);

  if (node.input_size() > 2)
  { // This means we also have a bias
    string bias_name = node.input(2);
    vector<float> *bias = &(map_init_values[bias_name]);
    vector<int> bias_shape;
    bias_shape.push_back(bias->size());
    temp = new Tensor(bias_shape, nullptr, DEV_CPU);
    COPY_FROM_VECTOR_PTR_TO_TENSOR(bias, temp);
    conv_tensors.push_back(temp);
  }

  return conv_tensors;
}

#endif // defined(cPROTO)
