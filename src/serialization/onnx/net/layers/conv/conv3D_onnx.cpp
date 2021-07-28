#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/conv/conv3D_onnx.h"

void build_conv3D_node(LConv3D *layer, onnx::GraphProto *graph, bool gradients)
{
  // Add an empty node to the graph
  onnx::NodeProto *node = graph->add_node();
  node->set_op_type("Conv");
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

  ////////////////////////// Attributes of the Conv operation //////////////////////////////////
  // Attr dilations
  onnx::AttributeProto *conv_dilations = node->add_attribute();
  conv_dilations->set_name("dilations");
  conv_dilations->set_type(onnx::AttributeProto::INTS);
  for (int i : layer->cd->dilation_rate)
    conv_dilations->add_ints(i);

  //Attr group
  //onnx::AttributeProto *conv_group = node->add_attribute();
  //conv_group->set_name("group");
  //conv_group->set_type(onnx::AttributeProto::INT);
  //conv_group->set_i(1);

  // Attr kernel_shape
  onnx::AttributeProto *conv_kernel_shape = node->add_attribute();
  conv_kernel_shape->set_name("kernel_shape");
  conv_kernel_shape->set_type(onnx::AttributeProto::INTS);
  conv_kernel_shape->add_ints(layer->cd->kd);
  conv_kernel_shape->add_ints(layer->cd->kr);
  conv_kernel_shape->add_ints(layer->cd->kc);

  // Attr pads
  onnx::AttributeProto *conv_pads = node->add_attribute();
  conv_pads->set_name("pads");
  conv_pads->set_type(onnx::AttributeProto::INTS);
  conv_pads->add_ints(layer->cd->paddf);
  conv_pads->add_ints(layer->cd->padrt);
  conv_pads->add_ints(layer->cd->padcl);
  conv_pads->add_ints(layer->cd->paddb);
  conv_pads->add_ints(layer->cd->padrb);
  conv_pads->add_ints(layer->cd->padcr);

  // Attr strides
  onnx::AttributeProto *conv_strides = node->add_attribute();
  conv_strides->set_name("strides");
  conv_strides->set_type(onnx::AttributeProto::INTS);
  conv_strides->add_ints(layer->cd->sd);
  conv_strides->add_ints(layer->cd->sr);
  conv_strides->add_ints(layer->cd->sc);

  // Check if we are exporting weights or accumulated gradients
  if (!gradients)
  {
    // Weights input
    onnx::TensorProto *conv_w = graph->add_initializer();
    conv_w->set_name(layer->name + "_W");
    conv_w->set_data_type(onnx::TensorProto::FLOAT);
    conv_w->mutable_dims()->Add(layer->cd->K->shape.begin(), layer->cd->K->shape.end());        // Set the shape of the weights
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
    conv_w->mutable_dims()->Add(layer->cd->acc_gK->shape.begin(), layer->cd->acc_gK->shape.end());             // Set the accumulated gradiens shape (weights)
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

#endif // defined(cPROTO)
