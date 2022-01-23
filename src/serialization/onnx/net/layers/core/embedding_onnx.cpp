#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/core/embedding_onnx.h"
#include "eddl/serialization/onnx/layers/core/squeeze_onnx.h"
#include "eddl/serialization/onnx/layers/onnx_nodes/onnx_node_conversion.h"

void build_embedding_node(LEmbedding *layer, onnx::GraphProto *graph, bool gradients)
{
  /*
   * To create the embedding operation in ONNX we have to use the following steps:
   *     1. Squeeze the last dim if the input shape is [batch, 1]
   *     2. Create a Cast op to int type for the indexes of the words
   *     3. Create a Gather op to select the embeddings from the indexes from the Cast
   */

  // 1. Create the Squeeze node for dim 2
  string cast_node_input;
  const int input_dim = layer->input->getShape().size();
  if (layer->length == 1 && input_dim == 2)
  {
    string squeeze_node_name = layer->name + "_squeeze";
    string squeeze_node_input = layer->parent[0]->name;
    string squeeze_node_output = layer->name + "_squeeze";
    squeeze_node_builder(
        squeeze_node_name,
        squeeze_node_input,
        squeeze_node_output,
        {2},
        graph);
    cast_node_input = squeeze_node_output;
  }
  else if (layer->length == 1)
    cast_node_input = layer->parent[0]->name;
  else
    msg("The input of the embedding layer must have length 1 in order to export it", "ONNX::ExportNet");

  // 2. Create the Cast op
  string cast_node_name = layer->name + "_indexes_cast";
  string cast_node_output = layer->name + "_cast";
  build_cast_node(
      cast_node_name,
      cast_node_input,
      cast_node_output,
      6, // cast type to int32
      graph);

  // 3. Creathe the Gahter op
  build_gather_node(
      layer->name,      // node name
      cast_node_output, // node input from cast node
      layer->name,      // node output name
      layer,
      graph,
      gradients);
}

/*
 * DISTRIBUTED TRAINING
 */

vector<Tensor *> get_embedding_tensors(onnx::NodeProto &node,
                                       map<string, vector<float>> &map_init_values,
                                       map<string, vector<int>> &map_init_dims)
{
  // Get weights and dims
  string weights_name = node.input(0);
  vector<float> *weights = &(map_init_values[weights_name]);
  vector<int> dims = map_init_dims[weights_name];

  Tensor *weights_tensor = new Tensor(dims, nullptr, DEV_CPU);
  COPY_FROM_VECTOR_PTR_TO_TENSOR(weights, weights_tensor);

  return {weights_tensor};
}

#endif // defined(cPROTO)
