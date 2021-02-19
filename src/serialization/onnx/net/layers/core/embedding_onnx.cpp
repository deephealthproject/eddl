#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/core/embedding_onnx.h"
#include "eddl/serialization/onnx/layers/core/squeeze_onnx.h"
#include "eddl/serialization/onnx/layers/onnx_helpers/export_helpers.h"

void build_embedding_node(LEmbedding *layer, onnx::GraphProto *graph)
{
  /*
   * To create the embedding operation in ONNX we have to use the following steps:
   *     1. Squeeze the last dim if the input shape is [batch, seq_len, 1]
   *     2. Create a Cast op to int type for the indexes of the words
   *     3. Create a Gather op to select the embeddings from the indexes from the Cast
   */

  // 1. Create the Squeeze node for dim 2
  string cast_node_input;
  if (layer->length == 1)
  {
    string squeeze_node_name = layer->name + "_squeeze";
    string squeeze_node_input = layer->parent[0]->name;
    string squeeze_node_output = layer->name + "_squeeze";
    build_squeeze_node(
        squeeze_node_name,
        squeeze_node_input,
        squeeze_node_output,
        {2},
        graph);
    cast_node_input = squeeze_node_output;
  }
  else
  {
    msg("The input of the embedding layer must have length 1 in order to export it", "ONNX::ExportNet");
  }

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
      graph);
}

#endif // defined(cPROTO)
