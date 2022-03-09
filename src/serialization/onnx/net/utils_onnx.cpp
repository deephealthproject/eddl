#if defined(cPROTO)
#include "eddl/serialization/onnx/utils_onnx.h"
#include "eddl/serialization/onnx/layers/core/unsqueeze_onnx.h"
#include "eddl/serialization/onnx/layers/core/repeat_onnx.h"
#include "eddl/layers/auxiliar/layer_auxiliar.h"

void log_string(string log, LOG_LEVEL actual_log_level, LOG_LEVEL string_log_level){
  if (actual_log_level <= string_log_level){
    std::cerr << "[ONNX::LOG] " << log << std::endl;
  }
}

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

void collect_params(Net *net)
{
  if (net->snets[0]->dev != DEV_CPU)
    for (int i = 0; i < net->layers.size(); i++)
      for (int j = 0; j < net->layers[i]->params.size(); j++)
        collectTensor(net->layers[i], "param", j);
}

vector<Layer *> expand_broadcast(vector<Layer *> layers)
{
  if (layers.size() != 2)
    msg("Error: Expected 2 layers to check for broadcasting, got " + to_string(layers.size()), "ONNX::expand_broadcast");

  int n_dims0 = layers[0]->output->shape.size();
  int n_dims1 = layers[1]->output->shape.size();
  if (n_dims0 != n_dims1)
    msg("Error: The number of dimensions of the input layers is not the same (" + to_string(n_dims0) + " != " + to_string(n_dims1) + ")",
        "ONNX::expand_broadcast");

  int new_size = -1; // The new size to use in the LExpand layer
  // Find the dimension that must be expanded
  for (int i = 1/*skip batch*/; i < n_dims0; ++i)
  {
    int dim0 = layers[0]->output->shape[i];
    int dim1 = layers[1]->output->shape[i];
    if (dim0 == dim1)
      continue;
    else if (dim0 > dim1 && dim1 == 1)
    {
      layers[1] = new LExpand(layers[1], dim0, layers[1]->name + "_expanded", layers[1]->dev, layers[1]->mem_level);
      break;
    }
    else if (dim0 < dim1 && dim0 == 1)
    {
      layers[0] = new LExpand(layers[0], dim1, layers[0]->name + "_expanded", layers[0]->dev, layers[0]->mem_level);
      break;
    }
  }

  return layers;
}

tuple<bool, vector<string>> mlayer_check_and_fix_recurrent_input(MLayer *layer, onnx::GraphProto *graph, int seq_len) {
  const int n_parents = layer->parent.size();
  // Count the number of recurrent parents
  const int n_recurrent = std::count_if(layer->parent.begin(), layer->parent.end(),
                                  [](Layer *l) { return l->isrecurrent || l->isdecoder; });

  if (!n_recurrent || n_recurrent == n_parents)
  {
    // In this case we don't need to fix anything
    vector<string> all_parents;
    for (Layer *parentl : layer->parent)
      all_parents.push_back(parentl->name);
    return {n_recurrent > 0, all_parents};
  }
  else // case: n_recurrent < n_parents
  {
    // For each parent layer that is not recurrent we have to repeat the parent to
    // create a sequence. To do it in ONNX we use an Unsqueeze operator to add the sequence
    // dimension and a Tile operator to repeat the data along the sequence dimension
    vector<string> parents;
    for (Layer *parentl : layer->parent)
    {
      if (!parentl->isrecurrent && !parentl->isdecoder)
      {
        const string unsq_node_name = parentl->name + "_EDDL-unsqueeze";
        unsqueeze_node_builder(unsq_node_name, // Node name
                               parentl->name,  // Input name
                               unsq_node_name, // Output name
                               {0},            // Axes, add the sequence dimension before the batch
                               graph);

        if (seq_len < 1)
          msg("Error exporting the merge layer " + layer->name + ". To export this model you need to provide the 'seq_len' argument "
              "with a value higher than 0 in the export function.", "ONNX::ExportNet");

        const string tile_node_name = parentl->name + "_EDDL-tile";
        // Create the tiles vector with the repetitions for each dimension
        // We only have to repeat the first dimension ({secuence, batch, dim0, dim1, ..., dimN})
        vector<int> tiles (parentl->output->getShape().size() + 1, 1);
        tiles[0] = seq_len;
        tile_node_builder(tile_node_name,
                          unsq_node_name,
                          tile_node_name,
                          tiles,
                          graph);
        parents.push_back(tile_node_name);
      }
      else
        parents.push_back(parentl->name);
    }
    return {true, parents};
  }
}

#endif // defined(cPROTO)
