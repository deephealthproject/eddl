#if defined(cPROTO)
#include "eddl/serialization/onnx/utils_onnx.h"
#include "eddl/layers/auxiliar/layer_auxiliar.h"

void log_string(string log, LOG_LEVEL actual_log_level, LOG_LEVEL string_log_level)
{
  if (actual_log_level <= string_log_level)
  {
    cout << "[ONNX::LOG] " << log << endl;
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

#endif // defined(cPROTO)
