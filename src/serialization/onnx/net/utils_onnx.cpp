#if defined(cPROTO)
#include "eddl/serialization/onnx/utils_onnx.h"

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

void sync_snets_with_orig(Net *net, bool acc_gradients)
{
  if (net->snets[0]->dev != DEV_CPU)
  {
    sync_params(net);
    if (acc_gradients)
      sync_acc_gradients(net);
  }
}

void sync_params(Net *net)
{
  for (int j = 0; j < net->layers.size(); j++)
  {
    for (int k = 0; k < net->layers[j]->params.size(); k++)
    {
      net->layers[j]->params[k]->fill_(0.0);
      for (int i = 0; i < net->snets.size(); i++)
      {
        Tensor::inc(net->snets[i]->layers[j]->params[k], net->layers[j]->params[k]);
      }
      net->layers[j]->params[k]->div_(net->snets.size());
    }
  }
}

void sync_acc_gradients(Net *net)
{
  for (int j = 0; j < net->layers.size(); j++)
  {
    for (int k = 0; k < net->layers[j]->acc_gradients.size(); k++)
    {
      net->layers[j]->acc_gradients[k]->fill_(0.0);
      for (int i = 0; i < net->snets.size(); i++)
      {
        Tensor::inc(net->snets[i]->layers[j]->acc_gradients[k], net->layers[j]->acc_gradients[k]);
      }
      net->layers[j]->acc_gradients[k]->div_(net->snets.size());
    }
  }
}

#endif // defined(cPROTO)
