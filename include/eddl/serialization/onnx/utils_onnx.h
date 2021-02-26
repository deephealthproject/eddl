#ifndef EDDL_UTILS_ONNX_H
#define EDDL_UTILS_ONNX_H
#include "eddl/net/net.h"

#if defined(cPROTO)
#include "eddl/serialization/onnx/onnx.pb.h"

// Parses the values of the onnx tensor to a c++ vector of that type
vector<float> parseTensorValues(onnx::TensorProto t);

// Converts a raw onnx value tensor and writes it to a vector of that value type.
template <class T>
bool TryConvertingTensorRawValues(const onnx::TensorProto &onnx_tensor, vector<T> &field)
{
  if (!onnx_tensor.has_raw_data())
    return false;

  size_t raw_size = onnx_tensor.raw_data().size();
  if (raw_size % sizeof(T) != 0)
    return false;

  size_t num_elements = raw_size / sizeof(T);
  const void *src_ptr = static_cast<const void *>(onnx_tensor.raw_data().data());
  field.resize(num_elements, 0);
  void *target_ptr = static_cast<void *>(field.data());
  memcpy(target_ptr, src_ptr, raw_size);
  return true;
}

#endif // defined(cPROTO)

#define COPY_FROM_VECTOR_PTR_TO_TENSOR(v, t) (copy((v)->begin(), (v)->end(), t->ptr))
#define COPY_FROM_VECTOR_PTR_TO_FLOAT_PTR(v, ptr) (copy((v)->begin(), (v)->end(), ptr))

enum LOG_LEVEL {
  TRACE = 0,
  DEBUG = 1,
  INFO = 2,
  WARN = 3,
  ERROR = 4,
  NO_LOGS = 5
};

// If the actual_log_level is <= than string_log_level prints log
void log_string(string log, LOG_LEVEL actual_log_level, LOG_LEVEL string_log_level);

// Return a new vector of ints from the float vector passed as a parameter
std::vector<int> vf2vi(const std::vector<float> &vf);

// Synchronize the weights of the snets with the original net
void collect_params(Net *net);

#endif // EDDL_UTILS_ONNX_H
