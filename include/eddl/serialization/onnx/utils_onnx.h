#ifndef EDDL_UTILS_ONNX_H
#define EDDL_UTILS_ONNX_H
#include "eddl/net/net.h"
#include <tuple>

#if defined(cPROTO)
#include "eddl/serialization/onnx/onnx.pb.h"

enum INPUT_TYPE { NORMAL, SEQUENCE_ENCODER, SEQUENCE_DECODER };

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

// Checks if any of the inputs of the MLayer is recurrent and if it is needed to repeat the other non-recurrent input
// to be able to compute the merge operation. In the EDDL this case don't cause any error but to export the model to
// ONNX the dimensions of the MLayer inputs must be compatible taking into account the sequence dimension.
//
// In case that some of the parents were recurrent, the function creates Unsqueeze and Tile operators to fix the problem
// converting de non-recurrent inputs to a sequence.
//
// The "seq_len" argument is used to set a fixed sequence length in the repeat operation to create a secuence from the
// non-secuence input of the MLayer.
//
// Returns a tuple:
//   - If one or more parent layers of the mlayer are sequences
//   - The list of the parent names to use as inputs when exporting the MLayer to ONNX
tuple<bool, vector<string>> mlayer_check_and_fix_recurrent_input(MLayer *layer, onnx::GraphProto *graph, int seq_len = 0);

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

// Given a vector with 2 layers, compares the output tensors and in case of finding a dimension that differs, uses a
// LExpand layer to simulate broadcasting. The returned vector will be the original layers or in case of using a LExpand,
// that layer will be replaced by the LExpand pointer
vector<Layer *> expand_broadcast(vector<Layer *> layers);

#endif // EDDL_UTILS_ONNX_H
