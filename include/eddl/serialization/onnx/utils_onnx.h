#ifndef EDDL_UTILS_ONNX_H
#define EDDL_UTILS_ONNX_H
#include "eddl/net/net.h"

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
// Also synchronizes the accumulated gradients if specified
void sync_snets_with_orig(Net *net, bool acc_gradients=false);

// Synchronize the weights of the snets
void sync_params(Net *net);

// Synchronize the accumulated gradients of the snets
void sync_acc_gradients(Net *net);

#endif // EDDL_UTILS_ONNX_H
