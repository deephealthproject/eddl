#ifndef EDDL_UTILS_ONNX_H
#define EDDL_UTILS_ONNX_H
#include "eddl/net/net.h"

enum LOG_LEVEL {
  TRACE = 0,
  DEBUG = 1,
  INFO = 2,
  WARN = 3,
  ERROR = 4,
  NO_LOGS = 5
};

// Synchronize the weights of the snets with the original net
// Also synchronizes the accumulated gradients if specified
void sync_snets_with_orig(Net *net, bool acc_gradients=false);

// Synchronize the weights of the snets
void sync_params(Net *net);

// Synchronize the accumulated gradients of the snets
void sync_acc_gradients(Net *net);

#endif // EDDL_UTILS_ONNX_H
