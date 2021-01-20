#ifndef EDDL_EDDL_ONNX_H
#define EDDL_EDDL_ONNX_H

#include "eddl/layers/conv/layer_conv.h"
#include "eddl/layers/core/layer_core.h"
#include "eddl/layers/layer.h"
#include "eddl/layers/merge/layer_merge.h"
#include "eddl/layers/normalization/layer_normalization.h"
#include "eddl/layers/operators/layer_operators.h"
#include "eddl/layers/pool/layer_pool.h"
#include "eddl/layers/recurrent/layer_recurrent.h"
#include "eddl/net/net.h"
#include <map>
#include <string>
#include <vector>

enum LOG_LEVEL {
  TRACE = 0,
  DEBUG = 1,
  INFO = 2,
  WARN = 3,
  ERROR = 4,
  NO_LOGS = 5
};

// Importing module
//------------------------------------------------------------------------------

/**
 * @brief Imports ONNX Net from file
 *
 * @param path Path to the file where the net in ONNX format is saved
 * @param mem (default 0)
 * @param log_level Available: LOG_LEVEL::{TRACE, DEBUG, INFO, WARN, ERROR,
 * NO_LOGS}. (default LOG_LEVEL::INFO)
 *
 * @return Net
 */
Net *import_net_from_onnx_file(std::string path, int mem = 0, int log_level = LOG_LEVEL::INFO);

Net *import_net_from_onnx_pointer(void *serialized_model, size_t model_size, int mem = 0);

Net *import_net_from_onnx_string(std::string *model_string, int mem = 0);


// Exporting module
//----------------------------------------------------------------------------------------

/**
 * @brief Saves a model with the onnx format in the file path provided
 *
 * @param net Net to be saved
 * @param path Path to the file where the net in ONNX format will be saved
 *
 * @return (void)
 */
void save_net_to_onnx_file(Net *net, string path);

// Returns a pointer to the serialized model in Onnx
size_t serialize_net_to_onnx_pointer(Net *net, void *&serialized_model, bool gradients = false);

// Returns a string containing the serialized model in Onnx
std::string *serialize_net_to_onnx_string(Net *net, bool gradients = false);


// Distributed Module
// ---------------------------------------------------------------------------------------

void set_weights_from_onnx(Net *net, std::string *model_string);
void set_weights_from_onnx_pointer(Net *net, void *ptr_model, size_t model_size);

void apply_grads_from_onnx(Net *net, std::string *model_string);
void apply_grads_from_onnx_pointer(Net *net, void *ptr_onnx, size_t count);


#endif // EDDL_EDDL_ONNX_H
