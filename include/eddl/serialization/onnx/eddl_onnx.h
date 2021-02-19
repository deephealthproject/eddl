#ifndef EDDL_EDDL_ONNX_H
#define EDDL_EDDL_ONNX_H

#include <cstddef>
#include <map>
#include <string>
#include <vector>

#include "eddl/serialization/onnx/utils_onnx.h"
#include "eddl/net/compserv.h"
#include "eddl/net/net.h"
#include "eddl/optimizers/optim.h"

//------------------------------------------------------------------------------
// Importing module
//------------------------------------------------------------------------------

// Net loaders
//-------------

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
Net *import_net_from_onnx_file(string path, int mem = 0, int log_level = LOG_LEVEL::INFO);

/**
 * @brief Imports ONNX Net from file and changes its input shape. Note that this function is 
 *        for models with only one input layer, in other case it will fail.
 *
 * @param path Path to the file where the net in ONNX format is saved
 * @param input_shape Shape of the input data (without batch dimension)
 * @param mem (default 0)
 * @param log_level Available: LOG_LEVEL::{TRACE, DEBUG, INFO, WARN, ERROR,
 * NO_LOGS}. (default LOG_LEVEL::INFO)
 *
 * @return Net
 */
Net *import_net_from_onnx_file(string path, vector<int> input_shape, int mem = 0, int log_level = LOG_LEVEL::INFO);

Net *import_net_from_onnx_pointer(void *serialized_model, size_t model_size, int mem = 0);

Net *import_net_from_onnx_string(string *model_string, int mem = 0);


// Optimizer loaders
//-------------------

/**
 * @brief Creates an Optimizer from the definition provided in an ONNX file. The
 *        ONNX will provide the Optimizer type and its attributes.
 *
 * @param path Path to the file where the Optimizer configuration is saved
 *
 * @return Optimizer*
 */
Optimizer *import_optimizer_from_onnx_file(string path);

Optimizer *import_optimizer_from_onnx_pointer(void *serialized_optimizer, size_t optimizer_size);

Optimizer *import_optimizer_from_onnx_string(string *optimizer_string);


// Computing Service loaders
//---------------------------

/**
 * @brief Creates a CompServ from the definition provided in an ONNX file. The
 *        ONNX will provide the CompServ type and its configuration.
 *
 * @param path Path to the file where the CompServ configuration is saved
 *
 * @return CompServ*
 */
CompServ *import_compserv_from_onnx_file(string path);

CompServ *import_compserv_from_onnx_pointer(void *serialized_cs, size_t cs_size);

CompServ *import_compserv_from_onnx_string(string *cs_string);


//----------------------------------------------------------------------------------------
// Exporting module
//----------------------------------------------------------------------------------------

// Net exporters
//---------------
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
string *serialize_net_to_onnx_string(Net *net, bool gradients = false);


// Optimizer exporters
//---------------------

/**
 * @brief Saves the configuration of an optimizer using the ONNX format. It will contain
 *        the Optimizer type and attributes like learning rate, momentum, weight decay, etc.
 *
 * @param optimizer Optimizer to be saved
 * @param path Path to the file where the Optimizer configuration will be saved
 *
 * @return (void)
 */
void save_optimizer_to_onnx_file(Optimizer *optimizer, string path);

size_t serialize_optimizer_to_onnx_pointer(Optimizer *optimizer, void *&serialized_optimizer);

string *serialize_optimizer_to_onnx_string(Optimizer *optimizer);


// Computing Service exporters
//-----------------------------

/**
 * @brief Saves the configuration of a computing service using the ONNX format. It will 
 *        contain the computing service type and configuration of devices.
 *
 * @param cs CompServ to be saved
 * @param path Path to the file where the computing service configuration will be saved
 *
 * @return (void)
 */
void save_compserv_to_onnx_file(CompServ *cs, string path);

size_t serialize_compserv_to_onnx_pointer(CompServ *cs, void *&serialized_cs);

string *serialize_compserv_to_onnx_string(CompServ *cs);


// ---------------------------------------------------------------------------------------
// Distributed Module
// ---------------------------------------------------------------------------------------

void set_weights_from_onnx(Net *net, string *model_string);
void set_weights_from_onnx_pointer(Net *net, void *ptr_model, size_t model_size);

void apply_grads_from_onnx(Net *net, string *model_string);
void apply_grads_from_onnx_pointer(Net *net, void *ptr_onnx, size_t count);


#endif // EDDL_EDDL_ONNX_H
