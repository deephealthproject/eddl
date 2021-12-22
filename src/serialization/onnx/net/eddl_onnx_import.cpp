#include <queue>
#include <fstream>
#include <map>
#include <set>
#include <algorithm>

#include "eddl/serialization/onnx/eddl_onnx.h"
#include "eddl/serialization/onnx/utils_onnx.h"
#include "eddl/serialization/onnx/import_helpers.h"

#if defined(cPROTO)
#include "eddl/serialization/onnx/onnx.pb.h"
#endif

using namespace std;

#ifdef cPROTO
// Imports a net stored in a onnx file
Net *import_net_from_onnx_file(std::string path, int mem, LOG_LEVEL log_level)
{
  // Check if the path exists
  if (!pathExists(path))
    msg("The specified path does not exist: " + path, "ONNX::ImportNet");

  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  onnx::ModelProto model;
  // Read the existing net.
  fstream input(path, ios::in | ios::binary);
  if (!model.ParseFromIstream(&input))
  {
    cerr << "Failed to parse model. Returning nullptr" << endl;
    input.close();
    return nullptr;
  }
  input.close();
  return build_net_onnx(model, {}, mem, log_level);
}

// Imports a net stored in a onnx file
Net *import_net_from_onnx_file(std::string path, vector<int> input_shape, int mem, LOG_LEVEL log_level)
{
  // Check if the path exists
  if (!pathExists(path))
    msg("The specified path does not exist: " + path, "ONNX::ImportNet");

  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  onnx::ModelProto model;
  // Read the existing net.
  fstream input(path, ios::in | ios::binary);
  if (!model.ParseFromIstream(&input))
  {
    cerr << "Failed to parse model. Returning nullptr" << endl;
    input.close();
    return nullptr;
  }
  input.close();
  return build_net_onnx(model, input_shape, mem, log_level);
}

// Imports a net from a pointer passed as argument
Net *import_net_from_onnx_pointer(void *serialized_model, size_t size, int mem)
{
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  onnx::ModelProto model;
  if (!model.ParseFromArray(serialized_model, size))
  {
    cerr << "Failed to parse model. Returning nullptr" << endl;
    return nullptr;
  }
  return build_net_onnx(model, {}, mem, LOG_LEVEL::INFO);
}

// Imports a net from a c++ string passed as argument.
Net *import_net_from_onnx_string(string *model_string, int mem)
{
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  onnx::ModelProto model;
  if (!model.ParseFromString(*model_string))
  {
    cerr << "Failed to parse model. Returning nullptr" << endl;
    return nullptr;
  }
  return build_net_onnx(model, {}, mem, LOG_LEVEL::INFO);
}


// -----------------------------------
// Functions for distributed training 
// -----------------------------------

// Sets the weights of a input Net to the ones stored in the onnx net inside the pointer
void set_weights_from_onnx_pointer(Net *net, void *ptr_model, size_t model_size)
{
  onnx::ModelProto model_proto;
  if (!model_proto.ParseFromArray(ptr_model, model_size))
    cerr << "Failed to parse model." << endl;

  set_weights_from_model_proto(net, model_proto);
}

// Sets the weights of a input Net to the ones stored in the onnx net inside the c++ string
void set_weights_from_onnx(Net *net, std::string *model_string)
{
  onnx::ModelProto model_proto;
  if (!model_proto.ParseFromString(*model_string))
    cerr << "Failed to parse model." << endl;

  set_weights_from_model_proto(net, model_proto);
}

// Accumulates the gradients stored in the pointer to the input net
void apply_grads_from_onnx_pointer(Net *net, void *ptr_onnx, size_t count)
{
  onnx::ModelProto model_proto;
  if (!model_proto.ParseFromArray(ptr_onnx, count))
    cerr << "Failed to parse model." << endl;

  apply_grads_from_model_proto(net, model_proto);
}

// Accumulates the gradients stored in the c++ string to the input net
void apply_grads_from_onnx(Net *net, std::string *model_string)
{
  onnx::ModelProto model_proto;
  if (!model_proto.ParseFromString(*model_string))
    cerr << "Failed to parse model." << endl;

  apply_grads_from_model_proto(net, model_proto);
}

#else // If protobuf is not enabled

Net *import_net_from_onnx_file(std::string path, int mem, LOG_LEVEL log_level)
{
  cerr << "Not compiled for ONNX. Missing protobuf. Returning nullptr" << endl;
  return nullptr;
}

Net *import_net_from_onnx_file(std::string path, vector<int> input_shape, int mem, LOG_LEVEL log_level)
{
  cerr << "Not compiled for ONNX. Missing protobuf. Returning nullptr" << endl;
  return nullptr;
}

Net *import_net_from_onnx_pointer(void *serialized_model, size_t model_size)
{
  cerr << "Not compiled for ONNX. Missing protobuf. Returning nullptr" << endl;
  return nullptr;
}

Net *import_net_from_onnx_string(std::string *model_string)
{
  cerr << "Not compiled for ONNX. Missing protobuf. Returning nullptr" << endl;
  return nullptr;
}

#endif // cPROTO
