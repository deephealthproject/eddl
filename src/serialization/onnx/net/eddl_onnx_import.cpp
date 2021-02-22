#include <queue>
#include <fstream>
#include <map>
#include <set>
#include <algorithm>

#include "eddl/serialization/onnx/eddl_onnx.h"
#include "eddl/serialization/onnx/utils_onnx.h"
#include "eddl/serialization/onnx/import_helpers.h"

#include "eddl/layers/core/layer_core.h"
#include "eddl/layers/conv/layer_conv.h"

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
  // TODO: Change implementation and use a generic one changing the params vectors
  onnx::ModelProto model;
  if (!model.ParseFromArray(ptr_model, model_size))
    cerr << "Failed to parse model." << endl;

  map<string, vector<Tensor *>> tensors = get_tensors_from_onnx(model);
  LConv *conv;
  LDense *dense;
  for (Layer *l : net->layers)
  {
    if (!tensors.count(l->name))
    {
      //cout << "Layer with name " << l->name << " is not trainable " << endl;
      continue;
    }
    vector<Tensor *> layer_tensors = tensors[l->name];
    if ((conv = dynamic_cast<LConv *>(l)))
    {
      if (layer_tensors.size() > 1)
        conv->update_weights(layer_tensors[0], layer_tensors[1]);
      else
      {
        cerr << "EDDL has not implemented convolutional without bias " << endl;
        //conv.update_weights(layer_tensors[0]);
      }
    }
    else if ((dense = dynamic_cast<LDense *>(l)))
    {
      if (layer_tensors.size() > 1)
        dense->update_weights(layer_tensors[0], layer_tensors[1]);
      else
        dense->update_weights(layer_tensors[0]);
    }
    else
      cerr << "not implemented layer type" << endl;
  }

  // copy the new weights to devices
  share_weights(net);

  // erase the map we used to free the memory
  map<string, vector<Tensor *>>::iterator it;
  vector<Tensor *> delete_tensors;
  for (it = tensors.begin(); it != tensors.end(); ++it)
  {
    delete_tensors = it->second;
    for (int i = 0; i < delete_tensors.size(); ++i)
    {
      delete delete_tensors[i];
    }
  }
}

// Sets the weights of a input Net to the ones stored in the onnx net inside the c++ string
void set_weights_from_onnx(Net *net, std::string *model_string)
{
  // TODO: Change implementation and use a generic one changing the params vectors
  onnx::ModelProto model;
  if (!model.ParseFromString(*model_string))
  {
    cerr << "Failed to parse model." << endl;
  }

  map<string, vector<Tensor *>> tensors = get_tensors_from_onnx(model);
  LConv *conv;
  LDense *dense;
  for (Layer *l : net->layers)
  {
    if (!tensors.count(l->name))
    {
      //cout << "Layer with name " << l->name << " is not trainable " << endl;
      continue;
    }
    vector<Tensor *> layer_tensors = tensors[l->name];
    if ((conv = dynamic_cast<LConv *>(l)))
    {
      if (layer_tensors.size() > 1)
        conv->update_weights(layer_tensors[0], layer_tensors[1]);
      else
      {
        cerr << "EDDL has not implemented convolutional without bias " << endl;
        //conv.update_weights(layer_tensors[0]);
      }
    }
    else if ((dense = dynamic_cast<LDense *>(l)))
    {
      if (layer_tensors.size() > 1)
        dense->update_weights(layer_tensors[0], layer_tensors[1]);
      else
        dense->update_weights(layer_tensors[0]);
    }
    else
      cerr << "not implemented layer type" << endl;
  }

  // copy the new weights to devices
  share_weights(net);

  // erase the map we used to free the memory
  map<string, vector<Tensor *>>::iterator it;
  vector<Tensor *> delete_tensors;
  for (it = tensors.begin(); it != tensors.end(); ++it)
  {
    delete_tensors = it->second;
    for (int i = 0; i < delete_tensors.size(); ++i)
    {
      delete delete_tensors[i];
    }
  }
}

// Accumulates the gradients stored in the pointer to the input net
void apply_grads_from_onnx_pointer(Net *net, void *ptr_onnx, size_t count)
{
  // TODO: Change implementation and use a generic one adding the grads to the params vectors
  onnx::ModelProto model;
  if (!model.ParseFromArray(ptr_onnx, count))
  {
    cerr << "Failed to parse model." << endl;
  }

  map<string, vector<Tensor *>> tensors = get_tensors_from_onnx(model);
  LConv *conv;
  LDense *dense;
  for (Layer *l : net->layers)
  {
    if (!tensors.count(l->name))
    {
      continue;
    }
    vector<Tensor *> layer_tensors = tensors[l->name];
    if ((conv = dynamic_cast<LConv *>(l)))
    {
      if (layer_tensors.size() > 1)
      {
        conv->accumulate_accumulated_gradients(layer_tensors[0], layer_tensors[1]);
      }
      else
      {
        cerr << "EDDL has not implemented convolutional without bias." << endl;
        //conv.update_weights(layer_tensors[0]);
      }
    }
    else if ((dense = dynamic_cast<LDense *>(l)))
    {
      if (layer_tensors.size() > 1)
      {
        dense->accumulate_accumulated_gradients(layer_tensors[0], layer_tensors[1]);
      }
      else
      {
        dense->accumulate_accumulated_gradients(layer_tensors[0]);
      }
    }
    else
      cerr << "not implemented layer type" << endl;
  }
  // erase the map we used to free the memory
  map<string, vector<Tensor *>>::iterator it;
  vector<Tensor *> delete_tensors;
  for (it = tensors.begin(); it != tensors.end(); ++it)
  {
    delete_tensors = it->second;
    for (int i = 0; i < delete_tensors.size(); ++i)
    {
      delete delete_tensors[i];
    }
  }

  // copy the new weights to devices
  share_weights(net);
}

// Accumulates the gradients stored in the c++ string to the input net
void apply_grads_from_onnx(Net *net, std::string *model_string)
{
  // TODO: Change implementation and use a generic one adding the grads to the params vectors
  onnx::ModelProto model;
  if (!model.ParseFromString(*model_string))
  {
    cerr << "Failed to parse model." << endl;
  }

  map<string, vector<Tensor *>> tensors = get_tensors_from_onnx(model);
  LConv *conv;
  LDense *dense;
  for (Layer *l : net->layers)
  {
    if (!tensors.count(l->name))
      continue;
    vector<Tensor *> layer_tensors = tensors[l->name];
    if ((conv = dynamic_cast<LConv *>(l)))
    {
      if (layer_tensors.size() > 1)
        conv->accumulate_accumulated_gradients(layer_tensors[0], layer_tensors[1]);
      else
      {
        cerr << "EDDL has not implemented convolutional without bias " << endl;
        //conv.update_weights(layer_tensors[0]);
      }
    }
    else if ((dense = dynamic_cast<LDense *>(l)))
    {
      if (layer_tensors.size() > 1)
        dense->accumulate_accumulated_gradients(layer_tensors[0], layer_tensors[1]);
      else
        dense->accumulate_accumulated_gradients(layer_tensors[0]);
    }
    else
      cerr << "not implemented layer type" << endl;
  }
  // erase the map we used to free the memory
  map<string, vector<Tensor *>>::iterator it;
  vector<Tensor *> delete_tensors;
  for (it = tensors.begin(); it != tensors.end(); ++it)
  {
    delete_tensors = it->second;
    for (int i = 0; i < delete_tensors.size(); ++i)
    {
      delete delete_tensors[i];
    }
  }

  // copy the new weights to devices
  share_weights(net);
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
