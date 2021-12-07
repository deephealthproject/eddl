#include <cstdio>
#include <fstream>

#include "eddl/net/net.h"

#include "eddl/serialization/onnx/eddl_onnx.h"
#include "eddl/serialization/onnx/utils_onnx.h"
#include "eddl/serialization/onnx/export_helpers.h"

using namespace std;

#if defined(cPROTO)
#include "eddl/serialization/onnx/onnx.pb.h"

void save_net_to_onnx_file(Net *net, string path)
{
  // Check if the folder exists
  string folder = path.substr(0, path.find_last_of("\\/"));
  if (folder != path && !pathExists(folder))
  {
    msg("The file could not be saved. Check if the directory exists or if you have permissions to write in it.", "ONNX::ExportNet");
  }

  collect_params(net); // sync weights from device

  bool export_gradients = false; // We always store weights to file
  onnx::ModelProto model = build_onnx_model(net, export_gradients);
  // Create the file stream and save the serialization of the onnx model in it
  fstream ofs(path, ios::out | ios::binary);
  if (!model.SerializeToOstream(&ofs))
  { // The serialization is automated by the protobuf library
    cerr << "Failed to write the model in onnx." << endl;
  }
  ofs.close();
}

size_t serialize_net_to_onnx_pointer(Net *net, void *&serialized_model, bool gradients)
{
  collect_params(net); // sync weights from device
  if (gradients && net->snets[0]->dev != DEV_CPU)
      net->collect_acc_grads();

  onnx::ModelProto model = build_onnx_model(net, gradients);
  // Serialization of the model to an array of bytes
  size_t size = model.ByteSizeLong(); // Get the size of the serialized model
  serialized_model = new char[size];
  memset(serialized_model, 0, size);
  if (!model.SerializeToArray(serialized_model, size))
  {
    cerr << "Failed to serialize the model in onnx into the buffer." << endl;
  }
  return size;
}

string *serialize_net_to_onnx_string(Net *net, bool gradients)
{
  collect_params(net); // sync weights from device
  if (gradients && net->snets[0]->dev != DEV_CPU)
      net->collect_acc_grads();

  onnx::ModelProto model = build_onnx_model(net, gradients);
  // Serialization of the model to an array of bytes
  string *model_string = new string();
  if (!model.SerializeToString(model_string))
  {
    cerr << "Failed to serialize the model in onnx into a string ." << endl;
  }
  return model_string;
}

#else // In case of compiling without protobuf

void save_net_to_onnx_file(Net *net, string path)
{
  cerr << "Not compiled for ONNX. Missing Protobuf. The net is not going to be exported to ONNX." << endl;
}

size_t serialize_net_to_onnx_pointer(Net *net, void *&serialized_model, bool gradients)
{
  cerr << "Not compiled for ONNX. Missing Protobuf. Returning -1" << endl;
  return -1;
}

std::string *serialize_net_to_onnx_string(Net *net, bool gradients)
{
  cerr << "Not compiled for ONNX. Missing Protobuf. Returning nullptr" << endl;
  return nullptr;
}

#endif //cPROTO
