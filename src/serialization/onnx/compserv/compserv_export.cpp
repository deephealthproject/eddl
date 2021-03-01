#include "eddl/net/compserv.h"
#include "eddl/serialization/onnx/eddl_onnx.h"

using namespace std;

#if defined(cPROTO)
#include "eddl/serialization/onnx/onnx.pb.h"
#endif

#ifdef cPROTO

// Helper functions
//--------------------------------------------------------------------------------------------

void build_nodeproto_from_compserv(onnx::NodeProto *node_cs, CompServ *cs) {
  node_cs->set_name("CompServ");
  node_cs->set_op_type(cs->type);

  /*
   * Store compserv attributes
   */
  onnx::AttributeProto *threads_attr = node_cs->add_attribute();
  threads_attr->set_name("local_threads");
  threads_attr->set_type(onnx::AttributeProto::INT);
  threads_attr->set_i(cs->threads_arg);

  onnx::AttributeProto *gpus_attr = node_cs->add_attribute();
  gpus_attr->set_name("local_gpus");
  gpus_attr->set_type(onnx::AttributeProto::INTS);
  for (int i : cs->local_gpus)
    gpus_attr->add_ints(i);

  onnx::AttributeProto *fpgas_attr = node_cs->add_attribute();
  fpgas_attr->set_name("local_fpgas");
  fpgas_attr->set_type(onnx::AttributeProto::INTS);
  for (int i : cs->local_fpgas)
    fpgas_attr->add_ints(i);

  // Local Sync Batches
  onnx::AttributeProto *lsb_attr = node_cs->add_attribute();
  lsb_attr->set_name("lsb");
  lsb_attr->set_type(onnx::AttributeProto::INT);
  lsb_attr->set_i(cs->lsb);

  // Mem level
  onnx::AttributeProto *mem_attr = node_cs->add_attribute();
  mem_attr->set_name("mem");
  mem_attr->set_type(onnx::AttributeProto::INT);
  mem_attr->set_i(cs->mem_level);
}

onnx::NodeProto serialize_compserv(CompServ *cs) {
  // We save the compserv data in a NodeProto object from ONNX
  onnx::NodeProto node_cs = onnx::NodeProto();

  build_nodeproto_from_compserv(&node_cs, cs);

  return node_cs;
}

// Main functions for API
//--------------------------------------------------------------------------------------------

void save_compserv_to_onnx_file(CompServ *cs, string path) {
  // Check if the folder exists
  string folder = path.substr(0, path.find_last_of("\\/"));
  if (folder != path && !pathExists(folder))
    msg("The file could not be saved. Check if the directory exists or if you "
        "have permissions to write in it.",
        "ONNX::save_compserv_to_onnx_file");

  onnx::NodeProto node_cs = serialize_compserv(cs); // create the NodeProto with the compserv data

  // Create the file stream and save the serialization of the compserv in it
  fstream ofs(path, ios::out | ios::trunc | ios::binary);
  if (!node_cs.SerializeToOstream(&ofs)) // The serialization is automated by the protobuf library
    msg("Failed to write the serialized compserv into the file.",
        "ONNX::save_compserv_to_onnx_file");

  ofs.close();
}

size_t serialize_compserv_to_onnx_pointer(CompServ *cs, void *&serialized_cs) {
  onnx::NodeProto node_cs = serialize_compserv(cs); // create the NodeProto with the compserv data

  // Serialization of the compserv to an array of bytes
  size_t size = node_cs.ByteSizeLong();
  serialized_cs = new char[size];
  memset(serialized_cs, 0, size);
  if (!node_cs.SerializeToArray(serialized_cs, size))
    msg("Failed to write the serialized compserv into the buffer.",
        "ONNX::save_compserv_to_onnx_pointer");

  return size;
}

string *serialize_compserv_to_onnx_string(CompServ *cs) {
  onnx::NodeProto node_cs = serialize_compserv(cs); // create the NodeProto with the compserv data

  // Serialization of the compserv to a string object
  string *cs_string = new string();
  if (!node_cs.SerializeToString(cs_string))
    msg("Failed to write the serialized compserv into the string.",
        "ONNX::save_compserv_to_onnx_string");

  return cs_string;
}

#else

void save_compserv_to_onnx_file(CompServ *cs, string path) {
  cerr << "Not compiled for ONNX. Missing Protobuf. The CompServ is not going to be exported to file" << endl;
}

size_t serialize_compsev_to_onnx_pointer(CompServ *cs, void *&serialized_cs) {
  cerr << "Not compiled for ONNX. Missing Protobuf. Returning -1" << endl;
  return -1;
}

string *serialize_compserv_to_onnx_string(CompServ *cs) {
  cerr << "Not compiled for ONNX. Missing Protobuf. Returning nullptr" << endl;
  return nullptr;
}

#endif //cPROTO
