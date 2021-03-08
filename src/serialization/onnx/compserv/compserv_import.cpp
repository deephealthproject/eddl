#include "eddl/net/compserv.h"
#include "eddl/serialization/onnx/eddl_onnx.h"

using namespace std;

#if defined(cPROTO)
#include "eddl/serialization/onnx/onnx.pb.h"
#endif

#ifdef cPROTO

// CompServ builders
//--------------------------------------------------------------------------------------------

CompServ *get_compserv_from_nodeproto(onnx::NodeProto *node_cs) {
  // initialize the compserv params and the found check variables
  //   Note: We require all the params to be in the node
  int threads; bool threads_found = false;
  vector<int> gpus; bool gpus_found = false;
  vector<int> fpgas; bool fpgas_found = false;
  int lsb; bool lsb_found = false;
  int mem; bool mem_found = false;

  // Collect all the node attributes
  for (int i = 0; i < node_cs->attribute_size(); i++) {
    onnx::AttributeProto attribute = node_cs->attribute(i);
    string attr_name = attribute.name();
    if (!attr_name.compare("local_threads")) {
      threads = attribute.i();
      threads_found = true;
    } else if (!attr_name.compare("local_gpus")) {
      // Read the list of gpus
      for (int j = 0; j < attribute.ints_size(); ++j)
        gpus.push_back(attribute.ints(j));
      gpus_found = true;
    } else if (!attr_name.compare("local_fpgas")) {
      // Read the list of fpgas
      for (int j = 0; j < attribute.ints_size(); ++j)
        fpgas.push_back(attribute.ints(j));
      fpgas_found = true;
    } else if (!attr_name.compare("lsb")) {
      lsb = attribute.i();
      lsb_found = true;
    } else if (!attr_name.compare("mem")) {
      mem = attribute.i();
      mem_found = true;
    }
  }

  // Check that we got all the parameters
  if (!threads_found)
    msg("\"local_threads\" was not found.", "ONNX::get_compserv_from_nodeproto");
  else if (!gpus_found)
    msg("\"local_gpus\" was not found.", "ONNX::get_compserv_from_nodeproto");
  else if (!fpgas_found)
    msg("\"local_fpgas\" was not found.", "ONNX::get_compserv_from_nodeproto");
  else if (!lsb_found)
    msg("\"lsb\" was not found.", "ONNX::get_compserv_from_nodeproto");
  else if (!mem_found)
    msg("\"mem\" was not found.", "ONNX::get_compserv_from_nodeproto");
  
  return new CompServ(threads, gpus, fpgas, lsb, mem);
}

// Helper functions
//--------------------------------------------------------------------------------------------

CompServ *build_compserv_from_node(onnx::NodeProto *node_cs) {
  string node_name = node_cs->name();

  // Check that the node is a compserv
  if (node_name.compare("CompServ"))
    msg("The node provided is not an \"CompServ\". Is \"" + node_name + "\".",
        "ONNX::build_compserv_from_node");

  return get_compserv_from_nodeproto(node_cs);
}

// Main functions for API
//--------------------------------------------------------------------------------------------

CompServ *import_compserv_from_onnx_file(string path) {
  // Check if the path exists
  if (!pathExists(path))
    msg("The specified path does not exist: " + path, 
        "ONNX::import_compserv_from_onnx_file");

  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  onnx::NodeProto node_cs;
  // Read the serialized compserv from file stream
  fstream input(path, ios::in | ios::binary);
  if (!node_cs.ParseFromIstream(&input))
    msg("Failed to parse the optimizer.", 
        "ONNX::import_compserv_from_onnx_file");
  input.close();

  return build_compserv_from_node(&node_cs);
}

CompServ *import_compserv_from_onnx_pointer(void *serialized_cs, size_t cs_size) {
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  onnx::NodeProto node_cs;
  if (!node_cs.ParseFromArray(serialized_cs, cs_size))
    msg("Failed to parse the optimizer.", 
        "ONNX::import_compserv_from_onnx_pointer");

  return build_compserv_from_node(&node_cs);
}

CompServ *import_compserv_from_onnx_string(string *cs_string) {
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  onnx::NodeProto node_cs;
  if (!node_cs.ParseFromString(*cs_string))
    msg("Failed to parse the optimizer.", 
        "ONNX::import_compserv_from_onnx_string");

  return build_compserv_from_node(&node_cs);
}

#else

CompServ *import_compserv_from_onnx_file(string path) {
  cerr << "Not compiled for ONNX. Missing protobuf. Returning nullptr" << endl;
  return nullptr;
}

CompServ *import_compserv_from_onnx_pointer(void *serialized_cs, size_t cs_size) {
  cerr << "Not compiled for ONNX. Missing protobuf. Returning nullptr" << endl;
  return nullptr;
}

CompServ *import_compserv_from_onnx_string(string *cs_string) {
  cerr << "Not compiled for ONNX. Missing protobuf. Returning nullptr" << endl;
  return nullptr;
}

#endif // cPROTO
