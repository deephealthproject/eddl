#include "eddl/optimizers/optim.h"
#include "eddl/serialization/onnx/eddl_onnx.h"
#include "eddl/utils.h"

using namespace std;

#if defined(cPROTO)
#include "eddl/serialization/onnx/onnx.pb.h"
#endif

#ifdef cPROTO

// Optimizers builders
//--------------------------------------------------------------------------------------------

Adam *get_adam_from_nodeproto(onnx::NodeProto *node_opt) {
  // initialize the optimizer params and the found check variables
  //   Note: We require all the params to be in the node
  float lr; bool lr_found = false;
  float beta_1; bool beta_1_found = false;
  float beta_2; bool beta_2_found = false;
  float epsilon; bool epsilon_found = false;
  float weight_decay; bool weight_decay_found = false;
  bool amsgrad; bool amsgrad_found = false;

  // Collect all the node attributes
  for (int i = 0; i < node_opt->attribute_size(); i++) {
    onnx::AttributeProto attribute = node_opt->attribute(i);
    string attr_name = attribute.name();
    if (!attr_name.compare("learning_rate")) {
      lr = attribute.f();
      lr_found = true;
    } else if (!attr_name.compare("beta_1")) {
      beta_1 = attribute.f();
      beta_1_found = true;
    } else if (!attr_name.compare("beta_2")) {
      beta_2 = attribute.f();
      beta_2_found = true;
    } else if (!attr_name.compare("epsilon")) {
      epsilon = attribute.f();
      epsilon_found = true;
    } else if (!attr_name.compare("weight_decay")) {
      weight_decay = attribute.f();
      weight_decay_found = true;
    } else if (!attr_name.compare("amsgrad")) {
      amsgrad = (bool)attribute.i();
      amsgrad_found = true;
    }
  }

  // Check that we got all the parameters
  if (!lr_found)
    msg("\"learning_rate\" was not found.", "ONNX::get_adam_from_nodeproto");
  else if (!beta_1_found)
    msg("\"beta_1\" was not found.", "ONNX::get_adam_from_nodeproto");
  else if (!beta_2_found)
    msg("\"beta_2\" was not found.", "ONNX::get_adam_from_nodeproto");
  else if (!epsilon_found)
    msg("\"epsilon\" was not found.", "ONNX::get_adam_from_nodeproto");
  else if (!weight_decay_found)
    msg("\"weight_decay\" was not found.", "ONNX::get_adam_from_nodeproto");
  else if (!amsgrad_found)
    msg("\"amsgrad\" was not found.", "ONNX::get_adam_from_nodeproto");

  return new Adam(lr, beta_1, beta_2, epsilon, weight_decay, amsgrad);
}

SGD *get_sgd_from_nodeproto(onnx::NodeProto *node_opt) {
  // initialize the optimizer params and the found check variables
  //   Note: We require all the params to be in the node
  float lr; bool lr_found = false;
  float momentum; bool momentum_found = false;
  float weight_decay; bool weight_decay_found = false;
  bool nesterov; bool nesterov_found = false;

  // Collect all the node attributes
  for (int i = 0; i < node_opt->attribute_size(); i++) {
    onnx::AttributeProto attribute = node_opt->attribute(i);
    string attr_name = attribute.name();
    if (!attr_name.compare("learning_rate")) {
      lr = attribute.f();
      lr_found = true;
    } else if (!attr_name.compare("momentum")) {
      momentum = attribute.f();
      momentum_found = true;
    } else if (!attr_name.compare("weight_decay")) {
      weight_decay = attribute.f();
      weight_decay_found = true;
    } else if (!attr_name.compare("nesterov")) {
      nesterov = (bool)attribute.i();
      nesterov_found = true;
    }
  }

  // Check that we got all the parameters
  if (!lr_found)
    msg("\"learning_rate\" was not found.", "ONNX::get_sgd_from_nodeproto");
  else if (!momentum_found)
    msg("\"momentum\" was not found.", "ONNX::get_sgd_from_nodeproto");
  else if (!weight_decay_found)
    msg("\"weight_decay\" was not found.", "ONNX::get_sgd_from_nodeproto");
  else if (!nesterov_found)
    msg("\"nesterov\" was not found.", "ONNX::get_sgd_from_nodeproto");

  return new SGD(lr, momentum, weight_decay, nesterov);
}

RMSProp *get_rmsprop_from_nodeproto(onnx::NodeProto *node_opt) {
  // initialize the optimizer params and the found check variables
  //   Note: We require all the params to be in the node
  float lr; bool lr_found = false;
  float rho; bool rho_found = false;
  float epsilon; bool epsilon_found = false;
  float weight_decay; bool weight_decay_found = false;

  // Collect all the node attributes
  for (int i = 0; i < node_opt->attribute_size(); i++) {
    onnx::AttributeProto attribute = node_opt->attribute(i);
    string attr_name = attribute.name();
    if (!attr_name.compare("learning_rate")) {
      lr = attribute.f();
      lr_found = true;
    } else if (!attr_name.compare("rho")) {
      rho = attribute.f();
      rho_found = true;
    } else if (!attr_name.compare("epsilon")) {
      epsilon = attribute.f();
      epsilon_found = true;
    } else if (!attr_name.compare("weight_decay")) {
      weight_decay = attribute.f();
      weight_decay_found = true;
    }
  }

  // Check that we got all the parameters
  if (!lr_found)
    msg("\"learning_rate\" was not found.", "ONNX::get_rmsprop_from_nodeproto");
  else if (!rho_found)
    msg("\"rho\" was not found.", "ONNX::get_rmsprop_from_nodeproto");
  else if (!epsilon_found)
    msg("\"epsilon\" was not found.", "ONNX::get_rmsprop_from_nodeproto");
  else if (!weight_decay_found)
    msg("\"weight_decay\" was not found.", "ONNX::get_rmsprop_from_nodeproto");

  return new RMSProp(lr, rho, epsilon, weight_decay);
}

// Helper functions
//--------------------------------------------------------------------------------------------

Optimizer *build_optimizer_from_node(onnx::NodeProto *node_opt) {
  string node_name = node_opt->name();
  string node_type = node_opt->op_type();

  // Check that the node is an optimizer
  if (node_name.compare("Optimizer"))
    msg("The node provided is not an \"Optimizer\". Is \"" + node_name + "\".",
        "ONNX::build_optimizer_from_node");

  // Detect the optimizer type, build it, and return it
  if (!node_type.compare("Adam"))
    return get_adam_from_nodeproto(node_opt);
  else if (!node_type.compare("SGD"))
    return get_sgd_from_nodeproto(node_opt);
  else if (!node_type.compare("RMSProp"))
    return get_rmsprop_from_nodeproto(node_opt);
  else {
    msg("The optimizer type \"" + node_type + "\" is not valid",
        "ONNX::build_optimizer_from_node");
    return nullptr;
  }
}

// Main functions for API
//--------------------------------------------------------------------------------------------

Optimizer *import_optimizer_from_onnx_file(string path) {
  // Check if the path exists
  if (!pathExists(path))
    msg("The specified path does not exist: " + path, 
        "ONNX::import_optimizer_from_onnx_file");

  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  onnx::NodeProto node_opt;
  // Read the serialized optimizer from file stream
  fstream input(path, ios::in | ios::binary);
  if (!node_opt.ParseFromIstream(&input))
    msg("Failed to parse the optimizer.", 
        "ONNX::import_optimizer_from_onnx_file");
  input.close();

  return build_optimizer_from_node(&node_opt);
}

Optimizer *import_optimizer_from_onnx_pointer(void *serialized_optimizer, size_t optimizer_size) {
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  onnx::NodeProto node_opt;
  if (!node_opt.ParseFromArray(serialized_optimizer, optimizer_size))
    msg("Failed to parse the optimizer.", 
        "ONNX::import_optimizer_from_onnx_pointer");

  return build_optimizer_from_node(&node_opt);
}

Optimizer *import_optimizer_from_onnx_string(string *optimizer_string) {
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  onnx::NodeProto node_opt;
  if (!node_opt.ParseFromString(*optimizer_string))
    msg("Failed to parse the optimizer.", 
        "ONNX::import_optimizer_from_onnx_string");

  return build_optimizer_from_node(&node_opt);
}

#else

Optimizer *import_optimizer_from_onnx_file(string path) {
  cerr << "Not compiled for ONNX. Missing protobuf. Returning nullptr" << endl;
  return nullptr;
}

Optimizer *import_optimizer_from_onnx_pointer(void *serialized_optimizer, size_t optimizer_size) {
  cerr << "Not compiled for ONNX. Missing protobuf. Returning nullptr" << endl;
  return nullptr;
}

Optimizer *import_optimizer_from_onnx_string(string *optimizer_string) {
  cerr << "Not compiled for ONNX. Missing protobuf. Returning nullptr" << endl;
  return nullptr;
}

#endif // cPROTO
