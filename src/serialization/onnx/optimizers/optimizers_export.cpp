#include "eddl/optimizers/optim.h"
#include "eddl/serialization/onnx/eddl_onnx.h"

using namespace std;

#if defined(cPROTO)
#include "eddl/serialization/onnx/onnx.pb.h"
#endif

#ifdef cPROTO

// Optimizers builders
//--------------------------------------------------------------------------------------------

void prepare_nodeproto_from_adam(onnx::NodeProto *node_opt, Adam *optimizer) {
  node_opt->set_name("Optimizer");
  node_opt->set_op_type("Adam");

  /*
   * Store optimizer attributes
   */
  onnx::AttributeProto *lr_attr = node_opt->add_attribute();
  lr_attr->set_name("learning_rate");
  lr_attr->set_type(onnx::AttributeProto::FLOAT);
  lr_attr->set_f(optimizer->lr);

  onnx::AttributeProto *beta_1_attr = node_opt->add_attribute();
  beta_1_attr->set_name("beta_1");
  beta_1_attr->set_type(onnx::AttributeProto::FLOAT);
  beta_1_attr->set_f(optimizer->beta_1);

  onnx::AttributeProto *beta_2_attr = node_opt->add_attribute();
  beta_2_attr->set_name("beta_2");
  beta_2_attr->set_type(onnx::AttributeProto::FLOAT);
  beta_2_attr->set_f(optimizer->beta_2);

  onnx::AttributeProto *epsilon_attr = node_opt->add_attribute();
  epsilon_attr->set_name("epsilon");
  epsilon_attr->set_type(onnx::AttributeProto::FLOAT);
  epsilon_attr->set_f(optimizer->epsilon);

  onnx::AttributeProto *weight_decay_attr = node_opt->add_attribute();
  weight_decay_attr->set_name("weight_decay");
  weight_decay_attr->set_type(onnx::AttributeProto::FLOAT);
  weight_decay_attr->set_f(optimizer->weight_decay);

  onnx::AttributeProto *amsgrad_attr = node_opt->add_attribute();
  amsgrad_attr->set_name("amsgrad");
  amsgrad_attr->set_type(onnx::AttributeProto::INT);
  amsgrad_attr->set_i((int)optimizer->amsgrad);
}

void prepare_nodeproto_from_sgd(onnx::NodeProto *node_opt, SGD *optimizer) {
  node_opt->set_name("Optimizer");
  node_opt->set_op_type("SGD");

  /*
   * Store optimizer attributes
   */
  onnx::AttributeProto *lr_attr = node_opt->add_attribute();
  lr_attr->set_name("learning_rate");
  lr_attr->set_type(onnx::AttributeProto::FLOAT);
  lr_attr->set_f(optimizer->lr);

  onnx::AttributeProto *momentum_attr = node_opt->add_attribute();
  momentum_attr->set_name("momentum");
  momentum_attr->set_type(onnx::AttributeProto::FLOAT);
  momentum_attr->set_f(optimizer->mu);

  onnx::AttributeProto *weight_decay_attr = node_opt->add_attribute();
  weight_decay_attr->set_name("weight_decay");
  weight_decay_attr->set_type(onnx::AttributeProto::FLOAT);
  weight_decay_attr->set_f(optimizer->weight_decay);

  onnx::AttributeProto *nesterov_attr = node_opt->add_attribute();
  nesterov_attr->set_name("nesterov");
  nesterov_attr->set_type(onnx::AttributeProto::INT);
  nesterov_attr->set_i((int)optimizer->nesterov);
}

void prepare_nodeproto_from_rmsprop(onnx::NodeProto *node_opt, RMSProp *optimizer) {
  node_opt->set_name("Optimizer");
  node_opt->set_op_type("RMSProp");

  /*
   * Store optimizer attributes
   */
  onnx::AttributeProto *lr_attr = node_opt->add_attribute();
  lr_attr->set_name("learning_rate");
  lr_attr->set_type(onnx::AttributeProto::FLOAT);
  lr_attr->set_f(optimizer->lr);

  onnx::AttributeProto *rho_attr = node_opt->add_attribute();
  rho_attr->set_name("rho");
  rho_attr->set_type(onnx::AttributeProto::FLOAT);
  rho_attr->set_f(optimizer->rho);

  onnx::AttributeProto *epsilon_attr = node_opt->add_attribute();
  epsilon_attr->set_name("epsilon");
  epsilon_attr->set_type(onnx::AttributeProto::FLOAT);
  epsilon_attr->set_f(optimizer->epsilon);

  onnx::AttributeProto *weight_decay_attr = node_opt->add_attribute();
  weight_decay_attr->set_name("weight_decay");
  weight_decay_attr->set_type(onnx::AttributeProto::FLOAT);
  weight_decay_attr->set_f(optimizer->weight_decay);
}

// Helper functions
//--------------------------------------------------------------------------------------------

void build_nodeproto_from_optimizer(onnx::NodeProto *node_opt, Optimizer *optimizer) {
  if (Adam *aux_opt = dynamic_cast<Adam *>(optimizer))
    prepare_nodeproto_from_adam(node_opt, aux_opt);
  else if (SGD *aux_opt = dynamic_cast<SGD *>(optimizer))
    prepare_nodeproto_from_sgd(node_opt, aux_opt);
  else if (RMSProp *aux_opt = dynamic_cast<RMSProp *>(optimizer))
    prepare_nodeproto_from_rmsprop(node_opt, aux_opt);
  else
    msg("The optimizer type can't be recognized.", 
        "ONNX::get_optimizer_type_name");
}

onnx::NodeProto serialize_optimizer(Optimizer *optimizer) {
  // We save the optimizer data in a NodeProto object from ONNX
  onnx::NodeProto node_opt = onnx::NodeProto();

  build_nodeproto_from_optimizer(&node_opt, optimizer);

  return node_opt;
}

// Main functions for API
//--------------------------------------------------------------------------------------------

void save_optimizer_to_onnx_file(Optimizer *optimizer, string path) {
  // Check if the folder exists
  string folder = path.substr(0, path.find_last_of("\\/"));
  if (folder != path && !pathExists(folder))
    msg("The file could not be saved. Check if the directory exists or if you "
        "have permissions to write in it.",
        "ONNX::save_optimizer_to_onnx_file");

  onnx::NodeProto node_opt = serialize_optimizer(optimizer); // create the NodeProto with the optimizer data

  // Create the file stream and save the serialization of the optimizer in it
  fstream ofs(path, ios::out | ios::trunc | ios::binary);
  if (!node_opt.SerializeToOstream(&ofs)) // The serialization is automated by the protobuf library
    msg("Failed to write the serialized optimizer into the file.",
        "ONNX::save_optimizer_to_onnx_file");

  ofs.close();
}

size_t serialize_optimizer_to_onnx_pointer(Optimizer *optimizer, void *&serialized_optimizer) {
  onnx::NodeProto node_opt = serialize_optimizer(optimizer); // create the NodeProto with the optimizer data

  // Serialization of the optimizer to an array of bytes
  size_t size = node_opt.ByteSizeLong(); // Get the size of the serialized model
  serialized_optimizer = new char[size];
  memset(serialized_optimizer, 0, size);
  if (!node_opt.SerializeToArray(serialized_optimizer, size))
    msg("Failed to write the serialized optimizer into the buffer.",
        "ONNX::save_optimizer_to_onnx_pointer");

  return size;
}

string *serialize_optimizer_to_onnx_string(Optimizer *optimizer) {
  onnx::NodeProto node_opt = serialize_optimizer(optimizer); // create the NodeProto with the optimizer data

  // Serialization of the optimizer to a string object
  string *opt_string = new string();
  if (!node_opt.SerializeToString(opt_string))
    msg("Failed to write the serialized optimizer into the string.",
        "ONNX::save_optimizer_to_onnx_string");

  return opt_string;
}

#else

void save_optimizer_to_onnx_file(Optimizer *optimizer, string path) {
  cerr << "Not compiled for ONNX. Missing Protobuf. The optimizer is not going to be exported to file" << endl;
}

size_t serialize_optimizer_to_onnx_pointer(Optimizer *optimizer, void *&serialized_optimizer) {
  cerr << "Not compiled for ONNX. Missing Protobuf. Returning -1" << endl;
  return -1;
}

string *serialize_optimizer_to_onnx_string(Optimizer *optimizer) {
  cerr << "Not compiled for ONNX. Missing Protobuf. Returning nullptr" << endl;
  return nullptr;
}

#endif //cPROTO
