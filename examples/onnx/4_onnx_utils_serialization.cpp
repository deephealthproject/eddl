/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.1
* copyright (c) 2022, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: March 2022
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/apis/eddl.h"

#include "eddl/net/compserv.h"
#include "eddl/optimizers/optim.h"
#include "eddl/serialization/onnx/eddl_onnx.h"

using namespace eddl;

/////////////////////////////////////////
// 4_utils_serialization.cpp:          //
//   Export and import of serialized   //
//   objects like optimizers and       //
//   computing services using protobuf //
/////////////////////////////////////////

enum SERIALIZE { ONNX_FILE, PTR, STRING };

void optimizer_info(optimizer opt) {
  if (Adam *aux_opt = dynamic_cast<Adam *>(opt)) {
    cout << "\n####### Adam #######\n";
    cout << "lr: " << aux_opt->lr << endl;
    cout << "beta_1: " << aux_opt->beta_1 << endl;
    cout << "beta_2: " << aux_opt->beta_2 << endl;
    cout << "epsilon: " << aux_opt->epsilon << endl;
    cout << "weight_decay: " << aux_opt->weight_decay << endl;
    cout << "amsgrad: " << aux_opt->amsgrad << endl;
    cout << "####################\n\n";
  } else if (SGD *aux_opt = dynamic_cast<SGD *>(opt)) {
    cout << "\n####### SGD #######\n";
    cout << "lr: " << aux_opt->lr << endl;
    cout << "momentum: " << aux_opt->mu << endl;
    cout << "weight_decay: " << aux_opt->weight_decay << endl;
    cout << "nesterov: " << aux_opt->nesterov << endl;
    cout << "###################\n\n";
  } else if (RMSProp *aux_opt = dynamic_cast<RMSProp *>(opt)) {
    cout << "\n####### RMSProp #######\n";
    cout << "lr: " << aux_opt->lr << endl;
    cout << "rho: " << aux_opt->rho << endl;
    cout << "epsilon: " << aux_opt->epsilon << endl;
    cout << "weight_decay: " << aux_opt->weight_decay << endl;
    cout << "#######################\n\n";
  } else {
    cout << "\n Optimizer type not valid!\n";
  }
}

void compserv_info(compserv cs) {
  cout << "\n####### Compserv #######\n";
  cout << "threads_arg: " << cs->threads_arg << endl;
  cout << "local_threads: " << cs->local_threads << endl;
  cout << "local_gpus: {";
  for (int i : cs->local_gpus)
    cout << i << ", ";
  cout << "}\n";
  cout << "local_fpgas: {";
  for (int i : cs->local_fpgas)
    cout << i << ", ";
  cout << "}\n";
  cout << "lsb: " << cs->lsb << endl;
  cout << "mem_level: " << cs->mem_level << endl;
  cout << "########################\n\n";
}

model create_model(int num_classes) {
  // Define network
  layer in = Input({784});
  layer l = in; // Aux var

  l = Activation(Dense(l, 128), "relu");
  l = Activation(Dense(l, 128), "relu");
  layer out = Softmax(Dense(l, num_classes));

  // Net define input and output layers list
  return Model({in}, {out}); // Build model
}

int main(int argc, char **argv) {
  // Download mnist dataset
  download_mnist();

  // Settings
  int epochs = 1;
  int batch_size = 100;
  int num_classes = 10;
  // Define how to serialize the optimizer and compserv
  int opt_serialization = SERIALIZE::PTR; // SERIALIZE::STRING;
  int cs_serialization = SERIALIZE::PTR; // SERIALIZE::STRING;

  optimizer opt = adam(0.001);
  //compserv cs = CS_GPU({1}, "full_mem");
  compserv cs = CS_CPU();
  optimizer_info(opt); // Print the optimizer data
  compserv_info(cs);   // Print compserv data

  model net = create_model(num_classes);

  build(net,
        opt,                       // Optimizer
        {"softmax_cross_entropy"}, // Losses
        {"categorical_accuracy"},  // Metrics
        cs                         // one GPU
  );

  // View model
  summary(net);

  // Load dataset
  Tensor *x_train = Tensor::load("mnist_trX.bin");
  Tensor *y_train = Tensor::load("mnist_trY.bin");
  Tensor *x_test = Tensor::load("mnist_tsX.bin");
  Tensor *y_test = Tensor::load("mnist_tsY.bin");

  // Preprocessing
  x_train->div_(255.0f);
  x_test->div_(255.0f);

  // Train model
  fit(net, {x_train}, {y_train}, batch_size, epochs);

  // Evaluate
  evaluate(net, {x_test}, {y_test}, batch_size);

  /*
   * Optimizer serialization (export and import)
   */
  optimizer opt2;
  switch (opt_serialization) {
  case SERIALIZE::ONNX_FILE: {
    save_optimizer_to_onnx_file(opt, "optimizer.onnx");
    cout << "\nSaved optmizer to protobuf file" << endl;

    opt2 = import_optimizer_from_onnx_file("optimizer.onnx");
    cout << "Optimizer imported!" << endl;
  } break;
  case SERIALIZE::PTR: {
    void *opt_ptr;
    size_t opt_size = serialize_optimizer_to_onnx_pointer(opt, opt_ptr);
    cout << "\nSaved optmizer to ptr" << endl;

    opt2 = import_optimizer_from_onnx_pointer(opt_ptr, opt_size);
    cout << "Optimizer imported!" << endl;
    delete [] (char *)opt_ptr; // check how memory is allocated in serialize_optimizer_to_onnx_pointer()
  } break;
  case SERIALIZE::STRING: {
    string *opt_str = serialize_optimizer_to_onnx_string(opt);
    cout << "\nSaved optmizer to string" << endl;

    opt2 = import_optimizer_from_onnx_string(opt_str);
    cout << "Optimizer imported!" << endl;
    delete opt_str;
  } break;
  default:
    cout << "\nThe optimizer serialization type is not valid!" << endl;
    return 1;
  }

  /*
   * Compserv serialization (export and import)
   */
  compserv cs2;
  switch (cs_serialization) {
  case SERIALIZE::ONNX_FILE: {
    save_compserv_to_onnx_file(cs, "compserv.onnx");
    cout << "\nSaved compsrv to protobuf file" << endl;

    cs2 = import_compserv_from_onnx_file("compserv.onnx");
    cout << "Compserv imported!" << endl;
  } break;
  case SERIALIZE::PTR: {
    void *cs_ptr;
    size_t cs_size = serialize_compserv_to_onnx_pointer(cs, cs_ptr);
    cout << "\nSaved compserv to ptr" << endl;

    cs2 = import_compserv_from_onnx_pointer(cs_ptr, cs_size);
    cout << "Compserv imported!" << endl;
    delete [] (char *)cs_ptr; // check how memory is allocated in serialize_compserv_to_onnx_pointer()
  } break;
  case SERIALIZE::STRING: {
    string *cs_str = serialize_compserv_to_onnx_string(cs);
    cout << "\nSaved compserv to string" << endl;

    cs2 = import_compserv_from_onnx_string(cs_str);
    cout << "Compserv imported!" << endl;
    delete cs_str;
  } break;
  default:
    cout << "\nThe compserv serialization type is not valid!" << endl;
    return 1;
  }

  optimizer_info(opt2); // Print the optimizer data
  compserv_info(cs2);   // Print compserv data

  model net2 = create_model(num_classes);

  build(net2,
        opt2,                      // Optimizer
        {"softmax_cross_entropy"}, // Losses
        {"categorical_accuracy"},  // Metrics
        cs2                        // one GPU
  );

  // View model
  summary(net2);

  // Train model
  fit(net2, {x_train}, {y_train}, batch_size, epochs);

  // Evaluate
  evaluate(net2, {x_test}, {y_test}, batch_size);

  delete x_train;
  delete y_train;
  delete x_test;
  delete y_test;
  delete net;
  delete net2;

  return 0;
}
