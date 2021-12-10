/*
 * EDDL Library - European Distributed Deep Learning Library.
 * Version: 1.0
 * copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research
 * Centre Date: November 2021 Author: PRHLT Research Centre, UPV,
 * (rparedes@prhlt.upv.es), (jon@prhlt.upv.es) All rights reserved
 */

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "eddl/apis/eddl.h"

#include "eddl/serialization/onnx/eddl_onnx.h" // Not allowed

using namespace eddl;

///////////////////////////////////////////
// 7_onnx_test_gradients_recurrent.cpp:
// An example on how to use the functions
// for exporting weights and gradients
// using the ONNX format, with a recurrent
// network
///////////////////////////////////////////

int main(int argc, char **argv) {
  // Read arguments
  bool export_cpu = false;
  bool import_cpu = false;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--export-cpu") == 0)
      export_cpu = true;
    else if (strcmp(argv[i], "--import-cpu") == 0)
      import_cpu = true;
  }

  // Download dataset
  download_imdb_2000();

  // Settings
  int epochs = 2;
  int batch_size = 64;
  CompServ *export_CS = export_cpu ? CS_CPU() : CS_GPU({1});
  CompServ *import_CS = import_cpu ? CS_CPU() : CS_GPU({1});

  int length = 250;
  int embed_dim = 33;
  int vocsize = 2000;

  // Define network
  layer in = Input({1}); // 1 word
  layer l = in;

  layer l_embed = RandomUniform(Embedding(l, vocsize, 1, embed_dim), -0.05, 0.05);

  l = LSTM(l_embed, 37);
  l = ReLu(Dense(l, 256));
  layer out = Sigmoid(Dense(l, 1));

  cout << "Creating model" << endl;
  model net = Model({in}, {out});
  cout << "Model created" << endl;

  // Build model
  cout << "Building the model" << endl;
  build(net,
        adam(0.001),              // Optimizer
        {"binary_cross_entropy"}, // Losses
        {"binary_accuracy"},      // Metrics
        export_CS,                // Computing service
        true                      // Enable parameters initialization
  );
  cout << "Model is correctly built" << endl;

  cout << "Enabling distributed training" << endl;
  net->enable_distributed();
  cout << "Distributed training enabled" << endl;

  // Export the net before training
  void *serialized_net;
  cout << "Serializing net (without training) to pointer" << endl;
  size_t model_size = serialize_net_to_onnx_pointer(net, serialized_net, false);
  cout << "Net serialized to pointer" << endl;

  // View model
  summary(net);

  // Load dataset
  Tensor *x_train = Tensor::load("imdb_2000_trX.bin");
  Tensor *y_train = Tensor::load("imdb_2000_trY.bin");
  Tensor *x_test = Tensor::load("imdb_2000_tsX.bin");
  Tensor *y_test = Tensor::load("imdb_2000_tsY.bin");

  x_train->reshape_({x_train->shape[0], length, 1}); // batch x timesteps x input_dim
  x_test->reshape_({x_test->shape[0], length, 1}); // batch x timesteps x input_dim

  y_train->reshape_({y_train->shape[0], 1, 1}); // batch x timesteps x input_dim
  y_test->reshape_({y_test->shape[0], 1, 1});   // batch x timesteps x input_dim

  // Train model
  cout << "Training the first model" << endl;
  fit(net, {x_train}, {y_train}, batch_size, epochs);

  // Evaluate
  cout << "Evaluating the first model" << endl;
  evaluate(net, {x_test}, {y_test}, batch_size);

  // Export gradients
  void *serialized_gradients;
  string path("mnist.onnx");
  cout << "Exporting gradients" << endl;
  size_t gradients_size = serialize_net_to_onnx_pointer(net, serialized_gradients, true);
  cout << "Gradients exported" << endl;

  // Export trained model
  void *serialized_net_once_trained;
  cout << "Exporting trained weights" << endl;
  size_t snet_size = serialize_net_to_onnx_pointer(net, serialized_net_once_trained, false);
  cout << "Trained weights exported" << endl;

  // Import net topology without trained weights
  cout << "Importing original net topology (without training)" << endl;
  Net *imported_net = import_net_from_onnx_pointer(serialized_net, model_size);
  cout << "Untrained net imported" << endl;

  // Build model
  cout << "Building the loaded topology" << endl;
  build(imported_net,
        adam(0.001),              // Optimizer
        {"binary_cross_entropy"}, // Losses
        {"binary_accuracy"},      // Metrics
        import_CS,                // Computing service
        false                     // Disable parameters initialization
  );
  cout << "Model is correctly built" << endl;

  // View loaded model
  summary(imported_net);

  // Evaluate with untrained model
  cout << "Evaluating test with the untrained weights" << endl;
  evaluate(imported_net, {x_test}, {y_test}, batch_size);

  // Apply grads
  cout << "Applying grads from training" << endl;
  apply_grads_from_onnx_pointer(imported_net, serialized_gradients, gradients_size);
  cout << "Grads applied" << endl;

  // Evaluate net with accumulated gradients applied
  cout << "Evaluating test after applying gradients" << endl;
  evaluate(imported_net, {x_test}, {y_test}, batch_size);

  // Set trained weights
  cout << "Putting the trained weights" << endl;
  set_weights_from_onnx_pointer(imported_net, serialized_net_once_trained, snet_size);
  cout << "Trained weights set" << endl;

  // Evaluate with trained weights
  cout << "Evaluating test after putting the trained weights" << endl;
  evaluate(imported_net, {x_test}, {y_test}, batch_size);

  return 0;
}
