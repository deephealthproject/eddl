/*
 * EDDL Library - European Distributed Deep Learning Library.
 * Version: 0.9
 * copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research
 * Centre Date: November 2020 Author: PRHLT Research Centre, UPV,
 * (rparedes@prhlt.upv.es), (jon@prhlt.upv.es) All rights reserved
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/apis/eddl.h"
#include "eddl/serialization/onnx/eddl_onnx.h"

using namespace eddl;

////////////////////////////////////
// test_onnx_rnn_mnist.cpp:
// A MNIST example with recurrent
// (RNN) net.
// For ONNX testing
////////////////////////////////////

int main(int argc, char **argv) {
  bool use_cpu = false;
  bool only_import = false;
  string onnx_model_path("model_test_onnx_rnn_mnist.onnx");
  string target_metric_file("");
  // Process provided args
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--cpu") == 0)
      use_cpu = true;
    else if (strcmp(argv[i], "--import") == 0)
      only_import = true;
    else if (strcmp(argv[i], "--onnx-file") == 0) {
      onnx_model_path = argv[i+1];
      ++i; // Skip model file path for next iteration
    } else if (strcmp(argv[i], "--target-metric") == 0) {
      target_metric_file = argv[i+1];
      ++i; // Skip metric file path for next iteration
    }
  }

  // Download mnist
  download_mnist();

  // Settings
  int epochs = use_cpu ? 1 : 2;
  int batch_size = 100;

  // Load dataset
  Tensor *x_train = Tensor::load("mnist_trX.bin");
  Tensor *y_train = Tensor::load("mnist_trY.bin");
  Tensor *x_test = Tensor::load("mnist_tsX.bin");
  Tensor *y_test = Tensor::load("mnist_tsY.bin");

  // Preprocessing
  x_train->div_(255.0f);
  x_test->div_(255.0f);

  x_train->reshape_({x_train->shape[0], 28, 28}); // batch x timesteps x input_dim
  x_test->reshape_({x_test->shape[0], 28, 28});   // batch x timesteps x input_dim

  y_train->reshape_({y_train->shape[0], 1, 10}); // batch x timesteps x output_dim
  y_test->reshape_({y_test->shape[0], 1, 10});   // batch x timesteps x output_dim

  float net_loss = -1;
  float net_acc = -1;

  if (!only_import) {
    // Define network
    layer in = Input({28});
    layer l = in;

    l = Dense(l, 32);
    l = LeakyReLu(l);
    l = RNN(l, 64);
    l = Dense(l, 32);
    l = LeakyReLu(l);
    layer out = Softmax(Dense(l, 10));

    model net = Model({in}, {out});

    compserv cs = nullptr;
    if (use_cpu)
      cs = CS_CPU();
    else
      cs = CS_GPU({1}, "low_mem"); // one GPU

    // Build model
    build(net,
          adam(0.001),              // Optimizer
          {"softmax_cross_entropy"},// Losses
          {"categorical_accuracy"}, // Metrics
          cs);                      // Computing Service

    // View model
    summary(net);

    // Train model
    fit(net, {x_train}, {y_train}, batch_size, epochs);

    // Evaluate
    evaluate(net, {x_test}, {y_test}, batch_size);
    net_loss = get_losses(net->rnet)[0];
    net_acc = get_metrics(net->rnet)[0];

    // Export the model to ONNX
    save_net_to_onnx_file(net, onnx_model_path);
    delete net;
  }

  // Import the trained model from ONNX
  model net2 = import_net_from_onnx_file(onnx_model_path);

  compserv cs2 = nullptr;
  if (use_cpu)
    cs2 = CS_CPU();
  else
    cs2 = CS_GPU({1}, "low_mem"); // one GPU

  // Build model
  build(net2,
        adam(0.001),               // Optimizer
        {"softmax_cross_entropy"}, // Losses
        {"categorical_accuracy"},  // Metrics
        cs2,                       // Computing Service
        false);                    // Avoid weights initialization

  // View model
  summary(net2);

  // Evaluate
  evaluate(net2, {x_test}, {y_test}, batch_size);
  float net2_loss = get_losses(net2->rnet)[0];
  float net2_acc = get_metrics(net2->rnet)[0];

  if (!only_import) {
    cout << "Original Net vs Imported Net" << endl;
    cout << "loss: " << net_loss << " == " << net2_loss << endl;
    cout << "acc: " << net_acc << " == " << net2_acc << endl;
    // Write metric to file
    ofstream ofile;
    ofile.open(target_metric_file);
    ofile << net_acc;
    ofile.close();
  } else {
    cout << "Imported net results:" << endl;
    cout << "loss: " << net2_loss << endl;
    cout << "acc: " << net2_acc << endl;
  }

  bool ok_test = true;
  if (!target_metric_file.empty() && only_import) {
    // Check if we got the same metric value than the target value provided
    ifstream ifile;
    ifile.open(target_metric_file);
    float target_metric = -1.0;
    ifile >> target_metric;
    ifile.close();
    float metrics_diff = abs(target_metric - net2_acc);
    if (metrics_diff > 0.01) {
      cout << "Test failed: Metric difference too high target=" << target_metric << ", pred=" << net2_acc << endl;
      ok_test = false;
    } else {
      cout << "Test passed!: target=" << target_metric << ", pred=" << net2_acc << endl;
    }
  }

  delete x_train;
  delete y_train;
  delete x_test;
  delete y_test;
  delete net2;

  if (ok_test)
    return EXIT_SUCCESS;
  else
    return EXIT_FAILURE;
}
