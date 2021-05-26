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

//////////////////////////////////
// test_onnx_conv2D.cpp:
// A MNIST example with conv2D net
// to test ONNX module
//////////////////////////////////

int main(int argc, char **argv) {
  bool use_cpu = false;
  bool only_import = false;
  string onnx_model_path("model_test_onnx_conv2D.onnx");
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

  // Settings
  int epochs = use_cpu ? 1 : 2;
  int batch_size = 100;
  int num_classes = 10;

  // Download mnist
  download_mnist();

  // Load dataset
  Tensor *x_train = Tensor::load("mnist_trX.bin");
  Tensor *y_train = Tensor::load("mnist_trY.bin");
  Tensor *x_test = Tensor::load("mnist_tsX.bin");
  Tensor *y_test = Tensor::load("mnist_tsY.bin");

  // Preprocessing
  x_train->div_(255.0f);
  x_test->div_(255.0f);

  float net_loss = -1;
  float net_acc = -1;

  if (!only_import) {
    // Define network
    layer in = Input({784});
    layer l = in; // Aux var

    l = Reshape(l, {1, 28, 28});
    l = MaxPool2D(ReLu(Conv2D(l, 32, {3, 3}, {1, 1})), {2, 2}, {2, 2}, "same");
    l = BatchNormalization(l);
    l = AveragePool2D(ReLu(Conv2D(l, 32, {3, 3}, {1, 1}, "valid")), {3, 3}, {2, 2}, "none");
    l = BatchNormalization(l);
    l = GlobalAveragePool2D(ReLu(Conv2D(l, 32, {3, 3}, {2, 2}, "same", false)));
    l = BatchNormalization(l);
    l = Flatten(l);
    layer out = Softmax(Dense(l, num_classes));

    model net = Model({in}, {out});

    compserv cs = nullptr;
    if (use_cpu)
      cs = CS_CPU();
    else
      cs = CS_GPU({1}, "low_mem"); // one GPU

    // Build model
    build(net,
          adam(0.001),               // Optimizer
          {"softmax_cross_entropy"}, // Losses
          {"categorical_accuracy"},  // Metrics
          cs);                       // Computing Service

    // View model
    summary(net);

    // Train model
    fit(net, {x_train}, {y_train}, batch_size, epochs);

    // Evaluate
    evaluate(net, {x_test}, {y_test}, batch_size);
    net_loss = get_losses(net)[0];
    net_acc = get_metrics(net)[0];

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
  float net2_loss = get_losses(net2)[0];
  float net2_acc = get_metrics(net2)[0];

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
    if (metrics_diff > 0.001) {
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
