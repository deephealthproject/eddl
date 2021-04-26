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
// test_onnx_conv3D.cpp:
// A synthetic example with conv3D
// net to test ONNX module
//////////////////////////////////

int main(int argc, char **argv) {
  bool use_cpu = false;
  bool only_import = false;
  string onnx_model_path("model_test_onnx_conv3D.onnx");
  string target_metric_file("metric_test_onnx_conv3D.txt");
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
  int epochs = 2;
  int batch_size = 2;
  int num_classes = 1;
  // Synthetic data shape
  int n_samples = 6;
  int channels = 3;
  int depth = 16;
  int height = 16;
  int width = 16;

  // Create synthetic dataset
  Tensor *aux_x_values = Tensor::linspace(0, 1, n_samples * channels * depth * height * width);
  Tensor *x = Tensor::reshape(aux_x_values, {n_samples, channels, depth, height, width});
  delete aux_x_values;
  Tensor *aux_y_values = Tensor::linspace(0, 1, n_samples * num_classes);
  Tensor *y = Tensor::reshape(aux_y_values, {n_samples, num_classes});
  delete aux_y_values;

  float net_loss = -1;

  if (!only_import) {
    // Define network
    layer in = Input({channels, depth, height, width});
    layer l = in; // Aux var

    l = MaxPool3D(ReLu(Conv3D(l, 4, {3, 3, 3}, {1, 1, 1}, "valid")), {2, 2, 2}, {2, 2, 2}, "same");
    l = ReLu(Conv3D(l, 4, {3, 3, 3}, {2, 2, 2}));
    l = GlobalAveragePool3D(ReLu(Conv3D(l, 4, {3, 3, 3}, {1, 1, 1}, "valid")));
    l = Flatten(l);
    layer out = Sigmoid(Dense(l, num_classes));

    model net = Model({in}, {out});

    compserv cs = nullptr;
    if (use_cpu)
      cs = CS_CPU();
    else
      cs = CS_GPU({1}, "low_mem"); // one GPU

    // Build model
    build(net,
          sgd(0.01), // Optimizer
          {"mse"},   // Losses
          {"mse"},   // Metrics
          cs);       // Computing Service

    // View model
    summary(net);

    // Train model
    fit(net, {x}, {y}, batch_size, epochs);

    // Evaluate
    evaluate(net, {x}, {y}, batch_size);
    net_loss = get_losses(net)[0];

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
        sgd(0.01), // Optimizer
        {"mse"},   // Losses
        {"mse"},   // Metrics
        cs2,       // Computing Service
        false);    // Avoid weights initialization

  // View model
  summary(net2);

  // Evaluate
  evaluate(net2, {x}, {y}, batch_size);
  float net2_loss = get_losses(net2)[0];

  if (!only_import) {
    cout << "Original Net vs Imported Net" << endl;
    cout << "loss: " << net_loss << " == " << net2_loss << endl;
    // Write metric to file
    ofstream ofile;
    ofile.open(target_metric_file);
    ofile << net_loss;
    ofile.close();
  } else {
    cout << "Imported net results:" << endl;
    cout << "loss: " << net2_loss << endl;
  }

  bool ok_test = true;
  if (!target_metric_file.empty() && only_import) {
    // Check if we got the same metric value than the target value provided
    ifstream ifile;
    ifile.open(target_metric_file);
    float target_metric = -1.0;
    ifile >> target_metric;
    ifile.close();
    float metrics_diff = abs(target_metric - net2_loss);
    if (metrics_diff > 0.1) {
      cout << "Test failed: Metric difference too high target=" << target_metric << ", pred=" << net2_loss << endl;
      ok_test = false;
    } else {
      cout << "Test passed!: target=" << target_metric << ", pred=" << net2_loss << endl;
    }
  }

  delete x;
  delete y;
  delete net2;

  if (ok_test)
    return EXIT_SUCCESS;
  else
    return EXIT_FAILURE;
}
