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
// test_onnx_convT3D.cpp:
// A synthetic encoder-decoder
// example with convT3D net to
// test the ONNX module
//////////////////////////////////

int main(int argc, char **argv) {
  bool use_cpu = false;
  bool only_import = false;
  bool channels_last = false;
  string onnx_model_path("model_test_onnx_convT3D.onnx");
  string target_metric_file("");
  // Process provided args
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--cpu") == 0)
      use_cpu = true;
    else if (strcmp(argv[i], "--import") == 0)
      only_import = true;
    else if (strcmp(argv[i], "--onnx-file") == 0) {
      onnx_model_path = argv[i + 1];
      ++i; // Skip model file path for next iteration
    } else if (strcmp(argv[i], "--target-metric") == 0) {
      target_metric_file = argv[i + 1];
      ++i; // Skip metric file path for next iteration
    } else if (strcmp(argv[i], "--channels-last") == 0) {
      channels_last = true;
    }
  }

  // Settings
  int epochs = use_cpu ? 1 : 2;
  int batch_size = 2;

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
  if (channels_last)
    x->permute_({0, 2, 3, 4, 1});

  float net_loss = -1;

  if (!only_import) {
    // Define network
    layer in = Input({channels, depth, height, width});
    layer l = in; // Aux var

    // Encoder
    // x = [batch, 3, 16, 16, 16]
    l = ReLu(Conv3D(l, 32, {3, 3, 3}));
    // x = [batch, 32, 16, 16, 16]
    l = MaxPool3D(l, {2, 2, 2});
    // x = [batch, 32, 8, 8, 8]
    l = ReLu(Conv3D(l, 64, {3, 3, 3}));
    // x = [batch, 64, 8, 8, 8]
    l = MaxPool3D(l, {2, 2, 2});
    // x = [batch, 64, 4, 4]

    // Decoder
    l = ReLu(ConvT3D(l, 64, {2, 2, 2}, {2, 2, 2}));
    // x = [batch, 64, 8, 8, 8]
    l = ReLu(Conv3D(l, 32, {1, 1, 1}));  // PointwiseConv3D
    // x = [batch, 32, 8, 8, 8]
    l = ReLu(ConvT3D(l, 32, {2, 2, 2}, {2, 2, 2}));
    // x = [batch, 32, 16, 16, 16]
    layer out = Sigmoid(Conv3D(l, channels, {1, 1, 1}));  // PointwiseConv3D
    // x = [batch, 3, 16, 16, 16]

    model net = Model({in}, {out});

    compserv cs = nullptr;
    if (use_cpu)
      cs = CS_CPU();
    else
      cs = CS_GPU({1}, "low_mem"); // one GPU

    // Build model
    build(net,
          adam(0.001), // Optimizer
          {"mse"},     // Losses
          {"mse"},     // Metrics
          cs);         // Computing Service

    // View model
    summary(net);

    // Train model
    fit(net, {x}, {x}, batch_size, epochs);

    // Evaluate
    evaluate(net, {x}, {x}, batch_size);
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
        adam(0.001), // Optimizer
        {"mse"},     // Losses
        {"mse"},     // Metrics
        cs2,         // Computing Service
        false);      // Avoid weights initialization

  // View model
  summary(net2);

  // Evaluate
  evaluate(net2, {x}, {x}, batch_size);
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
    if (metrics_diff > 0.001) {
      cout << "Test failed: Metric difference too high target=" << target_metric
           << ", pred=" << net2_loss << endl;
      ok_test = false;
    } else {
      cout << "Test passed!: target=" << target_metric << ", pred=" << net2_loss
           << endl;
    }
  }

  delete x;
  delete net2;

  if (ok_test)
    return EXIT_SUCCESS;
  else
    return EXIT_FAILURE;
}
