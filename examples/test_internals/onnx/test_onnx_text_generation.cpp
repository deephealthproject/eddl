/*
 * EDDL Library - European Distributed Deep Learning Library.
 * Version: 1.0
 * copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research
 * Centre Date: November 2021 Author: PRHLT Research Centre, UPV,
 * (rparedes@prhlt.upv.es), (jon@prhlt.upv.es) All rights reserved
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/apis/eddl.h"
#include "eddl/serialization/onnx/eddl_onnx.h"

using namespace eddl;

////////////////////////////////////
// test_onnx_text_generation.cpp:
// Decoder model for ONNX testing
////////////////////////////////////

Tensor *onehot(Tensor *in, int vocs) {
  int n = in->shape[0];
  int l = in->shape[1];
  int c = 0;

  Tensor *out = new Tensor({n, l, vocs});
  out->fill_(0.0);

  int p = 0;
  for (int i = 0; i < n * l; i++, p += vocs) {
    int w = in->ptr[i];
    if (w == 0)
      c++;
    out->ptr[p + w] = 1.0;
  }

  cout << "padding=" << (100.0 * c) / (n * l) << "%" << endl;
  return out;
}

int main(int argc, char **argv) {

  bool use_cpu = false;
  bool use_add = false;  // Use Add instead of Concat
  string onnx_model_path("model_test_onnx_text_generation.onnx");
  // Process provided args
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--cpu") == 0)
      use_cpu = true;
    else if (strcmp(argv[i], "--add") == 0)
      use_add = true;
    else if (strcmp(argv[i], "--onnx-file") == 0) {
      onnx_model_path = argv[i+1];
      ++i; // Skip model file path for next iteration
    }
  }

  // Download the datset
  download_flickr();

  // Settings
  int epochs = use_cpu ? 1 : 20;
  int batch_size = 24;

  int olength = 20;
  int outvs = 2000;
  int embdim = 32;

  model feature_extractor = download_resnet18(true, {3, 256, 256});

  layer lreshape = getLayer(feature_extractor, "top");

  // create a new model from input output
  layer image_in = getLayer(feature_extractor, "input");

  // Decoder
  layer ldecin = Input({outvs});
  layer ldec = ReduceArgMax(ldecin, {0});
  ldec = RandomUniform(Embedding(ldec, outvs, 1, embdim, false), -0.05, 0.05);

  if (use_add) {
    lreshape = ReLu(Dense(lreshape, embdim));  // Prepare the dim for the Add operation
    ldec = Add({ldec, lreshape});
  }
  else
    ldec = Concat({ldec, lreshape});

  layer l = LSTM(ldec, 512, false);

  layer out = Softmax(Dense(l, outvs));

  setDecoder(ldecin);

  model net = Model({image_in}, {out});

  delete feature_extractor;

  optimizer opt = adam(0.0001);
  compserv cs;
  if (use_cpu)
    cs = CS_CPU();
  else
    cs = CS_GPU({1});

  // Build model
  build(net,
        opt,                       // Optimizer
        {"softmax_cross_entropy"}, // Losses
        {"accuracy"},              // Metrics
        cs);

  // View model
  summary(net);

  // Load dataset
  Tensor *x_train = Tensor::load("flickr_trX.bin", "bin");
  Tensor *y_train = Tensor::load("flickr_trY.bin", "bin");

  Tensor *xtrain = Tensor::permute(x_train, {0, 3, 1, 2}); // 1000, 3, 256, 256
  Tensor *ytrain = y_train;
  y_train = onehot(ytrain, outvs);
  y_train->reshape_({y_train->shape[0], olength, outvs}); // batch x timesteps x input_dim

  // Train model
  fit(net, {xtrain}, {y_train}, batch_size, epochs);

  // Evaluate model
  evaluate(net, {xtrain}, {y_train}, batch_size);
  float net_loss = get_losses(net->rnet)[0];

  // Export the model to ONNX
  save_net_to_onnx_file(net, onnx_model_path, olength);

  // Import the trained model from ONNX
  model net2 = import_net_from_onnx_file(onnx_model_path);

  optimizer opt2 = adam(0.0001);
  compserv cs2;
  if (use_cpu)
    cs2 = CS_CPU();
  else
    cs2 = CS_GPU({1});

  // Build model
  build(net2,
        opt2,                      // Optimizer
        {"softmax_cross_entropy"}, // Losses
        {"accuracy"},              // Metrics
        cs2,
        false);                    // Avoid weights initialization

  // View model
  summary(net2);

  // Evaluate model
  evaluate(net2, {xtrain}, {y_train}, batch_size);
  float net2_loss = get_losses(net2->rnet)[0];

  cout << "Original Net vs Imported Net" << endl;
  cout << "loss: " << net_loss << " == " << net2_loss << endl;

  delete xtrain;
  delete ytrain;
  delete x_train;
  delete y_train;
  delete net;

  float loss_diff = abs(net_loss - net2_loss);
  if (loss_diff < 0.001) {
    cout << "Test passed!" << endl;
    return EXIT_SUCCESS;
  } else {
    cout << "Test failed!" << endl;
    return EXIT_FAILURE;
  }
}
