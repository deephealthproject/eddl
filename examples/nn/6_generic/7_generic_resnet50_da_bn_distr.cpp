/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.9
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/apis/eddl.h"

#include "utils.h"


using namespace eddl;

//////////////////////////////////
// cifar_resnet50_da_bn.cpp:
// Resnet50 with
// BatchNorm
// Data Augmentation
// Using fit for training
//////////////////////////////////

layer BN(layer l)
{
  return BatchNormalization(l);
  //return l;
}

layer BG(layer l) {
  //return GaussianNoise(BN(l),0.3);
  return BN(l);
}


layer ResBlock(layer l, int filters,int half, int expand=0) {
  layer in=l;

  l=ReLu(BG(Conv(l,filters,{1,1},{1,1},"same",false)));

  if (half)
    l=ReLu(BG(Conv(l,filters,{3,3},{2,2},"same",false)));
  else
    l=ReLu(BG(Conv(l,filters,{3,3},{1,1},"same",false)));

  l=BG(Conv(l,4*filters,{1,1},{1,1},"same",false));

  if (half)
    return ReLu(Add(BG(Conv(in,4*filters,{1,1},{2,2},"same",false)),l));
  else
    if (expand) return ReLu(Add(BG(Conv(in,4*filters,{1,1},{1,1},"same",false)),l));
    else return ReLu(Add(in,l));
}

int main(int argc, char **argv){
  int id;
    bool use_cpu = false;
    char model_name[64] = "resnet50_da_bn";
    char pdf_name[128];
    char onnx_name[128];
    
    char path[256] = "covid1";
    char tr_images[256];
    char tr_labels[256];
    char ts_images[256];
    char ts_labels[256];

    int epochs = 32;
    int batch_size = 100;
    int num_classes = 4;
    int channels = 1;
    int width = 256;
    int height = 256;
    float lr = 0.001;
    int initial_mpi_avg = 1;
    int chunks = 0;
    int use_bi8 = 0;
    int use_distr_dataset = 0;

    sprintf(pdf_name, "%s.pdf", model_name);
    sprintf(onnx_name, "%s.onnx", model_name);

    process_arguments(argc, argv,
            path, tr_images, tr_labels, ts_images, ts_labels,
            &epochs, &batch_size, &num_classes, &channels, &width, &height, &lr,
            &initial_mpi_avg,
            &chunks, &use_bi8, &use_distr_dataset);


    compserv cs = nullptr;
    if (use_cpu) {
        cs = CS_CPU();
    } else {
        cs = CS_GPU({1}); // one GPU
        // cs = CS_GPU({1,1},100); // two GPU with weight sync every 100 batches
        // cs = CS_CPU();
        // cs = CS_FPGA({1});
    }
    
  // network
    layer in = Input({channels, width, height});
    //layer in=Input({3,32,32});
    layer l = in;

  // Data augmentation

  l = RandomCropScale(l, {0.8f, 1.0f});
  l = RandomHorizontalFlip(l);

  // Resnet-50

  l=ReLu(BG(Conv(l,64,{3,3},{1,1},"same",false))); //{1,1}
  //l=MaxPool(l,{3,3},{1,1},"same");

  // Add explicit padding to avoid the asymmetric padding in the Conv layers
  l = Pad(l, {0, 1, 1, 0});

  for(int i=0;i<3;i++)
    l=ResBlock(l, 64, 0, i==0); // not half but expand the first

  for(int i=0;i<4;i++)
    l=ResBlock(l, 128,i==0);

  for(int i=0;i<6;i++)
    l=ResBlock(l, 256,i==0);

  for(int i=0;i<3;i++)
    l=ResBlock(l,512,i==0);

  l=MaxPool(l,{4,4});  // should be avgpool

  l=Reshape(l,{-1});

  layer out= Softmax(Dense(l, num_classes));

  // net define input and output layers list
  model net=Model({in},{out});

  

  // Build model
  build(net,
	adam(lr), // Optimizer
    {"softmax_cross_entropy"}, // Losses
    {"categorical_accuracy"}, // Metrics
    cs);

  setlogfile(net, model_name);

    // plot the model
    plot(net, pdf_name, "TB"); // TB --> Top-Bottom mode for dot (graphviz)

    // get some info from the network
    summary(net);


 // Load and preprocess training data
    Tensor* x_train;
    Tensor* y_train;
    //x_train = Tensor::load("apples/train-images.bi8");
    //        y_train = Tensor::load("apples/train-labels.bi8");
    //         x_train->div_(255.0f);

    printf("Test Images %s\n", ts_images);
    printf("Test Labels %s\n", ts_labels);

    Tensor* x_test = Tensor::load(ts_images);
    Tensor* y_test = Tensor::load(ts_labels);
    x_test->div_(255.0f);

    printf("Images %s\n", tr_images);
    printf("Labels %s\n", tr_labels);

    /* Load whole dataset */
    x_train = Tensor::load(tr_images);
    y_train = Tensor::load(tr_labels);

    x_train->div_(255.0f);
    
  //float lr=0.001;
    
  for(int j=0;j<3;j++) {
    lr/=10.0;

    setlr(net,{lr,0.9});
fit(net,{x_train},{y_train}, batch_size, epochs);
    // Evaluate
    evaluate(net,{x_test},{y_test});

    /*
    for(int i=0;i<epochs;i++) {
      // training, list of input and output tensors, batch, epochs
      fit(net,{x_train},{y_train},batch_size, 1);

      // Evaluate test
      std::cout << "Evaluate test:" << std::endl;
      evaluate(net,{x_test},{y_test});
    }*/
    
  }
  delete x_train;
  delete y_train;
  delete x_test;
  delete y_test;
  delete net;

  return EXIT_SUCCESS;
}
