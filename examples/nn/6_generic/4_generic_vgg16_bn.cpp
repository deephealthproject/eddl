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
#include "eddl/serialization/onnx/eddl_onnx.h" // Not allowed

#include "utils.h"

using namespace eddl;

/////////////////////////////////
// cifar_vgg16_bn.cpp:
// vgg16 with BatchNorm
// Using fit for training
//////////////////////////////////


layer Normalization(layer l)
{
  return BatchNormalization(l);
  //return LayerNormalization(l);
  //return GroupNormalization(l,8);
}

layer Block1(layer l,int filters) {
  return ReLu(Normalization(Conv(l,filters,{1,1},{1,1},"same",false)));
}
layer Block3_2(layer l,int filters) {
  l=ReLu(Normalization(Conv(l,filters,{3,3},{1,1},"same",false)));
  l=ReLu(Normalization(Conv(l,filters,{3,3},{1,1},"same",false)));
  return l;
}

int main(int argc, char **argv) {
    int id;
    char model_name[64] = "vgg16_bn";
    char pdf_name[128];
    char onnx_name[128];
        char test_file[128];
    
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
    int method=FIXED;
    int initial_mpi_avg = 1;
    int chunks = 0;
    int use_bi8 = 0;
    int use_distr_dataset = 0;
    int ptmodel=1;
    bool use_cpu=false;
    bool use_mpi=false; 

    sprintf(pdf_name, "%s.pdf", model_name);
    sprintf(onnx_name, "%s.onnx", model_name);

    process_arguments(argc, argv,
            path, tr_images, tr_labels, ts_images, ts_labels,
            &epochs, &batch_size, &num_classes, &channels, &width, &height, &lr,
            &method, &initial_mpi_avg,
            &chunks, &use_bi8, &use_distr_dataset, &ptmodel, test_file,
            &use_cpu, &use_mpi);
      
   
   
    
    // Init distribuited training
    //id = get_id_distributed();
    
            
    
    // Sync every batch, change every 2 epochs
    set_method_distributed(method,initial_mpi_avg,2);


    // download CIFAR data
    //download_cifar10();

    // Settings
    //int epochs = testing ? 2 : 32;
    //int batch_size = 100;
     //    int num_classes = 1000;

    // network
    layer in = Input({channels, width, height});
    //layer in = Input({3, 224, 224});
    layer l = in;


    // Data augmentation
  l = RandomCropScale(l, {0.8f, 1.0f});
  l = RandomFlip(l,1);

  l=MaxPool(Block3_2(l,64));
  l=MaxPool(Block3_2(l,128));
  l=MaxPool(Block1(Block3_2(l,256),256));
  l=MaxPool(Block1(Block3_2(l,512),512));
  l=MaxPool(Block1(Block3_2(l,512),512));

  l=Reshape(l,{-1});
//  l = Activation(Dense(l, 4096), "relu");
//  l = Activation(Dense(l, 4096), "relu");
  l = Activation(Dense(l, 512), "relu");

  layer out= Softmax(Dense(l, num_classes));


    // net define input and output layers list
    model net = Model({in}, {out});

 // Define computing service
    compserv cs = nullptr;
    if (use_cpu) {
        cs = CS_CPU();
    } else { 
	cs = CS_GPU({1}); // one GPU
    }


    // Build model
    build(net,
//            adam(0.0001), // Optimizer
            adam(lr), // Optimizer
    {"softmax_cross_entropy"}, // Losses
    {
        "categorical_accuracy"
    }, // Metrics
    cs);



   setlogfile(net, model_name);

    // plot the model
    if (id == 0)
        plot(net, pdf_name);

       // get some info from the network
    if (id == 0)
        summary(net);

    // Load and preprocess training data
    // Load and preprocess test data
    /*
    Tensor* x_train = Tensor::load("imagenet/train-images.bi8");
    Tensor* y_train = Tensor::load("imagenet/train-labels.bi8");
    Tensor* x_test = Tensor::load("imagenet/test-images.bi8");
    Tensor* y_test = Tensor::load("imagenet/test-labels.bi8");
     */
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

    //if (id==0)
    //y_train->print();



    if (chunks == 0) {
        printf("Images %s\n", tr_images);
        printf("Labels %s\n", tr_labels);

        /* Load whole dataset */
        x_train = Tensor::load(tr_images);
        y_train = Tensor::load(tr_labels);
        x_train->div_(255.0f);
        for (int i = 0; i < epochs; i++) {
            printf("Epoch: %d\n", i);
            // Train
            fit(net,{x_train},{y_train}, batch_size, 1);
            std::cout << "Evaluate test:" << std::endl;
            // Evaluate
            evaluate(net,{x_test},{y_test});
        }
        delete x_train;
        delete y_train;
    } else {
        /* Load chunks */
        for (int i = 0; i < epochs; i++) {
            for (int chunk = 0; chunk < chunks; chunk++) {
                //int selected= 1+(rand() % 3);
                int selected = chunk;
                printf("Chunk %d\n", chunk);
                if (use_bi8) {
                    sprintf(tr_images, "%s/%03d/train-images.bi8", path, chunk);
                    sprintf(tr_labels, "%s/%03d/train-labels.bi8", path, chunk);
                } else {
                    sprintf(tr_images, "%s/%03d/train-images.bin", path, chunk);
                    sprintf(tr_labels, "%s/%03d/train-labels.bin", path, chunk);
                }

                printf("%s\n", tr_images);
                printf("%s\n", tr_labels);
                x_train = Tensor::load(tr_images);
                y_train = Tensor::load(tr_labels);
                x_train->div_(255.0f);
                if (id == 0)
                    printf("Epoch: %d; chunk: %d\n", i, selected);
                // training, list of input and output tensors, batch, epochs
                fit(net,{x_train},
                {
                    y_train
                }, batch_size, 1);
                printf("Free\n");

                delete x_train;
                delete y_train;
            }
        }
    }
    std::cout << "Evaluate test:" << std::endl;
    // Evaluate
    evaluate(net,{x_test},{y_test});



    if (id == 0)
        save_net_to_onnx_file(net, onnx_name);

    //delete x_train;
    //delete y_train;
    delete x_test;
    delete y_test;
    delete net;

    return EXIT_SUCCESS;
}
