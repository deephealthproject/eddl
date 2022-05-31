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

//////////////////////////////////
// Generic example
// Model is mlp
// Dataset can be selected
// Using EDDL coarse-grain training
//////////////////////////////////

int main(int argc, char **argv) {
    int id;
    char model_name[64] = "mlp";
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
    int use_mpi=0;
    int dgt=0;
    
      // if executed without mpirun wrapper, error
  if (getenv("OMPI_COMM_WORLD_RANK") == NULL) {
      printf("ERROR: This program should be run with mpirun\n");
      exit (1);
  }

	// DISTR
    id=init_distributed();
    set_avg_method_distributed(method,initial_mpi_avg,2);

    
    sprintf(pdf_name, "%s.pdf", model_name);
    sprintf(onnx_name, "%s.onnx", model_name);

    process_arguments(argc, argv,
            path, tr_images, tr_labels, ts_images, ts_labels,
            &epochs, &batch_size, &num_classes, &channels, &width, &height, &lr,
            &method, &initial_mpi_avg,
            &chunks, &use_bi8, &use_distr_dataset, &ptmodel, test_file,
            &use_cpu, &use_mpi, &dgt);

   
    bool testing = false;

    bool use_mpiall = true;
    bool use_mpi1 = false;
    bool use_mpi2 = false;
    bool use_mpi4 = false;
    bool use_mpix = false;

    
    
    

    
   
    
    // Download mnist
    //download_mnist();


    // network
    layer in = Input({channels, height, width});
    layer l = in; // Aux var

    l = Flatten(l);
    l = LeakyReLu(Dense(l, 1024));
    l = LeakyReLu(Dense(l, 1024));
    l = LeakyReLu(Dense(l, 1024));

    layer out = Softmax(Dense(l, num_classes), -1); // Softmax axis optional (default=-1)
    model net = Model({in},{out});
    net->verbosity_level = 0;
  
   compserv cs = nullptr;
    if (use_cpu) {
        cs = CS_CPU();
    } else {
        cs = CS_GPU(); // one GPU
    }    

    // Build model
    build(net,
            adam(0.001), // Optimizer
    {"softmax_cross_entropy"}, // Losses
    {
        "categorical_accuracy"
    }, // Metrics
    cs);
    //    toGPU(net, {1}, 100,"low_mem"); // In two gpus, syncronize every 100 batches, low_mem setup

      
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
    Tensor* x_train ;
    Tensor* y_train ;
    //x_train = Tensor::load("apples/train-images.bi8");
    //        y_train = Tensor::load("apples/train-labels.bi8");
    //         x_train->div_(255.0f);
  
   

    
    Tensor* x_test = Tensor::load(ts_images);
    Tensor* y_test = Tensor::load(ts_labels);

  
    
     x_test->div_(255.0f);

     //if (id==0)
     //y_train->print();
   

    if (chunks == 0) {
        /* Load whole dataset */
        
        /* Split dataset into processes */
        if (use_distr_dataset) {
            sprintf(tr_images, "%s/train-images_0_%d.bi8", path, id);
            sprintf(tr_labels, "%s/train-labels_0_%d.bi8", path, id);
        }

        printf("%s\n", tr_images);
        printf("%s\n", tr_labels);

        /* Load whole dataset */
        x_train = Tensor::load(tr_images);
        y_train = Tensor::load(tr_labels);
        x_train->div_(255.0f);

        // Train
        fit(net,{x_train},
        {
            y_train
        }, batch_size, epochs);
        delete x_train;
        delete y_train;
    } else {
        /* Load chunks */
        for (int i = 0; i < epochs; i++) {
            for (int chunk = 0; chunk < chunks; chunk++) {
                int selected= rand() % chunks;
                //int selected = chunk;
                //printf("Chunk %d\n", chunk);
                if (use_distr_dataset) {
                    sprintf(tr_images, "%s/train-images_%d_%d.bi8", path, id, selected);
                    sprintf(tr_labels, "%s/train-labels_%d_%d.bi8", path, id, selected);
                } else {
                    sprintf(tr_images, "%s/train-images_0_%d.bi8", path, selected);
                    sprintf(tr_labels, "%s/train-labels_0_%d.bi8", path, selected);
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
                //printf("Free\n");

                delete x_train;
                delete y_train;
            }
        }
    }
    //std::cout << "Evaluate test:" << std::endl;
    // Evaluate
    evaluate(net,{x_test},{y_test});
    evaluate_distr(net,{x_test},{y_test});

    
    
    if (id==0)
        save_net_to_onnx_file (net,onnx_name);   

    //delete x_train;
    //delete y_train;
    delete x_test;
    delete y_test;
    delete net;
    
     end_distributed();

    return EXIT_SUCCESS;
}

   
