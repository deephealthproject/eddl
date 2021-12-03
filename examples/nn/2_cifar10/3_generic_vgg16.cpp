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
#include <unistd.h>
 


#include "eddl/apis/eddl.h"
#include "eddl/serialization/onnx/eddl_onnx.h" // Not allowed


using namespace eddl;

//////////////////////////////////
// cifar_vgg16.cpp:
// VGG-16
// Using fit for training
//////////////////////////////////

layer Block1(layer l, int filters) {
    return ReLu(Conv(l, filters,{1, 1},
    {
        1, 1
    }));
}

layer Block3_2(layer l, int filters) {
    l = ReLu(Conv(l, filters,{3, 3},
    {
        1, 1
    }));
    l = ReLu(Conv(l, filters,{3, 3},
    {
        1, 1
    }));
    return l;
}

int main(int argc, char **argv) {
    bool testing = false;
    bool use_cpu = false;
    char path[256] = "imagenet";
       char tr_images[256];
       char tr_labels[256];
       char ts_images[256];
       char ts_labels[256];
       
       
    int epochs = 32;
    int batch_size = 100;
    int num_classes = 1000;
    int channels = 3;
    int width =224;
    int height = 224; 
    double lr = 0.001;
    int initial_mpi_avg=1;

    int id;

    int ch;
    int opterr = 0;
    
    while ((ch = getopt(argc, argv, "m:w:h:c:z:b:e:a:l:")) != -1) {
        switch (ch) {
            case 'm':
                printf("model path:'%s'\n", optarg);
                sprintf(path,"%s",optarg);
                break;
            case 'w':
                printf("width:'%s'\n", optarg);
                width = atoi(optarg);
                break;
            case 'h':
                printf("height:'%s'\n", optarg);
                height = atoi(optarg);
                break;
                case 'z':
                printf("channels:'%s'\n", optarg);
                channels = atoi(optarg);
                break;
            case 'c':
                printf("classes:'%s'\n", optarg);
                num_classes = atoi(optarg);
                break;
            case 'b':
                printf("batch size:'%s'\n", optarg);
                batch_size = atoi(optarg);
                break;
            case 'e':
                printf("epochs:'%s'\n", optarg);
                epochs = atoi(optarg);
                break;
            case 'a':
                printf("mpi-average:'%s'\n", optarg);
                initial_mpi_avg = atoi(optarg);
                break;
            case 'l':
                printf("learning-rate:'%s'\n", optarg);
                lr = std::atof(optarg);
                break;
            default:
                printf("other %c\n", ch);
        }
    }
    sprintf(tr_images,"%s/%s",path,"train-images.bi8");
    sprintf(tr_labels,"%s/%s",path,"train-labels.bi8");
    sprintf(ts_images,"%s/%s",path,"test-images.bi8");
    sprintf(ts_labels,"%s/%s",path,"test-labels.bi8");
   
   
   
    // Define computing service
    compserv cs = nullptr;
    if (use_cpu) {
        cs = CS_CPU();
    } else { 
	cs = CS_GPU({1}); // one GPU
    }

    
    // Init distribuited training
    //id = get_id_distributed();
    
            
    
    // Sync every batch, change every 2 epochs
    //set_method_distributed(AUTO_TIME,1,2);


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


    l = MaxPool(Block3_2(l, 64));
    l = MaxPool(Block3_2(l, 128));
    l = MaxPool(Block1(Block3_2(l, 256), 256));
    l = MaxPool(Block1(Block3_2(l, 512), 512));
    l = MaxPool(Block1(Block3_2(l, 512), 512));

    l = Reshape(l,{-1});
    l = Activation(Dense(l, 512), "relu");

    layer out = Softmax(Dense(l, num_classes));

    // net define input and output layers list
    model net = Model({in}, {out});



    // Build model
    build(net,
//            adam(0.0001), // Optimizer
            adam(lr), // Optimizer
    {"softmax_cross_entropy"}, // Losses
    {
        "categorical_accuracy"
    }, // Metrics
    cs);



    setlogfile(net, "vgg16");

    // plot the model
    if (id == 0)
        plot(net, "model.pdf");

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

    int use_chunks = 1;

    if (use_chunks == 0) {
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
            for (int chunk = 3; chunk <5; chunk ++) {
            //int selected= 1+(rand() % 3);
            int selected=chunk;
            printf("Chunk %d\n", chunk);
            sprintf(tr_images,"%s/train-images_0_%d.bi8",path,selected);
            sprintf(tr_labels,"%s/train-labels_0_%d.bi8",path,selected);
            
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

    
    
    if (id==0)
        save_net_to_onnx_file (net,"vgg16.onnx");   

    //delete x_train;
    //delete y_train;
    delete x_test;
    delete y_test;
    delete net;

    return EXIT_SUCCESS;
}


///////////
