/*
 * EDDL Library - European Distributed Deep Learning Library.
 * Version: 1.1
 * copyright (c) 2022, Universitat Politècnica de València (UPV), PRHLT Research Centre
 * Date: March 2022
 * Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
 * All rights reserved
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <unistd.h>

#include "eddl/apis/eddl.h"


using namespace eddl;

//////////////////////////////////
// mnist_mlp_train_batch.cpp:
// A very basic MLP for mnist
// Using train_batch for training
// and eval_batch fot test
//////////////////////////////////

int main(int argc, char **argv) {
    bool testing = false;
    bool use_cpu = false;
    bool use_dg = false;
    bool use_dg_perfect = false;

    char jpgfile[128];
    char txtfile[128];

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--testing") == 0) testing = true;
        else if (strcmp(argv[i], "--cpu") == 0) use_cpu = true;
        else if (strcmp(argv[i], "--dp") == 0) {
            use_dg = true;
            use_dg_perfect = true;
        }
    }

    // Download mnist

    // Settings
    int epochs = 10;
    int batch_size = 100;


    int num_classes = 10;
    int width = 32;
    int height = 32;
    int channels = 3;


    // Define network
    layer in = Input({channels, height, width});
    layer l = in; // Aux var
    l = Flatten(l);

    l = ReLu(Dense(l, 1024));
    l = ReLu(Dense(l, 1024));
    l = ReLu(Dense(l, 1024));

    layer out = Softmax(Dense(l, num_classes));
    model net = Model({in},
    {
        out
    });

    // dot from graphviz should be installed:
    plot(net, "model.pdf");

    compserv cs = nullptr;
    if (use_cpu) {
        cs = CS_CPU();
    } else {
        cs = CS_GPU({1}, "low_mem"); // one GPU
        // cs = CS_GPU({1,1},100); // two GPU with weight sync every 100 batches
        // cs = CS_CPU();
        // cs = CS_FPGA({1});
    }

    // Build model

    build(net,
            sgd(0.001, 0.9), // Optimizer
    {"softmax_cross_entropy"}, // Losses
    {
        "categorical_accuracy"
    }, // Metrics
    cs);

    // View model
    summary(net);

    setlogfile(net, "mnist");


    // Load dataset
    Tensor* x_train;
    Tensor* y_train;
    Tensor* x_test;
    Tensor* y_test;

    if (use_dg == 0) {
        x_train = Tensor::load("train-images.bi8");
        y_train = Tensor::load("train-labels.bi8");
        x_test = Tensor::load("val-images.bi8");
        y_test = Tensor::load("val-labels.bi8");
        // Preprocessing

        x_train->div_(255.0f);
        x_test->div_(255.0f);
    }

    int num_batches = 0;
    int dataset_size = 0;
    int val_num_batches = 0;
    int val_dataset_size = 0;

    // DataGen* Train;
    //  DataGen* Val;

    struct DG_Data Train;
    struct DG_Data Val;
    struct DG_Data Val2;
    struct DG_Data Val3;
    //DG_Data* Train = new DG_Data;
    //DG_Data* Val= new DG_Data;

    if (use_dg) {
        new_DataGen(&Train, "train-images.bi8", "train-labels.bi8", batch_size, false, &dataset_size, &num_batches, DG_PERFECT, 1, 4);
        // imprime_DG(__func__, &Train);
        new_DataGen(&Val, "val-images.bi8", "val-labels.bi8", batch_size, false, &val_dataset_size, &val_num_batches, DG_PERFECT, 1, 2);
        new_DataGen(&Val2, "val-images.bi8", "val-labels.bi8", batch_size, false, &val_dataset_size, &val_num_batches, DG_RANDOM, 1, 2);
        new_DataGen(&Val3, "val-images.bi8", "val-labels.bi8", batch_size, false, &val_dataset_size, &val_num_batches, DG_LIN, 1, 2);
        
    }
    /*
        Fraction* frac3;
        Fraction frac2("train-images.bi8", "train-labels.bi8", batch_size,false,   &dataset_size, &num_batches, use_dg_perfect, 1, 8);
        Fraction frac1("val-images.bi8", "val-labels.bi8", batch_size,false,   &val_dataset_size, &val_num_batches, use_dg_perfect, 1, 2);
        Fraction frac4("val-images.bi8", "val-labels.bi8", batch_size,false,   &val_dataset_size, &val_num_batches, use_dg_perfect, 1, 2);
     */
    // DataGen Val=DataGen("val-images.bi8", "val-labels.bi8", batch_size,false,   &val_dataset_size, &val_num_batches, use_dg_perfect, 1, 4);

    //   Train=new DataGen("train-images.bi8", "train-labels.bi8", batch_size,false,   &dataset_size, &num_batches, use_dg_perfect, 1, 8);
    //   Val=new DataGen("val-images.bi8", "val-labels.bi8", batch_size,false,   &val_dataset_size, &val_num_batches, use_dg_perfect, 1, 4);
    //num_batches=Train->get_nbpp();
    //dataset_size=Train->get_dataset_size();

    Tensor* xbatch;
    Tensor* ybatch;

    xbatch = Tensor::empty({batch_size, channels, height, width});
    ybatch = Tensor::empty({batch_size, num_classes});

    if (use_dg == 0) {
        num_batches = x_train->getShape()[0] / batch_size;
        val_num_batches = x_test->getShape()[0] / batch_size;
    }
    printf("%s num_batches %d dataset %d %d %d \n", __func__, num_batches, dataset_size, val_num_batches, val_dataset_size);

    for (int i = 0; i < epochs; i++) {
        if (use_dg) {
            start_DataGen(&Train);
        }
        reset_loss(net); // Important
        fprintf(stdout, "Epoch %d/%d (%d batches)\n", i + 1, epochs, num_batches);
        for (int j = 0; j < num_batches; j++) {
            if (use_dg) {
                get_batch_DataGen(&Train, xbatch, ybatch);
                xbatch->div_(255.0f);
            } else {
                next_batch({x_train, y_train},
                {
                    xbatch, ybatch
                });
            }
            train_batch(net,{xbatch},
            {
                ybatch
            });
            print_loss(net, j);
            printf("\r");
        }
        if (use_dg) {
            stop_DataGen(&Train);
            //imprime_DG(__func__,Train);
        }
        printf("\n");
        
        if (use_dg) {
            start_DataGen(&Val);
        }
        reset_loss(net); // Important
        for (int j = 0; j < val_num_batches; j++) {
            if (use_dg) {
                get_batch_DataGen(&Val, xbatch, ybatch);
                xbatch->div_(255.0f);
                eval_batch(net,{xbatch},
                {
                    ybatch
                });
            } else {
                vector<int> indices(batch_size);
                for (int i = 0; i < indices.size(); i++)
                    indices[i] = (j * batch_size) + i;

                eval_batch(net,{x_test},
                {
                    y_test
                }, indices);
            }

            print_loss(net, j);
            printf("\r");


        }
        printf("\n");
        if (use_dg) 
            stop_DataGen(&Val);
        if (use_dg)
            start_DataGen(&Val2);
        
        reset_loss(net); // Important
        for (int j = 0; j < val_num_batches; j++) {
            if (use_dg) {
                get_batch_DataGen(&Val2, xbatch, ybatch);
                xbatch->div_(255.0f);
                eval_batch(net,{xbatch},
                {
                    ybatch
                });
            } else {
                vector<int> indices(batch_size);
                for (int i = 0; i < indices.size(); i++)
                    indices[i] = (j * batch_size) + i;

                eval_batch(net,{x_test},
                {
                    y_test
                }, indices);
            }

            print_loss(net, j);
            printf("\r");


        }
        printf("\n");
        if (use_dg)
            stop_DataGen(&Val2);      
        if (use_dg) 
            start_DataGen(&Val3);
        
        reset_loss(net); // Important
        for (int j = 0; j < val_num_batches; j++) {
            if (use_dg) {
                get_batch_DataGen(&Val3, xbatch, ybatch);
                xbatch->div_(255.0f);
                eval_batch(net,{xbatch},
                {
                    ybatch
                });
            } else {
                vector<int> indices(batch_size);
                for (int i = 0; i < indices.size(); i++)
                    indices[i] = (j * batch_size) + i;

                eval_batch(net,{x_test},
                {
                    y_test
                }, indices);
            }

            print_loss(net, j);
            printf("\r");


        }
        printf("\n");
        if (use_dg) stop_DataGen(&Val3);
    }
    if (use_dg) {
        end_DataGen(&Train);
        end_DataGen(&Val);
        end_DataGen(&Val2);
        end_DataGen(&Val3);
        // end_DataGen(&Test);
    }


    if (use_dg == 0) {
        delete xbatch;
        delete ybatch;
    }

    if (use_dg == 0) {
        delete x_train;
        delete y_train;
        delete x_test;
        delete y_test;
    }

    delete net;

    return EXIT_SUCCESS;
}