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
    
    int id;
    int n_procs;
    
    id= init_distributed();
    n_procs = get_n_procs_distributed();

     
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--testing") == 0) testing = true;
        else if (strcmp(argv[i], "--cpu") == 0) use_cpu = true;
    }
      
    // Download mnist
    download_mnist();

    // Settings
    int epochs = 2;
    int global_batch_size = 100;
    int batch_size;
    int num_classes = 10;

    
    // Define network
    layer in = Input({784});
    layer l = in;  // Aux var

    l = ReLu(Dense(l, 1024));
    l = ReLu(Dense(l, 1024));
    l = ReLu(Dense(l, 1024));

    layer out = Softmax(Dense(l, num_classes));
    model net = Model({in}, {out});

    // dot from graphviz should be installed:
    if (id == 0)
        plot(net, "model.pdf");

    // Define computing service
    compserv cs = nullptr;
    if (use_cpu) {
        cs = CS_CPU();
    } else { 
	cs=CS_GPU();
    }

    // Build model
    build(net,
          sgd(0.001, 0.9), // Optimizer
          {"softmax_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          cs);

    printf("after build\n");
    //Broadcast_params_distributed(net);

    // View model
    if (id==0)
        summary(net);

    setlogfile(net,"mnist");

    // Load dataset
    Tensor* x_train = Tensor::load("mnist_trX.bin");
    Tensor* y_train = Tensor::load("mnist_trY.bin");
    Tensor* x_test = Tensor::load("mnist_tsX.bin");
    Tensor* y_test = Tensor::load("mnist_tsY.bin");

    if (testing) {
        std::string _range_ = "0:" + std::to_string(2 * global_batch_size);
        Tensor* x_mini_train = x_train->select({_range_, ":"});
        Tensor* y_mini_train = y_train->select({_range_, ":"});
        Tensor* x_mini_test  = x_test->select({_range_, ":"});
        Tensor* y_mini_test  = y_test->select({_range_, ":"});

        delete x_train;
        delete y_train;
        delete x_test;
        delete y_test;

        x_train = x_mini_train;
        y_train = y_mini_train;
        x_test  = x_mini_test;
        y_test  = y_mini_test;
    }
    
    tshape s = x_train->getShape();
    
    batch_size=global_batch_size/n_procs;
    int num_batches_training=s[0]/batch_size;
    int nbpp_training=num_batches_training/n_procs;
   

   
    Tensor* xbatch = new Tensor({batch_size, 784});
    Tensor* ybatch = new Tensor({batch_size, 10});

    // Preprocessing
    x_train->div_(255.0f);
    x_test->div_(255.0f);


    // Train model
    int i,j;
    
     
    bcast_weights_distributed(net);
    
    for(i=0;i<epochs;i++) {
      reset_loss(net);
      if (id == 0) {
            fprintf(stdout, "Epoch %d/%d (%d batches, %d batches per proc)\n", i + 1, epochs, num_batches_training, nbpp_training);
        }
//      for(j=0;j<num_batches;j++)  {
      for(j=0;j<nbpp_training;j++)  {

        next_batch({x_train,y_train},{xbatch,ybatch});
        train_batch(net, {xbatch}, {ybatch});
        //sync_batch
        avg_weights_distributed(net, j, nbpp_training);  
        
        //if (id==0) {
            print_loss(net,j);
            if (id==0)
                printf("\r");
        //}
      }
      // adjust batches_avg
      if (id==0)
          printf("\n");
    }
    if (id==0) 
        printf("\n");   


    // Print loss and metrics
    vector<float> losses1 = get_losses(net);
    vector<float> metrics1 = get_metrics(net);
    if (id == 0) {
        for (int i = 0; i < losses1.size(); i++) {
            cout << "Loss: " << losses1[i] << "\t" << "Metric: " << metrics1[i] << "   |   ";
        cout << endl;
        }
    }

    // Evaluate model
    
    
    if (id==0)
        printf("Evaluate:\n");
    
    s=x_test->getShape();
    int num_batches_val=s[0]/batch_size;
    int nbpp_val=num_batches_val/n_procs;

    reset_loss(net); // Important
    if (id == 0) {
        for (j = 0; j < num_batches_val; j++) {
            vector<int> indices(batch_size);
            for (int i = 0; i < indices.size(); i++)
                indices[i] = (j * batch_size) + i;

            eval_batch(net,{x_test},
            {
                y_test
            }, indices);
            print_loss(net, j);

            printf("\r");

        }

        printf("\n");
    }
    
    // Print loss and metrics
    vector<float> losses2 = get_losses(net);
    vector<float> metrics2 = get_metrics(net);
    if (id == 0) {
        for (int i = 0; i < losses2.size(); i++) {
            cout << "Loss: " << losses2[i] << "\t" << "Metric: " << metrics2[i] << "   |   ";
        cout << endl;
        }
    }

    //last batch
    if (id == 0) {
        if (s[0] % batch_size) {
            int last_batch_size = s[0] % batch_size;
            vector<int> indices(last_batch_size);
            for (int i = 0; i < indices.size(); i++)
                indices[i] = (j * batch_size) + i;

            eval_batch(net,{x_test},
            {
                y_test
            }, indices);

            print_loss(net, j);

            printf("\r");
        }
    }
    


    // Print loss and metrics
    
    vector<float> losses3 = get_losses(net);
    vector<float> metrics3 = get_metrics(net);
    if (id == 0) {
        for (int i = 0; i < losses3.size(); i++) {
            cout << "Loss: " << losses3[i] << "\t" << "Metric: " << metrics3[i] << "   |   ";
        }
        cout << endl;
    }
    
    
    delete xbatch;
    delete ybatch;

    delete x_train;
    delete y_train;
    delete x_test;
    delete y_test;
    delete net;
    
    // Finalize distributed training
    end_distributed();

    
    return EXIT_SUCCESS;
}
