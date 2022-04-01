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
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--testing") == 0) testing = true;
        else if (strcmp(argv[i], "--cpu") == 0) use_cpu = true;
    }

    // Download mnist
    download_mnist();

    // Settings
    int epochs = 100;
    int batch_size = 100;
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
          {"categorical_accuracy"}, // Metrics
          cs);

    // View model
    summary(net);

    setlogfile(net,"mnist");

    // Load dataset
    Tensor* x_train = Tensor::load("mnist_trX.bin");
    Tensor* y_train = Tensor::load("mnist_trY.bin");
    Tensor* x_test = Tensor::load("mnist_tsX.bin");
    Tensor* y_test = Tensor::load("mnist_tsY.bin");

    if (testing) {
        std::string _range_ = "0:" + std::to_string(2 * batch_size);
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

    Tensor* xbatch = new Tensor({batch_size, 784});
    Tensor* ybatch = new Tensor({batch_size, 10});

    // Preprocessing
    x_train->div_(255.0f);
    x_test->div_(255.0f);

  
    // Train model
    int i,j;
    tshape s = x_train->getShape();
    int num_batches=s[0]/batch_size;

    for(i=0;i<epochs;i++) {
      reset_loss(net);
      fprintf(stdout, "Epoch %d/%d (%d batches)\n", i + 1, epochs,num_batches);
      for(j=0;j<num_batches;j++)  {

        next_batch({x_train,y_train},{xbatch,ybatch});
        train_batch(net, {xbatch}, {ybatch});

        print_loss(net,j);
        printf("\r");
        
      }
      printf("\n");
      if (early_stopping_on_loss_var (net, 0, 10, 0.1, i)) break;
      //if (early_stopping_on_metric_var (net, 0, 0.0001, 2, i)) break;
      if (early_stopping_on_metric (net, 0, 0.97, 2, i)) break;
    }
    printf("\n");   


    // Print loss and metrics
    vector<float> losses1 = get_losses(net);
    vector<float> metrics1 = get_metrics(net);
    for(int i=0; i<losses1.size(); i++) {
        cout << "Loss: " << losses1[i] << "\t" << "Metric: " << metrics1[i] << "   |   ";
    }
    cout << endl;

    // Evaluate model
    printf("Evaluate:\n");
    s=x_test->getShape();
    num_batches=s[0]/batch_size;

    reset_loss(net);  // Important
    for(j=0;j<num_batches;j++)  {
        vector<int> indices(batch_size);
        for(int i=0;i<indices.size();i++)
          indices[i]=(j*batch_size)+i;

        eval_batch(net, {x_test}, {y_test}, indices);

        print_loss(net,j);
        printf("\r");

      }

    printf("\n");

    // Print loss and metrics
    vector<float> losses2 = get_losses(net);
    vector<float> metrics2 = get_metrics(net);
    for(int i=0; i<losses2.size(); i++) {
        cout << "Loss: " << losses2[i] << "\t" << "Metric: " << metrics2[i] << "   |   ";
    }
    cout << endl;

    
    // Quantization
    CPU_quantize_network_distributed(net, 1, 5);

      // Evaluate model
    printf("Evaluate w/quantization:\n");
    s=x_test->getShape();
    num_batches=s[0]/batch_size;

    reset_loss(net);  // Important
    for(j=0;j<num_batches;j++)  {
        vector<int> indices(batch_size);
        for(int i=0;i<indices.size();i++)
          indices[i]=(j*batch_size)+i;

        eval_batch(net, {x_test}, {y_test}, indices);

        print_loss(net,j);
        printf("\r");

      }

    printf("\n");

    // Print loss and metrics
    losses2 = get_losses(net);
    metrics2 = get_metrics(net);
    for(int i=0; i<losses2.size(); i++) {
        cout << "Loss: " << losses2[i] << "\t" << "Metric: " << metrics2[i] << "   |   ";
    }
    cout << endl;
    
    
    
     // Train model
    
    s = x_train->getShape();
    num_batches=s[0]/batch_size;

    for(i=0;i<epochs;i++) {
      reset_loss(net);
      fprintf(stdout, "Epoch %d/%d (%d batches)\n", i + 1, epochs,num_batches);
      for(j=0;j<num_batches;j++)  {

        next_batch({x_train,y_train},{xbatch,ybatch});
        train_batch(net, {xbatch}, {ybatch});

        print_loss(net,j);
        printf("\r");
        
      }
      printf("\n");
      //if (early_stopping_on_loss_var (net, 0, 0.001, 2, i)) break;
      //if (early_stopping_on_metric_var (net, 0, 0.0001, 2, i)) break;
      if (early_stopping_on_metric (net, 0, 0.99, 2, i)) break;
    }
    printf("\n");   


    // Print loss and metrics
    losses1 = get_losses(net);
    metrics1 = get_metrics(net);
    for(int i=0; i<losses1.size(); i++) {
        cout << "Loss: " << losses1[i] << "\t" << "Metric: " << metrics1[i] << "   |   ";
    }
    cout << endl;

    // Evaluate model
    printf("Evaluate:\n");
    s=x_test->getShape();
    num_batches=s[0]/batch_size;

    reset_loss(net);  // Important
    for(j=0;j<num_batches;j++)  {
        vector<int> indices(batch_size);
        for(int i=0;i<indices.size();i++)
          indices[i]=(j*batch_size)+i;

        eval_batch(net, {x_test}, {y_test}, indices);

        print_loss(net,j);
        printf("\r");

      }

    printf("\n");

    // Print loss and metrics
    losses2 = get_losses(net);
    metrics2 = get_metrics(net);
    for(int i=0; i<losses2.size(); i++) {
        cout << "Loss: " << losses2[i] << "\t" << "Metric: " << metrics2[i] << "   |   ";
    }
    cout << endl;

    
    // Quantization
    CPU_quantize_network_distributed(net, 1, 7);

      // Evaluate model
    printf("Evaluate w/quantization:\n");
    s=x_test->getShape();
    num_batches=s[0]/batch_size;

    reset_loss(net);  // Important
    for(j=0;j<num_batches;j++)  {
        vector<int> indices(batch_size);
        for(int i=0;i<indices.size();i++)
          indices[i]=(j*batch_size)+i;

        eval_batch(net, {x_test}, {y_test}, indices);

        print_loss(net,j);
        printf("\r");

      }

    printf("\n");

    // Print loss and metrics
    losses2 = get_losses(net);
    metrics2 = get_metrics(net);
    for(int i=0; i<losses2.size(); i++) {
        cout << "Loss: " << losses2[i] << "\t" << "Metric: " << metrics2[i] << "   |   ";
    }
    cout << endl;
    
    
    

    //last batch
    if (s[0]%batch_size) {
      int last_batch_size=s[0]%batch_size;
      vector<int> indices(last_batch_size);
      for(int i=0;i<indices.size();i++)
        indices[i]=(j*batch_size)+i;

      eval_batch(net, {x_test}, {y_test}, indices);

//      print_loss(net,j);
//      printf("\r");
    }
    


    // Print loss and metrics
    vector<float> losses3 = get_losses(net);
    vector<float> metrics3 = get_metrics(net);
    for(int i=0; i<losses3.size(); i++) {
        cout << "Loss: " << losses3[i] << "\t" << "Metric: " << metrics3[i] << "   |   ";
    }
    cout << endl;

    delete xbatch;
    delete ybatch;

    delete x_train;
    delete y_train;
    delete x_test;
    delete y_test;
    delete net;
    
    return EXIT_SUCCESS;
}
