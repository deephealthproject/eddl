Training a simple MLP
---------------------

This example trains and evaluates a simple multilayer perceptron. `[source] <https://github.com/deephealthproject/eddl/blob/master/examples/nn/1_mnist/1_mnist_mlp.cpp>`__ 


.. image:: /_static/images/models/mlp.svg
  :scale: 100%


.. code:: c++

    #include <cstdio>
    #include <cstdlib>
    #include <iostream>

    #include "eddl/apis/eddl.h"


    using namespace eddl;

    //////////////////////////////////
    // mnist_mlp.cpp:
    // A very basic MLP for mnist
    // Using fit for training
    //////////////////////////////////

    int main(int argc, char **argv) {
        // Download mnist
        download_mnist();

        // Settings
        int epochs = 1;
        int batch_size = 100;
        int num_classes = 10;

        // Define network
        layer in = Input({784});
        layer l = in;  // Aux var

        l = LeakyReLu(Dense(l, 1024));
        l = LeakyReLu(Dense(l, 1024));
        l = LeakyReLu(Dense(l, 1024));

        layer out = Softmax(Dense(l, num_classes));
        model net = Model({in}, {out});
        net->verbosity_level = 0;

        // dot from graphviz should be installed:
        plot(net, "model.pdf");

        // Build model
        build(net,
              rmsprop(0.01), // Optimizer
              {"soft_cross_entropy"}, // Losses
              {"categorical_accuracy"}, // Metrics
              CS_GPU({1}) // one GPU
              //CS_GPU({1,1},100) // two GPU with weight sync every 100 batches
              //CS_CPU()
              //CS_FPGA({1})
        );
        //toGPU(net,{1},100,"low_mem"); // In two gpus, syncronize every 100 batches, low_mem setup

        // View model
        summary(net);

        // Load dataset
        Tensor* x_train = Tensor::load("mnist_trX.bin");
        Tensor* y_train = Tensor::load("mnist_trY.bin");
        Tensor* x_test = Tensor::load("mnist_tsX.bin");
        Tensor* y_test = Tensor::load("mnist_tsY.bin");

        // Preprocessing
        x_train->div_(255.0f);
        x_test->div_(255.0f);

        // Train model
        fit(net, {x_train}, {y_train}, batch_size, epochs);

        // Evaluate
        evaluate(net, {x_test}, {y_test});

    }
