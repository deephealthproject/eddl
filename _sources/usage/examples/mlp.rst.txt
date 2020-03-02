Training a simple MLP
---------------------

This example trains and evaluates a simple multilayer perceptron.


.. image:: /_static/images/models/mlp.svg


.. code:: c++

    #include <eddl/apis/eddl.h>
    #include <eddl/apis/eddlT.h>

    using namespace eddl;

    int main() {

        // Download mnist
        download_mnist();

        // Settings
        int epochs = 1;
        int batch_size = 100;
        int num_classes = 10;

        // Define network
        layer in = Input({784});  //
        layer l = in;  // Aux var

        l = LeakyReLu(Dense(l, 1024));
        l = LeakyReLu(Dense(l, 1024));
        l = LeakyReLu(Dense(l, 1024));

        layer out = Softmax(Dense(l, num_classes));
        model net = Model({in}, {out});

        // View model
        summary(net);

        // dot from graphviz should be installed:
        plot(net, "model_mlp.pdf");

        // Build model
        build(net,
              sgd(0.01f, 0.9), // Optimizer
              {"soft_cross_entropy"}, // Losses
              {"categorical_accuracy"}, // Metrics
              CS_GPU({1}, "low_mem") // one GPU
        );

        // Load dataset
        tensor x_train = eddlT::load("trX.bin");
        tensor y_train = eddlT::load("trY.bin");
        tensor x_test = eddlT::load("tsX.bin");
        tensor y_test = eddlT::load("tsY.bin");

        // Preprocessing
        eddlT::div_(x_train, 255.0);
        eddlT::div_(x_test, 255.0);

        // Train model
        fit(net, {x_train}, {y_train}, batch_size, epochs);

        // Evaluate
        evaluate(net, {x_test}, {y_test});
    }

