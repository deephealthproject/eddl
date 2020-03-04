Training a CNN
--------------

This example trains and evaluates a simple convolutional neural network. Additionally, we perform some data
augmentation to the input data.

.. image:: /_static/images/models/cnn_da.svg
  :scale: 100%


.. code:: c++


    #include "<eddl/apis/eddl.h>
    #include "<eddl/apis/eddlT.h>

    using namespace eddl;

    int main(int argc, char **argv){

        // download CIFAR data
        download_cifar10();

        // Settings
        int epochs = 25;
        int batch_size = 100;
        int num_classes = 10;

        // Network
        layer in=Input({3,32,32});
        layer l=in;

        // Set data augmentation
        l = RandomCrop(l, {28, 28});
        l = RandomCropScale(l, {0.f, 1.0f});
        l = RandomCutout(l, {0.0f, 0.3f}, {0.0f, 0.3f});

        // Set base network
        l = MaxPool(ReLu(Conv(l,32,{3,3},{1,1})),{2,2});
        l = MaxPool(ReLu(Conv(l,64,{3,3},{1,1})),{2,2});
        l = MaxPool(ReLu(Conv(l,128,{3,3},{1,1})),{2,2});
        l = GlobalMaxPool(l);

        l = Flatten(l);
        l = Activation(Dense(l,128),"relu");
        layer out = Activation(Dense(l,num_classes),"softmax");

        // net define input and output layers list
        model net = Model({in},{out});

        // Build model
        build(net,
              sgd(0.01, 0.9), // Optimizer
              {"soft_cross_entropy"}, // Losses
              {"categorical_accuracy"}, // Metrics
              CS_GPU({1}, "low_mem") // GPU with only one gpu
        );

        // plot the model
        plot(net,"model.pdf");

        // get some info from the network
        summary(net);

        // Load and preprocess training data
        tensor x_train = eddlT::load("cifar_trX.bin");
        tensor y_train = eddlT::load("cifar_trY.bin");
        eddlT::div_(x_train, 255.0);

        // Load and preprocess test data
        tensor x_test = eddlT::load("cifar_tsX.bin");
        tensor y_test = eddlT::load("cifar_tsY.bin");
        eddlT::div_(x_test, 255.0);

        // Train model
        fit(net,{x_train},{y_train},batch_size, epochs);

        // Evaluate train
        evaluate(net,{x_test},{y_test});
    }

