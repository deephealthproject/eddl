#include "eddl/apis/eddl.h"

using namespace eddl;

int main(int argc, char **argv) {
    // Download cifar
    download_cifar10();


    // Build model
    Net* net1=download_resnet18(false,{3, 32, 32});
    build(net1,
          adam(0.001), // Optimizer
          {"softmax_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          CS_GPU({1}), // one GPU
//            CS_CPU(), // CPU with maximum threads availables
          false       // Parameter that indicates that the weights of the net must not be initialized to random values.
    );

    Net* net2=download_resnet18(false,{3, 32, 32});
    build(net2,
          adam(0.001), // Optimizer
          {"softmax_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          CS_GPU({1}), // one GPU
//            CS_CPU(), // CPU with maximum threads availables
          false       // Parameter that indicates that the weights of the net must not be initialized to random values.
    );

    // Load and preprocess training data
    Tensor* x_train = Tensor::load("cifar_trX.bin");
    x_train->div_(255.0f);
    Tensor* input1 = x_train->select({ "0" });
    Tensor* input2 = x_train->select({ "0" });

    auto output1 = predict(net1, { input1 });
    auto output2 = predict(net2, { input2 });

    // Compare nets
    bool res = Net::compare_outputs(net1, net2, true, false);
    cout << "Both nets are equal? " << res << " (1=yes; 0=no)" << endl;

    return 0;
}