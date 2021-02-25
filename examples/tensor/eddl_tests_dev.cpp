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
//          CS_GPU({1}), // one GPU
            CS_CPU(), // CPU with maximum threads availables
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
    Tensor* x_train1 = Tensor::load("cifar_trX.bin"); x_train1->div_(255.0f);
    Tensor* input1 = x_train1->select({ "0" });
    set_mode(net1, TSMODE);
    auto output1 = predict(net1, { input1 });

    Tensor* x_train2 = Tensor::load("cifar_trX.bin"); x_train2->div_(255.0f);
    Tensor* input2 = x_train2->select({ "0" });
    set_mode(net2, TSMODE);
    auto output2 = predict(net2, { input2 });

    // Compare nets (outputs)
    bool res1 = Net::compare_outputs(net1, net2, true, false);
    cout << "Both nets are equal w.r.t outputs? " << res1 << " (1=yes; 0=no)" << endl;

    // Compare nets (params)
    bool res2 = Net::compare_params(net1, net2, true, false);
    cout << "Both nets are equal w.r.t params? " << res2 << " (1=yes; 0=no)" << endl;

    return 0;
}