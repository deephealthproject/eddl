#include "gtest/gtest.h"
#include <random>

#include "apis/eddl.h"
#include "serialization/onnx/eddl_onnx.h"
#include <typeinfo>

using namespace eddl;

// Random generators
const int MIN_RANDOM = 0;
const int MAX_RANDOM = 999999;
static std::random_device rd;
static std::mt19937 mt(rd());
static std::uniform_int_distribution<std::mt19937::result_type> dist6(MIN_RANDOM, MAX_RANDOM);


Net* get_network(){

    // Define network
    layer in = Input({784});
    layer l = in;  // Aux var

    l=Reshape(l,{-1});
    l = ReLu(Dense(l, 1024));
    l = BatchNormalization(l);
    l = ReLu(Dense(l, 1024));
    l = BatchNormalization(l);
    l = ReLu(Dense(l, 1024));
    l = BatchNormalization(l);

    layer out = Activation(Dense(l, 10), "softmax");
    model net = Model({in}, {out});

    return net;
}

TEST(ONNXTestSuite, onnx_import){
    // Generate random name
    int rdn_name = dist6(mt);
    string fname = "onnx_net_" + to_string(rdn_name) + ".onnx";

    // Get some network
    Net* net_export = get_network();
    build(net_export,
          sgd(0.01), // Optimizer
          {"soft_cross_entropy"}, // Losses
          {"categorical_accuracy"},  // Metrics)
          CS_CPU(), true);
    net_export->resize(1);

    // Export network to ONNX
    save_net_to_onnx_file(net_export, fname);

    // Import net
    Net* net_import = import_net_from_onnx_file(fname);
    build(net_import,
          sgd(0.01), // Optimizer
          {"soft_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics);
          CS_CPU(), false);
    net_import->resize(1);

    // Delete file
    int hasFailed = std::remove(fname.c_str());
    if(hasFailed) { cout << "Error deleting file: " << fname << endl; }

    // Test model properties
    ASSERT_TRUE(net_export->name == net_import->name);
    ASSERT_EQ(net_export->batch_size, net_import->batch_size);

    // Tests layers
    ASSERT_EQ(net_export->layers.size(), net_import->layers.size());
    for(int i=0; i<net_export->layers.size(); i++){
        // Cannot test the name because there is a static counter per layer
//        ASSERT_TRUE(net_export->layers[i]->name == net_import->layers[i]->name);

        // Check subclasses
        ASSERT_TRUE(typeid(*net_export->layers[i]) == typeid(*net_import->layers[i]));

        // Check array content
        ASSERT_TRUE(Tensor::equal2(net_export->layers[i]->input, net_import->layers[i]->input));
        ASSERT_TRUE(Tensor::equal2(net_export->layers[i]->output, net_import->layers[i]->output));
    }

    // Tests input layers
    ASSERT_EQ(net_export->lin.size(), net_import->lin.size());
    for(int i=0; i<net_export->lin.size(); i++){
        // Cannot test the name because there is a static counter per layer
//        ASSERT_TRUE(net_export->lin[i]->name == net_import->lin[i]->name);

        // Check subclasses
        ASSERT_TRUE(typeid(*net_export->lin[i]) == typeid(*net_import->lin[i]));

        // Check array content
        ASSERT_TRUE(Tensor::equal2(net_export->lin[i]->input, net_import->lin[i]->input));
        ASSERT_TRUE(Tensor::equal2(net_export->lin[i]->output, net_import->lin[i]->output));
    }


    // Tests output layers
    ASSERT_EQ(net_export->lout.size(), net_import->lout.size());
    for(int i=0; i<net_export->lout.size(); i++){
        // Cannot test the name because there is a static counter per layer
//        ASSERT_TRUE(net_export->lout[i]->name == net_import->lout[i]->name);
        ASSERT_TRUE(net_export->lout[i]->name == net_import->lout[i]->name);

        // Check subclasses
        ASSERT_TRUE(typeid(*net_export->lout[i]) == typeid(*net_import->lout[i]));

        // Check array content
        ASSERT_TRUE(Tensor::equal2(net_export->lout[i]->input, net_import->lout[i]->input));
        ASSERT_TRUE(Tensor::equal2(net_export->lout[i]->output, net_import->lout[i]->output));
    }

}

