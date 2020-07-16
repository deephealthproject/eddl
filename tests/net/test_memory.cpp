#include <gtest/gtest.h>


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/apis/eddl.h"

#include "eddl/tensor/tensor.h"
#include "eddl/tensor/nn/tensor_nn.h"
#include "eddl/descriptors/descriptors.h"


using namespace eddl;


//TEST(NetTestSuite, memory_leaks_select){
//    layer in=Input({3, 32, 32});
//    auto l = new LSelect(in, {":", "0:31", "0:31"}, "mylayer", DEV_CPU, 0);
//
//    delete l;
//    std::cout << "layer deleted" << std::endl;
//    delete in;
//
//    ASSERT_TRUE(true);
//}
//
//TEST(NetTestSuite, net1_memory_leaks){
//    // Define network
//    layer in=Input({3, 32, 32});
//    layer l=in;
//
//    l = Conv(l,32,{3,3},{1,1});
//    l = ReLu(l);
//    l = LeakyReLu(l);
//    l = Flatten(l);
//    l = Dense(l, 10);
//    layer out = Softmax(l);  // num_classes
//    model net = Model({in}, {out});
//
//    optimizer opt = rmsprop(0.01);
//    vector<string> lo = {"soft_cross_entropy"};
//    vector<string> me = {"categorical_accuracy"};
//    compserv cs = CS_CPU();
//
//    // Build model
//    build(net, opt, lo, me, cs);
//
//    delete net;
//    std::cout << "model deleted" << std::endl;
//
//    ASSERT_TRUE(true);
//}
