#include <gtest/gtest.h>


#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/apis/eddl.h"

#include "eddl/tensor/tensor.h"
#include "eddl/tensor/nn/tensor_nn.h"
#include "eddl/descriptors/descriptors.h"


using namespace eddl;

// Fast check for GPU: CS_GP-U({1}, "low_mem")
// Fast check for CPU: CS_CP-U()

TEST(NetTestSuite, memory_leaks_select){
    layer in=Input({3, 32, 32});
    auto l = new LSelect(in, {":", "0:31", "0:31"}, "mylayer", DEV_CPU, 0);

    delete l;
    std::cout << "layer deleted" << std::endl;
    delete in;

    ASSERT_TRUE(true);
}

TEST(NetTestSuite, net_delete_mnist_mlp){
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

    // Build model
    build(net,
          rmsprop(0.01), // Optimizer
          {"categorical_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          CS_CPU()
    );
    delete net;

    ASSERT_TRUE(true);
}


TEST(NetTestSuite, net_delete_mnist_initializers){
    int num_classes = 10;

    // Define network
    layer in = Input({784});
    layer l = in;  // Aux var

    l = ReLu(GlorotNormal(Dense(l, 1024)));
    l = ReLu(GlorotUniform(Dense(l, 1024)));
    l = ReLu(RandomNormal(Dense(l, 1024),0.0,0.1));

    layer out = Activation(Dense(l, num_classes), "softmax");
    model net = Model({in}, {out});
    net->verbosity_level = 0;

    // Build model
    build(net,
          sgd(0.01, 0.9), // Optimizer
          {"categorical_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          CS_CPU()
    );
    delete net;

    ASSERT_TRUE(true);
}



TEST(NetTestSuite, net_delete_mnist_regularizers){
    int num_classes = 10;

    // Define network
    layer in = Input({784});
    layer l = in;  // Aux var

    l = ReLu(L2(Dense(l, 1024),0.0001));
    l = ReLu(L1(Dense(l, 1024),0.0001));
    l = ReLu(L1L2(Dense(l, 1024),0.00001,0.0001));

    layer out = Activation(Dense(l, num_classes), "softmax");
    model net = Model({in}, {out});
    net->verbosity_level = 0;

    // Build model
    build(net,
          sgd(0.01, 0.9), // Optimizer
          {"categorical_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          CS_CPU()
    );
    delete net;
    ASSERT_TRUE(true);
}

TEST(NetTestSuite, net_delete_mnist_da){
    int num_classes = 10;

    // Define network
    layer in = Input({784});
    layer l = in;  // Aux var

    // Data augmentation assumes 3D tensors... images:
    l=Reshape(l,{1,28,28});

    // Data augmentation
    l = RandomCropScale(l, {0.9f, 1.0f});

    // Come back to 1D tensor for fully connected:
    l=Reshape(l,{-1});
    l = ReLu(GaussianNoise(BatchNormalization(Dense(l, 1024)),0.3));
    l = ReLu(GaussianNoise(BatchNormalization(Dense(l, 1024)),0.3));
    l = ReLu(GaussianNoise(BatchNormalization(Dense(l, 1024)),0.3));
    //l = ReLu(Dense(l, 1024));
    //l = ReLu(Dense(l, 1024));

    layer out = Activation(Dense(l, num_classes), "softmax");
    model net = Model({in}, {out});
    net->verbosity_level = 0;

    // Build model
    build(net,
          sgd(0.01, 0.9), // Optimizer
          {"categorical_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          CS_CPU()
    );
    delete net;

    ASSERT_TRUE(true);
}

// mnist_mlp_train_batch => No needed. Redundant
// mnist_mlp_auto_encoder => No needed. Redundant

TEST(NetTestSuite, net_delete_mnist_conv){
    int num_classes = 10;

    // Define network
    layer in = Input({784});
    layer l = in;  // Aux var

    l = Reshape(l,{1,28,28});
    l = MaxPool(ReLu(Conv(l,32, {3,3},{1,1})),{3,3}, {1,1}, "same");
    l = MaxPool(ReLu(Conv(l,64, {3,3},{1,1})),{2,2}, {2,2}, "same");
    l = MaxPool(ReLu(Conv(l,128,{3,3},{1,1})),{3,3}, {2,2}, "none");
    l = MaxPool(ReLu(Conv(l,256,{3,3},{1,1})),{2,2}, {2,2}, "none");
    l = Reshape(l,{-1});

    layer out = Activation(Dense(l, num_classes), "softmax");
    model net = Model({in}, {out});
    net->verbosity_level = 0;

    // Build model
    build(net,
          rmsprop(0.01), // Optimizer
          {"categorical_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          CS_CPU()
    );
    delete net;

    ASSERT_TRUE(true);
}


TEST(NetTestSuite, net_delete_mnist_rnn){
    int num_classes = 10;

    // Define network
    layer in = Input({28});
    layer l = in;  // Aux var

    l = LeakyReLu(Dense(l, 32));
    //l = L2(RNN(l, 128, "relu"),0.001);
    l = L2(LSTM(l, 128),0.001);
    l = LeakyReLu(Dense(l, 32));

    layer out = Softmax(Dense(l, num_classes));
    model net = Model({in}, {out});
    net->verbosity_level = 0;

    // Build model
    build(net,
          rmsprop(0.001), // Optimizer
          {"categorical_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          CS_CPU()
    );
    delete net;

    ASSERT_TRUE(true);
}

TEST(NetTestSuite, net_delete_mnist_conv1D){
    int num_classes = 10;

    // Define network
    layer in = Input({784});
    layer l = in;  // Aux var

    l = Reshape(l,{1,784}); //image as a 1D signal with depth=1
    l = MaxPool1D(ReLu(Conv1D(l,16, {3},{1})),{4},{4});  //MaxPool 4 stride 4
    l = MaxPool1D(ReLu(Conv1D(l,32, {3},{1})),{4},{4});
    l = MaxPool1D(ReLu(Conv1D(l,64,{3},{1})),{4},{4});
    l = MaxPool1D(ReLu(Conv1D(l,64,{3},{1})),{4},{4});
    l = Reshape(l,{-1});

    layer out = Activation(Dense(l, num_classes), "softmax");
    model net = Model({in}, {out});
    net->verbosity_level = 0;

    // Build model
    build(net,
          rmsprop(0.01), // Optimizer
          {"categorical_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          CS_CPU()
    );
    delete net;

    ASSERT_TRUE(true);
}



// Auxiliary function for: net_delete_cifar_resnet50_da_bg
layer BN(layer l){
    return BatchNormalization(l);
}

// Auxiliary function for: net_delete_cifar_resnet50_da_bg
layer BG(layer l) {
    return BN(l);
}

// Auxiliary function for: net_delete_cifar_resnet50_da_bg
layer ResBlock(layer l, int filters,int half, int expand=0){
    layer in=l;

    l=ReLu(BG(Conv(l,filters,{1,1},{1,1},"same",false)));

    if (half)
        l=ReLu(BG(Conv(l,filters,{3,3},{2,2},"same",false)));
    else
        l=ReLu(BG(Conv(l,filters,{3,3},{1,1},"same",false)));

    l=BG(Conv(l,4*filters,{1,1},{1,1},"same",false));

    if (half)
        return ReLu(Sum(BG(Conv(in,4*filters,{1,1},{2,2},"same",false)),l));
    else
    if (expand) return ReLu(Sum(BG(Conv(in,4*filters,{1,1},{1,1},"same",false)),l));
    else return ReLu(Sum(in,l));
}

TEST(NetTestSuite, net_delete_cifar_resnet50_da_bg){
    int num_classes = 10;

    // network
    layer in=Input({3,32,32});
    layer l=in;

    // Data augmentation
    l = RandomCropScale(l, {0.8f, 1.0f});
    l = RandomHorizontalFlip(l);

    // Resnet-50

    l=ReLu(BG(Conv(l,64,{3,3},{1,1},"same",false))); //{1,1}
    //l=MaxPool(l,{3,3},{1,1},"same");

    for(int i=0;i<3;i++)
        l=ResBlock(l, 64, 0, i==0); // not half but expand the first

    for(int i=0;i<4;i++)
        l=ResBlock(l, 128,i==0);

    for(int i=0;i<6;i++)
        l=ResBlock(l, 256,i==0);

    for(int i=0;i<3;i++)
        l=ResBlock(l,512,i==0);

    l=MaxPool(l,{4,4});  // should be avgpool

    l=Reshape(l,{-1});

    layer out=Activation(Dense(l,num_classes),"softmax");

    // net define input and output layers list
    model net=Model({in},{out});


    // Build model
    build(net,
          sgd(0.001,0.9), // Optimizer
          {"categorical_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          CS_CPU()
    );
    delete net;
}

// Auxiliary function for: net_delete_drive_seg
layer UNetWithPadding(layer x, bool use_concat){
    layer x2;
    layer x3;
    layer x4;
    layer x5;

    int depth=32;

    x = LeakyReLu(Conv(x, depth, { 3,3 }, { 1, 1 }, "same"));
    x = LeakyReLu(Conv(x, depth, { 3,3 }, { 1, 1 }, "same"));
    x2 = MaxPool(x, { 2,2 }, { 2,2 });
    x2 = LeakyReLu(Conv(x2, 2*depth, { 3,3 }, { 1, 1 }, "same"));
    x2 = LeakyReLu(Conv(x2, 2*depth, { 3,3 }, { 1, 1 }, "same"));
    x3 = MaxPool(x2, { 2,2 }, { 2,2 });
    x3 = LeakyReLu(Conv(x3, 4*depth, { 3,3 }, { 1, 1 }, "same"));
    x3 = LeakyReLu(Conv(x3, 4*depth, { 3,3 }, { 1, 1 }, "same"));
    x4 = MaxPool(x3, { 2,2 }, { 2,2 });
    x4 = LeakyReLu(Conv(x4, 8*depth, { 3,3 }, { 1, 1 }, "same"));
    x4 = LeakyReLu(Conv(x4, 8*depth, { 3,3 }, { 1, 1 }, "same"));
    x5 = MaxPool(x4, { 2,2 }, { 2,2 });
    x5 = LeakyReLu(Conv(x5, 8*depth, { 3,3 }, { 1, 1 }, "same"));
    x5 = LeakyReLu(Conv(x5, 8*depth, { 3,3 }, { 1, 1 }, "same"));
    x5 = Conv(UpSampling(x5, { 2,2 }), 8*depth, { 2,2 }, { 1, 1 }, "same");

    if (use_concat) x4 = Concat({x4,x5});
    else x4 = Sum(x4,x5);
    x4 = LeakyReLu(Conv(x4, 8*depth, { 3,3 }, { 1, 1 }, "same"));
    x4 = LeakyReLu(Conv(x4, 8*depth, { 3,3 }, { 1, 1 }, "same"));
    x4 = Conv(UpSampling(x4, { 2,2 }), 4*depth, { 2,2 }, { 1, 1 }, "same");

    if (use_concat) x3 = Concat({x3,x4});
    else x3 = Sum(x3,x4);
    x3 = LeakyReLu(Conv(x3, 4*depth, { 3,3 }, { 1, 1 }, "same"));
    x3 = LeakyReLu(Conv(x3, 4*depth, { 3,3 }, { 1, 1 }, "same"));
    x3 = Conv(UpSampling(x3, { 2,2 }), 2*depth, { 2,2 }, { 1, 1 }, "same");

    if (use_concat) x2 = Concat({x2,x3});
    else x2 = Sum(x2,x3);
    x2 = LeakyReLu(Conv(x2, 2*depth, { 3,3 }, { 1, 1 }, "same"));
    x2 = LeakyReLu(Conv(x2, 2*depth, { 3,3 }, { 1, 1 }, "same"));
    x2 = Conv(UpSampling(x2, { 2,2 }), depth, { 2,2 }, { 1, 1 }, "same");

    if (use_concat) x = Concat({x,x2});
    else x = Sum(x,x2);
    x = LeakyReLu(Conv(x, depth, { 3,3 }, { 1, 1 }, "same"));
    x = LeakyReLu(Conv(x, depth, { 3,3 }, { 1, 1 }, "same"));
    x = Conv(x, 1, { 1,1 });

    return x;
}

TEST(NetTestSuite, net_delete_drive_seg_da) {

    int input_size = 584/4;
    int crop_size = 512/4;

    // Network for Data Augmentation
    layer in1=Input({3,input_size,input_size});
    layer in2=Input({1,input_size,input_size});

    layer l=Concat({in1,in2});   // Cat image and mask
    l= RandomCropScale(l, {0.9f, 1.0f}); // Random Crop and Scale to orig size
    l= CenteredCrop(l,{crop_size,crop_size});         // Crop to work with sizes power 2
    layer img=Select(l,{"0:3"}); // UnCat [0-2] image
    layer mask=Select(l,{"3"});  // UnCat [3] mask
    // Both, image and mask, have the same augmentation

    // Define DA model inputs
    model danet=Model({in1,in2},{});

    // Build model for DA
    build(danet);
    delete danet;
}

TEST(NetTestSuite, net_delete_drive_seg_concat) {
    int input_size = 584/4;
    int crop_size = 512/4;

    // Build SegNet
    bool use_concat = true;
    layer in=Input({3,crop_size,crop_size});
    layer out=Sigmoid(UNetWithPadding(in, use_concat));
    model segnet=Model({in},{out});
    build(segnet,
          adam(0.00001), // Optimizer
          {"mse"}, // Losses
          {"mse"}, // Metrics
          CS_CPU()
    );
    delete segnet;
}


TEST(NetTestSuite, net_delete_drive_seg_sum){
    int input_size = 584/4;
    int crop_size = 512/4;

    // Build SegNet
    bool use_concat = false;
    layer in=Input({3,crop_size,crop_size});
    layer out=Sigmoid(UNetWithPadding(in, use_concat));
    model segnet=Model({in},{out});
    build(segnet,
          adam(0.00001), // Optimizer
          {"mse"}, // Losses
          {"mse"}, // Metrics
          CS_CPU()
    );
    delete segnet;
}



TEST(NetTestSuite, net_delete_nlp_sentiment_rnn){
    // ERROR => malloc_consolidate(): invalid chunk size
    int embdim=32;
    int vocsize= 2000;

    // Define network
    layer in = Input({1}); //1 word
    layer l = in;

    layer lE = RandomUniform(Embedding(l, vocsize, 1,embdim),-0.05,0.05);

    l = RNN(lE,32);
    l = ReLu(Dense(l,256));

    layer out = Sigmoid(Dense(l, 1));
    model net = Model({in}, {out});

    optimizer opt=adam(0.001);
    //opt->set_clip_val(0.01);

    // Build model
    build(net,
          opt, // Optimizer
          {"categorical_cross_entropy"}, // Losses
          {"binary_accuracy"}, // Metrics
          CS_CPU()
    );

    delete net;
}


TEST(NetTestSuite, net_delete_nlp_sentiment_lstm){
    // ERROR => malloc_consolidate(): invalid chunk size
    int embdim=32;
    int vocsize=2000;

    // Define network
    layer in = Input({1}); //1 word
    layer l = in;

    layer lE = RandomUniform(Embedding(l, vocsize, 1,embdim),-0.05,0.05);

    l = LSTM(lE,32);
    l = ReLu(Dense(l,256));

    layer out = Sigmoid(Dense(l, 1));
    model net = Model({in}, {out});

    optimizer opt=adam(0.001);
    //opt->set_clip_val(0.01);

    // Build model
    build(net,
          opt, // Optimizer
          {"categorical_cross_entropy"}, // Losses
          {"binary_accuracy"}, // Metrics
          CS_CPU()
    );
    delete net;
}


TEST(NetTestSuite, net_delete_nlp_machine_translation){

}
