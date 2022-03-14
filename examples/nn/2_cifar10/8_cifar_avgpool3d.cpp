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
#include "eddl/serialization/onnx/eddl_onnx.h"


using namespace eddl;


//////////////////////////////////
// Text generation
// Only Decoder
//////////////////////////////////


int main(int argc, char **argv) {

    bool testing = false;
    bool use_cpu = false;
    
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--testing") == 0) testing = true;
        else if (strcmp(argv[i], "--cpu") == 0) use_cpu = true;
    }

/*
     // Image (force padding manually, I don't want surprises)
    auto *ptr_img = new float[3*7*7]{
            0, 0, 0, 0, 0, 0, 0,
            0, 0, 2, 1, 0, 1, 0,
            0, 2, 2, 1, 0, 0, 0,
            0, 0, 0, 2, 0, 1, 0,
            0, 0, 2, 1, 2, 0, 0,
            0, 2, 2, 0, 2, 0, 0,
            0, 0, 0, 0, 0, 0, 0,

            0, 0, 0, 0, 0, 0, 0,
            0, 2, 0, 2, 1, 2, 0,
            0, 0, 2, 0, 1, 0, 0,
            0, 1, 2, 0, 2, 2, 0,
            0, 0, 0, 2, 1, 2, 0,
            0, 1, 1, 1, 1, 1, 0,
            0, 0, 0, 0, 0, 0, 0,

            0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 2, 0, 0, 0,
            0, 2, 0, 2, 0, 0, 0,
            0, 2, 1, 1, 2, 1, 0,
            0, 0, 2, 0, 1, 2, 0,
            0, 1, 1, 1, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0,
    };
    auto* t_image = new Tensor({1, 3, 7, 7}, ptr_img, DEV_CPU);
    t_image->toGPU();

    // Forward
    auto *ptr_fwrd = new float[2*3*3]{-3, -2, -2,
                                      3, -9, -7,
                                      -2, -1, -2,

                                      -1, 1, 2,
                                      -3, 3, 5,
                                      3, 1, 2};
    auto* t_fwrd = new Tensor({1, 2, 3, 3}, ptr_fwrd, DEV_CPU);
    t_fwrd->toGPU();


        // Kernels (=> w0(3x 3x3), w1(3x 3x3))
    auto *ptr_kernels = new float[2*3*3*3]{
            -1,  1,  0,
            1, -1,  0,
            -1, -1, -1,

            -1,  0, 0,
            -1,  0, 1,
            -1, -1, 0,

            -1, 0, -1,
            0, 0,  0,
            1, 0,  0,


            0, -1, 0,
            -1, 0, 1,
            1, 0, -1,

            0, 0, -1,
            1, 0,  0,
            1, 0, -1,

            1, 1, 0,
            0, 0, 1,
            0, 0, 0,
    };

    // Biases (One per kernel)
    auto *ptr_bias = new float[2]{0.0, 0.0};


     // Operation
    auto *cd = new ConvolDescriptorT(2, {3, 3}, {2, 2}, "none", {}, 1, {1, 1}, true);
    cd->build(t_image);
    //cd->K = new Tensor({2, 3, 3, 3}, ptr_kernels, DEV_CPU);
    cd->K = new Tensor({3, 2, 3, 3}, ptr_kernels, DEV_CPU);
    cd->K->toGPU();
    cd->bias = new Tensor({2}, ptr_bias, DEV_CPU); //Tensor::zeros(cd->bias->getShape());
    cd->bias->toGPU();
    cd->ID = Tensor::zeros(cd->I->getShape());
    cd->D = Tensor::ones(cd->O->getShape());
cd->O->toCPU();
    cout<<"Output"<<endl;
    cd->O->print(2);
    //cout<<"I: "<<cd->I->getShape()<<", O: "<<cd->O->getShape()<<endl;
    // Forward
    tensorNN::Conv2DT(cd);
    cd->O->toCPU();
    cout<<"Output"<<endl;
    cd->O->print(2);
    //ASSERT_TRUE((bool) Tensor::equivalent(t_fwrd, cd->O, 1e-3f, 0.0f, true, true));
    //cd->O->toCPU();
    cd->I->toCPU();
    //cout<<"Conv2D"<<endl;
    cout<<"Input"<<endl;
    cd->I->print(2);
   // cout<<"Output"<<endl;
   // cd->O->print(2);

    auto *cdt = new ConvolDescriptorT(2, {3, 3}, {2, 2}, "none", {}, 1, {1, 1}, true);
    cd->O->toGPU();
    cdt->build(cd->O);
    cdt->I=cd->O;
    cdt->I->toGPU();
    cdt->K = new Tensor({2, 3, 3, 3}, ptr_kernels, DEV_CPU);
    cdt->K->toGPU();
    cdt->bias = new Tensor({2}, ptr_bias, DEV_CPU); //Tensor::zeros(cd->bias->getShape());
    cdt->bias->toGPU();
    cdt->ID = Tensor::zeros(cdt->I->getShape());
    //cout<<"I: "<<cd->I->getShape()<<endl;
    cdt->D = Tensor::ones(cdt->O->getShape());
    //cout<<"I: "<<cdt->I->getShape()<<", O: "<<cdt->O->getShape()<<endl;

    cout<<"LISTO!"<<endl;
    tensorNN::Conv2DT(cdt);
    cout<<"ConvT2D"<<endl;
    cdt->O->toCPU();
    cdt->I->toCPU();
    cout<<"Input"<<endl;
    cdt->I->print(2);
    cout<<"Output"<<endl;
    cdt->O->print(2);

*/

    int data_length = 2;
    int batch_size = 2;

    int n_kernels = 2;
    int channels = 3;
    int size = 16;
    int outs=128;
    layer in = Input({channels, size, size});
    layer l = in;


    l=Conv2D(l, n_kernels, {3, 3}, {2, 2});
    l=ConvT2D(l, n_kernels+2, {3,3}, {2,2});
    l=Conv2D(l, n_kernels, {3, 3}, {2, 2});
    l = Reshape(l, {-1});
    layer out = l;
    model net = Model({in},{out});

    compserv cs = nullptr;
    if (use_cpu) {
        cs = CS_CPU();
    } else {
        cs = CS_GPU(); // one GPU
    }

    build(net,
          adam(0.01),
          {"mse"},
          {"mse"},
          cs
          );
    plot(net,"model.pdf","LR");
    summary(net);

    Tensor * x_train = Tensor::ones({batch_size*100,channels, size,size});
    Tensor * y_train = Tensor::ones({batch_size*100,outs});
    fit(net, {x_train}, {y_train}, batch_size, testing ? 100 : 10);


    Tensor* inpp = getOutput(net->layers[0]);
    Tensor* conv_out = getOutput(net->layers[1]);
    Tensor* convt_out = getOutput(net->layers[2]);

    cout << "Input data:" << endl;
    inpp->print(2,true);
    cout << "Conv output:" << endl;
    conv_out->print(2,true);
    cout << "convt output:" << endl;
    convt_out->print(2,true);

    delete net;
   
    
    return EXIT_SUCCESS;
}
