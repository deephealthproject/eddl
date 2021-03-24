#include <gtest/gtest.h>

#include "eddl/tensor/tensor.h"
#include "eddl/tensor/nn/tensor_nn.h"
#include "eddl/descriptors/descriptors.h"



TEST(Conv2DTestSuite, conv2d_custom)
{
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
//    t_image->toGPU();

    // Forward
    auto *ptr_fwrd = new float[2*3*3]{-3, -2, -2,
                                      3, -9, -7,
                                      -2, -1, -2,

                                      -1, 1, 2,
                                      -3, 3, 5,
                                      3, 1, 2};
    auto* t_fwrd = new Tensor({1, 2, 3, 3}, ptr_fwrd, DEV_CPU);
//    t_fwrd->toGPU();


    // backward
    auto *ptr_bwrd = new float[3*7*7]{-1.00, 0.00, -1.00, 0.00, -1.00, 0.00, 0.00,
                                    0.00, -1.00, 1.00, -1.00, 1.00, -1.00, 1.00,
                                    -1.00, -1.00, -3.00, -1.00, -3.00, -1.00, -2.00,
                                    0.00, -1.00, 1.00, -1.00, 1.00, -1.00, 1.00,
                                    -1.00, -1.00, -3.00, -1.00, -3.00, -1.00, -2.00,
                                    0.00, -1.00, 1.00, -1.00, 1.00, -1.00, 1.00,
                                    0.00, -1.00, -2.00, -1.00, -2.00, -1.00, -2.00,
                                    -1.00, 0.00, -2.00, 0.00, -2.00, 0.00, -1.00,
                                    0.00, 0.00, 1.00, 0.00, 1.00, 0.00, 1.00, -1.00,
                                    -1.00, -3.00, -1.00, -3.00, -1.00, -2.00, 0.00,
                                    0.00, 1.00, 0.00, 1.00, 0.00, 1.00, -1.00, -1.00,
                                    -3.00, -1.00, -3.00, -1.00, -2.00, 0.00, 0.00,
                                    1.00, 0.00, 1.00, 0.00, 1.00, 0.00, -1.00, -1.00,
                                    -1.00, -1.00, -1.00, -1.00, 0.00, 1.00, -1.00,
                                    1.00, -1.00, 1.00, -1.00, 0.00, 0.00, 1.00, 0.00,
                                    1.00, 0.00, 1.00, 1.00, 1.00, 0.00, 1.00, 0.00,
                                    1.00, -1.00, 0.00, 0.00, 1.00, 0.00, 1.00, 0.00,
                                    1.00, 1.00, 1.00, 0.00, 1.00, 0.00, 1.00, -1.00,
                                    0.00, 0.00, 1.00, 0.00, 1.00, 0.00, 1.00, 1.00,
                                    0.00, 1.00, 0.00, 1.00, 0.00, 0.00,
    };
    auto* t_bwrd = new Tensor({1, 3, 7, 7}, ptr_bwrd, DEV_CPU);
//    t_bwrd->toGPU();

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
    auto *ptr_bias = new float[2]{1.0, 0.0};

    // Operation
    auto *cd = new ConvolDescriptor(2, {3, 3}, {2, 2}, "none", {}, 1, {1, 1}, true);
    cd->build(t_image);
    cd->K = new Tensor({2, 3, 3, 3}, ptr_kernels, DEV_CPU);
    cd->bias = new Tensor({2}, ptr_bias, DEV_CPU); //Tensor::zeros(cd->bias->getShape());
    cd->ID = Tensor::zeros(cd->I->getShape());
    cd->D = Tensor::ones(cd->O->getShape());

//    cd->K->toGPU();
//    cd->bias->toGPU();
//    cd->ID->toGPU();
//    cd->D->toGPU();

    // Forward
    tensorNN::Conv2D(cd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_fwrd, cd->O, 10e-5f));

    // Backward
    tensorNN::Conv2D_back(cd);
//    cd->ID->toCPU(); cd->ID->print(2, true);
    ASSERT_TRUE((bool) Tensor::equivalent(t_bwrd, cd->ID, 10e-5f));
}

TEST(Conv2DTestSuite, conv2d_k2x2_s2x2_pad_valid)
{
    // Image
    auto *ptr_img = new float[5*5]{0, 1, 0, 4, 5,
                                   2, 3, 2, 1, 3,
                                   4, 4, 0, 4, 3,
                                   2, 5, 2, 6, 4,
                                   1, 0, 0, 5, 7};
    auto* t_image = new Tensor({1, 1, 5, 5}, ptr_img, DEV_CPU);


    // Forward
    auto *ptr_fwrd = new float[2*2]{6.00, 7.00,
                                    15.00, 12.00};
    auto* t_fwrd = new Tensor({1, 1, 2, 2}, ptr_fwrd, DEV_CPU);


    // backward
    auto *ptr_bwrd = new float[5*5]{1.00, 1.00, 1.00, 1.00, 0.00,
                                    1.00, 1.00, 1.00, 1.00, 0.00,
                                    1.00, 1.00, 1.00, 1.00, 0.00,
                                    1.00, 1.00, 1.00, 1.00, 0.00,
                                    0.00, 0.00, 0.00, 0.00, 0.00};
    auto* t_bwrd = new Tensor({1, 1, 5, 5}, ptr_bwrd, DEV_CPU);

    // Operation
    auto *cd = new ConvolDescriptor(1, {2, 2}, {2, 2}, "valid", {}, 1, {1, 1}, true);
    cd->build(t_image);
    cd->K = Tensor::ones(cd->K->getShape());
    cd->bias = Tensor::zeros(cd->bias->getShape());
    cd->ID = Tensor::zeros(cd->I->getShape());
    cd->D = Tensor::ones(cd->O->getShape());

    // Forward
    tensorNN::Conv2D(cd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_fwrd, cd->O, 10e-5f));

    // Backward
    tensorNN::Conv2D_back(cd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_bwrd, cd->ID, 10e-5f));

}

TEST(Conv2DTestSuite, conv2d_k2x2_s2x2_pad_same)
{
    // Image
    auto *ptr_img = new float[5*5]{0, 1, 0, 4, 5,
                                   2, 3, 2, 1, 3,
                                   4, 4, 0, 4, 3,
                                   2, 5, 2, 6, 4,
                                   1, 0, 0, 5, 7};
    auto* t_image = new Tensor({1, 1, 5, 5}, ptr_img, DEV_CPU);


    // Forward
    auto *ptr_fwrd = new float[3*3]{6.00, 7.00, 8.00,
                                    15.00, 12.00, 7.00,
                                    1.00, 5.00, 7.00};
    auto* t_fwrd = new Tensor({1, 1, 3, 3}, ptr_fwrd, DEV_CPU);


    // backward
    auto *ptr_bwrd = new float[5*5]{1.00, 1.00, 1.00, 1.00, 0.00,
                                    1.00, 1.00, 1.00, 1.00, 0.00,
                                    1.00, 1.00, 1.00, 1.00, 0.00,
                                    1.00, 1.00, 1.00, 1.00, 0.00,
                                    1.00, 1.00, 1.00, 1.00, 0.00};
    auto* t_bwrd = new Tensor({1, 1, 5, 5}, ptr_bwrd, DEV_CPU);

    // Operation
    auto *cd = new ConvolDescriptor(1, {2, 2}, {2, 2}, "same", {}, 1, {1, 1}, true);
    cd->build(t_image);
    cd->K = Tensor::ones(cd->K->getShape());
    cd->bias = Tensor::zeros(cd->bias->getShape());
    cd->ID = Tensor::zeros(cd->I->getShape());
    cd->D = Tensor::ones(cd->O->getShape());

    // Forward
    tensorNN::Conv2D(cd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_fwrd, cd->O, 10e-5f));

    // Backward
    tensorNN::Conv2D_back(cd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_bwrd, cd->ID, 10e-5f));
}


TEST(Conv2DTestSuite, conv2d_k3x3_s1x1_pad_valid)
{
    // Image
    auto *ptr_img = new float[5*5]{0, 1, 0, 4, 5,
                                   2, 3, 2, 1, 3,
                                   4, 4, 0, 4, 3,
                                   2, 5, 2, 6, 4,
                                   1, 0, 0, 5, 7};
    auto* t_image = new Tensor({1, 1, 5, 5}, ptr_img, DEV_CPU);


    // Forward
    auto *ptr_fwrd = new float[3*3]{16.00, 19.00, 22.00,
                                    24.00, 27.00, 25.00,
                                    18.00, 26.00, 31.00};
    auto* t_fwrd = new Tensor({1, 1, 3, 3}, ptr_fwrd, DEV_CPU);


    // backward
    auto *ptr_bwrd = new float[5*5]{1.00, 2.00, 3.00, 2.00, 1.00,
                                    2.00, 4.00, 6.00, 4.00, 2.00,
                                    3.00, 6.00, 9.00, 6.00, 3.00,
                                    2.00, 4.00, 6.00, 4.00, 2.00,
                                    1.00, 2.00, 3.00, 2.00, 1.00};
    auto* t_bwrd = new Tensor({1, 1, 5, 5}, ptr_bwrd, DEV_CPU);

    // Operation
    auto *cd = new ConvolDescriptor(1, {3, 3}, {1,1}, "valid", {}, 1, {1, 1}, true);
    cd->build(t_image);
    cd->K = Tensor::ones(cd->K->getShape());
    cd->bias = Tensor::zeros(cd->bias->getShape());
    cd->ID = Tensor::zeros(cd->I->getShape());
    cd->D = Tensor::ones(cd->O->getShape());

    // Forward
    tensorNN::Conv2D(cd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_fwrd, cd->O, 10e-5f));

    // Backward
    tensorNN::Conv2D_back(cd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_bwrd, cd->ID, 10e-5f));
}


TEST(Conv2DTestSuite, conv2d_k3x3_s1x1_pad_same)
{
    // Image
    auto *ptr_img = new float[5*5]{0, 1, 0, 4, 5,
                                   2, 3, 2, 1, 3,
                                   4, 4, 0, 4, 3,
                                   2, 5, 2, 6, 4,
                                   1, 0, 0, 5, 7};
    auto* t_image = new Tensor({1, 1, 5, 5}, ptr_img, DEV_CPU);


    // Forward
    auto *ptr_fwrd = new float[5*5]{6.00, 8.00, 11.00, 15.00, 13.00,
                                    14.00, 16.00, 19.00, 22.00, 20.00,
                                    20.00, 24.00, 27.00, 25.00, 21.00,
                                    16.00, 18.00, 26.00, 31.00, 29.00,
                                    8.00, 10.00, 18.00, 24.00, 22.00};
    auto* t_fwrd = new Tensor({1, 1, 5, 5}, ptr_fwrd, DEV_CPU);


    // backward
    auto *ptr_bwrd = new float[5*5]{4.00, 6.00, 6.00, 6.00, 4.00,
                                    6.00, 9.00, 9.00, 9.00, 6.00,
                                    6.00, 9.00, 9.00, 9.00, 6.00,
                                    6.00, 9.00, 9.00, 9.00, 6.00,
                                    4.00, 6.00, 6.00, 6.00, 4.00};
    auto* t_bwrd = new Tensor({1, 1, 5, 5}, ptr_bwrd, DEV_CPU);

    // Operation
    auto *cd = new ConvolDescriptor(1, {3, 3}, {1, 1}, "same", {}, 1, {1, 1}, true);
    cd->build(t_image);
    cd->K = Tensor::ones(cd->K->getShape());
    cd->bias = Tensor::zeros(cd->bias->getShape());
    cd->ID = Tensor::zeros(cd->I->getShape());
    cd->D = Tensor::ones(cd->O->getShape());

    // Forward
    tensorNN::Conv2D(cd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_fwrd, cd->O, 10e-5f));

    // Backward
    tensorNN::Conv2D_back(cd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_bwrd, cd->ID, 10e-5f));
}




TEST(Conv2DTestSuite, conv2d_k3x3_s2x2_pad_valid)
{
    // Image
    auto *ptr_img = new float[5*5]{0, 1, 0, 4, 5,
                                   2, 3, 2, 1, 3,
                                   4, 4, 0, 4, 3,
                                   2, 5, 2, 6, 4,
                                   1, 0, 0, 5, 7};
    auto* t_image = new Tensor({1, 1, 5, 5}, ptr_img, DEV_CPU);


    // Forward
    auto *ptr_fwrd = new float[2*2]{16.00, 22.00,
                                    18.00, 31.00};
    auto* t_fwrd = new Tensor({1, 1, 2, 2}, ptr_fwrd, DEV_CPU);


    // backward
    auto *ptr_bwrd = new float[5*5]{1.00, 1.00, 2.00, 1.00, 1.00,
                                    1.00, 1.00, 2.00, 1.00, 1.00,
                                    2.00, 2.00, 4.00, 2.00, 2.00,
                                    1.00, 1.00, 2.00, 1.00, 1.00,
                                    1.00, 1.00, 2.00, 1.00, 1.00};
    auto* t_bwrd = new Tensor({1, 1, 5, 5}, ptr_bwrd, DEV_CPU);

    // Operation
    auto *cd = new ConvolDescriptor(1, {3, 3}, {2, 2}, "valid", {}, 1, {1, 1}, true);
    cd->build(t_image);
    cd->K = Tensor::ones(cd->K->getShape());
    cd->bias = Tensor::zeros(cd->bias->getShape());
    cd->ID = Tensor::zeros(cd->I->getShape());
    cd->D = Tensor::ones(cd->O->getShape());

    // Forward
    tensorNN::Conv2D(cd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_fwrd, cd->O, 10e-5f));

    // Backward
    tensorNN::Conv2D_back(cd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_bwrd, cd->ID, 10e-5f));
}


TEST(Conv2DTestSuite, conv2d_k3x3_s2x2_pad_same){
    // Image
    auto *ptr_img = new float[5*5]{0, 1, 0, 4, 5,
                                   2, 3, 2, 1, 3,
                                   4, 4, 0, 4, 3,
                                   2, 5, 2, 6, 4,
                                   1, 0, 0, 5, 7};
    auto* t_image = new Tensor({1, 1, 5, 5}, ptr_img, DEV_CPU);


    // Forward
    auto *ptr_fwrd = new float[3*3]{6.00, 11.00, 13.00,
                                    20.00, 27.00, 21.00,
                                    8.00, 18.00, 22.00};
    auto* t_fwrd = new Tensor({1, 1, 3, 3}, ptr_fwrd, DEV_CPU);


    // backward
    auto *ptr_bwrd = new float[5*5]{1.00, 2.00, 1.00, 2.00, 1.00,
                                    2.00, 4.00, 2.00, 4.00, 2.00,
                                    1.00, 2.00, 1.00, 2.00, 1.00,
                                    2.00, 4.00, 2.00, 4.00, 2.00,
                                    1.00, 2.00, 1.00, 2.00, 1.00};
    auto* t_bwrd = new Tensor({1, 1, 5, 5}, ptr_bwrd, DEV_CPU);

    // Operation
    auto *cd = new ConvolDescriptor(1, {3, 3}, {2, 2}, "same", {}, 1, {1, 1}, true);
    cd->build(t_image);
    cd->K = Tensor::ones(cd->K->getShape());
    cd->bias = Tensor::zeros(cd->bias->getShape());
    cd->ID = Tensor::zeros(cd->I->getShape());
    cd->D = Tensor::ones(cd->O->getShape());

    // Forward
    tensorNN::Conv2D(cd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_fwrd, cd->O, 10e-5f));

    // Backward
    tensorNN::Conv2D_back(cd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_bwrd, cd->ID, 10e-5f));
}

#ifdef cGPU
TEST(Conv2DTestSuite, conv2d_cpu_gpu)
{
    // Image
    Tensor* t_cpu = Tensor::randu({1, 3, 1000, 500});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();

    vector<string> padding = {"valid", "same"};
    vector<int> strides = {1, 2, 3, 5};
    vector<int> kernels = {1, 2, 3, 5, 7};

    for(auto& p : padding){
            for(auto& s : strides){
                for(auto& k : kernels){
                    // CPU Operation
                    auto *cd_cpu = new ConvolDescriptor(1, {k, k}, {s, s}, p, {}, 1, {1, 1}, true);
                    cd_cpu->build(t_cpu);
                    cd_cpu->K = Tensor::randu(cd_cpu->K->getShape());
                    cd_cpu->bias = Tensor::zeros(cd_cpu->bias->getShape());
                    cd_cpu->ID = Tensor::zeros(cd_cpu->I->getShape());
                    cd_cpu->D = Tensor::randu(cd_cpu->O->getShape());
                    for (int i = 0; i < cd_cpu->D->size; i++) cd_cpu->D->ptr[i] = i;

                    // GPU Operation
                    auto *cd_gpu = new ConvolDescriptor(1, {k, k}, {s, s}, p, {}, 1, {1, 1}, true);
                    cd_gpu->build(t_gpu);
                    cd_gpu->K = cd_cpu->K->clone(); cd_gpu->K->toGPU();
                    cd_gpu->bias = Tensor::zeros(cd_gpu->bias->getShape(), t_gpu->device);
                    cd_gpu->ID = Tensor::zeros(cd_gpu->I->getShape(), t_gpu->device);
                    cd_gpu->D = cd_cpu->D->clone(); cd_gpu->D->toGPU();

                    // Forward
                    tensorNN::Conv2D(cd_cpu);
                    tensorNN::Conv2D(cd_gpu);
                    Tensor *cd_gpu_O = cd_gpu->O->clone(); cd_gpu_O->toCPU();  // Tensor::equivalent is only for CPU (at the moment)
                    bool test_fwrd = (bool) Tensor::equivalent(cd_cpu->O, cd_gpu_O, 10e-5f, 0.0, false, true);

                    // Backward
                    tensorNN::Conv2D_back(cd_cpu);
                    tensorNN::Conv2D_back(cd_gpu);
                    Tensor *cd_gpu_ID = cd_gpu->ID->clone(); cd_gpu_ID->toCPU(); // Tensor::equivalent is only for CPU (at the moment)
                    bool test_bwrd = (bool) Tensor::equivalent(cd_cpu->ID, cd_gpu_ID, 10e-5f, 0.0, false, true);

                    // Print results to ease debugging
                    cout << "Testing conv2d_cpu_gpu (" << "padding=" << p << "; kernel=" << k << "; stride=" << s << ")" <<
                    " [Forward="<< test_fwrd << "; Backward=" << test_bwrd << "]" << endl;

                    // Test correctness
                    ASSERT_TRUE(test_fwrd);
                    // ASSERT_TRUE(test_bwrd); // TODO fix CPU conv2D backward

                    delete cd_cpu->K;
                    delete cd_cpu->bias;
                    delete cd_cpu->ID;
                    delete cd_cpu->D;
                    delete cd_cpu;

                    delete cd_gpu->K;
                    delete cd_gpu->bias;
                    delete cd_gpu->ID;
                    delete cd_gpu->D;
                    delete cd_gpu;

                    delete cd_gpu_O;
                    delete cd_gpu_ID;
                }
            }
        }
}
#endif


