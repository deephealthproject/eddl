#include <gtest/gtest.h>

#include "eddl/tensor/tensor.h"
#include "eddl/tensor/nn/tensor_nn.h"
#include "eddl/descriptors/descriptors.h"


TEST(AvgPoolTestSuite, avgpool_k2x2_s2x2_pad_valid)
{
    // Image
    auto *ptr_img = new float[5*5]{0, 1, 0, 4, 5,
                                   2, 3, 2, 1, 3,
                                   4, 4, 0, 4, 3,
                                   2, 5, 2, 6, 4,
                                   1, 0, 0, 5, 7};
    auto* t_image = new Tensor({1, 1, 5, 5}, ptr_img, DEV_CPU);


    // Forward
    auto *ptr_fwrd = new float[2*2]{1.5, 1.75,
                                    3.75, 3.0};
    auto* t_fwrd = new Tensor({1, 1, 2, 2}, ptr_fwrd, DEV_CPU);


    // backward
    auto *ptr_bwrd = new float[5*5]{0.25000, 0.25000, 0.25000, 0.25000, 0.00000,
                                    0.25000, 0.25000, 0.25000, 0.25000, 0.00000,
                                    0.25000, 0.25000, 0.25000, 0.25000, 0.00000,
                                    0.25000, 0.25000, 0.25000, 0.25000, 0.00000,
                                    0.00000, 0.00000, 0.00000, 0.00000, 0.00000
    };
    auto* t_bwrd = new Tensor({1, 1, 5, 5}, ptr_bwrd, DEV_CPU);

    // Operation
    auto *pd = new PoolDescriptor({2, 2}, {2, 2}, "valid");
    pd->build(t_image);
    pd->ID = Tensor::zeros(pd->I->getShape());
    pd->D = Tensor::ones(pd->O->getShape());
    pd->indX = new Tensor(pd->O->getShape());
    pd->indY = new Tensor(pd->O->getShape());

    // Forward
    tensorNN::AvgPool2D(pd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_fwrd, pd->O, 1e-3f, 0.0f, true, true));

    // Backward
    tensorNN::AvgPool2D_back(pd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_bwrd, pd->ID, 1e-3f, 0.0f, true, true));
}


TEST(AvgPoolTestSuite, avgpool_k2x2_s2x2_pad_same)
{
    // Image
    auto *ptr_img = new float[5*5]{0, 1, 0, 4, 5,
                                   2, 3, 2, 1, 3,
                                   4, 4, 0, 4, 3,
                                   2, 5, 2, 6, 4,
                                   1, 0, 0, 5, 7};
    auto* t_image = new Tensor({1, 1, 5, 5}, ptr_img, DEV_CPU);


    // Forward
    auto *ptr_fwrd = new float[3*3]{1.5, 1.75, 2.0,
                                    3.75, 3.0, 1.75,
                                    0.25, 1.25, 1.75 };
    auto* t_fwrd = new Tensor({1, 1, 3, 3}, ptr_fwrd, DEV_CPU);


    // backward
    auto *ptr_bwrd = new float[5*5]{0.25, 0.25, 0.25, 0.25, 0.25,
                                    0.25, 0.25, 0.25, 0.25, 0.25,
                                    0.25, 0.25, 0.25, 0.25, 0.25,
                                    0.25, 0.25, 0.25, 0.25, 0.25,
                                    0.25, 0.25, 0.25, 0.25, 0.25 };
    auto* t_bwrd = new Tensor({1, 1, 5, 5}, ptr_bwrd, DEV_CPU);

    // Operation
    auto *pd = new PoolDescriptor({2, 2}, {2, 2}, "same");
    pd->build(t_image);
    pd->ID = Tensor::zeros(pd->I->getShape());
    pd->D = Tensor::ones(pd->O->getShape());
    pd->indX = new Tensor(pd->O->getShape());
    pd->indY = new Tensor(pd->O->getShape());

    // Forward
    tensorNN::AvgPool2D(pd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_fwrd, pd->O, 1e-3f, 0.0f, true, true));

    // Backward
    tensorNN::AvgPool2D_back(pd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_bwrd, pd->ID, 1e-3f, 0.0f, true, true));
}


TEST(AvgPoolTestSuite, avgpool_k3x3_s1x1_pad_valid)
{
    // Image
    auto *ptr_img = new float[5*5]{0, 1, 0, 4, 5,
                                   2, 3, 2, 1, 3,
                                   4, 4, 0, 4, 3,
                                   2, 5, 2, 6, 4,
                                   1, 0, 0, 5, 7};
    auto* t_image = new Tensor({1, 1, 5, 5}, ptr_img, DEV_CPU);


    // Forward
    auto *ptr_fwrd = new float[3*3]{1.77778, 2.11111, 2.44444,
                                    2.66667, 3.00000, 2.77778,
                                    2.00000, 2.88889, 3.44444};
    auto* t_fwrd = new Tensor({1, 1, 3, 3}, ptr_fwrd, DEV_CPU);


    // backward
    auto *ptr_bwrd = new float[5*5]{0.111111, 0.222222, 0.333333, 0.222222, 0.111111,
                                    0.222222, 0.444444, 0.666667, 0.444444, 0.222222,
                                    0.333333, 0.666667, 1.00000, 0.666667, 0.333333,
                                    0.222222, 0.444444, 0.666667, 0.444444, 0.222222,
                                    0.111111, 0.222222, 0.333333, 0.222222, 0.111111 };
    auto* t_bwrd = new Tensor({1, 1, 5, 5}, ptr_bwrd, DEV_CPU);

    // Operation
    auto *pd = new PoolDescriptor({3, 3}, {1,1}, "valid");
    pd->build(t_image);
    pd->ID = Tensor::zeros(pd->I->getShape());
    pd->D = Tensor::ones(pd->O->getShape());
    pd->indX = new Tensor(pd->O->getShape());
    pd->indY = new Tensor(pd->O->getShape());

    // Forward
    tensorNN::AvgPool2D(pd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_fwrd, pd->O, 1e-3f, 0.0f, true, true));

    // Backward
    tensorNN::AvgPool2D_back(pd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_bwrd, pd->ID, 1e-3f, 0.0f, true, true));
}


TEST(AvgPoolTestSuite, avgpool_k3x3_s1x1_pad_same)
{
    // Image
    auto *ptr_img = new float[5*5]{0, 1, 0, 4, 5,
                                   2, 3, 2, 1, 3,
                                   4, 4, 0, 4, 3,
                                   2, 5, 2, 6, 4,
                                   1, 0, 0, 5, 7};
    auto* t_image = new Tensor({1, 1, 5, 5}, ptr_img, DEV_CPU);


    // Forward
    auto *ptr_fwrd = new float[5*5]{0.666667, 0.888889, 1.22222, 1.66667, 1.44444,
                                    1.55556, 1.77778, 2.11111, 2.44444, 2.22222,
                                    2.22222, 2.66667, 3.00000, 2.77778, 2.33333,
                                    1.77778, 2.00000, 2.88889, 3.44444, 3.22222,
                                    0.888889, 1.11111, 2.00000, 2.66667, 2.44444 };
    auto* t_fwrd = new Tensor({1, 1, 5, 5}, ptr_fwrd, DEV_CPU);


    // backward
    auto *ptr_bwrd = new float[5*5]{0.444444, 0.666667, 0.666667, 0.666667, 0.444444,
                                    0.666667, 1.00000, 1.00000, 1.00000, 0.666667,
                                    0.666667, 1.00000, 1.00000, 1.00000, 0.666667,
                                    0.666667, 1.00000, 1.00000, 1.00000, 0.666667,
                                    0.444444, 0.666667, 0.666667, 0.666667, 0.444444 };
    auto* t_bwrd = new Tensor({1, 1, 5, 5}, ptr_bwrd, DEV_CPU);

    // Operation
    auto *pd = new PoolDescriptor({3, 3}, {1, 1}, "same");
    pd->build(t_image);
    pd->ID = Tensor::zeros(pd->I->getShape());
    pd->D = Tensor::ones(pd->O->getShape());
    pd->indX = new Tensor(pd->O->getShape());
    pd->indY = new Tensor(pd->O->getShape());

    // Forward
    tensorNN::AvgPool2D(pd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_fwrd, pd->O, 1e-3f, 0.0f, true, true));

    // Backward
    tensorNN::AvgPool2D_back(pd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_bwrd, pd->ID, 1e-3f, 0.0f, true, true));
}




TEST(AvgPoolTestSuite, avgpool_k3x3_s2x2_pad_valid)
{
    // Image
    auto *ptr_img = new float[5*5]{0, 1, 0, 4, 5,
                                   2, 3, 2, 1, 3,
                                   4, 4, 0, 4, 3,
                                   2, 5, 2, 6, 4,
                                   1, 0, 0, 5, 7};
    auto* t_image = new Tensor({1, 1, 5, 5}, ptr_img, DEV_CPU);


    // Forward
    auto *ptr_fwrd = new float[2*2]{1.77778, 2.44444,
                                    2, 3.44444};
    auto* t_fwrd = new Tensor({1, 1, 2, 2}, ptr_fwrd, DEV_CPU);


    // backward
    auto *ptr_bwrd = new float[5*5]{0.111111, 0.111111, 0.222222, 0.111111, 0.111111,
                                    0.111111, 0.111111, 0.222222, 0.111111, 0.111111,
                                    0.222222, 0.222222, 0.444444, 0.222222, 0.222222,
                                    0.111111, 0.111111, 0.222222, 0.111111, 0.111111,
                                    0.111111, 0.111111, 0.222222, 0.111111, 0.111111};
    auto* t_bwrd = new Tensor({1, 1, 5, 5}, ptr_bwrd, DEV_CPU);

    // Operation
    auto *pd = new PoolDescriptor({3, 3}, {2, 2}, "valid");
    pd->build(t_image);
    pd->ID = Tensor::zeros(pd->I->getShape());
    pd->D = Tensor::ones(pd->O->getShape());
    pd->indX = new Tensor(pd->O->getShape());
    pd->indY = new Tensor(pd->O->getShape());

    // Forward
    tensorNN::AvgPool2D(pd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_fwrd, pd->O, 1e-3f, 0.0f, true, true));

    // Backward
    tensorNN::AvgPool2D_back(pd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_bwrd, pd->ID, 1e-3f, 0.0f, true, true));
}


TEST(AvgPoolTestSuite, avgpool_k3x3_s2x2_pad_same)
{
    // Image
    auto *ptr_img = new float[5*5]{0, 1, 0, 4, 5,
                                   2, 3, 2, 1, 3,
                                   4, 4, 0, 4, 3,
                                   2, 5, 2, 6, 4,
                                   1, 0, 0, 5, 7};
    auto* t_image = new Tensor({1, 1, 5, 5}, ptr_img, DEV_CPU);


    // Forward
    auto *ptr_fwrd = new float[3*3]{0.666667, 1.22222, 1.44444,
                                    2.22222, 3.0, 2.33333,
                                    0.888889, 2.0, 2.44444
    };
    auto* t_fwrd = new Tensor({1, 1, 3, 3}, ptr_fwrd, DEV_CPU);


    // backward
    auto *ptr_bwrd = new float[5*5]{0.111111, 0.222222, 0.111111, 0.222222, 0.111111,
                                    0.222222, 0.444444, 0.222222, 0.444444, 0.222222,
                                    0.111111, 0.222222, 0.111111, 0.222222, 0.111111,
                                    0.222222, 0.444444, 0.222222, 0.444444, 0.222222,
                                    0.111111, 0.222222, 0.111111, 0.222222, 0.111111};
    auto* t_bwrd = new Tensor({1, 1, 5, 5}, ptr_bwrd, DEV_CPU);

    // Operation
    auto *pd = new PoolDescriptor({3, 3}, {2, 2}, "same");
    pd->build(t_image);
    pd->ID = Tensor::zeros(pd->I->getShape());
    pd->D = Tensor::ones(pd->O->getShape());
    pd->indX = new Tensor(pd->O->getShape());
    pd->indY = new Tensor(pd->O->getShape());

    // Forward
    tensorNN::AvgPool2D(pd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_fwrd, pd->O, 1e-3f, 0.0f, true, true));

    // Backward
    tensorNN::AvgPool2D_back(pd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_bwrd, pd->ID, 1e-3f, 0.0f, true, true));
}


#ifdef cGPU
TEST(MaxPoolTestSuite, avgpool_cpu_gpu){
    // Image
    Tensor* t_cpu = Tensor::randu({1, 3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();

    vector<string> padding = {"valid", "same"};
    vector<int> strides = {1, 2, 3, 5};
    vector<int> kernels = {1, 2, 3, 5, 7};

    for(auto& p : padding){
        for(auto& s : strides){
            for(auto& k : kernels){


                // CPU Operation
                auto *pd_cpu = new PoolDescriptor({k, k}, {s, s}, p);
                pd_cpu->build(t_cpu);
                pd_cpu->ID = Tensor::zeros(pd_cpu->I->getShape());
                pd_cpu->D = Tensor::ones(pd_cpu->O->getShape());
                pd_cpu->indX = new Tensor(pd_cpu->O->getShape());
                pd_cpu->indY = new Tensor(pd_cpu->O->getShape());

                // GPU Operation
                auto *pd_gpu = new PoolDescriptor({k, k}, {s, s}, p);
                pd_gpu->build(t_gpu);
                pd_gpu->ID = Tensor::zeros(pd_gpu->I->getShape(), t_gpu->device);
                pd_gpu->D = Tensor::ones(pd_gpu->O->getShape(), t_gpu->device);
                pd_gpu->indX = new Tensor(pd_gpu->O->getShape(), t_gpu->device);
                pd_gpu->indY = new Tensor(pd_gpu->O->getShape(), t_gpu->device);

                // Forward
                tensorNN::AvgPool2D(pd_cpu);
                tensorNN::AvgPool2D(pd_gpu);
                Tensor *pd_gpu_O = pd_gpu->O->clone(); pd_gpu_O->toCPU();  // Tensor::equivalent is only for CPU (at the moment)
                bool test_fwrd = (bool) Tensor::equivalent(pd_cpu->O, pd_gpu_O, 1e-3f, 0.0f, true, true);

                // Backward
                tensorNN::AvgPool2D_back(pd_cpu);
                tensorNN::AvgPool2D_back(pd_gpu);
                Tensor *pd_gpu_ID = pd_gpu->ID->clone(); pd_gpu_ID->toCPU(); // Tensor::equivalent is only for CPU (at the moment)
                bool test_bwrd = (bool) Tensor::equivalent(pd_cpu->ID, pd_gpu_ID, 1e-3f, 0.0f, true, true);

                // Print results to ease debugging
                cout << "Testing avgpool_cpu_gpu (" << "padding=" << p << "; kernel=" << k << "; stride=" << s << ")" <<
                     " [Forward="<< test_fwrd << "; Backward=" << test_bwrd << "]" << endl;

                // Test correctness
                ASSERT_TRUE(test_fwrd);
                ASSERT_TRUE(test_bwrd);

                delete pd_cpu->ID;
                delete pd_cpu->D;
                delete pd_cpu;

                delete pd_gpu->ID;
                delete pd_gpu->D;
                delete pd_gpu;

                delete pd_gpu_O;
                delete pd_gpu_ID;
            }
        }
    }
}
#endif


TEST(MaxPoolTestSuite, avgpool3d_k3x3x3_s3x3x3_pad_valid)
{
    // Image
    auto *ptr_img = new float[1*1*5*5*5]{
            0.13, 0.13, 0.98, -0.50, 0.26,
            1.74, -2.55, -0.60, -0.81, -1.35,
            1.21, -0.29, 0.44, -0.87, -0.73,
            -1.46, 0.12, -0.65, 0.86, 0.84,
            -1.08, -0.21, 0.54, -0.31, -1.04,
            0.07, 0.53, 0.02, 0.54, -0.88,
            0.67, -0.29, 0.77, -1.19, 0.90,
            2.29, 1.43, -1.06, 0.09, -0.09,
            -1.23, 0.78, 1.32, 1.10, -0.17,
            -0.18, 0.41, 0.72, 0.56, -0.42,
            0.57, -0.54, 0.23, 0.71, -1.83,
            0.94, -0.07, -0.18, -0.77, 2.35,
            0.42, -0.87, 2.12, -0.36, -0.06,
            1.02, 0.07, 0.20, 0.59, 0.82,
            -0.98, -0.41, 1.22, -1.95, 0.28,
            1.30, 1.22, 0.47, -1.32, -0.15,
            0.03, 0.71, 0.60, 0.08, 0.99,
            -0.81, -0.14, -0.31, 1.80, -0.90,
            -0.38, 0.28, -1.25, 1.38, 1.75,
            -0.62, 1.69, -0.94, 0.49, 0.56,
            0.65, 0.43, 0.36, 0.62, -1.44,
            -0.30, 0.24, -1.04, -0.21, -0.97,
            -0.12, 0.81, 1.17, -0.01, -0.41,
            0.12, 1.37, 0.37, -0.62, 1.33,
            -0.04, -2.55, -1.93, 0.21, 1.65,
    };

    auto* t_image = new Tensor({1, 1, 5, 5, 5}, ptr_img, DEV_CPU);
    t_image->print(2);

    // Forward
    auto *ptr_fwrd = new float[1]{0.31};
    auto* t_fwrd = new Tensor({1, 1, 1, 1, 1}, ptr_fwrd, DEV_CPU);


    // backward
//    auto *ptr_bwrd = new float[5*5]{0, 0, 0, 1, 0,
//                                    0, 1, 0, 0, 0,
//                                    0, 0, 0, 0, 0,
//                                    0, 1, 0, 1, 0,
//                                    0, 0, 0, 0, 0};
//    auto* t_bwrd = new Tensor({1, 1, 5, 5}, ptr_bwrd, DEV_CPU);

    // Operation
    auto *pd = new PoolDescriptor3D({3, 3, 3}, {3, 3, 3}, "valid");
    pd->build(t_image);
    pd->ID = Tensor::zeros(pd->I->getShape());
    pd->D = Tensor::ones(pd->O->getShape());
    pd->indX = new Tensor(pd->O->getShape());
    pd->indY = new Tensor(pd->O->getShape());
    pd->indZ = new Tensor(pd->O->getShape());

    // Forward
    tensorNN::AvgPool3D(pd);
    pd->O->print(2);
    ASSERT_TRUE((bool) Tensor::equivalent(t_fwrd, pd->O, 1e-1f, 0.0f, true, true));

//    // Backward
//    tensorNN::AvgPool3D_back(pd);
//    pd->ID->print(2);
//    ASSERT_TRUE((bool) Tensor::equivalent(t_bwrd, pd->ID, 1e-3f, 0.0f, true, true));

}


TEST(MaxPoolTestSuite, avgpool3d_k3x3x3_s1x1x1_pad_valid)
{
    // Image
    auto *ptr_img = new float[1*1*5*5*5]{
            0.13, 0.13, 0.98, -0.50, 0.26,
            1.74, -2.55, -0.60, -0.81, -1.35,
            1.21, -0.29, 0.44, -0.87, -0.73,
            -1.46, 0.12, -0.65, 0.86, 0.84,
            -1.08, -0.21, 0.54, -0.31, -1.04,
            0.07, 0.53, 0.02, 0.54, -0.88,
            0.67, -0.29, 0.77, -1.19, 0.90,
            2.29, 1.43, -1.06, 0.09, -0.09,
            -1.23, 0.78, 1.32, 1.10, -0.17,
            -0.18, 0.41, 0.72, 0.56, -0.42,
            0.57, -0.54, 0.23, 0.71, -1.83,
            0.94, -0.07, -0.18, -0.77, 2.35,
            0.42, -0.87, 2.12, -0.36, -0.06,
            1.02, 0.07, 0.20, 0.59, 0.82,
            -0.98, -0.41, 1.22, -1.95, 0.28,
            1.30, 1.22, 0.47, -1.32, -0.15,
            0.03, 0.71, 0.60, 0.08, 0.99,
            -0.81, -0.14, -0.31, 1.80, -0.90,
            -0.38, 0.28, -1.25, 1.38, 1.75,
            -0.62, 1.69, -0.94, 0.49, 0.56,
            0.65, 0.43, 0.36, 0.62, -1.44,
            -0.30, 0.24, -1.04, -0.21, -0.97,
            -0.12, 0.81, 1.17, -0.01, -0.41,
            0.12, 1.37, 0.37, -0.62, 1.33,
            -0.04, -2.55, -1.93, 0.21, 1.65,
    };

    auto* t_image = new Tensor({1, 1, 5, 5, 5}, ptr_img, DEV_CPU);
    t_image->print(2);

    // Forward
    auto *ptr_fwrd = new float[1*1*3*3*3]{
            0.31, -0.11, -0.07,
            0.23, -0.02, 0.13,
            0.22, 0.21, 0.15,
            0.37, 0.16, 0.10,
            0.26, 0.25, 0.39,
            0.18, 0.33, 0.28,
            0.29, 0.21, 0.06,
            0.19, 0.22, 0.31,
            -0.02, 0.09, 0.27,
    };
    auto* t_fwrd = new Tensor({1, 1, 3, 3, 3}, ptr_fwrd, DEV_CPU);


    // backward
//    auto *ptr_bwrd = new float[5*5]{0, 0, 0, 1, 0,
//                                    0, 1, 0, 0, 0,
//                                    0, 0, 0, 0, 0,
//                                    0, 1, 0, 1, 0,
//                                    0, 0, 0, 0, 0};
//    auto* t_bwrd = new Tensor({1, 1, 5, 5}, ptr_bwrd, DEV_CPU);

    // Operation
    auto *pd = new PoolDescriptor3D({3, 3, 3}, {1, 1, 1}, "valid");
    pd->build(t_image);
    pd->ID = Tensor::zeros(pd->I->getShape());
    pd->D = Tensor::ones(pd->O->getShape());
    pd->indX = new Tensor(pd->O->getShape());
    pd->indY = new Tensor(pd->O->getShape());
    pd->indZ = new Tensor(pd->O->getShape());

    // Forward
    tensorNN::AvgPool3D(pd);
    pd->O->print(2);
    ASSERT_TRUE((bool) Tensor::equivalent(t_fwrd, pd->O, 1e-1f, 0.0f, true, true));

//    // Backward
//    tensorNN::MPool3D_back(pd);
//    pd->ID->print(2);
//    ASSERT_TRUE((bool) Tensor::equivalent(t_bwrd, pd->ID, 1e-3f, 0.0f, true, true));

}


#ifdef cGPU
TEST(AvgPoolTestSuite, pool3d_cpu_gpu){
    // Image
    Tensor* t_cpu = Tensor::randu({1, 3, 10, 10, 10});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();

    vector<string> padding = {"valid", "same"};
    vector<int> strides = {1, 2, 3, 5};
    vector<int> kernels = {1, 2, 3, 5, 7};

    for(auto& p : padding){
        for(auto& s : strides){
            for(auto& k : kernels){
                try{
                    // CPU Operation
                    auto *pd_cpu = new PoolDescriptor3D({k, k, k}, {s, s, s}, p);
                    pd_cpu->build(t_cpu);
                    pd_cpu->ID = Tensor::zeros(pd_cpu->I->getShape());
                    pd_cpu->D = Tensor::ones(pd_cpu->O->getShape());
                    pd_cpu->indX = new Tensor(pd_cpu->O->getShape());
                    pd_cpu->indY = new Tensor(pd_cpu->O->getShape());
                    pd_cpu->indZ = new Tensor(pd_cpu->O->getShape());

                    // GPU Operation
                    auto *pd_gpu = new PoolDescriptor3D({k, k, k}, {s, s, s}, p);
                    pd_gpu->build(t_gpu);
                    pd_gpu->ID = Tensor::zeros(pd_gpu->I->getShape()); pd_gpu->ID->toGPU();
                    pd_gpu->D = Tensor::ones(pd_gpu->O->getShape()); pd_gpu->D->toGPU();
                    pd_gpu->indX = new Tensor(pd_gpu->O->getShape()); pd_gpu->indX->toGPU();
                    pd_gpu->indY = new Tensor(pd_gpu->O->getShape()); pd_gpu->indY->toGPU();
                    pd_gpu->indZ = new Tensor(pd_gpu->O->getShape()); pd_gpu->indZ->toGPU();

                    // Forward
                    tensorNN::AvgPool3D(pd_cpu);
                    tensorNN::AvgPool3D(pd_gpu);
                    Tensor *pd_gpu_O = pd_gpu->O->clone(); pd_gpu_O->toCPU();  // Tensor::equivalent is only for CPU (at the moment)
                    bool test_fwrd = (bool) Tensor::equivalent(pd_cpu->O, pd_gpu_O, 1e-1f, 0.0f, true, true);

                    // Backward
                    tensorNN::AvgPool3D_back(pd_cpu);
                    tensorNN::AvgPool3D_back(pd_gpu);
                    Tensor *pd_gpu_ID = pd_gpu->ID->clone(); pd_gpu_ID->toCPU(); // Tensor::equivalent is only for CPU (at the moment)
                    bool test_bwrd =(bool) Tensor::equivalent(pd_cpu->ID, pd_gpu_ID, 1e-3f, 0.0f, true, true);

                    // Print results to ease debugging
                    cout << "Testing pool3d_cpu_gpu (" << "padding=" << p << "; kernel=" << k << "; stride=" << s << ")" <<
                         " [Forward="<< test_fwrd << "; Backward=" << test_bwrd << "]" << endl;

                    // Test correctness
                    ASSERT_TRUE(test_fwrd);
                    ASSERT_TRUE(test_bwrd);

                    delete pd_cpu->ID;
                    delete pd_cpu->D;
                    delete pd_cpu;

                    delete pd_gpu->ID;
                    delete pd_gpu->D;
                    delete pd_gpu;

                    delete pd_gpu_O;
                    delete pd_gpu_ID;
                }
                catch (...) {
                    cout << "[FAILED] Testing pool3d_cpu_gpu (" << "padding=" << p << "; kernel=" << k << "; stride=" << s << ")" <<endl;
                }
            }
        }
    }
}
#endif