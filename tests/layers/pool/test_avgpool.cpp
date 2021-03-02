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
    ASSERT_TRUE((bool) Tensor::equivalent(t_fwrd, pd->O, 10e-5f));

    // Backward
    tensorNN::AvgPool2D_back(pd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_bwrd, pd->ID, 10e-5f));
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
    ASSERT_TRUE((bool) Tensor::equivalent(t_fwrd, pd->O, 10e-5f));

    // Backward
    tensorNN::AvgPool2D_back(pd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_bwrd, pd->ID, 10e-5f));
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
    ASSERT_TRUE((bool) Tensor::equivalent(t_fwrd, pd->O, 10e-5f));

    // Backward
    tensorNN::AvgPool2D_back(pd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_bwrd, pd->ID, 10e-5f));
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
    ASSERT_TRUE((bool) Tensor::equivalent(t_fwrd, pd->O, 10e-5f));

    // Backward
    tensorNN::AvgPool2D_back(pd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_bwrd, pd->ID, 10e-5f));
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
    ASSERT_TRUE((bool) Tensor::equivalent(t_fwrd, pd->O, 10e-5f));

    // Backward
    tensorNN::AvgPool2D_back(pd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_bwrd, pd->ID, 10e-5f));
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
    ASSERT_TRUE((bool) Tensor::equivalent(t_fwrd, pd->O, 10e-5f));

    // Backward
    tensorNN::AvgPool2D_back(pd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_bwrd, pd->ID, 10e-5f));
}


#ifdef cGPU
TEST(MaxPoolTestSuite, avgpool_k3x3_s1x1_pad_valid_gpu)
{
    // Image
    Tensor* t_cpu = Tensor::randu({1, 3, 1000, 1000});
    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();

    // CPU Operation
    auto *pd_cpu = new PoolDescriptor({3, 3}, {1, 1}, "valid");
    pd_cpu->build(t_cpu);
    pd_cpu->ID = Tensor::zeros(pd_cpu->I->getShape());
    pd_cpu->D = Tensor::ones(pd_cpu->O->getShape());
    pd_cpu->indX = new Tensor(pd_cpu->O->getShape());
    pd_cpu->indY = new Tensor(pd_cpu->O->getShape());

    // GPU Operation
    auto *pd_gpu = new PoolDescriptor({3, 3}, {1, 1}, "valid");
    pd_gpu->build(t_gpu);
    pd_gpu->ID = Tensor::zeros(pd_gpu->I->getShape(), t_gpu->device);
    pd_gpu->D = Tensor::ones(pd_gpu->O->getShape(), t_gpu->device);
    pd_gpu->indX = new Tensor(pd_gpu->O->getShape(), t_gpu->device);
    pd_gpu->indY = new Tensor(pd_gpu->O->getShape(), t_gpu->device);

    // Forward
    tensorNN::AvgPool2D(pd_cpu);
    tensorNN::AvgPool2D(pd_gpu);
    Tensor *pd_gpu_O = pd_gpu->O->clone(); pd_gpu_O->toCPU();  // Tensor::equivalent is only for CPU (at the moment)
    ASSERT_TRUE((bool) Tensor::equivalent(pd_cpu->O, pd_gpu_O, 10e-5f));

    // Backward
    tensorNN::AvgPool2D_back(pd_cpu);
    tensorNN::AvgPool2D_back(pd_gpu);
    Tensor *pd_gpu_ID = pd_gpu->ID->clone(); pd_gpu_ID->toCPU(); // Tensor::equivalent is only for CPU (at the moment)
    ASSERT_TRUE((bool) Tensor::equivalent(pd_cpu->ID, pd_gpu_ID, 10e-5f));
}
#endif
