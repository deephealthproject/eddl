#include <gtest/gtest.h>

#include "eddl/tensor/tensor.h"
#include "eddl/tensor/nn/tensor_nn.h"
#include "eddl/descriptors/descriptors.h"



TEST(MaxPoolTestSuite, mpool_k2x2_s2x2_pad_valid)
{
    // Image
    auto *ptr_img = new float[5*5]{0, 1, 0, 4, 5,
                                   2, 3, 2, 1, 3,
                                   4, 4, 0, 4, 3,
                                   2, 5, 2, 6, 4,
                                   1, 0, 0, 5, 7};
    auto* t_image = new Tensor({1, 1, 5, 5}, ptr_img, DEV_CPU);


    // Forward
    auto *ptr_fwrd = new float[2*2]{3, 4,
                                    5, 6};
    auto* t_fwrd = new Tensor({1, 1, 2, 2}, ptr_fwrd, DEV_CPU);


    // backward
    auto *ptr_bwrd = new float[5*5]{0, 0, 0, 1, 0,
                                    0, 1, 0, 0, 0,
                                    0, 0, 0, 0, 0,
                                    0, 1, 0, 1, 0,
                                    0, 0, 0, 0, 0};
    auto* t_bwrd = new Tensor({1, 1, 5, 5}, ptr_bwrd, DEV_CPU);

    // Operation
    auto *pd = new PoolDescriptor({2, 2}, {2, 2}, "valid");
    pd->build(t_image);
    pd->ID = Tensor::zeros(pd->I->getShape());
    pd->D = Tensor::ones(pd->O->getShape());
    pd->indX = new Tensor(pd->O->getShape());
    pd->indY = new Tensor(pd->O->getShape());

    // Forward
    tensorNN::MPool2D(pd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_fwrd, pd->O, 1e-3f, 0.0f, true, true));

    // Backward
    tensorNN::MPool2D_back(pd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_bwrd, pd->ID, 1e-3f, 0.0f, true, true));
    
}


TEST(MaxPoolTestSuite, mpool_k2x2_s2x2_pad_same)
{
    // Image
    auto *ptr_img = new float[5*5]{0, 1, 0, 4, 5,
                                   2, 3, 2, 1, 3,
                                   4, 4, 0, 4, 3,
                                   2, 5, 2, 6, 4,
                                   1, 0, 0, 5, 7};
    auto* t_image = new Tensor({1, 1, 5, 5}, ptr_img, DEV_CPU);


    // Forward
    auto *ptr_fwrd = new float[3*3]{3, 4, 5,
                                    5, 6, 4,
                                    1, 5, 7};
    auto* t_fwrd = new Tensor({1, 1, 3, 3}, ptr_fwrd, DEV_CPU);


    // backward
    auto *ptr_bwrd = new float[5*5]{0, 0, 0, 1, 1,
                                    0, 1, 0, 0, 0,
                                    0, 0, 0, 0, 0,
                                    0, 1, 0, 1, 1,
                                    1, 0, 0, 1, 1};
    auto* t_bwrd = new Tensor({1, 1, 5, 5}, ptr_bwrd, DEV_CPU);

    // Operation
    auto *pd = new PoolDescriptor({2, 2}, {2, 2}, "same");
    pd->build(t_image);
    pd->ID = Tensor::zeros(pd->I->getShape());
    pd->D = Tensor::ones(pd->O->getShape());
    pd->indX = new Tensor(pd->O->getShape());
    pd->indY = new Tensor(pd->O->getShape());

    // Forward
    tensorNN::MPool2D(pd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_fwrd, pd->O, 1e-3f, 0.0f, true, true));

    // Backward
    tensorNN::MPool2D_back(pd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_bwrd, pd->ID, 1e-3f, 0.0f, true, true));
}


TEST(MaxPoolTestSuite, mpool_k3x3_s1x1_pad_valid)
{
    // Image
    auto *ptr_img = new float[5*5]{0, 1, 0, 4, 5,
                                   2, 3, 2, 1, 3,
                                   4, 4, 0, 4, 3,
                                   2, 5, 2, 6, 4,
                                   1, 0, 0, 5, 7};
    auto* t_image = new Tensor({1, 1, 5, 5}, ptr_img, DEV_CPU);


    // Forward
    auto *ptr_fwrd = new float[3*3]{4,4,5,
                                    5,6,6,
                                    5,6,7};
    auto* t_fwrd = new Tensor({1, 1, 3, 3}, ptr_fwrd, DEV_CPU);


    // backward
    auto *ptr_bwrd = new float[5*5]{0, 0, 0, 1, 1,
                                    0, 0, 0, 0, 0,
                                    1, 0, 0, 0, 0,
                                    0, 2, 0, 3, 0,
                                    0, 0, 0, 0, 1};
    auto* t_bwrd = new Tensor({1, 1, 5, 5}, ptr_bwrd, DEV_CPU);

    // Operation
    auto *pd = new PoolDescriptor({3, 3}, {1,1}, "valid");
    pd->build(t_image);
    pd->ID = Tensor::zeros(pd->I->getShape());
    pd->D = Tensor::ones(pd->O->getShape());
    pd->indX = new Tensor(pd->O->getShape());
    pd->indY = new Tensor(pd->O->getShape());

    // Forward
    tensorNN::MPool2D(pd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_fwrd, pd->O, 1e-3f, 0.0f, true, true));

    // Backward
    tensorNN::MPool2D_back(pd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_bwrd, pd->ID, 1e-3f, 0.0f, true, true));
}


TEST(MaxPoolTestSuite, mpool_k3x3_s1x1_pad_same)
{
    // Image
    auto *ptr_img = new float[5*5]{0, 1, 0, 4, 5,
                                   2, 3, 2, 1, 3,
                                   4, 4, 0, 4, 3,
                                   2, 5, 2, 6, 4,
                                   1, 0, 0, 5, 7};
    auto* t_image = new Tensor({1, 1, 5, 5}, ptr_img, DEV_CPU);


    // Forward
    auto *ptr_fwrd = new float[5*5]{3, 3, 4, 5, 5,
                                    4, 4, 4, 5, 5,
                                    5, 5, 6, 6, 6,
                                    5, 5, 6, 7, 7,
                                    5, 5, 6, 7, 7};
    auto* t_fwrd = new Tensor({1, 1, 5, 5}, ptr_fwrd, DEV_CPU);


    // backward
    auto *ptr_bwrd = new float[5*5]{0, 0, 0, 2, 4,
                                    0, 2, 0, 0, 0,
                                    2, 0, 0, 0, 0,
                                    0, 6, 0, 5, 0,
                                    0, 0, 0, 0, 4};
    auto* t_bwrd = new Tensor({1, 1, 5, 5}, ptr_bwrd, DEV_CPU);

    // Operation
    auto *pd = new PoolDescriptor({3, 3}, {1, 1}, "same");
    pd->build(t_image);
    pd->ID = Tensor::zeros(pd->I->getShape());
    pd->D = Tensor::ones(pd->O->getShape());
    pd->indX = new Tensor(pd->O->getShape());
    pd->indY = new Tensor(pd->O->getShape());

    // Forward
    tensorNN::MPool2D(pd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_fwrd, pd->O, 1e-3f, 0.0f, true, true));

    // Backward
    tensorNN::MPool2D_back(pd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_bwrd, pd->ID, 1e-3f, 0.0f, true, true));
}




TEST(MaxPoolTestSuite, mpool_k3x3_s2x2_pad_valid)
{
    // Image
    auto *ptr_img = new float[5*5]{0, 1, 0, 4, 5,
                                   2, 3, 2, 1, 3,
                                   4, 4, 0, 4, 3,
                                   2, 5, 2, 6, 4,
                                   1, 0, 0, 5, 7};
    auto* t_image = new Tensor({1, 1, 5, 5}, ptr_img, DEV_CPU);


    // Forward
    auto *ptr_fwrd = new float[2*2]{4, 5,
                                    5, 7};
    auto* t_fwrd = new Tensor({1, 1, 2, 2}, ptr_fwrd, DEV_CPU);


    // backward
    auto *ptr_bwrd = new float[5*5]{0, 0, 0, 0, 1,
                                    0, 0, 0, 0, 0,
                                    1, 0, 0, 0, 0,
                                    0, 1, 0, 0, 0,
                                    0, 0, 0, 0, 1};
    auto* t_bwrd = new Tensor({1, 1, 5, 5}, ptr_bwrd, DEV_CPU);

    // Operation
    auto *pd = new PoolDescriptor({3, 3}, {2, 2}, "valid");
    pd->build(t_image);
    pd->ID = Tensor::zeros(pd->I->getShape());
    pd->D = Tensor::ones(pd->O->getShape());
    pd->indX = new Tensor(pd->O->getShape());
    pd->indY = new Tensor(pd->O->getShape());

    // Forward
    tensorNN::MPool2D(pd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_fwrd, pd->O, 1e-3f, 0.0f, true, true));

    // Backward
    tensorNN::MPool2D_back(pd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_bwrd, pd->ID, 1e-3f, 0.0f, true, true));
}


TEST(MaxPoolTestSuite, mpool_k3x3_s2x2_pad_same){
    // Image
    auto *ptr_img = new float[5*5]{0, 1, 0, 4, 5,
                                   2, 3, 2, 1, 3,
                                   4, 4, 0, 4, 3,
                                   2, 5, 2, 6, 4,
                                   1, 0, 0, 5, 7};
    auto* t_image = new Tensor({1, 1, 5, 5}, ptr_img, DEV_CPU);


    // Forward
    auto *ptr_fwrd = new float[3*3]{3, 4, 5,
                                    5, 6, 6,
                                    5, 6, 7};
    auto* t_fwrd = new Tensor({1, 1, 3, 3}, ptr_fwrd, DEV_CPU);


    // backward
    auto *ptr_bwrd = new float[5*5]{0, 0, 0, 1, 1,
                                    0, 1, 0, 0, 0,
                                    0, 0, 0, 0, 0,
                                    0, 2, 0, 3, 0,
                                    0, 0, 0, 0, 1};
    auto* t_bwrd = new Tensor({1, 1, 5, 5}, ptr_bwrd, DEV_CPU);

    // Operation
    auto *pd = new PoolDescriptor({3, 3}, {2, 2}, "same");
    pd->build(t_image);
    pd->ID = Tensor::zeros(pd->I->getShape());
    pd->D = Tensor::ones(pd->O->getShape());
    pd->indX = new Tensor(pd->O->getShape());
    pd->indY = new Tensor(pd->O->getShape());

    // Forward
    tensorNN::MPool2D(pd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_fwrd, pd->O, 1e-3f, 0.0f, true, true));

    // Backward
    tensorNN::MPool2D_back(pd);
    ASSERT_TRUE((bool) Tensor::equivalent(t_bwrd, pd->ID, 1e-3f, 0.0f, true, true));
}

#ifdef cGPU
TEST(MaxPoolTestSuite, maxpool_cpu_gpu){
//    // Image
//    Tensor* t_cpu = Tensor::randu({1, 3, 1000, 1000});
//    Tensor* t_gpu = t_cpu->clone(); t_gpu->toGPU();
//
//    vector<string> padding = {"valid", "same"};
//    vector<int> strides = {1, 2, 3, 5};
//    vector<int> kernels = {1, 2, 3, 5, 7};
//
//    for(auto& p : padding){
//        for(auto& s : strides){
//            for(auto& k : kernels){
//
//
//                // CPU Operation
//                auto *pd_cpu = new PoolDescriptor({k, k}, {s, s}, p);
//                pd_cpu->build(t_cpu);
//                pd_cpu->ID = Tensor::zeros(pd_cpu->I->getShape());
//                pd_cpu->D = Tensor::ones(pd_cpu->O->getShape());
//                pd_cpu->indX = new Tensor(pd_cpu->O->getShape());
//                pd_cpu->indY = new Tensor(pd_cpu->O->getShape());
//
//                // GPU Operation
//                auto *pd_gpu = new PoolDescriptor({k, k}, {s, s}, p);
//                pd_gpu->build(t_gpu);
//                pd_gpu->ID = Tensor::zeros(pd_gpu->I->getShape(), t_gpu->device);
//                pd_gpu->D = Tensor::ones(pd_gpu->O->getShape(), t_gpu->device);
//                pd_gpu->indX = new Tensor(pd_gpu->O->getShape(), t_gpu->device);
//                pd_gpu->indY = new Tensor(pd_gpu->O->getShape(), t_gpu->device);
//
//                // Forward
//                tensorNN::MPool2D(pd_cpu);
//                tensorNN::MPool2D(pd_gpu);
//                Tensor *pd_gpu_O = pd_gpu->O->clone(); pd_gpu_O->toCPU();  // Tensor::equivalent is only for CPU (at the moment)
//                bool test_fwrd = (bool) Tensor::equivalent(pd_cpu->O, pd_gpu_O, 1e-5f, 0.0, false, true);
//
//                // Backward
//                tensorNN::MPool2D_back(pd_cpu);
//                tensorNN::MPool2D_back(pd_gpu);
//                Tensor *pd_gpu_ID = pd_gpu->ID->clone(); pd_gpu_ID->toCPU(); // Tensor::equivalent is only for CPU (at the moment)
//                bool test_bwrd = (bool) Tensor::equivalent(pd_cpu->ID, pd_gpu_ID, 1e-5f, 0.0, false, true);
//
//                // Print results to ease debugging
//                cout << "Testing maxpool_cpu_gpu (" << "padding=" << p << "; kernel=" << k << "; stride=" << s << ")" <<
//                     " [Forward="<< test_fwrd << "; Backward=" << test_bwrd << "]" << endl;
//
//                // Test correctness
//                ASSERT_TRUE(test_fwrd);
//                ASSERT_TRUE(test_bwrd);
//
//                delete pd_cpu->ID;
//                delete pd_cpu->D;
//                delete pd_cpu;
//
//                delete pd_gpu->ID;
//                delete pd_gpu->D;
//                delete pd_gpu;
//
//                delete pd_gpu_O;
//                delete pd_gpu_ID;
//            }
//        }
//    }
}
#endif