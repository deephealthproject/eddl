#include <gtest/gtest.h>

#include "tensor/tensor.h"
#include "tensor/nn/tensor_nn.h"
#include "descriptors/descriptors.h"



TEST(MaxPoolTestSuite, mpool_k2x2_s2x2_pad_valid)
{
    // Image
    auto *ptr_img = new float[5*5]{0, 1, 0, 4, 5,
                                   2, 3, 2, 1, 3,
                                   4, 4, 0, 4, 3,
                                   2, 5, 2, 6, 4,
                                   1, 0, 0, 5, 7};
    auto* t_image = new Tensor({1, 1, 5, 5}, ptr_img);


    // Forward
    auto *ptr_fwrd = new float[2*2]{3, 4,
                                    5, 6};
    auto* t_fwrd = new Tensor({1, 1, 2, 2}, ptr_fwrd);


    // backward
    auto *ptr_bwrd = new float[5*5]{0, 0, 0, 1, 0,
                                    0, 1, 0, 0, 0,
                                    0, 0, 0, 0, 0,
                                    0, 1, 0, 1, 0,
                                    0, 0, 0, 0, 0};
    auto* t_bwrd = new Tensor({1, 1, 5, 5}, ptr_bwrd);

    // Operation
    auto *pd = new PoolDescriptor({2, 2}, {2, 2}, "valid");
    pd->build(t_image);
    pd->ID = Tensor::zeros(pd->I->getShape());
    pd->D = Tensor::ones(pd->O->getShape());
    pd->indX = new Tensor(pd->O->getShape());
    pd->indY = new Tensor(pd->O->getShape());

    // Forward
    MPool2D(pd);
    ASSERT_TRUE((bool)Tensor::equal2(t_fwrd, pd->O, 10e-5f));

    // Backward
    MPool2D_back(pd);
    ASSERT_TRUE((bool)Tensor::equal2(t_bwrd, pd->ID, 10e-5f));
}


TEST(MaxPoolTestSuite, mpool_k2x2_s2x2_pad_same)
{
    // Image
    auto *ptr_img = new float[5*5]{0, 1, 0, 4, 5,
                                   2, 3, 2, 1, 3,
                                   4, 4, 0, 4, 3,
                                   2, 5, 2, 6, 4,
                                   1, 0, 0, 5, 7};
    auto* t_image = new Tensor({1, 1, 5, 5}, ptr_img);


    // Forward
    auto *ptr_fwrd = new float[3*3]{3, 4, 5,
                                    5, 6, 4,
                                    1, 5, 7};
    auto* t_fwrd = new Tensor({1, 1, 3, 3}, ptr_fwrd);


    // backward
    auto *ptr_bwrd = new float[5*5]{0, 0, 0, 1, 1,
                                    0, 1, 0, 0, 0,
                                    0, 0, 0, 0, 0,
                                    0, 1, 0, 1, 1,
                                    1, 0, 0, 1, 1};
    auto* t_bwrd = new Tensor({1, 1, 5, 5}, ptr_bwrd);

    // Operation
    auto *pd = new PoolDescriptor({2, 2}, {2, 2}, "same");
    pd->build(t_image);
    pd->ID = Tensor::zeros(pd->I->getShape());
    pd->D = Tensor::ones(pd->O->getShape());
    pd->indX = new Tensor(pd->O->getShape());
    pd->indY = new Tensor(pd->O->getShape());

    // Forward
    MPool2D(pd);
    ASSERT_TRUE((bool)Tensor::equal2(t_fwrd, pd->O, 10e-5f));

    // Backward
    MPool2D_back(pd);
    ASSERT_TRUE((bool)Tensor::equal2(t_bwrd, pd->ID, 10e-5f));
}


TEST(MaxPoolTestSuite, mpool_k3x3_s1x1_pad_valid)
{
    // Image
    auto *ptr_img = new float[5*5]{0, 1, 0, 4, 5,
                                   2, 3, 2, 1, 3,
                                   4, 4, 0, 4, 3,
                                   2, 5, 2, 6, 4,
                                   1, 0, 0, 5, 7};
    auto* t_image = new Tensor({1, 1, 5, 5}, ptr_img);


    // Forward
    auto *ptr_fwrd = new float[3*3]{4,4,5,
                                    5,6,6,
                                    5,6,7};
    auto* t_fwrd = new Tensor({1, 1, 3, 3}, ptr_fwrd);


    // backward
    auto *ptr_bwrd = new float[5*5]{0, 0, 0, 1, 1,
                                    0, 0, 0, 0, 0,
                                    1, 0, 0, 0, 0,
                                    0, 2, 0, 3, 0,
                                    0, 0, 0, 0, 1};
    auto* t_bwrd = new Tensor({1, 1, 5, 5}, ptr_bwrd);

    // Operation
    auto *pd = new PoolDescriptor({3, 3}, {1,1}, "valid");
    pd->build(t_image);
    pd->ID = Tensor::zeros(pd->I->getShape());
    pd->D = Tensor::ones(pd->O->getShape());
    pd->indX = new Tensor(pd->O->getShape());
    pd->indY = new Tensor(pd->O->getShape());

    // Forward
    MPool2D(pd);
    ASSERT_TRUE((bool)Tensor::equal2(t_fwrd, pd->O, 10e-5f));

    // Backward
    MPool2D_back(pd);
    ASSERT_TRUE((bool)Tensor::equal2(t_bwrd, pd->ID, 10e-5f));
}


TEST(MaxPoolTestSuite, mpool_k3x3_s1x1_pad_same)
{
    // Image
    auto *ptr_img = new float[5*5]{0, 1, 0, 4, 5,
                                   2, 3, 2, 1, 3,
                                   4, 4, 0, 4, 3,
                                   2, 5, 2, 6, 4,
                                   1, 0, 0, 5, 7};
    auto* t_image = new Tensor({1, 1, 5, 5}, ptr_img);


    // Forward
    auto *ptr_fwrd = new float[5*5]{3, 3, 4, 5, 5,
                                    4, 4, 4, 5, 5,
                                    5, 5, 6, 6, 6,
                                    5, 5, 6, 7, 7,
                                    5, 5, 6, 7, 7};
    auto* t_fwrd = new Tensor({1, 1, 5, 5}, ptr_fwrd);


    // backward
    auto *ptr_bwrd = new float[5*5]{0, 0, 0, 2, 4,
                                    0, 2, 0, 0, 0,
                                    2, 0, 0, 0, 0,
                                    0, 6, 0, 5, 0,
                                    0, 0, 0, 0, 4};
    auto* t_bwrd = new Tensor({1, 1, 5, 5}, ptr_bwrd);

    // Operation
    auto *pd = new PoolDescriptor({3, 3}, {1, 1}, "same");
    pd->build(t_image);
    pd->ID = Tensor::zeros(pd->I->getShape());
    pd->D = Tensor::ones(pd->O->getShape());
    pd->indX = new Tensor(pd->O->getShape());
    pd->indY = new Tensor(pd->O->getShape());

    // Forward
    MPool2D(pd);
    ASSERT_TRUE((bool)Tensor::equal2(t_fwrd, pd->O, 10e-5f));

    // Backward
    MPool2D_back(pd);
    ASSERT_TRUE((bool)Tensor::equal2(t_bwrd, pd->ID, 10e-5f));
}




TEST(MaxPoolTestSuite, mpool_k3x3_s2x2_pad_valid)
{
    // Image
    auto *ptr_img = new float[5*5]{0, 1, 0, 4, 5,
                                   2, 3, 2, 1, 3,
                                   4, 4, 0, 4, 3,
                                   2, 5, 2, 6, 4,
                                   1, 0, 0, 5, 7};
    auto* t_image = new Tensor({1, 1, 5, 5}, ptr_img);


    // Forward
    auto *ptr_fwrd = new float[2*2]{4, 5,
                                    5, 7};
    auto* t_fwrd = new Tensor({1, 1, 2, 2}, ptr_fwrd);


    // backward
    auto *ptr_bwrd = new float[5*5]{0, 0, 0, 0, 1,
                                    0, 0, 0, 0, 0,
                                    1, 0, 0, 0, 0,
                                    0, 1, 0, 0, 0,
                                    0, 0, 0, 0, 1};
    auto* t_bwrd = new Tensor({1, 1, 5, 5}, ptr_bwrd);

    // Operation
    auto *pd = new PoolDescriptor({3, 3}, {2, 2}, "valid");
    pd->build(t_image);
    pd->ID = Tensor::zeros(pd->I->getShape());
    pd->D = Tensor::ones(pd->O->getShape());
    pd->indX = new Tensor(pd->O->getShape());
    pd->indY = new Tensor(pd->O->getShape());

    // Forward
    MPool2D(pd);
    ASSERT_TRUE((bool)Tensor::equal2(t_fwrd, pd->O, 10e-5f));

    // Backward
    MPool2D_back(pd);
    ASSERT_TRUE((bool)Tensor::equal2(t_bwrd, pd->ID, 10e-5f));
}


TEST(MaxPoolTestSuite, mpool_k3x3_s2x2_pad_same)
{
    // Image
    auto *ptr_img = new float[5*5]{0, 1, 0, 4, 5,
                                   2, 3, 2, 1, 3,
                                   4, 4, 0, 4, 3,
                                   2, 5, 2, 6, 4,
                                   1, 0, 0, 5, 7};
    auto* t_image = new Tensor({1, 1, 5, 5}, ptr_img);


    // Forward
    auto *ptr_fwrd = new float[3*3]{3, 4, 5,
                                    5, 6, 6,
                                    5, 6, 7};
    auto* t_fwrd = new Tensor({1, 1, 3, 3}, ptr_fwrd);


    // backward
    auto *ptr_bwrd = new float[5*5]{0, 0, 0, 1, 1,
                                    0, 1, 0, 0, 0,
                                    0, 0, 0, 0, 0,
                                    0, 2, 0, 3, 0,
                                    0, 0, 0, 0, 1};
    auto* t_bwrd = new Tensor({1, 1, 5, 5}, ptr_bwrd);

    // Operation
    auto *pd = new PoolDescriptor({3, 3}, {2, 2}, "same");
    pd->build(t_image);
    pd->ID = Tensor::zeros(pd->I->getShape());
    pd->D = Tensor::ones(pd->O->getShape());
    pd->indX = new Tensor(pd->O->getShape());
    pd->indY = new Tensor(pd->O->getShape());

    // Forward
    MPool2D(pd);
    ASSERT_TRUE((bool)Tensor::equal2(t_fwrd, pd->O, 10e-5f));

    // Backward
    MPool2D_back(pd);
    ASSERT_TRUE((bool)Tensor::equal2(t_bwrd, pd->ID, 10e-5f));
}
