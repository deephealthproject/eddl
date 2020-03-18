#include "gtest/gtest.h"

#include "tensor/tensor.h"
#include "tensor/nn/tensor_nn.h"
#include "descriptors/descriptors.h"

TEST(maxpoolTest, mpool_forward)
{
    bool isCorrect = false;

    // Image
    auto *ptr_img = new float[5*5]{0, 1, 0, 4, 5,
                                   2, 3, 2, 1, 3,
                                   4, 4, 0, 4, 3,
                                   2, 5, 2, 6, 4,
                                   1, 0, 0, 5, 7};
    auto* t_image = new Tensor({1, 1, 5, 5}, ptr_img);


    // Solution
    auto *ptr_mp_3x3_s1_padv = new float[3*3]{4,4,5,
                                              5,6,6,
                                              5,6,7};
    auto* t_sol_fwd = new Tensor({1, 1, 3, 3}, ptr_mp_3x3_s1_padv);


    // Operation
    auto *pd = new PoolDescriptor({3, 3}, {1,1}, "valid");
    pd->build(t_image);
    pd->indX = new Tensor(pd->O->getShape());
    pd->indY = new Tensor(pd->O->getShape());
    pd->ID = Tensor::zeros(pd->I->getShape());
    pd->D = Tensor::ones(pd->O->getShape());

    // Forward
    MPool2D(pd);
    t_sol_fwd->toCPU();

    // Result
    isCorrect = (bool)Tensor::equal2(t_sol_fwd, pd->O, 10e-1f);
    ASSERT_TRUE(isCorrect);
}