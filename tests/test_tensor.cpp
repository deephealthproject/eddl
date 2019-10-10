/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include "gtest/gtest.h"
#include "tensor/tensor.h"


// Test tensor constructor
TEST(TensorTest, Constructor) {
    ASSERT_EQ(DEV_CPU, Tensor(vector<int>{1, 1, 1}, DEV_CPU).device);
    ASSERT_EQ(3, Tensor(vector<int>{1, 1, 1}, DEV_CPU).ndim);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
