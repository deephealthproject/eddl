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