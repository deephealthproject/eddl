#include <gtest/gtest.h>

#include "eddl/tensor/tensor.h"
#include "eddl/tensor/nn/tensor_nn.h"
#include "eddl/descriptors/descriptors.h"


TEST(MathTestSuite, layer_argmax)
{
    vector<int> dev_list = {DEV_CPU, DEV_GPU};

    for(auto & dev : dev_list){
        Tensor *input = new Tensor({
                                          0, 0, 1, 7, 0, 3,
                                          0, 4, 1, 0, 0, 0,
                                          0, 0, 1, 2, 0, 5,
                                          7, 0, 1, 6, 0, 4,

                                          0, 9, 1, 2, 0, 3,
                                          0, 4, 9, 0, 0, 0,
                                          8, 0, 1, 2, 0, 5,
                                          7, 0, 1, 6, 0, 9

                                          }, {2, 4, 6}, dev);
        Tensor *output_ref = new Tensor({3,1,5,0,
                                              1,2,0,5}, {2, 4}, dev);
        Tensor *parent_delta_ref = new Tensor({
                                                    0, 0, 0, 1, 0, 0,
                                                    0, 1, 0, 0, 0, 0,
                                                    0, 0, 0, 0, 0, 1,
                                                    1, 0, 0, 0, 0, 0,

                                                    0, 1, 0, 0, 0, 0,
                                                    0, 0, 1, 0, 0, 0,
                                                    1, 0, 0, 0, 0, 0,
                                                    0, 0, 0, 0, 0, 1}, {2, 4, 6}, dev);

        // Forward
        auto *RD2 = new ReduceDescriptor2({1 +1}, false, dev);  // Dimension extra for the batch
        RD2->build(input->shape);

        // Create output tensor
        Tensor *output = Tensor::empty(RD2->oshape, dev);

        // Operation
        Tensor::argmax(input, output, RD2);
        output->print(0);
        ASSERT_TRUE((bool) Tensor::equivalent(output, output_ref, 10e-5f));

        // Backward
        Tensor *delta = Tensor::ones(RD2->oshape, dev);
        Tensor *parent_delta = Tensor::zeros(RD2->ishape, dev);

        // Operation
        Tensor::argmax_d(delta, output, parent_delta);
        parent_delta->print(0);
        ASSERT_TRUE((bool) Tensor::equivalent(parent_delta, parent_delta_ref, 10e-5f));
    }

}
