#include <gtest/gtest.h>
#include <string>

#include "eddl/tensor/tensor.h"
#include "eddl/tensor/nn/tensor_nn.h"
#include "eddl/descriptors/descriptors.h"
#include "eddl/layers/normalization/layer_normalization.h"

using namespace std;


TEST(NormalizationTestSuite, batchnorm){
//
//    // Image
//    auto *ptr_img = new float[1*3*5*5]{-0.66, 1.88, -0.09, 2.00, -1.26,
//                                       -0.96, 1.49, -0.34, -0.12, -0.09,
//                                       -0.19, -0.60, -1.60, -0.84, -1.44,
//                                       -0.83, -0.06, 0.01, -0.81, -0.90,
//                                       0.43, 0.82, -0.46, -0.10, -0.17,
//
//                                       0.32, -1.09, 0.52, 1.19, 0.76,
//                                       0.16, 1.07, -1.08, 0.14, -2.00,
//                                       0.94, 1.24, 0.23, -0.77, 0.23,
//                                       0.09, -1.64, 2.31, 0.09, 0.98,
//                                       0.23, -2.14, -1.47, 1.18, -0.02,
//
//                                       0.75, -0.97, 0.47, 0.67, -0.03,
//                                       0.77, 0.27, 1.16, 0.62, 1.39,
//                                       -0.23, 0.51, 0.26, 0.75, -0.08,
//                                       0.14, 0.17, 0.89, 0.19, -0.44,
//                                       0.98, 0.66, 0.48, -0.20, 0.10,};
//    auto* t_image = new Tensor({1, 3, 5, 5}, ptr_img, DEV_CPU);
//
//    // Forward
//    auto *ptr_fwrd_ref = new float[1*3*5*5]{
//            -0.51, 2.27, 0.12, 2.41, -1.16,
//            -0.84, 1.84, -0.15, 0.08, 0.12,
//            0.00, -0.45, -1.53, -0.70, -1.36,
//            -0.70, 0.15, 0.23, -0.67, -0.77,
//            0.68, 1.11, -0.29, 0.10, 0.02,
//
//            0.24, -1.05, 0.42, 1.03, 0.64,
//            0.09, 0.92, -1.04, 0.07, -1.87,
//            0.81, 1.07, 0.15, -0.75, 0.15,
//            0.02, -1.55, 2.05, 0.03, 0.84,
//            0.15, -2.00, -1.40, 1.02, -0.07,
//
//            0.73, -2.58, 0.19, 0.58, -0.78,
//            0.77, -0.20, 1.52, 0.47, 1.96,
//            -1.16, 0.26, -0.21, 0.73, -0.86,
//            -0.44, -0.38, 1.00, -0.35, -1.56,
//            1.17, 0.56, 0.20, -1.09, -0.53,
//    };
//    auto* t_fwrd_ref = new Tensor({1, 3, 5, 5}, ptr_fwrd_ref, DEV_CPU);
//
//    // Mean
//    auto *ptr_mean_ref = new float[3]{-0.0196,  0.0058,  0.0370};
//    auto* t_mean_ref = new Tensor({3}, ptr_mean_ref, DEV_CPU);
//
//    // Var
//    auto *ptr_var_ref = new float[3]{0.9870, 1.0254, 0.9280};
//    auto* t_var_ref = new Tensor({3}, ptr_var_ref, DEV_CPU);
//    t_var_ref->add(10e-5);
//    t_var_ref->sqrt();
//    t_var_ref->inv();
//
//
//    // Forward
//    auto* t_output = Tensor::empty_like(t_image);
//    auto* t_opa = Tensor::empty_like(t_image);
//
//    auto* t_mean_acc = Tensor::zeros({3});
//    auto* t_var_acc = Tensor::ones({3});
//
//    auto* t_mean = Tensor::zeros({3});
//    auto* t_var = Tensor::ones({3});
//
//    auto* t_gamma = Tensor::ones({3});
//    auto* t_beta = Tensor::zeros({3});
////
////
////
////BN_forward(t_image, t_gamma, t_beta, t_mean, t_var,0.1f,10e-5, TRMODE);
//
//
//// Andres
//    tensorNN::BatchNormForward(t_image, t_output, t_opa,
//                               t_mean_acc, t_var_acc,
//                               t_gamma, t_beta,
//                               t_mean, t_var, TRMODE==TRMODE, 1e-5, 0.1f);
//
//
//    t_output->print(3);
//
//    cout << "Mean" << endl;
//    t_mean_ref->print(3);
//    t_mean->print(3);
//
//    cout << "Var" << endl;
//    t_var_ref->print(3);
//    t_var->print(3);
//    ASSERT_TRUE((bool) Tensor::equivalent(t_fwrd_ref, t_output, 10e-2f));
//    ASSERT_TRUE((bool) Tensor::equivalent(t_mean_ref, t_mean, 10e-2f));
//    ASSERT_TRUE((bool) Tensor::equivalent(t_var_ref, t_var, 10e-2f));
    int asd = 33;
}
