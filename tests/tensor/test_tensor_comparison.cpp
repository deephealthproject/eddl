#include <gtest/gtest.h>
#include <random>
#include <string>

#include "eddl/tensor/tensor.h"
#include "eddl/tensor/nn/tensor_nn.h"
#include "eddl/descriptors/descriptors.h"


using namespace std;



TEST(TensorTestSuite, tensor_comparison_all){
    // Test #1
    Tensor* t1 = new Tensor({1.0f, 1.0f, 1.0f,
                                 1.0f, 1.0f, 1.0f,
                                 1.0f, 1.0f, 1.0f}, {3, 3}, DEV_CPU);

    Tensor* t2 = new Tensor({1.0f, 1.0f, 1.0f,
                                  1.0f, 1.0f, 0.0f,
                                  1.0f, 1.0f, 1.0f}, {3, 3}, DEV_CPU);

    ASSERT_TRUE(Tensor::all(t1));
    ASSERT_FALSE(Tensor::all(t2));

    delete t1;
    delete t2;

    // Test GPU
#ifdef cGPU
    Tensor* t1_cpu = Tensor::ones({3, 1000, 1000});
    Tensor* t1_gpu = t1_cpu->clone(); t1_gpu->toGPU();
    ASSERT_TRUE(Tensor::all(t1_cpu) && Tensor::all(t1_gpu));

    Tensor* t2_cpu = Tensor::ones({3, 1000, 1000}); t2_cpu->ptr[5] = 0.0f;
    Tensor* t2_gpu = t2_cpu->clone(); t2_gpu->toGPU();
    ASSERT_FALSE(Tensor::all(t2_cpu) || Tensor::all(t2_gpu));

    delete t1_cpu;
    delete t1_gpu;
    delete t2_cpu;
    delete t2_gpu;

#endif

}

TEST(TensorTestSuite, tensor_comparison_any){
    // Test #1
    Tensor* t1 = Tensor::ones({3, 3}, DEV_CPU);
    Tensor* t2 = Tensor::zeros({3, 3}, DEV_CPU);
    Tensor* t3 = new Tensor({0.0f, 0.0f, 0.0f,
                                  1.0f, 0.0f, 0.0f,
                                  0.0f, 1.0f, 1.0f}, {3, 3}, DEV_CPU);

    ASSERT_TRUE(Tensor::any(t1));
    ASSERT_FALSE(Tensor::any(t2));
    ASSERT_TRUE(Tensor::any(t3));

    delete t1;
    delete t2;
    delete t3;

    // Test GPU
#ifdef cGPU
    Tensor* t1_cpu = Tensor::ones({3, 1000, 1000});
    Tensor* t1_gpu = t1_cpu->clone(); t1_gpu->toGPU();
    ASSERT_TRUE(Tensor::any(t1_cpu) && Tensor::any(t1_gpu));

    Tensor* t2_cpu = Tensor::zeros({3, 1000, 1000});
    Tensor* t2_gpu = t2_cpu->clone(); t2_gpu->toGPU();
    ASSERT_FALSE(Tensor::any(t2_cpu) || Tensor::any(t2_gpu));

    Tensor* t3_cpu = Tensor::zeros({3, 1000, 1000}); t3_cpu->ptr[5] = 1.0f;
    Tensor* t3_gpu = t3_cpu->clone(); t3_gpu->toGPU();
    ASSERT_TRUE(Tensor::any(t3_cpu) && Tensor::any(t3_gpu));

    delete t1_cpu;
    delete t1_gpu;
    delete t2_cpu;
    delete t2_gpu;
    delete t3_cpu;
    delete t3_gpu;

#endif
}

//TEST(TensorTestSuite, tensor_comparison_allclose){
//    // Test #1
//    Tensor* t1 = new Tensor({1.0f, 1.0f, 1.0f,
//                             1.0f, 1.0f, 1.0f,
//                             1.0f, 1.0f, 1.0f}, {3, 3}, DEV_CPU);
//
//    Tensor* t2 = new Tensor({1.0f, 1.0f, 1.0f,
//                             1.0f, 1.0f, 0.0f,
//                             1.0f, 1.0f, 1.0f}, {3, 3}, DEV_CPU);
//
//    ASSERT_TRUE(Tensor::allclose(t1, t1));
//    ASSERT_FALSE(Tensor::allclose(t1, t2));
//
//
//    // Test GPU
//#ifdef cGPU
//    Tensor* t1_cpu = Tensor::ones({3, 1000, 1000});
//    Tensor* t1_gpu = t1_cpu->clone(); t1_gpu->toGPU();
//
//    Tensor* t2_cpu = Tensor::ones({3, 1000, 1000}); t2_cpu->ptr[5] = 1.0f + 10e-3f;
//    Tensor* t2_gpu = t2_cpu->clone(); t2_gpu->toGPU();
//
//    bool t1_cpu_res = Tensor::allclose(t1_cpu, t2_cpu, 1e-5);
//    bool t1_gpu_res = Tensor::allclose(t1_gpu, t2_gpu, 1e-5);
//    ASSERT_TRUE(t1_cpu_res && t1_gpu_res);
//
//    bool t2_cpu_res = Tensor::allclose(t1_cpu, t2_cpu, 1e-5);
//    bool t2_gpu_res = Tensor::allclose(t1_gpu, t2_gpu, 1e-5);
//    ASSERT_TRUE(t2_cpu_res && t2_gpu_res);
//#endif
//}


TEST(TensorTestSuite, tensor_comparison_greaterT){
    // Test #1
    vector<int> t1_shape_ref = {2, 2};
    vector<float> d_t1_ref = {1, 1, 1, 1};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    Tensor* t1_A = Tensor::full(t1_shape_ref, 5, DEV_CPU);
    Tensor* t1_B = Tensor::full(t1_shape_ref, 3, DEV_CPU);

    Tensor* new_t = t1_A->greater(t1_B);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 1e-3f, 0.0f, true, true));

    delete t1_ref;
    delete t1_A;
    delete t1_B;
    delete new_t;

    // Test GPU
    #ifdef cGPU
        Tensor* t_cpu_A = Tensor::randu({3, 1000, 1000});
        Tensor* t_gpu_A = t_cpu_A->clone(); t_gpu_A->toGPU();

        Tensor* t_cpu_B = Tensor::full(t_cpu_A->shape, 0.5);
        Tensor* t_gpu_B = t_cpu_B->clone(); t_gpu_B->toGPU();

        Tensor* new_t_cpu = t_cpu_A->greater(t_cpu_B);
        Tensor* new_t_gpu = t_gpu_A->greater(t_gpu_B); new_t_gpu->toCPU();

        ASSERT_TRUE(Tensor::equivalent(new_t_cpu, new_t_gpu, 1e-3f, 0.0f, true, true));

        delete t_cpu_A;
        delete t_gpu_A;
        delete t_cpu_B;
        delete t_gpu_B;
        delete new_t_cpu;
        delete new_t_gpu;

    #endif
}

TEST(TensorTestSuite, tensor_comparison_greater_equalT){
    // Test #1
    vector<int> t1_shape_ref = {2, 2};
    vector<float> d_t1_ref = {1, 1, 1, 1};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    Tensor* t1_A = Tensor::full(t1_shape_ref, 5, DEV_CPU);
    Tensor* t1_B = Tensor::full(t1_shape_ref, 3, DEV_CPU);

    Tensor* new_t = t1_A->greater_equal(t1_B);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 1e-3f, 0.0f, true, true));

    Tensor* t2_A = Tensor::full(t1_shape_ref, 5, DEV_CPU);
    Tensor* t2_B = Tensor::full(t1_shape_ref, 5, DEV_CPU);

    Tensor* new_t2 = t2_A->greater_equal(t2_B);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t2, 1e-3f, 0.0f, true, true));

    delete t1_ref;
    delete t1_A;
    delete t1_B;
    delete new_t;
    delete t2_A;
    delete t2_B;
    delete new_t2;


    // Test GPU
#ifdef cGPU
    Tensor* t_cpu_A = Tensor::randu({3, 1000, 1000});
        Tensor* t_gpu_A = t_cpu_A->clone(); t_gpu_A->toGPU();

        Tensor* t_cpu_B = Tensor::full(t_cpu_A->shape, 0.5);
        Tensor* t_gpu_B = t_cpu_B->clone(); t_gpu_B->toGPU();

        Tensor* new_t_cpu = t_cpu_A->greater_equal(t_cpu_B);
        Tensor* new_t_gpu = t_gpu_A->greater_equal(t_gpu_B); new_t_gpu->toCPU();

        ASSERT_TRUE(Tensor::equivalent(new_t_cpu, new_t_gpu, 1e-3f, 0.0f, true, true));

        delete t_cpu_A;
        delete t_gpu_A;
        delete t_cpu_B;
        delete t_gpu_B;
        delete new_t_cpu;
        delete new_t_gpu;

#endif
}

TEST(TensorTestSuite, tensor_comparison_lessT){
    // Test #1
    vector<int> t1_shape_ref = {2, 2};
    vector<float> d_t1_ref = {1, 1, 1, 1};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    Tensor* t1_A = Tensor::full(t1_shape_ref, 3, DEV_CPU);
    Tensor* t1_B = Tensor::full(t1_shape_ref, 5, DEV_CPU);

    Tensor* new_t = t1_A->less(t1_B);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 1e-3f, 0.0f, true, true));

    delete t1_ref;
    delete t1_A;
    delete t1_B;
    delete new_t;

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu_A = Tensor::randu({3, 1000, 1000});
        Tensor* t_gpu_A = t_cpu_A->clone(); t_gpu_A->toGPU();

        Tensor* t_cpu_B = Tensor::full(t_cpu_A->shape, 0.5);
        Tensor* t_gpu_B = t_cpu_B->clone(); t_gpu_B->toGPU();

        Tensor* new_t_cpu = t_cpu_A->less(t_cpu_B);
        Tensor* new_t_gpu = t_gpu_A->less(t_gpu_B); new_t_gpu->toCPU();

        ASSERT_TRUE(Tensor::equivalent(new_t_cpu, new_t_gpu, 1e-3f, 0.0f, true, true));

        delete t_cpu_A;
        delete t_gpu_A;
        delete t_cpu_B;
        delete t_gpu_B;
        delete new_t_cpu;
        delete new_t_gpu;

#endif
}

TEST(TensorTestSuite, tensor_comparison_less_equalT){
    // Test #1
    vector<int> t1_shape_ref = {2, 2};
    vector<float> d_t1_ref = {1, 1, 1, 1};
    Tensor* t1_ref = new Tensor(t1_shape_ref, d_t1_ref.data(), DEV_CPU);

    Tensor* t1_A = Tensor::full(t1_shape_ref, 3, DEV_CPU);
    Tensor* t1_B = Tensor::full(t1_shape_ref, 5, DEV_CPU);

    Tensor* new_t = t1_A->less_equal(t1_B);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 1e-3f, 0.0f, true, true));

    Tensor* t2_A = Tensor::full(t1_shape_ref, 5, DEV_CPU);
    Tensor* t2_B = Tensor::full(t1_shape_ref, 5, DEV_CPU);

    Tensor* new_t2 = t2_A->greater_equal(t2_B);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t2, 1e-3f, 0.0f, true, true));

    delete t1_ref;
    delete t1_A;
    delete t1_B;
    delete new_t;
    delete t2_A;
    delete t2_B;
    delete new_t2;

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu_A = Tensor::randu({3, 1000, 1000});
        Tensor* t_gpu_A = t_cpu_A->clone(); t_gpu_A->toGPU();

        Tensor* t_cpu_B = Tensor::full(t_cpu_A->shape, 0.5);
        Tensor* t_gpu_B = t_cpu_B->clone(); t_gpu_B->toGPU();

        Tensor* new_t_cpu = t_cpu_A->less_equal(t_cpu_B);
        Tensor* new_t_gpu = t_gpu_A->less_equal(t_gpu_B); new_t_gpu->toCPU();

        ASSERT_TRUE(Tensor::equivalent(new_t_cpu, new_t_gpu, 1e-3f, 0.0f, true, true));

        delete t_cpu_A;
        delete t_gpu_A;
        delete t_cpu_B;
        delete t_gpu_B;
        delete new_t_cpu;
        delete new_t_gpu;

#endif
}

TEST(TensorTestSuite, tensor_comparison_equalT){
    // Test #1
    Tensor* t1_ref = Tensor::ones({2, 2});
    Tensor* t1_A = Tensor::full(t1_ref->shape, 5, DEV_CPU);
    Tensor* t1_B = Tensor::full(t1_ref->shape, 5, DEV_CPU);

    Tensor* new_t = t1_A->equal(t1_B);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 1e-3f, 0.0f, true, true));

    Tensor* t2_ref = Tensor::zeros({2, 2});
    Tensor* t2_A = Tensor::full(t2_ref->shape, 5, DEV_CPU);
    Tensor* t2_B = Tensor::full(t2_ref->shape, 4, DEV_CPU);

    Tensor* new_t2 = t2_A->equal(t2_B);
    ASSERT_TRUE(Tensor::equivalent(t2_ref, new_t2, 1e-3f, 0.0f, true, true));

    delete t1_ref;
    delete t1_A;
    delete t1_B;
    delete new_t;
    delete t2_ref;
    delete t2_A;
    delete t2_B;
    delete new_t2;

    // Test GPU
#ifdef cGPU
    Tensor* t_cpu_A = Tensor::randu({3, 1000, 1000});
        Tensor* t_gpu_A = t_cpu_A->clone(); t_gpu_A->toGPU();

        Tensor* t_cpu_B = Tensor::full(t_cpu_A->shape, 0.5);
        Tensor* t_gpu_B = t_cpu_B->clone(); t_gpu_B->toGPU();

        Tensor* new_t_cpu = t_cpu_A->equal(t_cpu_B);
        Tensor* new_t_gpu = t_gpu_A->equal(t_gpu_B); new_t_gpu->toCPU();

        ASSERT_TRUE(Tensor::equivalent(new_t_cpu, new_t_gpu, 1e-3f, 0.0f, true, true));

        delete t_cpu_A;
        delete t_gpu_A;
        delete t_cpu_B;
        delete t_gpu_B;
        delete new_t_cpu;
        delete new_t_gpu;

#endif
}

TEST(TensorTestSuite, tensor_comparison_not_equalT){
// Test #1
    Tensor* t1_ref = Tensor::zeros({2, 2});
    Tensor* t1_A = Tensor::full(t1_ref->shape, 5, DEV_CPU);
    Tensor* t1_B = Tensor::full(t1_ref->shape, 5, DEV_CPU);

    Tensor* new_t = t1_A->not_equal(t1_B);
    ASSERT_TRUE(Tensor::equivalent(t1_ref, new_t, 1e-3f, 0.0f, true, true));

    Tensor* t2_ref = Tensor::ones({2, 2});
    Tensor* t2_A = Tensor::full(t2_ref->shape, 5, DEV_CPU);
    Tensor* t2_B = Tensor::full(t2_ref->shape, 4, DEV_CPU);

    Tensor* new_t2 = t2_A->not_equal(t2_B);
    ASSERT_TRUE(Tensor::equivalent(t2_ref, new_t2, 1e-3f, 0.0f, true, true));

    delete t1_ref;
    delete t1_A;
    delete t1_B;
    delete new_t;
    delete t2_ref;
    delete t2_A;
    delete t2_B;
    delete new_t2;


    // Test GPU
#ifdef cGPU
        Tensor* t_cpu_A = Tensor::randu({3, 1000, 1000});
        Tensor* t_gpu_A = t_cpu_A->clone(); t_gpu_A->toGPU();

        Tensor* t_cpu_B = Tensor::full(t_cpu_A->shape, 0.5);
        Tensor* t_gpu_B = t_cpu_B->clone(); t_gpu_B->toGPU();

        Tensor* new_t_cpu = t_cpu_A->not_equal(t_cpu_B);
        Tensor* new_t_gpu = t_gpu_A->not_equal(t_gpu_B); new_t_gpu->toCPU();

        ASSERT_TRUE(Tensor::equivalent(new_t_cpu, new_t_gpu, 1e-3f, 0.0f, true, true));

        delete t_cpu_A;
        delete t_gpu_A;
        delete t_cpu_B;
        delete t_gpu_B;
        delete new_t_cpu;
        delete new_t_gpu;
        
#endif
}
