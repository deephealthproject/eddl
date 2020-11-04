#include <gtest/gtest.h>
#include <random>
#include <string>

#include "eddl/tensor/tensor.h"
#include "eddl/tensor/nn/tensor_nn.h"
#include "eddl/descriptors/descriptors.h"

using namespace std;

// Demo dataset
static auto *ptr_iris = new float[150*4]{
        5.10, 3.50, 1.40, 0.20,
        4.90, 3.00, 1.40, 0.20,
        4.70, 3.20, 1.30, 0.20,
        4.60, 3.10, 1.50, 0.20,
        5.00, 3.60, 1.40, 0.20,
        5.40, 3.90, 1.70, 0.40,
        4.60, 3.40, 1.40, 0.30,
        5.00, 3.40, 1.50, 0.20,
        4.40, 2.90, 1.40, 0.20,
        4.90, 3.10, 1.50, 0.10,
        5.40, 3.70, 1.50, 0.20,
        4.80, 3.40, 1.60, 0.20,
        4.80, 3.00, 1.40, 0.10,
        4.30, 3.00, 1.10, 0.10,
        5.80, 4.00, 1.20, 0.20,
        5.70, 4.40, 1.50, 0.40,
        5.40, 3.90, 1.30, 0.40,
        5.10, 3.50, 1.40, 0.30,
        5.70, 3.80, 1.70, 0.30,
        5.10, 3.80, 1.50, 0.30,
        5.40, 3.40, 1.70, 0.20,
        5.10, 3.70, 1.50, 0.40,
        4.60, 3.60, 1.00, 0.20,
        5.10, 3.30, 1.70, 0.50,
        4.80, 3.40, 1.90, 0.20,
        5.00, 3.00, 1.60, 0.20,
        5.00, 3.40, 1.60, 0.40,
        5.20, 3.50, 1.50, 0.20,
        5.20, 3.40, 1.40, 0.20,
        4.70, 3.20, 1.60, 0.20,
        4.80, 3.10, 1.60, 0.20,
        5.40, 3.40, 1.50, 0.40,
        5.20, 4.10, 1.50, 0.10,
        5.50, 4.20, 1.40, 0.20,
        4.90, 3.10, 1.50, 0.20,
        5.00, 3.20, 1.20, 0.20,
        5.50, 3.50, 1.30, 0.20,
        4.90, 3.60, 1.40, 0.10,
        4.40, 3.00, 1.30, 0.20,
        5.10, 3.40, 1.50, 0.20,
        5.00, 3.50, 1.30, 0.30,
        4.50, 2.30, 1.30, 0.30,
        4.40, 3.20, 1.30, 0.20,
        5.00, 3.50, 1.60, 0.60,
        5.10, 3.80, 1.90, 0.40,
        4.80, 3.00, 1.40, 0.30,
        5.10, 3.80, 1.60, 0.20,
        4.60, 3.20, 1.40, 0.20,
        5.30, 3.70, 1.50, 0.20,
        5.00, 3.30, 1.40, 0.20,
        7.00, 3.20, 4.70, 1.40,
        6.40, 3.20, 4.50, 1.50,
        6.90, 3.10, 4.90, 1.50,
        5.50, 2.30, 4.00, 1.30,
        6.50, 2.80, 4.60, 1.50,
        5.70, 2.80, 4.50, 1.30,
        6.30, 3.30, 4.70, 1.60,
        4.90, 2.40, 3.30, 1.00,
        6.60, 2.90, 4.60, 1.30,
        5.20, 2.70, 3.90, 1.40,
        5.00, 2.00, 3.50, 1.00,
        5.90, 3.00, 4.20, 1.50,
        6.00, 2.20, 4.00, 1.00,
        6.10, 2.90, 4.70, 1.40,
        5.60, 2.90, 3.60, 1.30,
        6.70, 3.10, 4.40, 1.40,
        5.60, 3.00, 4.50, 1.50,
        5.80, 2.70, 4.10, 1.00,
        6.20, 2.20, 4.50, 1.50,
        5.60, 2.50, 3.90, 1.10,
        5.90, 3.20, 4.80, 1.80,
        6.10, 2.80, 4.00, 1.30,
        6.30, 2.50, 4.90, 1.50,
        6.10, 2.80, 4.70, 1.20,
        6.40, 2.90, 4.30, 1.30,
        6.60, 3.00, 4.40, 1.40,
        6.80, 2.80, 4.80, 1.40,
        6.70, 3.00, 5.00, 1.70,
        6.00, 2.90, 4.50, 1.50,
        5.70, 2.60, 3.50, 1.00,
        5.50, 2.40, 3.80, 1.10,
        5.50, 2.40, 3.70, 1.00,
        5.80, 2.70, 3.90, 1.20,
        6.00, 2.70, 5.10, 1.60,
        5.40, 3.00, 4.50, 1.50,
        6.00, 3.40, 4.50, 1.60,
        6.70, 3.10, 4.70, 1.50,
        6.30, 2.30, 4.40, 1.30,
        5.60, 3.00, 4.10, 1.30,
        5.50, 2.50, 4.00, 1.30,
        5.50, 2.60, 4.40, 1.20,
        6.10, 3.00, 4.60, 1.40,
        5.80, 2.60, 4.00, 1.20,
        5.00, 2.30, 3.30, 1.00,
        5.60, 2.70, 4.20, 1.30,
        5.70, 3.00, 4.20, 1.20,
        5.70, 2.90, 4.20, 1.30,
        6.20, 2.90, 4.30, 1.30,
        5.10, 2.50, 3.00, 1.10,
        5.70, 2.80, 4.10, 1.30,
        6.30, 3.30, 6.00, 2.50,
        5.80, 2.70, 5.10, 1.90,
        7.10, 3.00, 5.90, 2.10,
        6.30, 2.90, 5.60, 1.80,
        6.50, 3.00, 5.80, 2.20,
        7.60, 3.00, 6.60, 2.10,
        4.90, 2.50, 4.50, 1.70,
        7.30, 2.90, 6.30, 1.80,
        6.70, 2.50, 5.80, 1.80,
        7.20, 3.60, 6.10, 2.50,
        6.50, 3.20, 5.10, 2.00,
        6.40, 2.70, 5.30, 1.90,
        6.80, 3.00, 5.50, 2.10,
        5.70, 2.50, 5.00, 2.00,
        5.80, 2.80, 5.10, 2.40,
        6.40, 3.20, 5.30, 2.30,
        6.50, 3.00, 5.50, 1.80,
        7.70, 3.80, 6.70, 2.20,
        7.70, 2.60, 6.90, 2.30,
        6.00, 2.20, 5.00, 1.50,
        6.90, 3.20, 5.70, 2.30,
        5.60, 2.80, 4.90, 2.00,
        7.70, 2.80, 6.70, 2.00,
        6.30, 2.70, 4.90, 1.80,
        6.70, 3.30, 5.70, 2.10,
        7.20, 3.20, 6.00, 1.80,
        6.20, 2.80, 4.80, 1.80,
        6.10, 3.00, 4.90, 1.80,
        6.40, 2.80, 5.60, 2.10,
        7.20, 3.00, 5.80, 1.60,
        7.40, 2.80, 6.10, 1.90,
        7.90, 3.80, 6.40, 2.00,
        6.40, 2.80, 5.60, 2.20,
        6.30, 2.80, 5.10, 1.50,
        6.10, 2.60, 5.60, 1.40,
        7.70, 3.00, 6.10, 2.30,
        6.30, 3.40, 5.60, 2.40,
        6.40, 3.10, 5.50, 1.80,
        6.00, 3.00, 4.80, 1.80,
        6.90, 3.10, 5.40, 2.10,
        6.70, 3.10, 5.60, 2.40,
        6.90, 3.10, 5.10, 2.30,
        5.80, 2.70, 5.10, 1.90,
        6.80, 3.20, 5.90, 2.30,
        6.70, 3.30, 5.70, 2.50,
        6.70, 3.00, 5.20, 2.30,
        6.30, 2.50, 5.00, 1.90,
        6.50, 3.00, 5.20, 2.00,
        6.20, 3.40, 5.40, 2.30,
        5.90, 3.00, 5.10, 1.80,
};
static auto* t_iris = new Tensor({150, 4}, ptr_iris, DEV_CPU);

// Demo image
static auto* t_image = Tensor::arange(0, 100);

// Random generators
const int MIN_RANDOM = 0;
const int MAX_RANDOM = 999999;
static std::random_device rd;
static std::mt19937 mt(rd());
static std::uniform_int_distribution<std::mt19937::result_type> dist6(MIN_RANDOM, MAX_RANDOM);


TEST(TensorTestSuite, tensor_io_jpg)
{
    // Generate random name
    int rdn_name = dist6(mt);
    string fname = "image_" + to_string(rdn_name) + ".jpg";

    // Save file
    Tensor *t_ref = Tensor::concat({t_image, t_image, t_image}); // This jpeg needs 3 channels
    t_ref->reshape_({3, 10, 10});  // This jpeg needs 3 channels
    t_ref->save(fname);  // All values are cast to integers

    // Load saved file
    Tensor* t_load = Tensor::load(fname);

    // Delete file
    int hasFailed = std::remove(fname.c_str());
    if(hasFailed) { cout << "Error deleting file: " << fname << endl; }

    ASSERT_TRUE(Tensor::equivalent(t_ref, t_load, 10e-0));

    delete t_ref;
    delete t_load;
}

TEST(TensorTestSuite, tensor_io_png)
{
    // Generate random name
    int rdn_name = dist6(mt);
    string fname = "image_" + to_string(rdn_name) + ".png";

    // Save file
    Tensor *t_ref = Tensor::concat({t_image, t_image, t_image}); // This jpeg needs 3 channels
    t_ref->reshape_({3, 10, 10});  // This jpeg needs 3 channels
    t_ref->save(fname);  // All values are cast to integers

    // Load saved file
    Tensor* t_load = Tensor::load(fname);

    // Delete file
    int hasFailed = std::remove(fname.c_str());
    if(hasFailed) { cout << "Error deleting file: " << fname << endl; }

    ASSERT_TRUE(Tensor::equivalent(t_ref, t_load, 10e-0));

    delete t_ref;
    delete t_load;
}


TEST(TensorTestSuite, tensor_io_bmp)
{
    // Generate random name
    int rdn_name = dist6(mt);
    string fname = "image_" + to_string(rdn_name) + ".bmp";

    // Save file
    Tensor *t_ref = Tensor::concat({t_image, t_image, t_image}); // This jpeg needs 3 channels
    t_ref->reshape_({3, 10, 10});  // This jpeg needs 3 channels
    t_ref->save(fname);  // All values are cast to integers

    // Load saved file
    Tensor* t_load = Tensor::load(fname);

    // Delete file
    int hasFailed = std::remove(fname.c_str());
    if(hasFailed) { cout << "Error deleting file: " << fname << endl; }

    ASSERT_TRUE(Tensor::equivalent(t_ref, t_load, 10e-0));

    delete t_ref;
    delete t_load;
}


//TEST(TensorTestSuite, tensor_io_numpy)
//{
//    // Generate random name
//    int rdn_name = dist6(mt);
//    string fname = "iris_" + to_string(rdn_name) + ".npy";
//
//    // Save file
//    t_iris->save(fname);
//
//    // Load saved file
//    Tensor* t_load = Tensor::load<float>(fname);
//
//    // Delete file
//    int hasFailed = std::remove(fname.c_str());
//    if(hasFailed) { cout << "Error deleting file: " << fname << endl; }
//
//    ASSERT_TRUE(Tensor::equivalent(t_iris, t_load, 10e-5));
//}


//TEST(TensorTestSuite, tensor_io_csv)
//{
//    // Generate random name
//    int rdn_name = dist6(mt);
//    string fname = "iris_" + to_string(rdn_name) + ".csv";
//
//    // Save file
//    t_iris->save(fname);
//
//    // Load saved file
//    Tensor* t_load = Tensor::load(fname);
//
//    // Delete file
//    int hasFailed = std::remove(fname.c_str());
//    if(hasFailed) { cout << "Error deleting file: " << fname << endl; }
//
//    ASSERT_TRUE(Tensor::equivalent(t_iris, t_load, 10e-5));
//}


//TEST(TensorTestSuite, tensor_io_tsv)
//{
//    // Generate random name
//    int rdn_name = dist6(mt);
//    string fname = "iris_" + to_string(rdn_name) + ".tsv";
//
//    // Save file
//    t_iris->save(fname);
//
//    // Load saved file
//    Tensor* t_load = Tensor::load(fname);
//
//    // Delete file
//    int hasFailed = std::remove(fname.c_str());
//    if(hasFailed) { cout << "Error deleting file: " << fname << endl; }
//
//    ASSERT_TRUE(Tensor::equivalent(t_iris, t_load, 10e-5));
//}


//TEST(TensorTestSuite, tensor_io_txt)
//{
//    // Generate random name
//    int rdn_name = dist6(mt);
//    string fname = "iris_" + to_string(rdn_name) + ".txt";
//
//    // Save file
//    t_iris->save2txt(fname, ' ', {"sepal.length" , "sepal.width", "petal.length", "petal.width"});
//
//    // Load saved file
//    Tensor* t_load = Tensor::load_from_txt(fname, ' ', 1);
//
//    // Delete file
//    int hasFailed = std::remove(fname.c_str());
//    if(hasFailed) { cout << "Error deleting file: " << fname << endl; }
//
//    ASSERT_TRUE(Tensor::equivalent(t_iris, t_load, 10e-5));
//}


TEST(TensorTestSuite, tensor_io_bin)
{
    // Generate random name
    int rdn_name = dist6(mt);
    string fname = "iris_" + to_string(rdn_name) + ".bin";

    // Save file
    t_iris->save(fname);

    // Load saved file
    Tensor* t_load = Tensor::load(fname);

    // Delete file
    int hasFailed = std::remove(fname.c_str());
    if(hasFailed) { cout << "Error deleting file: " << fname << endl; }

    ASSERT_TRUE(Tensor::equivalent(t_iris, t_load, 10e-5));

    delete t_iris;
    delete t_load;
}