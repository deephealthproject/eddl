/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.8
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <string>

#include "eddl/tensor/tensor.h"


using namespace std;

int main(int argc, char **argv) {

    // Paths
    string fname = "../../examples/data/";
    string output = "./";

    // Default tensor

    // ****************************************************
    // LOAD METHODS ***************************************
    // ****************************************************

    Tensor* t = Tensor::randn({3, 3});
    t->print();
    t->print(1);
    t->print(0, true);

    delete t;

    // Load image (formats accepted: jpg, jpeg, png, bmp, hdr, psd, tga, gif, pic, pgm, ppm)
    Tensor *t1 = Tensor::load(fname + "elephant.jpg");
    cout << "Tensor loaded! (image)" << endl;
    cout << endl;

//    // Load Numpy
//    Tensor *t2 = Tensor::load<float>(fname + "iris.npy");  // <source data type>
//    cout << "Tensor loaded! (Numpy)" << endl;
//    cout << endl;
//    t2->print(2);
//
//    // Load CSV (Presumes a one row-header, and ',' as delimiter)
//    Tensor *t3 = Tensor::load(fname + "iris.csv");
//    cout << "Tensor loaded! (csv)" << endl;
//    cout << "isEqualToPrevious? " << Tensor::equivalent(t2, t3, 10e-6f) << endl;
//    cout << endl;
//
//    // Load TSV (Presumes a one row-header, and '\t' as delimiter)
//    Tensor *t4 = Tensor::load(fname + "iris.tsv");
//    cout << "Tensor loaded! (tsv)" << endl;
//    cout << "isEqualToPrevious? " << Tensor::equivalent(t3, t4, 10e-6f) << endl;
//    cout << endl;
//
//    // Load generic txt (csv, csv, tsv,...)
//    Tensor *t5 = Tensor::load_from_txt(fname + "iris.txt", ' ', 1);
//    cout << "Tensor loaded! (txt)" << endl;
//    cout << "isEqualToPrevious? " << Tensor::equivalent(t4, t5, 10e-6f) << endl;
//    cout << endl;

    // Load binary (EDDL format)
    Tensor *t6 = Tensor::load(fname + "iris.bin");
    cout << "Tensor loaded! (bin)" << endl;
    cout << "isEqualToPrevious? " << Tensor::equivalent(t1, t6, 10e-6f) << endl;
    cout << endl;

    // ****************************************************
    // SAVE METHODS ***************************************
    // ****************************************************
    // Save as image (png, bmp, tga, jpg)
    t1->save(output + "iris.jpg");
    cout << "Tensor saved! (image)" << endl;
    cout << endl;

    // Save as image (force format/codification)
    t1->save(output + "iris.bin", "jpg");  // Will be save with jpeg's codification but with the extension "*.bin"
    cout << "Tensor saved! (image - Force codification)" << endl;
    cout << endl;

//    // Save as numpy (npy)
//    t2->save(output + "iris.npy");
//    cout << "Tensor saved! (numpy)" << endl;
//    cout << endl;
//
//    // Save as CSV
//    t3->save(output + "iris.csv");
//    cout << "Tensor saved! (csv)" << endl;
//    cout << endl;
//
//    // Save as TSV
//    t4->save(output + "iris.tsv");
//    cout << "Tensor saved! (tsv)" << endl;
//    cout << endl;
//
//    // Save as TXT (csv, csv, tsv,...)
//    t5->save2txt(output + "iris.txt", ',', {"sepal.length" , "sepal.width", "petal.length", "petal.width"});
//    cout << "Tensor saved! (txt)" << endl;
//    cout << endl;

    // Save as binary (EDDL own format)
    t6->save(output + "iris.bin");
    cout << "Tensor saved! (bin)" << endl;
    cout << endl;

    // ptr serialization 
    Tensor* t7 = Tensor::randn({2, 3, 4, 4});
    cout << "GOING TO TEST PTR SERIALIZATION" << endl;
    cout << "Tensor before saving to ptr" << endl;
    t7->info();
    t7->print();
    cout << "Saving to ptr" << endl;
    std::pair<void*, size_t> serial_t7 = t7->save2ptr();
    cout << "Saved to ptr" << endl;

    cout << "Loading from ptr" << endl;
    Tensor* loaded_t7 = Tensor::load_from_ptr(serial_t7.first);
    cout << "Loaded from ptr" << endl;
    cout << "Tensor after loading from ptr" << endl;
    loaded_t7->info();
    loaded_t7->print();

    delete t1;
    delete t6;
    delete t7;
    delete loaded_t7;
    free(serial_t7.first);
}
