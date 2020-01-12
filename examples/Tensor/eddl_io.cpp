/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <string>

#include "tensor/tensor.h"


using namespace std;

int main(int argc, char **argv) {

    // Paths
    string fname = "../examples/data/";
    string output = "./output/";  // Create this folder!

    // Default tensor
    Tensor *t1 = nullptr;


    // ****************************************************
    // LOAD METHODS ***************************************
    // ****************************************************

    // Load image (formats accepted: jpg, jpeg, png, bmp, hdr, psd, tga, gif, pic, pgm, ppm)
    t1 = Tensor::load(fname + "elephant.jpg");
    t1->info();
    cout << "Tensor loaded! (image)" << endl;

    // Load Numpy
    t1 = Tensor::load<uint8_t>(fname + "iris.npy");  // <source data type>
    t1->info();
    cout << "Tensor loaded! (Numpy)" << endl;

    // Load CSV (Presumes a one row-header, and ',' as delimiter)
    t1 = Tensor::load(fname + "iris.csv");
    t1->info();
    cout << "Tensor loaded! (csv)" << endl;

    // Load TSV (Presumes a one row-header, and '\t' as delimiter)
    t1 = Tensor::load(fname + "iris.tsv");
    t1->info();
    cout << "Tensor loaded! (tsv)" << endl;

    // Load generic txt (csv, csv, tsv,...)
    t1 = Tensor::load_from_txt(fname + "iris.txt", ' ', 0); // false=No header
    t1->info();
    cout << "Tensor loaded! (txt)" << endl;

    // Load binary (EDDL format)
    t1 = Tensor::load(fname + "iris.bin");
    t1->info();
    cout << "Tensor loaded! (bin)" << endl;


    // ****************************************************
    // SAVE METHODS ***************************************
    // ****************************************************
    // Save as image (png, bmp, tga, jpg)
    t1->save(output + "iris.jpg");
    cout << "Tensor saved! (image)" << endl;

    // Save as image (force format/codification)
    t1->save(output + "iris.bin", "jpg");  // Will be save with jpeg's codification but with the extension "*.bin"
    cout << "Tensor saved! (image - Force codification)" << endl;

    // Save as numpy (npy)
    t1->save(output + "iris.npy");
    cout << "Tensor saved! (numpy)" << endl;

//     Save as CSV
    t1->save(output + "iris.csv");
    cout << "Tensor saved! (csv)" << endl;

    // Save as TSV
    t1->save(output + "iris.tsv");
    cout << "Tensor saved! (tsv)" << endl;

    // Save as TXT (csv, csv, tsv,...)
    t1->save2txt(output + "iris.txt", ',', {"sepal.length" , "sepal.width", "petal.length", "petal.width"});
    cout << "Tensor saved! (txt)" << endl;

    // Save as binary (EDDL own format)
    t1->save(output + "iris.bin");
    cout << "Tensor saved! (bin)" << endl;

}
