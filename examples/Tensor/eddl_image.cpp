/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.2
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
    string fname = "../examples/data/elephant.jpg";  // Some image
    string output = "output/";  // Create this folder!

    // Load image
    Tensor *t1 = Tensor::load(fname);
    t1->info();

    // Where to save the image
    Tensor *t2 = new Tensor(t1->shape);

    // Perform some manipulations ************************
    // [Shift] - Mode = constant
    Tensor::shift(t1, t2, {50, 100}, "constant", 0.0f);  // {y_offset, x_offset}
    t2->save(output + "example_shift_mC.png");
    cout << "Image saved! (shift 1)" << endl;

    // [Shift] - Mode = constant
    Tensor::shift(t1, t2, {50, 100}, "original");  // {y_offset, x_offset}
    t2->save(output + "example_shift_mO.bmp");
    cout << "Image saved! (shift 2)" << endl;

    // [Rotate]
     Tensor::rotate(t1, t2, 30, {0,0}, "original"); // angle in degrees
     t2->save(output + "example_rotate.png");
    cout << "Image saved! (rotate)" << endl;

    // [Scale]
    Tensor::scale(t1, t2, {200, 200}); // {height, width}
    t2->save(output + "example_scale.jpg");
    cout << "Image saved! (scale)" << endl;

    // [Flip]
    Tensor::flip(t1, t2, 0); // axis {vertical=0; horizontal=1}
    t2->save(output + "example_flip.jpg");
    cout << "Image saved! (flip)" << endl;

    // [Crop]
    Tensor::crop(t1, t2, {0, 250}, {200, 450}); // from: {y, x}; to: {y, x}
    t2->save(output + "example_crop.jpg");
    cout << "Image saved! (crop)" << endl;

    // [CropScale]
    Tensor::crop_scale(t1, t2, {0, 250}, {200, 450}); // from: {y, x}; to: {y, x}
    t2->save(output + "example_cropscale.jpg");
    cout << "Image saved! (cropscale)" << endl;

    // [Cutout]
    Tensor::cutout(t1, t2, {50, 100}, {100, 400}); // from: {y, x}; to: {y, x}
    t2->save(output + "example_cutout.jpg");
    cout << "Image saved! (cutout)" << endl;

    // Increase brightness
    t2 = t1->clone(); // Copy image for changes in-place
    t2->mult_(1.5);
    t2->clamp_(0.0f, 255.0f);  // Clamp values
    t2->save(output + "example_brightness.jpg");
    cout << "Image saved! (brightness)" << endl;


}
