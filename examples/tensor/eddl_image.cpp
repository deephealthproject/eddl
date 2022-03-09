/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.1
* copyright (c) 2022, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: March 2022
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <string>

#include "eddl/tensor/tensor.h"


using namespace std;

int main(int argc, char **argv) {

    Tensor* t1 = nullptr;
    Tensor* t2 = nullptr;
    Tensor* t3 = nullptr;

    // Paths
    string fname = "../../examples/data/elephant.jpg";  // Some image
    string output = "./";  // Create this folder!

    // Load image
    t1 = Tensor::load(fname);
    t1->unsqueeze_();
    t1->save(output + "original.jpg");
    cout << "Image saved! (Original)" << endl;

    // Downscale
    t2 = Tensor::zeros({1, 3, 100, 100});
    Tensor::scale(t1, t2, {100, 100}, WrappingMode::Constant, 0.0f, TransformationMode::HalfPixel);
    t1->set_select({":", ":", "100:200", "300:400"}, t2);  // "Paste" t2 in t1
    t2->save(output + "example_scale_resize.jpg");
    cout << "Image saved! (Scale resize)" << endl;

    // Rotate
    t3 = t2->clone();
    Tensor::rotate(t2, t3, 60.0f, {0,0}, WrappingMode::Original);
    t3->mult_(0.5f);
    t3->clampmax_(255.0f);
    t1->set_select({":", ":", "-150:-50", "-150:-50"}, t3);  // "Paste" t3 in t1

    // Save original
    t1->save(output + "original_modified.jpg");
    cout << "Image saved! (Original modified)" << endl;

    delete t1;
    delete t2;
    delete t3;

    // ***************************************************
    // Perform more manipulations ************************
    // ***************************************************
    // Load original
    t1 = Tensor::load(fname);
    t1->unsqueeze_();
    t2 = new Tensor(t1->shape);

    // [Shift] - Mode = constant
    Tensor::shift(t1, t2, {50, 100}, WrappingMode::Constant, 0.0f);  // {y_offset, x_offset}
    t2->save(output + "example_shift_mode_C.png");
    cout << "Image saved! (shift 1)" << endl;

    // [Shift] - Mode = constant
    Tensor::shift(t1, t2, {50, 100}, WrappingMode::Original);  // {y_offset, x_offset}
    t2->save(output + "example_shift_mode_O.bmp");
    cout << "Image saved! (shift 2)" << endl;

    // [Rotate]
     Tensor::rotate(t1, t2, 30, {0,0}, WrappingMode::Original); // angle in degrees
     t2->save(output + "example_rotate.png");
    cout << "Image saved! (rotate)" << endl;

    // [Scale]
    Tensor::scale(t1, t2, {200, 200}); // {height, width}
    t2->save(output + "example_scale_keep_size.jpg");
    cout << "Image saved! (scale keep)" << endl;

    // [Flip]
    Tensor::flip(t1, t2, 0); // axis {vertical=0; horizontal=1}
    t2->save(output + "example_flip.jpg");
    cout << "Image saved! (flip)" << endl;

    // [Crop]
    Tensor::crop(t1, t2, {0, 250}, {200, 450}); // from: {y, x}; to: {y, x}
    t2->save(output + "example_crop_keep_size.jpg");
    cout << "Image saved! (crop keep)" << endl;

    // [Crop]
    t3 = Tensor::zeros({1, 3, 200, 200});
    Tensor::crop(t1, t3, {0, 250}, {200, 450}); // from: {y, x}; to: {y, x}
    t3->save(output + "example_crop_resize.jpg");
    cout << "Image saved! (crop resize)" << endl;

    // [CropScale]
    Tensor::crop_scale(t1, t2, {0, 250}, {200, 450}); // from: {y, x}; to: {y, x}
    t2->save(output + "example_cropscale.jpg");
    cout << "Image saved! (cropscale)" << endl;

    // [Cutout]
    Tensor::cutout(t1, t2, {50, 100}, {100, 400}); // from: {y, x}; to: {y, x}
    t2->save(output + "example_cutout.jpg");
    cout << "Image saved! (cutout)" << endl;

    delete t2;

    // Increase brightness
    t2 = t1->clone(); // Copy image for changes in-place
    t2->mult_(1.5);
    t2->clamp_(0.0f, 255.0f);  // Clamp values
    t2->save(output + "example_brightness.jpg");
    cout << "Image saved! (brightness)" << endl;

    delete t1;
    delete t2;
    delete t3;
}
