#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "apis/eddl.h"
#include "apis/eddlT.h"

using namespace eddl;

int main(int argc, char** argv)
{

    // Define Network without Conv 3x3
    layer in_3x3 = Input({ 3, 767, 1022 });
    layer out_3x3 = Conv(in_3x3, 1, { 3,3 });
    model net_3x3 = Model({ in_3x3 }, {}); //no output, no losses

    // Define Network with Conv 1x1
    layer in_1x1 = Input({ 3, 767, 1022 });
    layer out_1x1 = Conv(in_1x1, 1, { 1,1 });
    model net_1x1 = Model({ in_1x1 }, {}); //no output, no losses

    // Build model
    build(net_3x3);
    //toGPU(net3x3);

    // Build model
    build(net_1x1);
    //toGPU(net1x1);

    summary(net_3x3);
    summary(net_1x1);

    // Load dataset
    tensor x_train = eddlT::load("ISIC_0000000.jpg", "jpg");
    tensor y_train = eddlT::load("ISIC_0000000_segmentation.png", "png");

    // Preprocessing
    eddlT::div_(x_train, 255.);
    eddlT::div_(y_train, 255.);

    // 3x3
    forward(net_3x3,{x_train});
    tensor img_3x3 = getTensor(out_3x3);
    img_3x3 = eddlT::select(img_3x3, 0);

    cout << "Saving 'img_conv_3x3.png'...\n";
    eddlT::reshape_(img_3x3, { 1, 1, 767, 1022 });
    eddlT::save(img_3x3, "img_conv_3x3.png", "png");

    //1x1
    forward(net_1x1,{x_train});
    tensor img_1x1 = getTensor(out_1x1);
    img_1x1 = eddlT::select(img_1x1, 0);

    cout << "Saving 'img_conv_1x1.png'...\n";
    eddlT::reshape_(img_1x1, { 1, 1, 767, 1022 });
    eddlT::save(img_1x1, "img_conv_1x1.png", "png");


}
