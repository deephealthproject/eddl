/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "apis/eddl.h"
#include "apis/eddlT.h"

using namespace eddl;

//////////////////////////////////
// Drive segmentation
// https://drive.grand-challenge.org/DRIVE/
// A Multi-GPU segmentation example
// Data Augmentation graph
// Segmentation graph
//////////////////////////////////


// from use case repo:
layer SegNet(layer x)
{
    x = ReLu(Conv(x, 64, { 3,3 }, { 1, 1 }, "same"));
    x = ReLu(Conv(x, 64, { 3,3 }, { 1, 1 }, "same"));
    x = MaxPool(x, { 2,2 }, { 2,2 });
    x = ReLu(Conv(x, 128, { 3,3 }, { 1, 1 }, "same"));
    x = ReLu(Conv(x, 128, { 3,3 }, { 1, 1 }, "same"));
    x = MaxPool(x, { 2,2 }, { 2,2 });
    x = ReLu(Conv(x, 256, { 3,3 }, { 1, 1 }, "same"));
    x = ReLu(Conv(x, 256, { 3,3 }, { 1, 1 }, "same"));
    x = ReLu(Conv(x, 256, { 3,3 }, { 1, 1 }, "same"));
    x = MaxPool(x, { 2,2 }, { 2,2 });
    x = ReLu(Conv(x, 512, { 3,3 }, { 1, 1 }, "same"));
    x = ReLu(Conv(x, 512, { 3,3 }, { 1, 1 }, "same"));
    x = ReLu(Conv(x, 512, { 3,3 }, { 1, 1 }, "same"));
    x = MaxPool(x, { 2,2 }, { 2,2 });
    x = ReLu(Conv(x, 512, { 3,3 }, { 1, 1 }, "same"));
    x = ReLu(Conv(x, 512, { 3,3 }, { 1, 1 }, "same"));
    x = ReLu(Conv(x, 512, { 3,3 }, { 1, 1 }, "same"));
    x = MaxPool(x, { 2,2 }, { 2,2 });

    x = UpSampling(x, { 2,2 });
    x = ReLu(Conv(x, 512, { 3,3 }, { 1, 1 }, "same"));
    x = ReLu(Conv(x, 512, { 3,3 }, { 1, 1 }, "same"));
    x = ReLu(Conv(x, 512, { 3,3 }, { 1, 1 }, "same"));
    x = UpSampling(x, { 2,2 });
    x = ReLu(Conv(x, 512, { 3,3 }, { 1, 1 }, "same"));
    x = ReLu(Conv(x, 512, { 3,3 }, { 1, 1 }, "same"));
    x = ReLu(Conv(x, 256, { 3,3 }, { 1, 1 }, "same"));
    x = UpSampling(x, { 2,2 });
    x = ReLu(Conv(x, 256, { 3,3 }, { 1, 1 }, "same"));
    x = ReLu(Conv(x, 256, { 3,3 }, { 1, 1 }, "same"));
    x = ReLu(Conv(x, 128, { 3,3 }, { 1, 1 }, "same"));
    x = UpSampling(x, { 2,2 });
    x = ReLu(Conv(x, 128, { 3,3 }, { 1, 1 }, "same"));
    x = ReLu(Conv(x, 64, { 3,3 }, { 1, 1 }, "same"));
    x = UpSampling(x, { 2,2 });
    x = ReLu(Conv(x, 64, { 3,3 }, { 1, 1 }, "same"));
    x = Conv(x, 1, { 3,3 }, { 1,1 }, "same");

    return x;
}

int main(int argc, char **argv){

  // Download Dataset
  download_drive();

  // Settings
  int epochs = 25;
  int batch_size =8;

  // Network for Data Augmentation
  // also images are downscale to 256x256
  layer in1=Input({3,584,584});
  layer in2=Input({1,584,584});

  layer l=Concat({in1,in2});   // Cat image and mask
  l=CropScaleRandom(l, {0.8f, 1.0f}); // Random Crop and Scale to orig size
  l=Crop(l,{512,512});         // Crop to work with sizes power 2
  l=Scale(l,{256,256},true);   // Downscale
  layer img=Select(l,{"0:3"}); // UnCat [0-2] image
  layer mask=Select(l,{"3"});  // UnCat [3] mask
  // Both, image and mask, have the same augmentation

  // Define DA model inputs
  model danet=Model({in1,in2},{});
  // Build model for DA
  build(danet);
  // Perform DA in Multi-GPU
  toGPU(danet,{1,1});
  summary(danet);


  // Build SegNet
  layer in=Input({3,256,256});
  layer out=Sigmoid(SegNet(in));
  model segnet=Model({in},{out});
  build(segnet,
    sgd(0.0001, 0.9), // Optimizer
    {"mse"}, // Losses
    {"mse"} // Metrics
  );
  // Train on multi-gpu with sync weights every 10 batches:
  toGPU(segnet,{1,1},10);
  summary(segnet);

  // Load and preprocess training data
  cout<<"Reading train numpy\n";
  tensor x_train_f = Tensor::load<unsigned char>("drive_x.npy");
  tensor x_train=Tensor::permute(x_train_f, {0,3,1,2});
  x_train->info();
  eddlT::div_(x_train,255.0);
  //permute

  cout<<"Reading test numpy\n";
  tensor y_train = Tensor::load<unsigned char>("drive_y.npy");
  y_train->info();
  eddlT::reshape_(y_train,{20,1,584,584});
  eddlT::div_(y_train,255.0);

  tensor xbatch = eddlT::create({batch_size,3,584,584});
  tensor ybatch = eddlT::create({batch_size,1,584,584});

  int num_batches=4;
  for(int i=0;i<epochs;i++) {
    for(int j=0;j<num_batches;j++)  {

      next_batch({x_train,y_train},{xbatch,ybatch});
      tensor yout = eddlT::select(ybatch,0);

      yout->save("./outb.jpg");
      // DA
      forward(danet, (vector<Tensor *>){xbatch,ybatch});

      // get tensors from DA
      tensor xbatch_da = getTensor(img);
      tensor ybatch_da = getTensor(mask);



      // SegNet
      train_batch(segnet, {xbatch_da},{ybatch_da});

      yout = eddlT::select(getTensor(out),0);

      yout->save("./out.jpg");

      print_loss(segnet,j);
      printf("\r");
    }
    printf("\n");
  }

}
