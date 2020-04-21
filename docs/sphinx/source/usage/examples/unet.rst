Training a U-Net
-----------------

This example trains and evaluates a U-Net

.. image:: /_static/images/models/segnet.svg
  :scale: 100%

.. code:: c++

    #include <stdio.h>
    #include <stdlib.h>
    #include <iostream>

    #include <eddl/apis/eddl.h>
    #include <eddl/apis/eddlT.h>

    using namespace eddl;

    layer UNetWithPadding(layer x){
        layer x2;
        layer x3;
        layer x4;
        layer x5;

        int depth = 32;

        x = LeakyReLu(Conv(x, depth, { 3,3 }, { 1, 1 }, "same"));
        x = LeakyReLu(Conv(x, depth, { 3,3 }, { 1, 1 }, "same"));
        x2 = MaxPool(x, { 2,2 }, { 2,2 });
        x2 = LeakyReLu(Conv(x2, 2*depth, { 3,3 }, { 1, 1 }, "same"));
        x2 = LeakyReLu(Conv(x2, 2*depth, { 3,3 }, { 1, 1 }, "same"));
        x3 = MaxPool(x2, { 2,2 }, { 2,2 });
        x3 = LeakyReLu(Conv(x3, 4*depth, { 3,3 }, { 1, 1 }, "same"));
        x3 = LeakyReLu(Conv(x3, 4*depth, { 3,3 }, { 1, 1 }, "same"));
        x4 = MaxPool(x3, { 2,2 }, { 2,2 });
        x4 = LeakyReLu(Conv(x4, 8*depth, { 3,3 }, { 1, 1 }, "same"));
        x4 = LeakyReLu(Conv(x4, 8*depth, { 3,3 }, { 1, 1 }, "same"));
        x5 = MaxPool(x4, { 2,2 }, { 2,2 });
        x5 = LeakyReLu(Conv(x5, 8*depth, { 3,3 }, { 1, 1 }, "same"));
        x5 = LeakyReLu(Conv(x5, 8*depth, { 3,3 }, { 1, 1 }, "same"));
        x5 = Conv(UpSampling(x5, { 2,2 }), 8*depth, { 2,2 }, { 1, 1 }, "same");

        x4 = Concat({x4,x5});
        x4 = LeakyReLu(Conv(x4, 8*depth, { 3,3 }, { 1, 1 }, "same"));
        x4 = LeakyReLu(Conv(x4, 8*depth, { 3,3 }, { 1, 1 }, "same"));
        x4 = Conv(UpSampling(x4, { 2,2 }), 4*depth, { 2,2 }, { 1, 1 }, "same");

        x3 = Concat({x3,x4});
        x3 = LeakyReLu(Conv(x3, 4*depth, { 3,3 }, { 1, 1 }, "same"));
        x3 = LeakyReLu(Conv(x3, 4*depth, { 3,3 }, { 1, 1 }, "same"));
        x3 = Conv(UpSampling(x3, { 2,2 }), 2*depth, { 2,2 }, { 1, 1 }, "same");

        x2 = Concat({x2,x3});
        x2 = LeakyReLu(Conv(x2, 2*depth, { 3,3 }, { 1, 1 }, "same"));
        x2 = LeakyReLu(Conv(x2, 2*depth, { 3,3 }, { 1, 1 }, "same"));
        x2 = Conv(UpSampling(x2, { 2,2 }), depth, { 2,2 }, { 1, 1 }, "same");

        x = Concat({x,x2});
        x = LeakyReLu(Conv(x, depth, { 3,3 }, { 1, 1 }, "same"));
        x = LeakyReLu(Conv(x, depth, { 3,3 }, { 1, 1 }, "same"));
        x = Conv(x, 1, { 1,1 });

        return x;
    }


    int main(int argc, char **argv){

        // Download Dataset
        download_drive();

        // Settings
        int epochs = 100000;
        int batch_size = 2;

        // Network for Data Augmentation
        layer in1=Input({3,584,584});
        layer in2=Input({1,584,584});

        layer l=Concat({in1,in2});   // Cat image and mask
        l= RandomCropScale(l, {0.9f, 1.0f}); // Random Crop and Scale to orig size
        l= CenteredCrop(l,{512,512});         // Crop to work with sizes power 2
        layer img = Select(l,{"0:3"}); // UnCat [0-2] image
        layer mask = Select(l,{"3"});  // UnCat [3] mask
        // Both, image and mask, have the same augmentation

        // Define DA model inputs
        model danet=Model({in1,in2},{});

        // Build model for DA
        build(danet);
        summary(danet);

        // Build SegNet
        layer in=Input({3,512,512});
        layer out=Sigmoid(UNetWithPadding(in));
        model segnet=Model({in},{out});
        build(segnet,
              adam(0.00001), // Optimizer
              {"mse"}, // Losses
              {"mse"}, // Metrics
              CS_GPU({1, 1}, 100, "low_mem")  // Training using two GPU (0/1=off/on) on low memory setup, and synchronize every 100 batches
        );

        // Print net information
        summary(segnet);
        plot(segnet,"segnet.pdf");

        // Load and preprocess training data
        cout << "Reading train numpy" << endl;
        tensor x_train_f = Tensor::load<uint8_t>("drive_x.npy");
        tensor x_train=Tensor::permute(x_train_f, {0,3,1,2});
        eddlT::div_(x_train,255.0);

        cout << "Reading test numpy" << endl;
        tensor y_train = Tensor::load<uint8_t>("drive_y.npy");
        eddlT::reshape_(y_train,{20,1,584,584});
        eddlT::div_(y_train,255.0);

        tensor xbatch = eddlT::create({batch_size,3,584,584});
        tensor ybatch = eddlT::create({batch_size,1,584,584});

        // Training
        int num_batches=1000;
        for(int i=0;i<epochs;i++) {
            reset_loss(segnet);
            for(int j=0;j<num_batches;j++)  {

                // Get batch
                next_batch({x_train,y_train},{xbatch,ybatch});

                // Perform data augmentation (image+mask)
                forward(danet, (vector<Tensor *>){xbatch,ybatch});

                // get tensors from DA
                tensor xbatch_da = getTensor(img);
                tensor ybatch_da = getTensor(mask);

                // SegNet
                train_batch(segnet, {xbatch_da},{ybatch_da});

                print_loss(segnet, j);

                // Save image
                tensor yout = eddlT::select(getTensor(out),0);
                yout->save("./out.jpg");
                delete yout;
            }
        }
    }
