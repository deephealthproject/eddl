/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.1
* copyright (c) 2022, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: March 2022
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "eddl/apis/eddl.h"

#include "eddl/serialization/onnx/eddl_onnx.h" // Not allowed

using namespace eddl;


//////////////////////////////////
// Text generation
// Only Decoder
//////////////////////////////////


Tensor *onehot(Tensor *in, int vocs)
{
  int n=in->shape[0];
  int l=in->shape[1];
  int c=0;

  Tensor *out=new Tensor({n,l,vocs});
  out->fill_(0.0);

  int p=0;
  for(int i=0;i<n*l;i++,p+=vocs) {
    int w=in->ptr[i];
    if (w==0) c++;
    out->ptr[p+w]=1.0;
  }

  cout<<"padding="<<(100.0*c)/(n*l)<<"%"<<endl;
  return out;
}

int main(int argc, char **argv) {

    bool testing = false;
    bool use_cpu = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--testing") == 0) testing = true;
        else if (strcmp(argv[i], "--cpu") == 0) use_cpu = true;
    }

    download_flickr();

    // Settings
    //int epochs = testing ? 2 : 50;
    int epochs = testing ? 2 : 4;
    int batch_size = 24;

    int olength=20;
    int outvs=2000;
    int embdim=32;

    model net=download_resnet18(true,{3, 256, 256});  
    // true: remove last layers and set new top=flatten 
    // new input_size {3,256,256} from {224,224,3}

    layer lreshape=getLayer(net,"top");
    
    // create a new model from input output
    layer image_in=getLayer(net,"input");

    
    // Decoder
    layer ldecin = Input({outvs});
    layer ldec = ReduceArgMax(ldecin,{0});
    ldec = RandomUniform(Embedding(ldec, outvs, 1,embdim,true),-0.05,0.05);

    ldec = Concat({ldec,lreshape});

    layer l = LSTM(ldec,512,true);

    layer out = Softmax(Dense(l, outvs));

    setDecoder(ldecin);

    model old_net = net;

    net = Model({image_in}, {out});

    delete old_net;

    plot(net, "model.pdf");

    optimizer opt=adam(0.01);
    //opt->set_clip_val(0.01);
    compserv cs = nullptr;
    if (use_cpu) {
        cs = CS_CPU();
    } else {
        //cs = CS_GPU({1}, "low_mem"); // one GPU
        cs = CS_GPU(); // one GPU
        // cs = CS_GPU({1,1},100); // two GPU with weight sync every 100 batches
        // cs = CS_CPU();
    }

    // Build model
    build(net,
          opt, // Optimizer
          {"softmax_cross_entropy"}, // Losses
          {"accuracy"}, // Metrics
          cs);

    // View model
    summary(net);


    // Load dataset
    Tensor *x_train=Tensor::load("flickr_trX.bin","bin");
    //x_train->info(); //1000,256,256,3

    Tensor *y_train=Tensor::load("flickr_trY.bin","bin");
    //y_train->info();

    if (testing) {
        x_train->info();
        y_train->info();
        std::string _range_ = "0:" + std::to_string(2 * batch_size);
        Tensor* x_mini_train = x_train->select({_range_, ":", ":", ":"});
        Tensor* y_mini_train = y_train->select({_range_, ":"});
        //Tensor* x_mini_test  = x_test->select({_range_, ":", ":", ":"});
        //Tensor* y_mini_test  = y_test->select({_range_, ":"});

        delete x_train;
        delete y_train;
        //delete x_test;
        //delete y_test;

        x_train = x_mini_train;
        y_train = y_mini_train;
        //x_test  = x_mini_test;
        //y_test  = y_mini_test;
    }

    Tensor *xtrain = Tensor::permute(x_train,{0,3,1,2});//1000,3,256,256
    Tensor *ytrain = y_train;
    y_train=onehot(ytrain,outvs);
    y_train->reshape_({y_train->shape[0],olength,outvs}); //batch x timesteps x input_dim
    //y_train->info();


    //load(net,"img2text.bin","bin");

    // Train model
    fit(net, {xtrain}, {y_train}, batch_size, epochs);
    

    save(net,"img2text.bin","bin");

    /////////////////////////////////////////////
    // INFERENCE
    /////////////////////////////////////////////


    cout<<"==================================\n";
    cout<<"===         INFERENCE          ===\n";
    cout<<"==================================\n";


    /////////////////////////////////////////////
    /// Get all the reshapes of the images
    /// Only use the CNN
    /////////////////////////////////////////////

    Tensor *timage=new Tensor({x_train->shape[0], 512}); //images reshape

    model cnn=Model({image_in},{lreshape});

    cs = nullptr;
    if (use_cpu) {
        cs = CS_CPU();
    } else {
        //cs = CS_GPU({1}, "low_mem"); // one GPU
        cs = CS_GPU(); // one GPU
        // cs = CS_GPU({1,1},100); // two GPU with weight sync every 100 batches
        // cs = CS_CPU();
    }

    build(cnn,
          adam(0.001), // not relevant
          {"mse"}, // not relevant
          {"mse"}, // not relevant
          cs
    );
    summary(cnn);
    plot(cnn,"cnn.pdf");

    // forward images
    Tensor* xbatch = new Tensor({batch_size,3,256,256});

    int numbatches=x_train->shape[0]/batch_size;
    for(int j=0;j<1;j++)  {
        cout<<"batch "<<j<<endl;

        next_batch({x_train},{xbatch});
        forward(cnn,{xbatch});

        Tensor* ybatch=getOutput(lreshape);

        string sample=to_string(j*batch_size)+":"+to_string((j+1)*batch_size);
        timage->set_select({sample,":"},ybatch);


        delete ybatch;
    }
    delete xbatch;



    /////////////////////////////////////////////
    /// Create Decoder non recurrent for n-best
    /////////////////////////////////////////////

    ldecin = Input({outvs});
    layer image = Input({512});
    layer lstate = States({2,512});

    ldec = ReduceArgMax(ldecin,{0});
    ldec = RandomUniform(Embedding(ldec, outvs, 1,embdim),-0.05,0.05);

    ldec = Concat({ldec,image});

    layer lstm = LSTM({ldec,lstate},512,true);

    lstm->isrecurrent=false; // Important

    out = Softmax(Dense(lstm, outvs));

    model decoder=Model({ldecin,image,lstate},{out});

    cs = nullptr;
    if (use_cpu) {
        cs = CS_CPU();
    } else {
        //cs = CS_GPU({1}, "low_mem"); // one GPU
        cs = CS_GPU(); // one GPU
        // cs = CS_GPU({1,1},100); // two GPU with weight sync every 100 batches
        // cs = CS_CPU();
    }

      // Build model
    build(decoder,
          adam(0.001), // not relevant
          {"softmax_cross_entropy"}, // not relevant
          {"accuracy"}, // not relevant
          cs
    );

    // View model
    summary(decoder);
    plot(decoder, "decoder.pdf");

    // Copy params from trained net
    copyParam(getLayer(net,"LSTM1"),getLayer(decoder,"LSTM2"));
    copyParam(getLayer(net,"dense1"),getLayer(decoder,"dense2"));
    copyParam(getLayer(net,"embedding1"),getLayer(decoder,"embedding2"));


   ////// N-best for sample s
   int s = testing ? 1 : 100; //sample 100
   // three input tensors with batch_size=1 (one sentence)
   Tensor *treshape=timage->select({to_string(s),":"});
   Tensor *text=y_train->select({to_string(s),":",":"}); //1 x olength x outvs
   //Tensor *state=Tensor::zeros({1,2,512}); // batch x num_states x dim_states

   for(int j=0;j<olength;j++) {
     cout<<"Word:"<<j<<endl;

     Tensor *word;
     if (j==0) word=Tensor::zeros({1,outvs});
     else {
       word=text->select({"0",to_string(j-1),":"});
       word->reshape_({1,outvs}); // batch=1
     }

     treshape->reshape_({1,512}); // batch=1
     Tensor *state=Tensor::zeros({1,2,512}); // batch=1
     
     vtensor input;
     input.push_back(word);
     input.push_back(treshape);
     input.push_back(state);
     forward(decoder, input);
     // forward(decoder,(vtensor){word,treshape,state});

     Tensor *outword=getOutput(out);

     vector<Tensor*> vstates=getStates(lstm); 
     for(int i=0;i<vstates.size();i++) {
       Tensor * temp = vstates[i]->reshape({1,1,512});
       state->set_select({":",to_string(i),":"}, temp);
       delete temp;
       delete vstates[i];
     }
     vstates.clear();
     delete state;
     delete word;
     delete outword;
   }

    delete xtrain;
    delete ytrain;
    delete x_train;
    delete y_train;

    //delete lstate;
    delete decoder;
    delete cnn;
    delete net;

    delete timage;
    delete treshape;
    delete text;

    return EXIT_SUCCESS;
}
