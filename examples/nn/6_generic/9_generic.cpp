/*
 * EDDL Library - European Distributed Deep Learning Library.
 * Version: 0.9
 * copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
 * Date: November 2020
 * Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
 * All rights reserved
 */

#include <cstdio>
#include <cstdlib>
#include <iostream>
 


#include "eddl/apis/eddl.h"
#include "eddl/serialization/onnx/eddl_onnx.h" // Not allowed

#include "utils.h"

using namespace eddl;

//////////////////////////////////
// vgg16 with BatchNorm
//////////////////////////////////

layer Normalization(layer l)
{
  return BatchNormalization(l);
  //return LayerNormalization(l);
  //return GroupNormalization(l,8);
}

layer Block1(layer l,int filters) {
  return ReLu(Normalization(Conv(l,filters,{1,1},{1,1},"same",false)));
}
layer Block3_2(layer l,int filters) {
  l=ReLu(Normalization(Conv(l,filters,{3,3},{1,1},"same",false)));
  l=ReLu(Normalization(Conv(l,filters,{3,3},{1,1},"same",false)));
  return l;
}

model vgg16bn (vector<int> in_shape, int num_classes, int size1, int size2)  {  
    layer in = Input(in_shape);
    //layer in = Input({3, 224, 224});
    layer l = in;


    // Data augmentation
    //l = RandomCropScale(l, {0.8f, 1.0f});
    //l = RandomFlip(l,1);

    l=MaxPool(Block3_2(l,64));
    l=MaxPool(Block3_2(l,128));
    l=MaxPool(Block1(Block3_2(l,256),256));
    l=MaxPool(Block1(Block3_2(l,512),512));
    l=MaxPool(Block1(Block3_2(l,512),512));

    l=Reshape(l,{-1});
    l = Activation(Dense(l, size1), "relu");
    l = Activation(Dense(l, size2), "relu");

    layer out = Softmax(Dense(l, num_classes));
    // net define input and output layers list
    model net = Model({in},{out});
    return net;
}    

//////////////////////////////////
// Resnet50 with
// BatchNorm
//////////////////////////////////

layer BN(layer l)
{
  return BatchNormalization(l);
  //return l;
}

layer BG(layer l) {
  //return GaussianNoise(BN(l),0.3);
  return BN(l);
}

layer ResBlock(layer l, int filters,int half, int expand=0) {
  layer in=l;

  l=ReLu(BG(Conv(l,filters,{1,1},{1,1},"same",false)));

  if (half)
    l=ReLu(BG(Conv(l,filters,{3,3},{2,2},"same",false)));
  else
    l=ReLu(BG(Conv(l,filters,{3,3},{1,1},"same",false)));

  l=BG(Conv(l,4*filters,{1,1},{1,1},"same",false));

  if (half)
    return ReLu(Add(BG(Conv(in,4*filters,{1,1},{2,2},"same",false)),l));
  else
    if (expand) return ReLu(Add(BG(Conv(in,4*filters,{1,1},{1,1},"same",false)),l));
    else return ReLu(Add(in,l));
}

// Resnet-50
model resnet50 (vector<int> in_shape, int num_classes, int size1, int size2)  {

    layer in = Input(in_shape);
    layer l = in;
    
    l = ReLu(BG(Conv(l, 64,{3, 3},{1, 1}, "same", false))); //{1,1}
    //l=MaxPool(l,{3,3},{1,1},"same");

    // Add explicit padding to avoid the asymmetric padding in the Conv layers
    l = Pad(l,{0, 1, 1, 0});

    for (int i = 0; i < 3; i++)
        l = ResBlock(l, 64, 0, i == 0); // not half but expand the first

    for (int i = 0; i < 4; i++)
        l = ResBlock(l, 128, i == 0);

    for (int i = 0; i < 6; i++)
        l = ResBlock(l, 256, i == 0);

    for (int i = 0; i < 3; i++)
        l = ResBlock(l, 512, i == 0);

    l = MaxPool(l,{4, 4}); // should be avgpool

    l = Reshape(l,{-1});
    //l = Activation(Dense(l, size1), "relu");
    //l = Activation(Dense(l, size2), "relu");

    layer out = Softmax(Dense(l, num_classes));

    // net define input and output layers list
    model net = Model({in}, {out});
    return net;
}


model select_model (int choice, vector<int> in_shape, int num_classes, int size1, int size2)  {
    if (in_shape[0]!=3) {
        printf("ERROR: Only 3 channels are supported");
        exit(1);
    }
    if (in_shape[1]!=224) {
        printf("ERROR: Only height 224 is supported");
        exit(1);
    }
    if (in_shape[2]!=224) {
        printf("ERROR: Only width 224 is supported");
        exit(1);
    }
    model pretrained_model;
    switch (choice) {
        case 0: pretrained_model = download_vgg16(true, in_shape);
            printf("model: vgg16\n");
            break;
        case 1: pretrained_model = download_vgg16_bn(true, in_shape);
            printf("model: vgg16bn\n");
            break;
        case 2: pretrained_model = download_vgg19(true, in_shape);
            printf("model: vgg19\n");
            break;
        case 3: pretrained_model = download_vgg19_bn(true, in_shape);
            printf("model: vgg19bn\n");
            break;
        case 4: pretrained_model = download_resnet18(true, in_shape);
            printf("model: resnet18\n");
            break;
        case 5: pretrained_model = download_resnet34(true, in_shape);
            printf("model: resnet34\n");
            break;
        case 6: pretrained_model = download_resnet50(true, in_shape);
            printf("model: resnet50\n");
            break;
        case 7: pretrained_model = download_resnet101(true, in_shape);
            printf("model: resnet101\n");
            break;
        case 8: pretrained_model = download_resnet152(true, in_shape);
            printf("model: resnet152\n");
            break;
        case 9: pretrained_model = download_densenet121(true, in_shape);
            printf("model: densenet121\n");
            break;
        default: 
                printf("ERROR: unknown model\n");
                exit(1);
            
    }

    // Get the input layer of the pretrained model
    layer in_ = getLayer(pretrained_model, "input");
    // Get the last layer of the pretrained model
    layer top_layer = getLayer(pretrained_model, "top");
    //layer top_layer=nullptr;
    // Create the new densely connected part
    // std::vector<std::string> layers2init{"dense1", "dense_out"};
    //const int input_units = top_layer->output->shape[1];
    //eddl::layer l = eddl::Dense(top_layer, input_units / 2, true, layers2init[0]);
    layer l = nullptr;
    l = Reshape(top_layer,{-1});
    l = Activation(Dense(l, size1, true), "relu");
    l = Activation(Dense(l, size2, true), "relu");
    // l = eddl::Dropout(l, 0.4);
    // Output layer
    layer out_ = Softmax(Dense(l, num_classes, true, "dense_out"));

    model net = Model({in_},{out_});
    return net;
}



void custom_fit(model net, Tensor* x_train, Tensor* y_train, int batch, int epochs, bool divide_batch, bool distr_dataset=false) {
   
    int i,j;
    int id=get_id_distributed();
    int n_procs=get_n_procs_distributed();
    
    tshape s = x_train->getShape();
    int dataset_size=s[0];
    int channels =s[1];
    int width=s[2];
    int height = s[3];
    int num_classes = y_train->getShape()[1];
    
    int num_batches;
    int local_batch;
    int global_batch;
    int nbpp;
    
    int use_cpu;
    int use_mpi;
    
    
    /*
    if (distr_dataset){
        mpi_id0(printf("custom_fit. dataset_size %d  Distributed dataset\n", dataset_size));
        dataset_size=dataset_size*n_procs;
    } else {
       // nbpp=num_batches/n_procs;
    }
    
    num_batches=dataset_size/batch;
    if (divide_batch) {
        global_batch=batch;
        local_batch=global_batch/n_procs;
        nbpp=num_batches;
        mpi_id0(printf("custom_fit. Divide batch: global: %d, local: %d\n", batch, local_batch));
    } else {
        global_batch=batch*n_procs;
        local_batch=global_batch/n_procs;
        nbpp=num_batches/n_procs;
        mpi_id0(printf("custom_fit. Mul batch: global: %d, local: %d\n", local_batch*n_procs, local_batch));
    }
    */
    
    set_batch_distributed(&global_batch, &local_batch, batch, divide_batch);
    nbpp=set_NBPP_distributed(dataset_size,local_batch, distr_dataset);
  
    Tensor* xbatch = new Tensor({local_batch, channels,width,height});
    Tensor* ybatch = new Tensor({local_batch, num_classes});
 
    
    for(i=0;i<epochs;i++) {
      reset_loss(net);
      mpi_id0(printf("custom_fit. Epoch %d/%d\n", i + 1, epochs));
      for(j=0;j<nbpp;j++)  {
          
        next_batch({x_train,y_train},{xbatch,ybatch});
        
        train_batch(net, {xbatch}, {ybatch});
        //sync_batch
        avg_weights_distributed(net, j+1, nbpp);  
        print_loss(net,j+1, false);
        mpi_id0(printf("\r"););
      }
      mpi_id0(printf("\n"));
     // if (early_stopping_on_loss_var (net, 0, 10, 0.1, i)) break;
      //if (early_stopping_on_metric_var (net, 0, 0.0001, 2, i)) break;
      //if (early_stopping_on_metric (net, 0, 0.97, 2, i)) break;
    }
    mpi_id0(printf("\n"));   
     mpi_id0(fflush(stdout));
    
    delete xbatch;
    delete ybatch;

}

void custom_evaluate(model net, Tensor* x_test, Tensor* y_test, int batch, bool divide_batch, bool distr_dataset=false) {
    int i,j,k;
    int id=get_id_distributed();
    int n_procs=get_n_procs_distributed();

    tshape s = x_test->getShape();
    int dataset_size=s[0];
    int channels =s[1];
    int width=s[2];
    int height = s[3];
    int num_classes = y_test->getShape()[1];
    
    int num_batches;
    int local_batch;
    int global_batch;
    int nbpp;

    set_batch_distributed(&global_batch, &local_batch, batch, divide_batch);
    nbpp = set_NBPP_distributed(dataset_size, local_batch, distr_dataset);

 
    
    vind sind;
    for (k = 0; k < local_batch; k++)
        sind.push_back(0);

    reset_loss(net);
    mpi_id0(printf("custom_evaluate\n"));
    for (j = 0; j < nbpp; j++) {
        for (k = 0; k < local_batch; k++) {
            if (distr_dataset)
                sind [k] = (j * local_batch) + k;
            else 
                sind [k] = (((id * nbpp) + j) * local_batch) + k;
            //printf("id=%d  %5d",id, sind[k]);
        }
        eval_batch(net, {x_test}, {y_test}, sind);
        //         print_loss(batches, num_batches);
        print_loss(net, j + 1, false);
        mpi_id0(fprintf(stdout, "\r"));
        mpi_id0(fflush(stdout));
    }
    // sync processes and print final results
    avg_metrics_distributed(net);
    print_loss(net, nbpp, false);
    mpi_id0(printf("\n"));
    mpi_id0(fflush(stdout));
    
    // if (early_stopping_on_loss_var (net, 0, 10, 0.1, i)) break;
    //if (early_stopping_on_metric_var (net, 0, 0.0001, 2, i)) break;
    //if (early_stopping_on_metric (net, 0, 0.97, 2, i)) break;
}


int main(int argc, char **argv) {
    int id;
    char model_name[64] = "vgg16_bn";
    char pdf_name[128];
    char onnx_name[128];
    char test_file[128];
    
    char path[256] = "covid1";
    char tr_images[256];
    char tr_labels[256];
    char ts_images[256];
    char ts_labels[256];

    int epochs = 32;
    int batch_size = 100;
    int num_classes = 4;
    int channels = 1;
    int width = 256;
    int height = 256;
    float lr = 0.001;
    int method=FIXED;
    int initial_mpi_avg = 1;
    int chunks = 1;
    int use_bi8 = 0;
    int use_distr_dataset = 0;
    int ptmodel=1;
    bool use_cpu=false;
    int use_mpi=0;
    
    double secs;
    
    //init_distributed();

    sprintf(pdf_name, "%s.pdf", model_name);
    sprintf(onnx_name, "%s.onnx", model_name);
  
    process_arguments(argc, argv,
            path, tr_images, tr_labels, ts_images, ts_labels,
            &epochs, &batch_size, &num_classes, &channels, &width, &height, &lr,
            &method, &initial_mpi_avg,
            &chunks, &use_bi8, &use_distr_dataset, &ptmodel, test_file,
            &use_cpu, &use_mpi);
    
    // Init distribuited training
    //id = get_id_distributed();
               
    // Sync every batch, change every 2 epochs
        set_method_distributed(method,initial_mpi_avg,2);


    // network
    auto in_shape = {channels, height, width};
    /*
    layer in = Input({channels, width, height});
    //layer in = Input({3, 224, 224});
    layer l = in;


    // Data augmentation
    //l = RandomCropScale(l, {0.8f, 1.0f});
    //l = RandomFlip(l,1);

    l=MaxPool(Block3_2(l,64));
    l=MaxPool(Block3_2(l,128));
    l=MaxPool(Block1(Block3_2(l,256),256));
    l=MaxPool(Block1(Block3_2(l,512),512));
    l=MaxPool(Block1(Block3_2(l,512),512));

    l=Reshape(l,{-1});
    l = Activation(Dense(l, 4096), "relu");
    l = Activation(Dense(l, 4096), "relu");

    layer out = Softmax(Dense(l, num_classes));
    // net define input and output layers list
    model net = Model({in},{out});
    */
    //model net = vgg16bn(in_shape, num_classes, 512, 512);
    //model net = resnet50(in_shape, num_classes, 512, 512);
    model net = select_model(ptmodel,in_shape, num_classes, 512, 512);

    /*
    model pretrained_model = download_vgg16_bn(true,{channels, width, height}); // With new input shape  
    summary(pretrained_model);
    // Get the input layer of the pretrained model
    layer in_ = getLayer(pretrained_model, "input");
    // Get the last layer of the pretrained model
    layer top_layer = getLayer(pretrained_model, "top");
    //layer top_layer=nullptr;
    // Create the new densely connected part
    // std::vector<std::string> layers2init{"dense1", "dense_out"};
    //const int input_units = top_layer->output->shape[1];
    //eddl::layer l = eddl::Dense(top_layer, input_units / 2, true, layers2init[0]);
    layer l = nullptr;
    l = Reshape(top_layer,{-1});
    l = Activation(Dense(l, 512, true), "relu");
    l = Activation(Dense(l, 512, true), "relu");
    // l = eddl::Dropout(l, 0.4);
    // Output layer
    layer out_ = Softmax(Dense(l, num_classes, true, "dense_out"));

    model net = Model({in_},{out_});
    */
    
    
    compserv cs = nullptr;
    if (use_cpu) {
        cs = CS_CPU();
    } else {
        cs = CS_GPU(); // one GPU
    }

    // Build model
    build(net,
//            adam(0.0001), // Optimizer
            //adam(lr), // Optimizer
            sgd(lr), // Optimizer
    {"softmax_cross_entropy"}, // Losses
    {"categorical_accuracy"}, // Metrics
    cs);

    setlogfile(net, model_name);

    // plot the model
    if (id == 0)
        plot(net, pdf_name);

    // get some info from the network
    if (id == 0)
        summary(net);

    // Later, we fill the training dataset
    Tensor* x_train;
    Tensor* y_train;


    // Load val dataset
    Tensor* x_test = Tensor::load(ts_images);
    Tensor* y_test = Tensor::load(ts_labels);
    x_test->div_(255.0f);


    for (int i = 0; i < epochs; i++) {
        mpi_id0(printf("== Epoch %d/%d ===\n", i + 1, epochs));
        for (int chunk = 0; chunk < chunks; chunk++) {
            mpi_id0(printf("-- Chunk %d/%d ---\n", chunk + 1, chunks));
            if (use_distr_dataset) { /* Distribute dataset into processes */
                sprintf(tr_images, "%s/%03d/train-images.bi8", path, id*chunks+chunk);
                sprintf(tr_labels, "%s/%03d/train-labels.bi8", path, id*chunks+chunk);
            } else {
                 if (chunks == 1) {
                    sprintf(tr_images, "%s/train-images.bi8", path);
                    sprintf(tr_labels, "%s/train-labels.bi8", path);
                } else {
                    sprintf(tr_images, "%s/%03d/train-images.bi8", path, chunk);
                    sprintf(tr_labels, "%s/%03d/train-labels.bi8", path, chunk);
                }
            }
            if (id == 0) {
                printf("Train: %s, %s\n ", tr_images, tr_labels);
                printf("Val: %s, %s\n", ts_images, ts_labels);
            }
            /* Load dataset */
            x_train = Tensor::load(tr_images);
            y_train = Tensor::load(tr_labels);
            x_train->div_(255.0f);
            // Train
            // printf("FIT:\n");
            //fit(net,{x_train},{y_train}, batch_size, epochs);
            
            //mpi_id0(printf("CUSTOM FIT Mul :\n"));        
            //custom_fit(net,{x_train},{y_train},batch_size, epochs, false, use_distr_dataset);
            mpi_id0(printf("CUSTOM FIT Div:\n"));
            custom_fit(net,{x_train},{y_train}, batch_size, 1, DIV_BATCH, use_distr_dataset);

            delete x_train;
            delete y_train;
        }
    }
    
    /*
    else {
        /// Load chunks 
        for (int i = 0; i < epochs; i++) {
            for (int chunk = 0; chunk < chunks; chunk++) {
                //int selected= 1+(rand() % 3);
                int selected = chunk;
                printf("Chunk %d\n", chunk);
                if (use_distr_dataset) {
                    sprintf(tr_images, "%s/%03d/train-images.bi8", path, chunk);
                    sprintf(tr_labels, "%s/%03d/train-labels.bi8", path, chunk);
                } else {
                    sprintf(tr_images, "%s/%03d/train-images.bi8", path, chunk);
                    sprintf(tr_labels, "%s/%03d/train-labels.bi8", path, chunk);
                }
                printf("%s\n", tr_images);
                printf("%s\n", tr_labels);
                x_train = Tensor::load(tr_images);
                y_train = Tensor::load(tr_labels);
                x_train->div_(255.0f);
                if (id == 0)
                    printf("Epoch: %d; chunk: %d\n", i, selected);
                // training, list of input and output tensors, batch, epochs
                fit(net,{x_train},{y_train}, batch_size, 1);
                printf("Free\n");

                delete x_train;
                delete y_train;
            }
        }
    }*/
    //std::cout << "Evaluate test:" << std::endl;
    // Evaluate
    //barrier_distributed();
    mpi_id0(printf("EVALUATE:\n"));
    evaluate(net,{x_test},{y_test});
    barrier_distributed();
    mpi_id0(printf("DISTR EVALUATE:\n"));
    evaluate_distr(net,{x_test},{y_test});
    barrier_distributed();
    mpi_id0(printf("CUSTOM EVALUATE:\n"));
    custom_evaluate(net,{x_test},{y_test}, batch_size, DIV_BATCH, false);
    
    if (id==0)
        save_net_to_onnx_file (net,"vgg16.onnx");   

   //delete x_train;
    //delete y_train;
    delete x_test;
    delete y_test;
    delete net;

    //end_distributed();

    return EXIT_SUCCESS;
}


///////////
