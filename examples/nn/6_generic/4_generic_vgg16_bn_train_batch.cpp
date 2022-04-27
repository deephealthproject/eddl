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
// cifar_vgg16_bn.cpp:
// vgg16 with BatchNorm
// Using fit for training
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

void custom_fit(model net, Tensor* x_train, Tensor* y_train, int batch, int epochs, bool divide_batch, bool split_dataset=false) {
   
    int i,j;
    int id=get_id_distributed();
    int n_procs=get_n_procs_distributed();
    
    tshape s = x_train->getShape();
    int dataset_size=s[0];
    int channels =s[1];
    int width=s[2];
    int height = s[3];
    int num_classes = y_train->getShape()[1];
    printf("Num classes %d\n", num_classes);
    
    int num_batches;
    int local_batch;
    int global_batch;
    int nbpp;
    
     if (split_dataset){
        mpi_id0(printf("custom_fit. Split dataset\n"));
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
       
  
    Tensor* xbatch = new Tensor({local_batch, channels*width*height});
    Tensor* ybatch = new Tensor({local_batch, num_classes});
 
    
    for(i=0;i<epochs;i++) {
      reset_loss(net);
      mpi_id0(printf("custom_fit. Epoch %d/%d. Batch (param %d, global %d, local %d). Num Batches: %d (of size %d) Per proc: %d (of size %d)\n", i + 1, epochs, batch, global_batch, local_batch, num_batches, global_batch, nbpp, local_batch));
      for(j=0;j<nbpp;j++)  {
          
        next_batch({x_train,y_train},{xbatch,ybatch});
        
        train_batch(net, {xbatch}, {ybatch});
        //sync_batch
        //avg_weights_distributed(net, j, nbpp);  
        print_loss(net,j, false);
        mpi_id0(printf("\r"););
      }
      mpi_id0(printf("\n"));
     // if (early_stopping_on_loss_var (net, 0, 10, 0.1, i)) break;
      //if (early_stopping_on_metric_var (net, 0, 0.0001, 2, i)) break;
      //if (early_stopping_on_metric (net, 0, 0.97, 2, i)) break;
    }
    mpi_id0(printf("\n"));   
    
    delete xbatch;
    delete ybatch;

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

    compserv cs = nullptr;
    if (use_cpu) {
        cs = CS_CPU();
    } else {
        cs = CS_GPU(); // one GPU
    }

    // Build model
    build(net,
//            adam(0.0001), // Optimizer
            adam(lr), // Optimizer
            //sgd(lr), // Optimizer
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
            if (use_distr_dataset) { /* Split dataset into processes */
                sprintf(tr_images, "%s/%03d/train-images.bi8", path, id);
                sprintf(tr_labels, "%s/%03d/train-labels.bi8", path, id);
                if (id == 0) {
                    printf("Train: %s, %s\n ", tr_images, tr_labels);
                    printf("Val: %s, %s\n", ts_images, ts_labels);
                }
            } else {
                if (chunks == 1) {
                    sprintf(tr_images, "%s/train-images.bi8", path);
                    sprintf(tr_labels, "%s/train-labels.bi8", path);
                } else {
                    sprintf(tr_images, "%s/%03d/train-images.bi8", path, chunk);
                    sprintf(tr_labels, "%s/%03d/train-labels.bi8", path, chunk);
                }
                if (id == 0) {
                    printf("Train: %s, %s\n ", tr_images, tr_labels);
                    printf("Val: %s, %s\n", ts_images, ts_labels);
                }
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
            custom_fit(net,{x_train},{y_train}, batch_size, 1, true, use_distr_dataset);

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
    evaluate(net,{x_test},{y_test});
    //evaluate_distr(net,{x_test},{y_test});
    
    
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
