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

layer Normalization(layer l) {
    return BatchNormalization(l);
    //return LayerNormalization(l);
    //return GroupNormalization(l,8);
}

layer Block1(layer l, int filters) {
    return ReLu(Normalization(Conv(l, filters,{1, 1},
    {
        1, 1
    }, "same", false)));
}

layer Block3_2(layer l, int filters) {
    l = ReLu(Normalization(Conv(l, filters,{3, 3},
    {
        1, 1
    }, "same", false)));
    l = ReLu(Normalization(Conv(l, filters,{3, 3},
    {
        1, 1
    }, "same", false)));
    return l;
}

void custom_fit(model net, Tensor* x_train, Tensor* y_train, int batch, int epochs, bool divide_batch, bool distr_dataset = false) {

    int i, j;
    int id = get_id_distributed();
    int n_procs = get_n_procs_distributed();

    tshape s = x_train->getShape();
    int dataset_size = s[0];
    int channels = s[1];
    int width = s[2];
    int height = s[3];
    int num_classes = y_train->getShape()[1];

    int num_batches;
    int local_batch;
    int global_batch;
    int nbpp;


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
    nbpp = set_NBPP_distributed(dataset_size, local_batch, distr_dataset);

    Tensor* xbatch = new Tensor({local_batch, channels, width, height});
    Tensor* ybatch = new Tensor({local_batch, num_classes});


    for (i = 0; i < epochs; i++) {
        reset_loss(net);
        mpi_id0(printf("custom_fit. Epoch %d/%d\n", i + 1, epochs));
        for (j = 0; j < nbpp; j++) {

            next_batch({x_train, y_train},
            {
                xbatch, ybatch
            });

            train_batch(net,{xbatch},
            {
                ybatch
            });
            //sync_batch
            avg_weights_distributed(net, j + 1, nbpp);
            print_loss(net, j + 1, false);
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

void custom_evaluate(model net, Tensor* x_test, Tensor* y_test, int batch, bool divide_batch, bool distr_dataset = false) {
    int i, j, k;
    int id = get_id_distributed();
    int n_procs = get_n_procs_distributed();

    tshape s = x_test->getShape();
    int dataset_size = s[0];
    int channels = s[1];
    int width = s[2];
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
        eval_batch(net,{x_test},
        {
            y_test
        }, sind);
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
    int ptmodel = 1;
    bool use_cpu=false;
    bool use_mpi=false;
    
    init_distributed();

    sprintf(pdf_name, "%s.pdf", model_name);
    sprintf(onnx_name, "%s.onnx", model_name);

    process_arguments(argc, argv,
            path, tr_images, tr_labels, ts_images, ts_labels,
            &epochs, &batch_size, &num_classes, &channels, &width, &height, &lr,
            &method, &initial_mpi_avg,
            &chunks, &use_bi8, &use_distr_dataset, &ptmodel, test_file,
            &use_cpu, &use_mpi);


    // Init distribuited training
    id = get_id_distributed();

    // Sync every batch, change every 2 epochs
     set_method_distributed(method,initial_mpi_avg,2);


    // network
    layer in = Input({channels, width, height});
    //layer in = Input({3, 224, 224});
    layer l = in;


    // Data augmentation
    //l = RandomCropScale(l, {0.8f, 1.0f});
    //l = RandomFlip(l,1);

    l = MaxPool(Block3_2(l, 64));
    l = MaxPool(Block3_2(l, 128));
    l = MaxPool(Block1(Block3_2(l, 256), 256));
    l = MaxPool(Block1(Block3_2(l, 512), 512));
    l = MaxPool(Block1(Block3_2(l, 512), 512));

    l = Reshape(l,{-1});
    l = Activation(Dense(l, 4096), "relu");
    l = Activation(Dense(l, 4096), "relu");

    layer out = Softmax(Dense(l, num_classes));
    // net define input and output layers list
    model net = Model({in},
    {
        out
    });

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
    {
        "categorical_accuracy"
    }, // Metrics
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
                sprintf(tr_images, "%s/%03d/train-images.bi8", path, id * chunks + chunk);
                sprintf(tr_labels, "%s/%03d/train-labels.bi8", path, id * chunks + chunk);
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
            custom_fit(net,{x_train},
            {
                y_train
            }, batch_size, 1, DIV_BATCH, use_distr_dataset);

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
    barrier_distributed();
    printf("EVALUATE:\n");
    evaluate(net,{x_test},
    {
        y_test
    });
    barrier_distributed();
    printf("DISTR EVALUATE:\n");
    evaluate_distr(net,{x_test},
    {
        y_test
    });
    barrier_distributed();
    printf("CUSTOM EVALUATE:\n");
    custom_evaluate(net,{x_test},
    {
        y_test
    }, batch_size, DIV_BATCH, use_distr_dataset);

    if (id == 0)
        save_net_to_onnx_file(net, "vgg16.onnx");

    //delete x_train;
    //delete y_train;
    delete x_test;
    delete y_test;
    delete net;

    end_distributed();

    return EXIT_SUCCESS;
}


///////////
