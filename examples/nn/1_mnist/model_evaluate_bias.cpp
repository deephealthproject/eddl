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
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include "eddl/apis/eddl.h"
#include "eddl/layers/core/layer_core.h"
#include "eddl/net/net.h"
#include "eddl/random.h"
#include "eddl/system_info.h"
#include "eddl/utils.h"
#include <numeric>      // std::accumulate
#include <bitset>
#include <limits.h> /* for CHAR_BIT */
#include <ctime> 
#include <fstream>
#include<tuple> // for tuple
#include "eddl/serialization/onnx/eddl_onnx.h" // Not allowed


using namespace eddl;

/** formatted output of ieee-754 representation of float */
void show_ieee754 (float f)
{
    union {
        float f;
        uint32_t u;
    } fu = { .f = f };
    int i = sizeof f * CHAR_BIT;

    printf ("  ");
    while (i--)
        printf ("%d ", (fu.u >> i) & 0x1);

    putchar ('\n');
    printf (" |- - - - - - - - - - - - - - - - - - - - - - "
            "- - - - - - - - - -|\n");
    printf (" |s|      exp      |                  mantissa"
            "                   |\n\n");
}

union fp_bit_twiddler {
    float f;
    int i;
} q;

layer Normalization(layer l)
{
	return l;
}



//////////////////////////////////
// mnist_mlp.cpp:
// A very basic CNN for mnist
// Using fit for training
//////////////////////////////////

int main(int argc, char **argv) {

    ofstream file;
    file.open(argv[2], fstream::app);


    bool testing = false;
    bool use_cpu = false;
    bool use_fails = false;

    srand(time(NULL)); 


    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--testing") == 0) testing = true;
        else if (strcmp(argv[i], "--cpu") == 0) use_cpu = true;
        else if (strcmp(argv[i], "--fails") == 0) use_fails = true;
    }


    // Settings
    int batch_size = 1;
    int num_classes = 10;

    vector<float> total_metric;
    Metric* m = getMetric("categorical_accuracy");
    float sum = 0., ca = 0.; 
    int dev;

    // Define network
    // Define network
    layer in, out;
    model net;
    //layer in = Input({784});
    //layer l = in;  // Aux var

    //l = Reshape(l,{1,28,28});
    //l = MaxPool2D(ReLu(Conv2D(l,32, {3,3},{1,1})),{3,3}, {1,1}, "same");
    //l = MaxPool2D(ReLu(Conv2D(l,64, {3,3},{1,1})),{2,2}, {2,2}, "same");
    //l = MaxPool2D(ReLu(Conv2D(l,128,{3,3},{1,1})),{3,3}, {2,2}, "none");
    //l = MaxPool2D(ReLu(Conv2D(l,256,{3,3},{1,1})),{2,2}, {2,2}, "none");
    //l = Reshape(l,{-1});


    //layer out = Softmax(Dense(l, num_classes));
    //model net = Model({in}, {out});


    //net->verbosity_level = 0;


    // network cifar10 conv
    //layer in=Input({3,32,32});
    //layer l=in;


    //l=MaxPool(ReLu(Normalization(Conv(l,32,{3,3},{1,1}))),{2,2});
    //l=MaxPool(ReLu(Normalization(Conv(l,64,{3,3},{1,1}))),{2,2});
    //l=MaxPool(ReLu(Normalization(Conv(l,128,{3,3},{1,1}))),{2,2});
    //l=MaxPool(ReLu(Normalization(Conv(l,256,{3,3},{1,1}))),{2,2});

    //l=GlobalMaxPool(l);


    //l=Flatten(l);

    //l=Activation(Dense(l,128),"relu");


    //layer out = Softmax(Dense(l, num_classes));
    //model net = Model({in}, {out});


    //net->verbosity_level = 0;

    // dot from graphviz should be installed:
    //plot(net, "model.pdf");

    compserv cs = nullptr;
    if (use_cpu) {
        cs = CS_CPU();
	dev = DEV_CPU;
    } else {
        cs = CS_GPU({1}, "low_mem"); // one GPU
	dev = DEV_GPU;
        // cs = CS_GPU({1,1},100); // two GPU with weight sync every 100 batches
        // cs = CS_CPU();
        // cs = CS_FPGA({1});
    }




    // Load weights
    if(testing){
        load(net, "saved-weights.bin");
    }else{
        //load(net, "saved-cifar-full.bin");
  	 net = import_net_from_onnx_file(argv[3]);
	 out = net->lout[0];
    }

    // Build model, fer un programa que escriga i altre que llisca, en este caso, creo dos modelos nuevos, que se inicializan con pesos diferentes, estaria mal. Hay que usar el mismo modelo
    build(net,
          adam(0.001), // Optimizer
          {"softmax_cross_entropy"}, // Losses
          {"categorical_accuracy"}, // Metrics
          cs, false);

    // View model
    //summary(net);


    // Load dataset (cargar imagenes mnnist)
    Tensor* x_test = Tensor::load("cifar_tsX.bin");
    Tensor* y_test = Tensor::load("cifar_tsY.bin");
    Tensor* output = new Tensor();
    Tensor* target = new Tensor();
    Tensor* result = new Tensor();
    Tensor *batch = Tensor::empty({batch_size, 3, 32, 32});

    if (testing) {
        std::string _range_ = "0:" + std::to_string(2 * batch_size);

        Tensor* x_mini_test  = x_test->select({_range_, ":"});
        Tensor* y_mini_test  = y_test->select({_range_, ":"});

        delete x_test;
        delete y_test;

        x_test  = x_mini_test;
        y_test  = y_mini_test;

    }

    // Preprocessing
    x_test->div_(255.0f);
    int random_param, num_fails, random_bit, random_capa;
    typedef vector< tuple<int, int, int> > my_tuple;
    my_tuple capas;
    int total_params;
    float num_antes, num_despues = 0.0;
	int parametro;
    //--------------- INSERCIÓN FALLOS ------------------------------------------------------------------//
    if(use_fails){
        //printf("----------------------------- INSERCIÓN FALLOS:------------------------------\n");
        // Train model

        total_params = 0;
        int params_capa = 0;
        //printf("Nº capas modelo %d\n", net->layers.size());
        for(int capa = 0; capa < net->layers.size(); capa++){
            if(net->layers[capa]->params.size() > 0){
               for(int p = 0; p < net->layers[capa]->params.size(); p++){
                    //cout << "Número capa " << capa << "(" <<  net->layers[capa]->name << ")";

                    //printf(" parámetro num %d con size: %d, numero params %d\n", p, net->layers[capa]->params[p]->size, net->layers[capa]->params.size());
                    params_capa += net->layers[capa]->params[p]->size;
                    if(p == net->layers[capa]->params.size() - 1){
                        capas.push_back(tuple<int, int, int>(capa, net->layers[capa]->params.size(), params_capa));
                    }
                    //for(int s = 0; s < net->layers[capa]->params[p]->shape.size(); s++){
                        //printf("%d ", net->layers[capa]->params[p]->shape[s]);
                    //}
               }
            }   
        }
        
        total_params = params_capa;

	    vector<vtensor> net_params;
	    net_params = get_parameters(net,true); //todos los parametros de todas las capas


	    for(int i = 0; i < atoi(argv[1]); i++) {
            random_param = (rand() % total_params);
            int capa_elem, num_params_capa, cont = 0;
            for (my_tuple::const_iterator i = capas.begin(); i != capas.end(); ++i) {
                //printf("Capa num %d, Num params %d, Params capa %d\n", get<0>(*i), get<1>(*i), get<2>(*i));
                if(get<2>(*i) < random_param) {
                    random_capa = get<0>(*(i+1));
                    capa_elem = random_param - get<2>(*(i)) ;
                    num_params_capa = get<1>(*(i+1));
                }else{
                if(cont==0){
                        random_capa = get<0>(*(i+1));
                        capa_elem = random_param - get<2>(*(i)) ;
                        num_params_capa = get<1>(*(i+1));		
                }		
                }
                cont++;
            }
            /*
		    printf("total_params %d, random_param %d, random_capa %d, capa_elem %d, num_params_capa %d, \n", total_params, random_param, random_capa, capa_elem, num_params_capa);

	
		
			for(int param = 0; param < num_params_capa; param++){
				
				if(net_params[random_capa][param]->size > capa_elem){
					random_bit = (rand() % 32);
					float before = net_params[random_capa][param]->ptr[capa_elem];
					printf("pos: %d, bit %d, numero antes %f , mask %f\n\n", capa_elem, random_bit, before, pow (2.0, random_bit));

					//show_ieee754 (before);
					num_antes = before;
					q.f = before;
					q.i ^= (1 << random_bit);
					net_params[random_capa][param]->ptr[capa_elem] = q.f;
					//show_ieee754 (net_params[random_capa][param]->ptr[capa_elem]);
					printf("\n numero despues %f\n", net_params[random_capa][param]->ptr[capa_elem]);
					parametro = param;
					num_despues = net_params[random_capa][param]->ptr[capa_elem];
					param = num_params_capa;
				}else{
					capa_elem = capa_elem - net_params[random_capa][param]->size;
				}
				
			}

			file << random_capa << " " << parametro << " " << random_param << " " << random_bit << " " << num_antes << " " << num_despues <<" ";
	        }
            */
            //bias test
            capa_elem = (rand() % net_params[random_capa][1]->size);
            random_bit = (rand() % 32);
            float before = net_params[random_capa][1]->ptr[capa_elem];
            printf("pos: %d, bit %d, numero antes %f , mask %f\n\n", capa_elem, random_bit, before, pow (2.0, random_bit));

            //show_ieee754 (before);
            num_antes = before;
            q.f = before;
            q.i ^= (1 << random_bit);
            net_params[random_capa][1]->ptr[capa_elem] = q.f;
            //show_ieee754 (net_params[random_capa][param]->ptr[capa_elem]);
            printf("\n numero despues %f\n", net_params[random_capa][1]->ptr[capa_elem]);
            parametro = 1;
            num_despues = net_params[random_capa][1]->ptr[capa_elem];
            random_param=capa_elem;

            //foult injection en la conv2d1-----------------------------------
            //inyectamos fallos despues de realizar el entrenamiento, donde los pesos de cada capa se han adaptado para obtener un resultado optimo.
            //ahora, cambiamos esos pesos para ver la repercusion.

            //num_fails = (rand() % net_params[capas[random_capa]][random_param]->size) + 1; //atoi(argv[1]);
            //if(num_fails > net_params[capas[random_capa]][random_param]->size) num_fails = net_params[capas[random_capa]][random_param]->size;


            //printf("Num Fallos: %d\n", num_fails);
            //net_params[capas[random_capa]][random_param]->info();//cojo el parametro 0 de la capa 2 (la conv)
            //net_params[capas[random_capa]][0]->print();


            //for(int i = 0; i < num_fails; i++){
                //random_pos = (rand() % net_params[capas[random_capa]][random_param]->size); //puede tocar aleatoriamente la misma pos varias veces, de forma que nunca se cambian todos los valores
                //random_bit = (rand() % 31) + 1;
                //float before = net_params[capas[random_capa]][random_param]->ptr[random_pos];
                //printf("pos: %d, bit %d, numero antes %f , mask %f\n\n", random_pos, random_bit, before, pow (2.0, random_bit));

                //show_ieee754 (before);

                //q.f = before;
                //q.i ^= (1 << random_bit);
                //net_params[capas[random_capa]][random_param]->ptr[random_pos] = q.f;

                //show_ieee754 (net_params[capas[random_capa]][0]->ptr[random_pos]);
                //printf("\n numero despues %f\n", net_params[capas[random_capa]][0]->ptr[random_pos]);
            //}
        }

        set_parameters(net, net_params);
        //net->layers[capas[random_capa]]->params[0]->print();

        //guardar model entrenat en onnx i altre programa que ho llisca
        //agafar i fer una distribucio de errors depenent del tipo de fallo bit flip, o canvi de parametres
    }
    //-------------------------------------------------------------------------------------------------------------//




    //-----------------------------INFERENCIA Y EVALUACION A MANO---------------------------------------------------------//

    int num_batches=x_test->shape[0]/batch_size;
    for(int j=0;j<num_batches;j++)  {

        //cout<<"Batch "<<j;

        next_batch({x_test},{batch});

        forward(net,{batch});

        output = getOutput(out);//resultados de la inferencia
        sum = 0.; 
        for (int j = 0; j < batch_size; ++j) {
            result = output->select({to_string(j)}); //resultado de la inferencia, lo que ha predicho
            target = y_test->select({to_string(j)}); //etiqueta con la que comparar

            ca = m->value(target, result);//categorical accuracy de cada batch (imagen) -> 1 pass, 0 not pass
            //printf("batc %d, ca %f\n", j, ca);
            total_metric.push_back(ca);
            sum += ca;

            delete result;
            delete target;
        }
       // cout << " categorical_accuracy: " << static_cast<float>(sum) / batch_size << endl;

    }

    //show_profile();
    float total_avg = accumulate(total_metric.begin(), total_metric.end(), 0.0f) / total_metric.size();
    cout << "Total categorical accuracy: " << total_avg << endl;

    
    //-----------------------------------------------------------------------------------------------------------//




    //-----------------EVALUACIÓN MEDIANTE MÉTODO EDDL-----------------------------------------------------------//
    evaluate(net, {x_test}, {y_test});
    vector<float> losses1 = get_losses(net);
    vector<float> metrics1 = get_metrics(net);
    for(int i=0; i<losses1.size(); i++) {
        cout << "Loss: " << losses1[i] << "\t" << "Metric: " << metrics1[i] << "   |   ";
    }
    cout << endl;
    if(use_fails){
        file << metrics1[0] << " " << random_capa << " " << parametro << " " << random_param << " " << random_bit << " " << num_antes << " " << num_despues <<"\n";
    }

    file.close();

    delete x_test;
    delete y_test;
    delete net;

    return EXIT_SUCCESS;
}
