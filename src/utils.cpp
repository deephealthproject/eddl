
/////////////////////////////////////////////////////////////////////////////
// This file is part of EDDLL an European Distributed Deep Learning Library.
// Developed within the DeepHealth project.
// Boosting AI in Europe.
//
// Main authors and developers:
//      Roberto Paredes: rparedes@prhlt.upv.es
//      Joan Ander Gómez: jon@prhlt.upv.es
//
//
// Collaborators:
//      Salva Carrión: salcarpo@prhlt.upv.es
//      Mario Parreño: maparla@prhlt.upv.es
//
//
// To collaborate please contact rparedes@prhlt.upv.es
//
/////////////////////////////////////////////////////////////////////////////


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <new>      // included for std::bad_alloc

#include "utils.h"
#include <random>

#define PI 3.1415926
#define MAX_RTABLE 100000

float *RTable=nullptr;
int posTable=0;

// Default seed
std::random_device rd;  //Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()


float uniform(float min, float max) {
    // rand() may not generate numbers uniformly and is therefore discouraged
    // Talk: https://channel9.msdn.com/Events/GoingNative/2013/rand-Considered-Harmful
    std::uniform_real_distribution<float> distr(min, max);
    return distr(gen);
}

float suniform() {
    return (2 * uniform()) - 1;
}

float gaussgen() {
    float x, u1, u2;
    int i;

    u1 = uniform();
    u2 = uniform();

    while (u1 == 0.0) u1 = uniform();

    x = sqrt(log(1 / u1)) * cos(2 * PI * u2);

    return x;
}


float gauss(float mean, float sd) {
    return (gaussgen() * sd) + mean;
}

void gen_rtable()
{
  if (RTable==nullptr) {
    printf("Generating Random Table\n");
    RTable=get_fmem(MAX_RTABLE, "gen_rtable");

    for(int i=0;i<MAX_RTABLE;i++)
      RTable[i]=gaussgen();
  }
}

float gauss(int s, float mean, float sd) {
    posTable=(posTable+s)%MAX_RTABLE;
    return (RTable[posTable] * sd) + mean;
}

float *get_fmem(int size, char *str){
    float* ptr = nullptr;
    try{
        ptr = new float[size];
    }
    catch (std::bad_alloc& badAlloc){
        fprintf(stderr, "Malloc error allocating %lu bytes in %s\n", size*sizeof(float), str);
    }
    return ptr;
}
