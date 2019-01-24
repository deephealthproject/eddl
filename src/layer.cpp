#include <stdio.h>
#include <stdlib.h>

#include "layer.h"

using namespace std;

////////////////////////////////////
///// BASE LAYER CLASS
////////////////////////////////////
Layer::Layer()
{
  mode=TRMODE;
  input=output=NULL;
  lin=lout=0;
  parent=(Layer **)malloc(MAX_CONNECT*sizeof(Layer*));
  child=(Layer **)malloc(MAX_CONNECT*sizeof(Layer*));

}
Layer::Layer(string n):Layer(){name=n;}












//////
