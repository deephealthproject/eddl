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
  delta=input=output=NULL;
  parent=NULL;
  child=NULL;

}
Layer::Layer(string n):Layer(){name=n;}












//////
