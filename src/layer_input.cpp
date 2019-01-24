#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "layer.h"

using namespace std;


Input::Input(Tensor *in,string name):Layer(name){input=output=in;}
Input::Input(Tensor *in):Layer(){name="input";}

// virtual
void Input::info()
{
  cout<<"\n===============\n";
  cout<< "Layer Input "<<name<<"\n";
  input->info();
  cout<<"===============\n\n";
}

void Input::addchild(Layer *l)
{
  child[lout]=l;
  lout++;
}
void Input::addparent(Layer *l)
{
  msg("Input layer can not have parent layers");
}

void Input::forward(){}
void Input::backward(){}
void Input::initialize(){}
void Input::applygrads(){}
void Input::reset(){}








//////
