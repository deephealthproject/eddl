#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "layer.h"

using namespace std;


Dense::Dense(Layer *parent,int dim):Dense(parent,dim,std::string("dense")){}

Dense::Dense(Layer *parent,int dim,string name):Layer(name)
{
  if (parent->output->dim!=2) msg("Dense only over 2D tensors");

  input=parent->output;
  output=new Tensor({input->sizes[0],dim});

  W=new Tensor({input->sizes[1],dim});
  gW=new Tensor({input->sizes[1],dim});
  D=new Tensor({input->sizes[1],dim});

}
// virtual
void Dense::info()
{
  cout<<"\n===============\n";
  cout<< "Layer Dense "<<name<<"\n";
  cout<<"Input:\n";
  input->info();
  cout<<"Output:\n";
  output->info();
  cout<<"Param:\n";
  W->info();
  cout<<"===============\n\n";
}

void Dense::addchild(Layer *l)
{
  child[lout]=l;
  lout++;
}
void Dense::addparent(Layer *l)
{
    if (lin>0) msg("Dense only can have one parent layer");

    parent[lin]=l;
    lin++;
}
void Dense::forward(){}
void Dense::backward(){}
void Dense::initialize(){}
void Dense::applygrads(){}
void Dense::reset(){}
