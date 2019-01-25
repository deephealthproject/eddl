#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "layer.h"

using namespace std;


Dense::Dense(Layer *parent,int dim):Dense(parent,dim,std::string("dense")){}

Dense::Dense(Layer *parent,int dim,string name):Layer(name)
{
  if (parent->output->dim!=2) msg("Dense only works over 2D tensors");

  input=parent->output;
  output=new Tensor({input->sizes[0],dim});
  delta=new Tensor(input->getshape());

  W=new Tensor({input->sizes[1],dim});
  gW=new Tensor({input->sizes[1],dim});
  bias=new Tensor({dim});
  gbias=new Tensor({dim});

  parent->addchild(this);
  addparent(parent);

}
// virtual
void Dense::addchild(Layer *l)
{
  if (child!=NULL) msg("Dense only can have one child layer\nTry with Split Layer");
  child=l;
}
void Dense::addparent(Layer *l)
{
    if (parent!=NULL) msg("Dense only can have one parent layer");
    parent=l;
}
void Dense::initialize()
{
}

void Dense::reset()
{
  //gW->set(0);
  //delta->set(0);
}

void Dense::forward()
{
  Tensor::mult2D(input,0,W,0,output);
  Tensor::sum2D_rowwise(output,bias,output);
}

void Dense::backward()
{
  if (child!=NULL){
    Tensor::mult2D(child->delta,0,W,1,delta);
    Tensor::mult2D(input,1,child->delta,0,gW);
  }
}
void Dense::applygrads(){}

void Dense::info()
{
  cout<<"\n===============\n";
  cout<< "Layer Dense "<<name<<"\n";
  cout<<"Input:\n";
  input->info();
  cout<<"Param:\n";
  W->info();
  cout<<"Output:\n";
  output->info();
  cout<<"===============\n\n";
}
