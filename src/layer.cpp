#include <stdio.h>      /* printf, scanf, NULL */
#include <stdlib.h>     /* malloc, free, rand */
#include <iostream>

#include "layer.h"
#include "utils.h"

#ifdef MKL 
#define EIGEN_USE_MKL_ALL
#endif
#include "Eigen/Dense"

#define VERBOSE_PARAM 1

using namespace Eigen;
using namespace std;

////////////////////////////////////
///// BASE LAYER CLASS
////////////////////////////////////

Layer::Layer()
{

}

Layer::Layer(int batch,char *name)
{

  strcpy(this->name,name);
  this->batch=batch;

  din=0;
  lin=0;
  lout=0;
  out=0;
  lambda=1.0;
  mu=0.0001;
  //MOMENTUM
  mmu=0.9;
  //NO_DROP
  drop=0.0;
  //NO_REG
  l2=l1=maxn=0.0;

  trmode=1;
  type=0;

  //ReLU
  act=1;
  optim=1; // SGD+MOM

  // GUASSIAN NOISE
  noiser=noisesd=0;
  //Binary noise
  noiseb=0.0;

  dev_done=0;
  reshape=0;

  //BATCH NORM
  bn=0;

  init=0;

  // IMAGE TRANSFORMS
  shift=flip=0;
  brightness=contrast=0;

  D=NULL;
  L=NULL;

  Lin=(Layer **)malloc(MAX_CONNECT*sizeof(Layer *));
  Lout=(Layer **)malloc(MAX_CONNECT*sizeof(Layer *));


}

void Layer::setbrightness(double f){ if (VERBOSE_PARAM) fprintf(stderr,"Layer %s setting brightness %f\n",name,f);brightness=f;}

void Layer::setcontrast(double f){ if (VERBOSE_PARAM) fprintf(stderr,"Layer %s setting contrast %f\n",name,f);contrast=f;}

void Layer::setflip(int f){if (VERBOSE_PARAM) fprintf(stderr,"Layer %s setting flip %d\n",name,f);flip=f;}
void Layer::setshift(int f){ if (VERBOSE_PARAM) fprintf(stderr,"Layer %s setting shift %d\n",name,f);shift=f;}
void Layer::setmu(double m){if (VERBOSE_PARAM) fprintf(stderr,"Layer %s setting mu %f\n",name,m);mu=m;resetmomentum();}
void Layer::setmmu(double m){ if (VERBOSE_PARAM) fprintf(stderr,"Layer %s setting mmu %f\n",name,m);mmu=m;}
void Layer::setdrop(double m){ if (VERBOSE_PARAM) fprintf(stderr,"Layer %s setting drop %f\n",name,m);drop=m;}
void Layer::setl2(double m){  if (VERBOSE_PARAM) fprintf(stderr,"Layer %s setting l2 %f\n",name,m);l2=m;}
void Layer::setl1(double m){if (VERBOSE_PARAM) fprintf(stderr,"Layer %s setting l1 %f\n",name,m);l1=m;}
void Layer::setmaxn(double m){ if (VERBOSE_PARAM) fprintf(stderr,"Layer %s setting maxn %f\n",name,m);maxn=m;}
void Layer::setlambda(double l){if (VERBOSE_PARAM) fprintf(stderr,"Layer %s setting lambda %f\n",name,l);lambda=l;}
void Layer::trainmode(){trmode=1;}
void Layer::testmode(){trmode=0;}
void Layer::setact(int i){if (VERBOSE_PARAM) fprintf(stderr,"Layer %s setact to %d\n",name,i);act=i;}
void Layer::setbn(int i){ if (VERBOSE_PARAM) fprintf(stderr,"Layer %s setting BN %d\n",name,i);bn=i;}
void Layer::setnoiser(double m){if (VERBOSE_PARAM) fprintf(stderr,"Layer %s setting noiser %f\n",name,m);noiser=m;}
void Layer::setnoisesd(double m){if (VERBOSE_PARAM) fprintf(stderr,"Layer %s setting noisesd %f\n",name,m);noisesd=m;}
void Layer::setnoiseb(double m){if (VERBOSE_PARAM) fprintf(stderr,"Layer %s setting binary noise %f\n",name,m);noiseb=m;}
void Layer::setthreads(int t){threads=t;}
void Layer::setoptim(int t){ if (VERBOSE_PARAM) fprintf(stderr,"Set optimization method layer %s to %d\n",name,t);optim=t;}


void Layer::save_param(FILE *fe)
{
  int i;


  fprintf(fe,"%s\n",name);
  fprintf(fe,"%d\n",din);
  fprintf(fe,"%d\n",lin);
  fprintf(fe,"%d\n",lout);

  fprintf(fe,"%d\n",act);
  fprintf(fe,"%d\n",bn);
  fprintf(fe,"%d\n",shift);
  fprintf(fe,"%d\n",flip);

  fprintf(fe,"%f\n",mu);
  fprintf(fe,"%f\n",mmu);
  fprintf(fe,"%f\n",drop);
  fprintf(fe,"%f\n",l2);
  fprintf(fe,"%f\n",l1);
  fprintf(fe,"%f\n",maxn);
  fprintf(fe,"%f\n",brightness);
  fprintf(fe,"%f\n",contrast);

  // FOR FUTURE PARAMS
  for(i=0;i<10;i++)
    fprintf(fe,"-1\n");
  
  fflush(fe);


}
void Layer::load_param(FILE *fe)
{
  char lname[100];
  int ldin,llin,llout;
  int fsd,i;


  fsd=fscanf(fe,"%s\n",lname);
  if (strcmp(name,lname)) {
    fprintf(stderr,"Error loading layer  %s (name) %s -- %s\n",name,name,lname);
    exit(1);
  }
  fsd=fscanf(fe,"%d\n",&ldin);

  /*if (ldin!=din) {
    fprintf(stderr,"Error loading layer  %s (din) %d -- %d\n",name,din,ldin);
    exit(1);
    }*/

  fsd=fscanf(fe,"%d\n",&llin);
  if (llin!=lin) {
    fprintf(stderr,"Error loading layer  %s (lin) %d -- %d\n",name,lin,llin);
    exit(1);
  }
  fsd=fscanf(fe,"%d\n",&llout);
  if (llout!=lout) {
    fprintf(stderr,"Error loading layer  %s (lout) %d -- %d\n",name,lout,llout);
    exit(1);
  }

  fsd=fscanf(fe,"%d\n",&act);
  fsd=fscanf(fe,"%d\n",&bn);
  fsd=fscanf(fe,"%d\n",&shift);
  fsd=fscanf(fe,"%d\n",&flip);
  fsd=fscanf(fe,"%lf\n",&mu);
  fsd=fscanf(fe,"%lf\n",&mmu);
  fsd=fscanf(fe,"%lf\n",&drop);
  fsd=fscanf(fe,"%lf\n",&l2);
  fsd=fscanf(fe,"%lf\n",&l1);
  fsd=fscanf(fe,"%lf\n",&maxn);
  fsd=fscanf(fe,"%lf\n",&brightness);
  fsd=fscanf(fe,"%lf\n",&contrast);

  // FOR FUTURE PARAMS
  for(i=0;i<10;i++)
    fsd=fscanf(fe,"-1\n");


}
