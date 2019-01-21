#include "data.h"

#ifdef MKL 
#define EIGEN_USE_MKL_ALL
#endif

#include "Eigen/Dense"
#include "types.h"
#include "tensor.h"

#define MAX_CONNECT 100
#define MAX_DATA 100

using namespace Eigen;
using namespace std;

class Net;

class Layer {
 public:
  char name[1000];
  int type;

  Tensor *N,*Delta;

  int din;
  int lin;
  int lout;
  int batch;
  int act;
  double mu;
  double mmu;
  double l2;
  double l1;
  double maxn;
  double drop;
  double noiser;
  double noiseb;
  double noisesd;
   
  int optim;
  int trmode;
  int dev_done;
  int threads;
  int opt;
  int reshape;

  int bn;
  int init;
  int out;
  double lambda;

  int shift,flip;
  double brightness,contrast;

  Data *D;
  Layer *L;

  Layer **Lin;
  Layer **Lout;
  Net *rnet;

  Layer();
  Layer(int b,char *name);

  void setflip(int f);
  void setshift(int f);
  void setbrightness(double f);
  void setcontrast(double f);

  void setmu(double m);
  void setmmu(double m);
  void setdrop(double m);
  void setl2(double m);
  void setl1(double m);
  void setmaxn(double m);
  void trainmode();
  void testmode();
  void setact(int i);
  void setbn(int a);
  void setnoiser(double n);
  void setnoisesd(double n);
  void setnoiseb(double n);
  void setlambda(double l);
  void setthreads(int t);
  void setoptim(int i);

  void save_param(FILE *fe);
  void load_param(FILE *fe);


  virtual void save(FILE *fe){}
  virtual void load(FILE *fe){}
  virtual void printkernels(FILE *fe){}
  virtual void addchild(Layer *l){}
  virtual void shared(Layer *l){}
  virtual void addparent(Layer *l){}
  virtual void forward(){}
  virtual void backward(){}
  virtual void initialize(){}
  virtual void applygrads(){}
  virtual void reset(){}
  virtual void resetmomentum(){}
  virtual void resetstats(){}
  virtual void getbatch(){}
  virtual void next(){}
  
  };


class FLayer : public Layer {
 public:

  FLayer();
  FLayer(int in,int batch,char *name);
  FLayer(int batch,char *name);
  FLayer(Layer *In,int batch,char *name);
  
  // Params
  Tensor *W,*gW,*pgW,*b,*gb,*pgb,*dvec;
  // Activations and deltas
  Tensor *E,*T,*dE;
  //Batch norm
  int bnc;
  Tensor *bn_mean,*bn_var,*bn_gmean,*bn_gvar,*bn_g,*bn_b,*bn_E,*BNE;  
  Tensor *gbn_mean,*gbn_var,*gbn_g,*gbn_b,*gbn_E;


  void mem();
  void addchild(Layer *l);
  void shared(Layer *l);
  void addparent(Layer *l);
  void forward();
  void fBN();
  void backward();
  void bBN();
  void initialize();
  void applygrads();
  void reset();
  void resetstats();
  void resetmomentum();

  void save(FILE *fe);
  void load(FILE *fe);

  void printkernels(FILE *fe);

};

class IFLayer : public FLayer {
 public:
  int dc;
  int itype[MAX_DATA];

  IFLayer(Data *D,FLayer *L,int b,char *name);

  void getbatch();
  void next();
  void addparent(Layer *l);
  void backward();
  void setsource(Data *newd);
  void setsource(FLayer *newl);

};


class OFLayer : public FLayer {
 public:
  
  double landa;  
  double rmse,mse,mae,cerr,ent,loss;

  OFLayer(Data *D,FLayer *T,int b,int opt,char *name);
  void settarget(Data *Dt);
  void settarget(FLayer *Lt);
  void backward();
  double get_err(int n);
};


/////////////////////////////////

class OLayer : public Layer {
 public:
  Tensor *dE;
  OLayer();
  OLayer(int batch,int op,char *name);
  void addchild(Layer *l);
  void shared(Layer *l);
  void addparent(Layer *l);
  void forward();
  void backward();
  void save(FILE *fe);
  void load(FILE *fe);
  void initialize();
  void applygrads();
  void reset();
  int op;

};

////////////////////
class CLayer : public Layer {
 public:

  int nk,kr,kc,kz;
  int outz,outr,outc;
  int stride;
  int zpad;
  int rpad,cpad;

  Tensor *K;
  Tensor *gK;
  Tensor *pgK;
  Tensor *bias;
  Tensor *gbias;

  Tensor *E;
  Tensor *BNE;
  Tensor *Dvec;
  Tensor *dE;
    
  // FOR BN
  Tensor *bn_mean;
  Tensor *bn_gmean;
  Tensor *bn_var;
  Tensor *bn_gvar;
  Tensor *bn_g;
  Tensor *bn_b;
  Tensor *bn_E;
  int bnc;

  Tensor *gbn_mean;
  Tensor *gbn_var;
  Tensor *gbn_g;
  Tensor *gbn_b;
  Tensor *gbn_E;

  CLayer();
  CLayer(int batch,char *name);
  CLayer(int nk,int kr, int kc,int batch,int rpad,int cpad,int stride,char *name);



  void addchild(Layer *l);
  void shared(Layer *l);
  void addparent(Layer *l);
  void forward();
  void backward();
  void initialize();
  void applygrads();
  void reset();
  void resetstats();
  void resetmomentum();

  void fBN();
  void bBN();

  void Convol();
  void ConvolOMP();
  void MaxPool();
  void ConvolB();
  void MaxPoolB();
  void printkernels(FILE *fe);


  void save(FILE *fe);
  void load(FILE *fe);


};

class ICLayer : public CLayer {
 public:

  
  int imr,imc;

  ICLayer(Data *D,Layer *K,int batch,int z,int r,int c,int ir,int ic,char *name);


  void getbatch();
  void addparent(Layer *l);
  void setsource(Data *newd);
  void setsource(FLayer *newl);

  void forward();
  void backward();
  void initialize();
  void applygrads();
  void reset();
  void resetmomentum();
  void save(FILE *fe);
  void load(FILE *fe);
  void next();

  void doflip(LMatrix& I);
  void doshift(LMatrix& I,int sx,int sy);
  void donoise(LMatrix& I,double ratio, double sd);
  void donoiseb(LMatrix& I,double ratio);
  double calc_brightness(LMatrix I,double factor);
  void dobrightness(LMatrix& I,double factor);
  void docontrast(LMatrix& I,double factor);
  void SaveImage(LMatrix R,LMatrix G,LMatrix B,char *name);

};




class PLayer : public CLayer {
 public:

  int sizer;
  int sizec;

  MatrixXi **maxR;
  MatrixXi **maxC;


  void addchild(Layer *l);
  void shared(Layer *l);
  void addparent(Layer *l);
  void forward();
  void backward();
  void initialize();
  void applygrads();
  void reset();



  void MaxPoolB();
  void MaxPool();

  void save(FILE *fe);
  void load(FILE *fe);

  PLayer();
  PLayer(int batch,int sizer,int sizec,char *name);
};



class CatLayer : public CLayer {
 public:
  int cat,catvec[100];

  void addchild(Layer *l);
  void shared(Layer *l);
  void addparent(Layer *l);
  void forward();
  void backward();
  void initialize();
  void applygrads();
  void reset();
  void resetmomentum();

  void save(FILE *fe);
  void load(FILE *fe);

  CatLayer();
  CatLayer(int batch,char *name);
};

class AddLayer : public CLayer {
 public:
  int add,addvec[100];

  void addchild(Layer *l);
  void shared(Layer *l);
  void addparent(Layer *l);
  void forward();
  void backward();
  void initialize();
  void applygrads();
  void reset();
  void resetmomentum();

  void save(FILE *fe);
  void load(FILE *fe);

  AddLayer();
  AddLayer(int batch,char *name);
};
