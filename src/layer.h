
#include <string>
#include "tensor.h"

#define MAX_CONNECT 100
#define TRMODE 0
#define TSMODE 1

using namespace std;

class Layer {
 public:
  string name;
  Tensor *input;
  Tensor *output;
  Tensor *delta;

  Layer *parent;
  Layer *child;

  int mode;
  Layer();
  Layer(string name);

  virtual void info(){};
  virtual void addchild(Layer *l){}
  virtual void addparent(Layer *l){}
  virtual void forward(){}
  virtual void backward(){}
  virtual void initialize(){}
  virtual void applygrads(){}
  virtual void reset(){}
};

/// INPUT Layer
class Input : public Layer {
 public:

  Input(Tensor *in,string name);
  Input(Tensor *in);

  void info();
  void addchild(Layer *l);
  void addparent(Layer *l);
  void forward();
  void backward();
  void initialize();
  void applygrads();
  void reset();

};

/// DENSE Layer
class Dense : public Layer {
 public:
  int dim;

  Dense(Layer *parent,int dim);
  Dense(Layer *parent,int dim,string name);

  // Paras
  Tensor *W;
  Tensor *gW;
  Tensor *bias;
  Tensor *gbias;



  void info();
  void addchild(Layer *l);
  void addparent(Layer *l);
  void forward();
  void backward();
  void initialize();
  void applygrads();
  void reset();

};
