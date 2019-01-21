#include "data.h"

#define MAX_LAYERS 100
#define MAX_DATA 100

class Layer;

class Net {
 public:
  
  Layer *lvec[MAX_LAYERS];
  Data *Dtrain;
  Data *Dtest;
  Data *Dval;

  FILE *flog;
  
  Layer *out[MAX_LAYERS];
  Layer *in[MAX_LAYERS];
  double err[MAX_LAYERS];
  

  Layer *fts[MAX_LAYERS];
  Layer *bts[MAX_LAYERS];
  
  char name[100];
  
  int layers,olayers,ilayers;
  int bn;
  int init;
  int cropmode;
  int crops;
  int batch;
  int bproc;
  int berr;

  double mu;
  double decay;

  double ftime;
  double btime;

  Net(char *name,int b);

  void addLayer(Layer *l);

  void preparebatch(Data *Dt,int code);
  void getbatch();
  void next();
  int calcbatch(Data *Dt);

  void initialize();
  void evaluate(Data *dt);

  void resetLayers();
  void resetstats();
  void build_fts();
  void build_bts();
  void forward();
  void backward();
  void applygrads();

  void net2dot();
  void trainmode();
  void testmode();  
  void copy(Layer *ld,Layer *ls);
  void setvalues();
  void train(int epochs);
  void testOut(FILE *fs); 
  void printOut(FILE *fs,int n);
  void preparetrainbatch();
  void calcerr(int n);
  void printerrors(int n);

  void doforward();
  void docounterr();
  void dobackward();
  void doresetstats();
  void doupdate();
  void doprinterrors();
  void doreseterrors();

  void reseterrors();
  void Init(FILE *flog);

  void setcropmode(int f);
  void setmu(double m);
  void setmmu(double m);
  void setshift(int f);
  void setflip(int f);
  void setbrightness(double f);
  void setcontrast(double f);
  void decmu(double decay);  
  void setact(int a);
  void setbn(int a);
  void setmaxn(double m);
  void setl2(double m);
  void setl1(double m);
  void setdrop(double m);
  void setoptim(int l);
  void setthreads(int l);
  void setnoiser(double n);
  void setnoisesd(double n);
  void setnoiseb(double n);
  void setdecay(double f);
  void setlambda(double f);

  int isIn(Layer *l);

  void save(FILE *fe);
  void load(FILE *fe);

};

