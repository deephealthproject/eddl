/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 1.0
* copyright (c) 2021, Universitat Politècnica de València (UPV), PRHLT Research Centre
* Date: November 2021
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <stdexcept>
#include "eddl/layers/core/layer_core.h"
#include "eddl/net/net.h"
#include "eddl/random.h"
#include "eddl/system_info.h"
#include "eddl/utils.h"

#include "eddl/mpi_distributed/mpi_distributed.h"

#ifdef cMPI
#include <mpi.h>
#endif


#ifdef cNCCL
#include <nccl.h>
#endif

#ifdef cFPGA
extern void _show_profile_fpga();
#endif

#define VERBOSE 0

#define mpi_id0(...)   \
    if (id==0) \
        __VA_ARGS__; 

int verboserec=1;

using namespace std;
using namespace std::chrono;

float loss1, loss2;

#ifdef cNCCL
// NCCL
extern ncclUniqueId nccl_id;
extern ncclComm_t nccl_comm;
extern cudaStream_t cuda_stream;
#endif


/////////////////////////////////////////
//// THREADS
struct tdata {
  Net *net;
};

/////////////////////////////////////////
void *train_batch_t(void *t) {
  auto *targs = (tdata *) t;

  Net *net = targs->net;
  net->do_reset();
  net->do_reset_grads();
  net->do_forward();
  net->do_compute_loss();

  net->do_delta();
  net->do_backward();
  net->do_applygrads();

  return nullptr;
}

void *eval_batch_t(void *t) {
  auto *targs = (tdata *) t;

  Net *net = targs->net;
  net->do_reset();
  net->do_reset_grads();
  net->do_forward();
  net->do_compute_loss();

  return nullptr;
}

/////////////////////////////////////////
void *forward_t(void *t) {
  auto *targs = (tdata *) t;

  Net *net = targs->net;

  net->do_forward();

  return nullptr;
}

/////////////////////////////////////////
void *reset_t(void *t) {
  auto *targs = (tdata *) t;

  Net *net = targs->net;

  net->do_reset();

  return nullptr;
}
void *reset_grads_t(void *t) {
  auto *targs = (tdata *) t;

  Net *net = targs->net;

  net->do_reset_grads();

  return nullptr;
}

/////////////////////////////////////////

void *delta_t(void *t) {
  auto *targs = (tdata *) t;

  Net *net = targs->net;

  net->do_delta();

  return nullptr;
}
/////////////////////////////////////////
void *backward_t(void *t) {
  auto *targs = (tdata *) t;

  Net *net = targs->net;


  net->do_delta();
  net->do_backward();

  return nullptr;
}

void *compute_loss_t(void *t)
{
  auto *targs = (tdata *) t;

  Net *net = targs->net;

  net->do_compute_loss();

  return nullptr;
}

/////////////////////////////////////////
void *update_t(void *t) {
  auto *targs = (tdata *) t;

  Net *net = targs->net;
  net->do_applygrads();

  return nullptr;
}
/////////////////////////////////////////

/////////////////////////////////////////
// "a ring to rule them all"
void Net::run_snets(void *(*F)(void *t))
{
  void *status;
  int rc;
  struct tdata td[100];

  int comp = snets.size();

  if((snets[0]->dev != DEV_CPU) && (comp > 1))
  {
    #pragma omp parallel for
    for (int i = 0; i < comp; i++) {
      // Thread params
      td[i].net = snets[i];
      // Call function
      F(&td[i]);
    }
  }
  else
  {
    // Thread params
    td[0].net = snets[0];
    // Call function
    F(&td[0]);
  }
}

//////////////////////////////////////////////////////////////
//////// SIMPLE ATOMICS FUNCS
void Net::setmode(int m) {
  trmode=m;
  for (int i = 0; i < snets.size(); i++){
      snets[i]->trmode=m;
      for (int j = 0; j < snets[i]->layers.size(); j++){
          snets[i]->layers[j]->setmode(m);
      }
  }
}

void Net::clamp(float min,float max)
{
  for (int i = 0; i < snets.size(); i++)
  for (int j = 0; j < snets[i]->layers.size(); j++)
  snets[i]->layers[j]->clamp(min,max);
}


void Net::setlr(vector <float> p)
{
  for(int i=0;i<snets.size();i++)
  snets[i]->optimizer->change(p);
}

vector<vtensor> Net::get_parameters(bool deepcopy){
    vector<vtensor> net_params;

    // Collect layer params
    for(int i=0; i<this->layers.size(); i++){

        // Clone parameters
        vtensor layer_params;
        for(int j=0; j<this->layers[i]->params.size(); j++){
            // Collect Tensors from Device to CPU
            collectTensor(this->layers[i], "param", j);

            // Add to layer vector of params
            if (deepcopy){
                layer_params.push_back(this->layers[i]->params[j]->clone());
            }else{
                layer_params.push_back(this->layers[i]->params[j]);
            }
        }

        // Add new params
        net_params.push_back(layer_params);
    }

    return net_params;
}

void Net::set_parameters(const vector<vtensor>& new_params) {
    // Check the number of layers
    if(new_params.size() != this->layers.size()){
        msg("AssertionError: The number of layers in params does not match the number of layers in this network ("
        + std::to_string(new_params.size())  + "!=" + std::to_string(this->layers.size()) +")",
        "Net::set_parameters");
    }

    // Check the number of params per layer
    for(int i=0; i<this->layers.size(); i++){
        Layer* l = this->layers[i];  // Alias

        // Check number of params
        if(new_params[i].size() != l->params.size()){
            msg("AssertionError: The number of parameters in layer '" + std::to_string(i) +
            "'(" + l->name +") does not match the number of parameters in the given vector<Tensor*>. (" +
            std::to_string(l->params.size()) + "!=" + std::to_string(new_params[i].size())  +")",
                "Net::set_parameters");
        }
    }

    // Set layer params
    for (int layer_index = 0; layer_index < this->layers.size(); ++layer_index) {

        // Copy current params
        for (int param_index = 0; param_index < this->layers[layer_index]->params.size(); ++param_index) {
            Tensor::copy(new_params[layer_index][param_index], this->layers[layer_index]->params[param_index]);
            // sync_weights();  // Send CPU tensors to devices -- 2021-07-05 to be definitively removed because it does just the opposite we need here
            // copy-back to devices
            for (int snet_index = 0; snet_index < snets.size(); ++snet_index) {
                Tensor::copy(this->layers[layer_index]->params[param_index], snets[snet_index]->layers[layer_index]->params[param_index]);
            }
        }
    }
}

//////////////////////////////////
// API functions

// FORWARD
void Net::forward(vector<Tensor*> in)
{

  if (isrecurrent) {
    verboserec=0;
    forward_recurrent(in);
  }
  else {

    reset();
    if (in.size()) {
      if (in.size()!=lin.size())
      msg("size missmatch in list of tensors","Net.forward(vtensor)");

      if (batch_size!=in[0]->shape[0]) {
        resize(in[0]->shape[0]);
      }

      for (int i = 0; i < in.size(); i++) {
        Tensor::copy(in[i],lin[i]->output);
      }

      // Distribute to snets inputs
      for (int i = 0; i < in.size(); i++)
        distributeTensor(lin[i]);


    }

    run_snets(forward_t);
  }

}

void Net::forward_recurrent(vector<Tensor*> tin)
{
  int i,j,k,l;

  if (isdecoder) {
    msg("Recurrent nets with decoders can not use atomic funcs","forward");
  }

  // prepare data for unroll net
  vtensor xt;
  vtensor xtd;
  vtensor yt;

  vtensor toutr;
  vtensor tinr;
  vtensor tout;

  int inl;
  int outl;

  prepare_recurrent(tin,tout,inl,outl,xt,xtd,yt,tinr,toutr);

  build_rnet(inl,outl);

  rnet->forward(tinr);

  if (snets[0]->dev!=DEV_CPU) rnet->sync_weights();

  for(i=0;i<tinr.size();i++) delete(tinr[i]);
  for(i=0;i<toutr.size();i++) delete(toutr[i]);


  for(i=0;i<xt.size();i++)
    delete xt[i];
  xt.clear();

  for(i=0;i<yt.size();i++)
    delete yt[i];
  yt.clear();


}


void Net::forward(vector<Layer *> in)
{
  netinput=in;

  reset();
  if (in.size()) {
    if (in.size()!=lin.size())
    msg("size missmatch in list of tensors","Net.forward(vtensor)");

    if (batch_size!=in[0]->output->shape[0]) {

      resize(in[0]->output->shape[0]);
    }
  }

  vector<Tensor *> vt;
  for (int i = 0; i < in.size(); i++) {
     collectTensor(in[i],"output");
     vt.push_back(in[i]->output);
   }


  forward(vt);

}

void Net::forward()
{
  reset();

  run_snets(forward_t);
}


//// BACKWARD
void Net::backward(Layer* (*f)(Layer *),Layer *out){
    msg("Not implemented error", "Net::backward(Layer* (*f)(Layer *),Layer *out)");
}


void Net::backward(vector<Tensor *> target)
{

  if (isrecurrent) {
    if (rnet==nullptr) {
      msg("Error backward without previous forward","backward_recurrent");
    }
    verboserec=0;
    backward_recurrent(target);
  }
  else  {

    if (target.size()) {
      if (target.size()!=lout.size())
      msg("size missmatch in list of targets","Net.backward(vtensor)");

      if (batch_size!=target[0]->shape[0]) {
        cout<<batch_size<<"!="<<target[0]->shape[0]<<endl;
        msg("bakcward step with different batch_size than forward","Net.backward(vtensor)");
      }

      int comp=snets.size();
      if (batch_size<comp) {
        msg("batch_size lower than computing service parallelism","backward");
      }

      int thread_batch_size=batch_size / comp;

      // Split data for each network
      for (int i = 0; i < comp; i++) {
        int start = i * thread_batch_size;
        int end = start + Ys[i][0]->shape[0];
        vector<int> sind(batch_size);
        for(int k=0;k<batch_size;k++) sind[k]=k;
        // Copy targets
        for (int j = 0; j < target.size(); j++) {
          Tensor::select(target[j], Ys[i][j], sind, start, end);
          snets[i]->lout[j]->check_target();
          Tensor::copy(Ys[i][j], snets[i]->lout[j]->target);
        }
      }
    }
    tr_batches++;

    compute_loss();

    run_snets(backward_t);

  }
}

void Net::backward_recurrent(vector<Tensor *> target)
{
  int i,j,k,l;

  if (isdecoder) {
    msg("Recurrent nets with decoders can not use atomic funcs","backward");
  }

  // prepare data for unroll net
  vtensor xt;
  vtensor xtd;
  vtensor yt;

  vtensor toutr;
  vtensor tinr;
  vtensor tin;

  int inl;
  int outl;

  prepare_recurrent(tin,target,inl,outl,xt,xtd,yt,tinr,toutr);

  rnet->backward(toutr);

  if (snets[0]->dev!=DEV_CPU) rnet->sync_weights();

  for(i=0;i<tinr.size();i++) delete(tinr[i]);
  for(i=0;i<toutr.size();i++) delete(toutr[i]);


  for(i=0;i<xt.size();i++)
    delete xt[i];
  xt.clear();

  for(i=0;i<yt.size();i++)
    delete yt[i];
  yt.clear();

}


void Net::backward(){

  vector<Net*> visited;
  tr_batches++;

  run_snets(backward_t);


  for(int i=0;i<netinput.size();i++) {
    if (netinput[i]->detached==false) {
      lin[i]->mem_delta();
      collectTensor(lin[i],"delta");
      netinput[i]->mem_delta();
      Tensor::copy(lin[i]->delta,netinput[i]->delta);
      distributeTensor(netinput[i],"delta");
    }
  }

  for(int i=0;i<netinput.size();i++) {
    if (netinput[i]->detached==false){
      bool enc=false;
      for(int j=0;j<visited.size();j++)
      if (visited[j]==netinput[i]->net) enc=true;

      if (!enc) {
        visited.push_back(netinput[i]->net);
        netinput[i]->net->backward();
      }
    }
  }

  netinput.clear();

}




//// Loss
void Net::reset_loss()
{
  if (isrecurrent) {
    if (rnet!=nullptr) rnet->reset_loss();
  }
  else {
    // Reset errors
    int p=0;
    for (int j = 0; j < lout.size(); j++,p+=2){
      total_loss[j] = 0.0;
      total_metric[j] = 0.0;
      fiterr[p] = fiterr[p + 1] = 0.0;
    }
    inferenced_samples=0;
  }
}


//// COMPUTE Loss
void Net::compute_loss()
{
  if (isrecurrent) {
    if (rnet==nullptr) {
      msg("Error compute loss unroll net","compute_loss");
    }
    rnet->compute_loss();
  }
  else {
    run_snets(compute_loss_t);

    int comp=snets.size();
    if (batch_size<comp) {
      msg("batch_size lower than computing service parallelism","compute_loss");
    }


    if (snets[0]->dev != DEV_CPU)
    for (int i = 0; i < comp; i++) {
      for (int j = 0; j < 2 * lout.size(); j++) {
        fiterr[j] += snets[i]->fiterr[j];
      }
    }

    int p=0;
    for(int k=0;k<lout.size();k+=decsize)
     for (int j = 0; j < decsize; j++,p+=2) {
       total_loss[k] += fiterr[p];  // losses
       total_metric[k] += fiterr[p + 1];  // metric
       fiterr[p] = fiterr[p + 1] = 0.0;
      }

    inferenced_samples+=batch_size;
  }
}

void Net::print_loss(int b, int nb, bool reduce) {
    int lbar = 50;
    int p = 0;
    int id, n_procs;
    float loss;
    float metric;
    char symbol[10];

    loss = 0;
    if (isrecurrent) {
        if (rnet != nullptr) rnet->print_loss(b, nb, reduce);
    } else {
        if (is_mpi_distributed()) {
            n_procs = get_n_procs_distributed();
            id = get_id_distributed();
            sprintf(symbol, "%s", ">");
        } else {
            n_procs = 1;
            id = 0;
            sprintf(symbol, "%s", "█");
        }
        if (id == 0) {
            if (nb != -1) {
                int pc = ((b + 1) * lbar) / nb;
                if (b >= nb) pc = lbar;

                printf("[");

                set_text_green();
                //                for (int k = 0; k < pc; k++) printf("█");
                for (int k = 0; k < pc; k++) printf(symbol);
                //if (pc<lbar) {
                //  if (b%4<2) printf(".");
                //  else printf(".");
                //}

                set_text_red();
                for (int k = pc + 1; k < lbar; k++) printf("-");

                set_text_default();
                printf("] ");
            }

            fprintf(stdout, "%d ", b);
        }
        int length = decsize;
        for (int k = 0; k < lout.size(); k += decsize) {
            string name = lout[k]->name;

            if (lout[k]->sorig != nullptr)
                name = lout[k]->sorig->name;

            mpi_id0(fprintf(stdout, "%s[", name.c_str()));

            if (losses.size() >= (k + 1)) {
                if (is_mpi_distributed() && reduce) {
#ifdef cMPI
                    //MPICHECK(MPI_Reduce(&total_loss[k], &total_loss[k], 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD));                        
                    MPICHECK(MPI_Reduce(&total_loss[k], &loss, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD));
                    loss = loss / n_procs;
#endif
                } else
                    loss = total_loss[k];

                //fprintf(stdout, "loss[%s]=%1.4f ", losses[k]->name.c_str(), total_loss[k] / (length*inferenced_samples));
                //mpi_id0(fprintf(stdout, "loss=%1.3f ", total_loss[k] / (length * inferenced_samples)));                
                mpi_id0(fprintf(stdout, "loss=%1.3f ", loss / (length * inferenced_samples)));

            }
            if (this->metrics.size() >= (k + 1)) {
                if (is_mpi_distributed() && reduce) {
#ifdef cMPI                                     
                    //MPICHECK(MPI_Reduce(&total_metric[k], &total_metric[k], 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD));
                    MPICHECK(MPI_Reduce(&total_metric[k], &metric, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD));
                    metric = metric / n_procs;
#endif                    
                } else {
                    metric = total_metric[k];
                }
                //fprintf(stdout, "metric[%s]=%1.4f ", this->metrics[k]->name.c_str(), total_metric[k] / (length*inferenced_samples));
                //mpi_id0(fprintf(stdout, "metric=%1.3f", total_metric[k] / (length * inferenced_samples)));
                mpi_id0(fprintf(stdout, "metric=%1.3f", metric / (length * inferenced_samples)));
            }

            mpi_id0(fprintf(stdout, "] "));


            // Log files. Only process 0
            if (id == 0) {
                if ((flog_tr != nullptr)&&(trmode)) {
                    fprintf(flog_tr, "%s ", name.c_str());
                    if (losses.size() >= (k + 1)) {
                        //fprintf(flog_tr, "loss[%s]=%1.4f ", losses[k]->name.c_str(), total_loss[k] / inferenced_samples);
                        fprintf(flog_tr, "loss[%s]=%1.4f ", losses[k]->name.c_str(), loss / inferenced_samples);
                    }
                    if (this->metrics.size() >= (k + 1)) {
                        if (this->metrics[k]->name != "none")
                            //fprintf(flog_tr, "metric[%s]=%1.4f ", this->metrics[k]->name.c_str(), total_metric[k] / inferenced_samples);
                            fprintf(flog_tr, "metric[%s]=%1.4f ", this->metrics[k]->name.c_str(), metric / inferenced_samples);
                    }

                    fprintf(flog_tr, " -- ");


                }
                if ((flog_ts != nullptr)&&(!trmode)) {
                    fprintf(flog_ts, "%s ", name.c_str());
                    if (losses.size() >= (k + 1)) {
                        //fprintf(flog_ts, "loss[%s]=%1.4f ", losses[k]->name.c_str(), total_loss[k] / inferenced_samples);
                        fprintf(flog_ts, "loss[%s]=%1.4f ", losses[k]->name.c_str(), loss / inferenced_samples);
                    }
                    if (this->metrics.size() >= (k + 1)) {
                        if (this->metrics[k]->name != "none")
                            //fprintf(flog_ts, "metric[%s]=%1.4f ", this->metrics[k]->name.c_str(), total_metric[k] / inferenced_samples);
                            fprintf(flog_ts, "metric[%s]=%1.4f ", this->metrics[k]->name.c_str(), metric / inferenced_samples);
                    }

                    fprintf(flog_ts, " -- ");
                }
            }


        }

        mpi_id0(fflush(stdout));

        // Log files. Only process 0
        if (id == 0) {
            if ((flog_tr != nullptr)&&(trmode)) {
                fprintf(flog_tr, "\n");
                fflush(flog_tr);
            }

            if ((flog_ts != nullptr)&&(!trmode)) {
                fprintf(flog_ts, "\n");
                fflush(flog_ts);
            }
        }
    }
}

vector<float> Net::get_losses() {
    // NOTE: I have no idea how the internals of the metrics/loss values work,
    // so I did a sort of copy and paste from print_loss fuction with minor workarounds
    vector<float> loss_values;

    if (this->isrecurrent) {
        if (this->rnet != nullptr) {
            return this->rnet->get_losses();
        } // Dangerous A.F.
    } else {
        int p = 0;

        // Copy total_loss / fiterr (I don't know how it works but i don't like it...)
        vector<float> tmp_total_error; //(total_loss);
        vector<float> tmp_fiterr; //(fiterr);
        for (auto _ : total_loss) tmp_total_error.push_back(_);
        for (auto _ : fiterr) tmp_fiterr.push_back(_);

        int length = decsize;
        for (int k = 0; k < lout.size(); k += decsize) {

            // Do stuff
            for (int l = 0; l < length; l++, p += 2) {
                tmp_total_error[k] += tmp_fiterr[p]; // loss
                tmp_fiterr[p] = tmp_fiterr[p + 1] = 0.0;
            }

            // Compute average loss
            if (losses.size() >= (k + 1)) {
                loss_values.push_back(tmp_total_error[k] / (float) (length * inferenced_samples));
            }
        }
    }

    return loss_values;
}

vector<float> Net::get_metrics(){
    // NOTE: I have no idea how the internals of the metrics/loss values work,
    // so I did a sort of copy and paste from print_loss fuction with minor workarounds
    vector<float> metrics_values;

    if (this->isrecurrent) {
        if (this->rnet!=nullptr) { return this->rnet->get_metrics(); } // Dangerous A.F.
    } else {
        int p = 0;

        // Copy total_loss / fiterr (I don't know how it works but i don't like it...)
        vector<float> tmp_total_metrics; //(total_metric);
        vector<float> tmp_fiterr; //(fiterr);
        for (auto _ : total_metric) tmp_total_metrics.push_back(_);
        for (auto _ : fiterr) tmp_fiterr.push_back(_);

        int length=decsize;
        for (int k = 0; k < lout.size(); k+=decsize) {

            for(int l=0;l<length;l++,p+=2) {
                tmp_total_metrics[k] += tmp_fiterr[p + 1];  // metric
                tmp_fiterr[p] = tmp_fiterr[p + 1] = 0.0;
            }

            if (this->metrics.size()>=(k+1)) {
                metrics_values.push_back( tmp_total_metrics[k] / (float)(length*inferenced_samples));
            }
        }
    }

    return metrics_values;
}

void Net::reset_grads()
{
  if (isrecurrent)
  if (rnet!=nullptr)
  rnet->reset_grads();

  do_reset_grads();
  run_snets(reset_grads_t);
}

void Net::reset()
{
  if (isrecurrent)
  if (rnet!=nullptr)
  rnet->reset();

  do_reset();
  run_snets(reset_t);
}




void Net::update()
{
  if (isrecurrent) {
    if (rnet!=nullptr) {
      rnet->update();
    }
  }
  else {
    run_snets(update_t);

    int comp=snets.size();

    if (batch_size<comp) {
      msg("batch_size lower than computing service parallelism","update");

    }

    if ((snets[0]->dev != DEV_CPU) && (comp > 1) && (tr_batches%cs->lsb==1)) {
      sync_weights();
    }
  }
}

void Net::delta()
{
  if (isrecurrent) {
    if (rnet!=nullptr)
    rnet->run_snets(delta_t);
  }
  else run_snets(delta_t);

}


//////////////////////////////////////////////////////////////
//////// HIGHER LEVEL FUNCS
void Net::fit(vtensor tin, vtensor tout, int batch, int epochs) {
    int i, j, k, n;
    int ii, jj;
    int id;
    int n_procs;
    float * myptr;
    int count;
   // int batches = 0;
    int batches_per_proc = 0;
   // int batches_avg = 0;
    int local_batch;
    double secs_epoch = 1e10;
    double secs_epoch_prev = 0;
    float SPEED_UP = 1.05;
    
    int mpi_avg ;
    int avg_method ;
    int x_avg ;
   

    

    if (isrecurrent) {
        fit_recurrent(tin, tout, batch, epochs);
    } else {
        if (is_mpi_distributed()) {
            //MPI_Comm_size(MPI_COMM_WORLD, &n_procs);
            //MPI_Comm_rank(MPI_COMM_WORLD, &id);
            n_procs = get_n_procs_distributed();
            id = get_id_distributed();
            get_params_distributed(&avg_method, &mpi_avg, &x_avg);

            // Set local batch size
            local_batch = batch / n_procs;           
            //mpi_id0(fprintf(stderr, "[DISTR] fit\n"));
        } else {
            n_procs=1;
            id = 0;
            local_batch = batch;
        }

        // Check current optimizer
        if (optimizer == nullptr)
            msg("Net is not build", "Net.fit");

        // Check if number of input/output network layers matches with the input/output tensor data
        if (tin.size() != lin.size()) {
            cout << tin.size() << "!=" << lin.size() << endl;
            msg("input tensor list does not match with defined input layers", "Net.fit");
        }
        if (tout.size() != lout.size()) {
            cout << tout.size() << "!=" << lout.size() << endl;
            msg("output tensor list does not match with defined output layers", "Net.fit");
        }


        // Check if all the data inputs has the same number of samples
        n = tin[0]->shape[0];
        for (i = 1; i < tin.size(); i++)
            if (tin[i]->shape[0] != n)
                msg("different number of samples in input tensor", "Net.fit");


        // Check if the size of the output layers matches with inputs sizes
        for (i = 1; i < tout.size(); i++)
            if (tout[i]->shape[0] != n)
                msg("different number of samples in output tensor", "Net.fit");


        // Set batch size
        resize(local_batch);


        // Create array to store batch indices (later random)
        vind sind;
        for (i = 0; i < batch_size; i++)
            sind.push_back(0);


        // Start training
        setmode(TRMODE);

        // Set some parameters
        int num_batches = n / batch_size;
        batches_per_proc = num_batches / n_procs;

        // Train network
        if (id == 0) {
            fprintf(stdout, "%d epochs of (global: %d batches of size %d),(local: %d batches of size %d) \n", epochs, num_batches/n_procs, n_procs * batch_size, num_batches, batch_size);
            if (is_mpi_distributed()) fprintf(stdout, "[DISTR] %d procs. %d batches per proc. sync every %d batches \n", n_procs, batches_per_proc, mpi_avg);
        }

        //batches_avg = mpi_avg;
        for (i = 0; i < epochs; i++) {
            high_resolution_clock::time_point e1 = high_resolution_clock::now();
            if (id == 0) {
                fprintf(stdout, "Epoch %d\n", i + 1);
                if (is_mpi_distributed()) {
                    fprintf(stdout, "[DISTR] batches_avg: %d\n", get_current_batch_avg_distributed ());
                } 
            }
            reset_loss();

            //batches = 0;
            // For each batch
            for (j = 0; j < (batches_per_proc); j++) {
                //batches = batches + n_procs;

                //printf("Batch nr %d\n", j);
                // Set random indices
                //printf("Proc: %d sind:\n", id);
                for (k = 0; k < batch_size; k++) {
                    sind[k] = rand() % n;
                    //printf("%5d ",sind[k]);
                }
                //printf("\n");
                
                // Train batch
                tr_batches++;

                train_batch(tin, tout, sind);
                 // gpu_layer_print (this, 3);
                  
                // synchronize
                if (is_mpi_distributed()) 
                    avg_weights_distributed(this, j+1, batches_per_proc);
                 
                // In training mode, do not reduce
                print_loss(j+1, batches_per_proc, false);
                //print_loss(j+1,num_batches);
                
                high_resolution_clock::time_point e2 = high_resolution_clock::now();
                duration<double> epoch_time_span = e2 - e1;
                if (id == 0) {
                    fprintf(stdout, "%1.4f secs/batch\r", epoch_time_span.count() / (j + 1)) ;
                    fflush(stdout);
                }
            }
            high_resolution_clock::time_point e2 = high_resolution_clock::now();
            duration<double> epoch_time_span = e2 - e1;
            secs_epoch = epoch_time_span.count();
            if (i == 0) secs_epoch_prev = 2 * secs_epoch; // Force first change with adaptive

            if (id == 0) {
                fprintf(stdout, "\n%1.4f secs/epoch\n", epoch_time_span.count());
                fflush(stdout);
            }
            /*
            if (((i+1) % 2)==1) {
                    loss1 =  get_losses()[lout.size()-1];
                    printf("measuring loss1 %f\n", loss1);
                }
            if (((i+1) % 2)==0) {
                    loss2 =  get_losses()[lout.size()-1];
                    printf("measuring loss2 %f\n", loss2);
                 }
            printf("loss1 %f\n", loss1);
            printf("loss2 %f\n", loss2);
             */
            update_batch_avg_distributed (i, secs_epoch, batches_per_proc);           
        }
        fflush(stdout);
    }
}


void Net::prepare_recurrent_dec(vtensor tin, vtensor tout, int &inl, int &outl, vtensor &xt, vtensor &xtd,vtensor &yt,vtensor &tinr,vtensor &toutr, Tensor *Z)
{
  int i, j, k, n;

  // Check whether is encoder, decoder or both.
  for(i=0;i<vfts.size();i++) {
    if (vfts[i]->isdecoder) {isdecoder=true;break;}
    else if (vfts[i]->isrecurrent) isencoder=true;
  }

  // Set the properties to snets
  for(i=0;i<snets.size();i++) {
    snets[i]->isdecoder=isdecoder;
    snets[i]->isencoder=isencoder;
  }

  inl=outl=1;

  for(i=0;i<tin.size();i++) {
    xt.push_back(tin[i]->clone());
    tinr.push_back(new Tensor(xt[i]->shape,xt[i]->ptr,xt[i]->device));
  }


  // PREPARE INPUT and OUTPUT DECODER
  for(i=0;i<tout.size();i++) {
    if (tout[i]->ndim<3)
      msg("Output tensor should be batch x timesteps x dims","Net::prepare_recurrent");
    vector<int> pshape;
    pshape.push_back(1);
    pshape.push_back(0);
    for(int j=2;j<tout[i]->ndim;j++)
     pshape.push_back(j);
    yt.push_back(Tensor::permute(tout[i],pshape)); // time x batch x dim
  }

  // outl=out time_steps.
  // Check that all the potential inputs have the same timesteps:
  outl=yt[0]->shape[0];
  for(i=0;i<yt.size();i++) {
    if (yt[i]->shape[0]!=outl)
    msg("Output tensors with different time steps","fit_recurrent");
  }
  if (verboserec) cout<<"Vec2Seq "<<inl<<" to "<<outl<<"\n";

  int offset;
  for(i=0;i<yt.size();i++) {
    offset=yt[i]->size/yt[i]->shape[0];
    vector<int>shape;
    for(j=1;j<yt[i]->ndim;j++)
      shape.push_back(yt[i]->shape[j]);

    //input, delayed
    vector<int>zero_shape;
    for(j=0;j<tout[i]->ndim;j++)
      if (j!=1) zero_shape.push_back(tout[i]->shape[j]);

    tinr.push_back(Tensor::zeros(zero_shape,tout[i]->device));
    for(j=0;j<outl-1;j++)
      tinr.push_back(new Tensor(shape,yt[i]->ptr+(j*offset),yt[i]->device));

    // output
    for(j=0;j<outl;j++)
      toutr.push_back(new Tensor(shape,yt[i]->ptr+(j*offset),yt[i]->device));
  }



}



void Net::prepare_recurrent_enc_dec(vtensor tin, vtensor tout, int &inl, int &outl, vtensor &xt, vtensor &xtd,vtensor &yt,vtensor &tinr,vtensor &toutr, Tensor *Z)
{

  int i, j, k, n;

  // Check whether is encoder, decoder or both.
  for(i=0;i<vfts.size();i++) {
    if (vfts[i]->isdecoder) {isdecoder=true;break;}
    else if (vfts[i]->isrecurrent) isencoder=true;
  }

  // Set the properties to snets
  for(i=0;i<snets.size();i++) {
    snets[i]->isdecoder=isdecoder;
    snets[i]->isencoder=isencoder;
  }

  inl=outl=1;

  // PREPARE INPUT ENCODER
  for(i=0;i<tin.size();i++) {
    if (tin[i]->ndim<3)
        msg("Input tensor should be batch x timesteps x dims","Net::prepare_recurrent");
    vector<int> pshape;
    pshape.push_back(1);
    pshape.push_back(0);
    for(int j=2;j<tin[i]->ndim;j++)
     pshape.push_back(j);
    xt.push_back(Tensor::permute(tin[i],pshape)); // time x batch x dims
  }

  // inl=input time_steps.
  // Check that all the potential inputs have the same timesteps:
  inl=xt[0]->shape[0];
  for(i=0;i<xt.size();i++) {
    if (xt[i]->shape[0]!=inl)
      msg("Input tensors with different time steps","Net::prepare_recurrent");
  }


  int offset;
  for(i=0;i<xt.size();i++) { // again xt.size() is normally 1
    offset=xt[i]->size/xt[i]->shape[0];
    vector<int>shape;
    for(j=1;j<xt[i]->ndim;j++)
      shape.push_back(xt[i]->shape[j]);
    for(j=0;j<inl;j++) // fot all timesteps create a share tensor
      tinr.push_back(new Tensor(shape,xt[i]->ptr+(j*offset),xt[i]->device));
  }


  // PREPARE INPUT and OUTPUT DECODER
  for(i=0;i<tout.size();i++) {
    if (tout[i]->ndim<3)
      msg("Output tensor should be batch x timesteps x dims","Net::prepare_recurrent");
    vector<int> pshape;
    pshape.push_back(1);
    pshape.push_back(0);
    for(int j=2;j<tout[i]->ndim;j++)
     pshape.push_back(j);
    yt.push_back(Tensor::permute(tout[i],pshape)); // time x batch x dim
  }

  // outl=out time_steps.
  // Check that all the potential inputs have the same timesteps:
  outl=yt[0]->shape[0];
  for(i=0;i<yt.size();i++) {
    if (yt[i]->shape[0]!=outl)
    msg("Output tensors with different time steps","fit_recurrent");
  }

  if (verboserec) std::cerr << "Seq2Seq " << inl << " to " << outl << std::endl;

  for(i=0;i<yt.size();i++) {
    offset=yt[i]->size/yt[i]->shape[0];
    vector<int>shape;
    for(j=1;j<yt[i]->ndim;j++)
      shape.push_back(yt[i]->shape[j]);

    //input, delayed
    vector<int>zero_shape;
    for(j=0;j<tout[i]->ndim;j++)
      if (j!=1) zero_shape.push_back(tout[i]->shape[j]);

    tinr.push_back(Tensor::zeros(zero_shape,tout[i]->device));
    for(j=0;j<outl-1;j++)
      tinr.push_back(new Tensor(shape,yt[i]->ptr+(j*offset),yt[i]->device));

    // output
    for(j=0;j<outl;j++)
      toutr.push_back(new Tensor(shape,yt[i]->ptr+(j*offset),yt[i]->device));
  }

}


void Net::prepare_recurrent_enc(vtensor tin, vtensor tout, int &inl, int &outl, vtensor &xt, vtensor &xtd,vtensor &yt,vtensor &tinr,vtensor &toutr, Tensor *Z)
{
  int i, j, k, n;


  // Check whether is encoder, decoder or both.
  for(i=0;i<vfts.size();i++) {
    if (vfts[i]->isdecoder) {isdecoder=true;break;}
    else if (vfts[i]->isrecurrent) isencoder=true;
  }

  // Set the properties to snets
  for(i=0;i<snets.size();i++) {
    snets[i]->isdecoder=isdecoder;
    snets[i]->isencoder=isencoder;
  }

  inl=outl=1;

  // PREPARE INPUT
  for(i=0;i<tin.size();i++) {
    if (tin[i]->ndim<3)
        msg("Input tensor should be batch x timesteps x dims","Net::prepare_recurrent");
    vector<int> pshape;
    pshape.push_back(1);
    pshape.push_back(0);
    for(int j=2;j<tin[i]->ndim;j++)
     pshape.push_back(j);
    xt.push_back(Tensor::permute(tin[i],pshape)); // time x batch x dims

  inl=xt[0]->shape[0];
  // inl=time_steps.
  // Check that all the potential inputs have the same timesteps:
  for(i=0;i<xt.size();i++) {
    if (xt[i]->shape[0]!=inl)
      msg("Input tensors with different time steps","Net::prepare_recurrent");
    }
  }

  int offset;
  for(i=0;i<xt.size();i++) { // again xt.size() is normally 1
    offset=xt[i]->size/xt[i]->shape[0];
    vector<int>shape;
    for(j=1;j<xt[i]->ndim;j++)
      shape.push_back(xt[i]->shape[j]);
    for(j=0;j<inl;j++) // fot all timesteps create a share tensor
      tinr.push_back(new Tensor(shape,xt[i]->ptr+(j*offset),xt[i]->device));
  }

  // PREPARE OUTPUT
  for(i=0;i<tout.size();i++) {
    if (tout[i]->ndim<3)
      msg("Output tensor should be batch x timesteps x dims","Net::prepare_recurrent");
    vector<int> pshape;
    pshape.push_back(1);
    pshape.push_back(0);
    for(int j=2;j<tout[i]->ndim;j++)
     pshape.push_back(j);
    yt.push_back(Tensor::permute(tout[i],pshape)); // time x batch x dims

    outl=yt[0]->shape[0];
    // outl=time_steps output
    // Check that all the potential outputs have the same timesteps:
    for(i=0;i<yt.size();i++) {
      if (yt[i]->shape[0]!=outl)
      msg("Output tensors with different time steps","Net::prepare_recurrent");
    }
  }

  if (verboserec)
    if (outl>1)
      cout<<"Synchronous Seq2Seq "<<inl<<" to "<<outl<<"\n";
    else
      cout<<"Recurrent "<<inl<<" to "<<outl<<"\n";

  for(i=0;i<yt.size();i++) {
    offset=yt[i]->size/yt[i]->shape[0];
    vector<int>shape;
    for(j=1;j<yt[i]->ndim;j++)
      shape.push_back(yt[i]->shape[j]);

    if (tout.size())
      for(j=0;j<outl;j++)
        toutr.push_back(new Tensor(shape,yt[i]->ptr+(j*offset),yt[i]->device));
  }
}



void Net::prepare_recurrent(vtensor tin, vtensor tout, int &inl, int &outl, vtensor &xt, vtensor &xtd,vtensor &yt,vtensor &tinr,vtensor &toutr, Tensor *Z)
{
  int i, j, k, n;

  // Check whether is encoder, decoder or both.
  for(i=0;i<vfts.size();i++) {
    if (vfts[i]->isdecoder) {isdecoder=true;break;}
    else if (vfts[i]->isrecurrent) isencoder=true;
  }

  // Set the properties to snets
  for(i=0;i<snets.size();i++) {
    snets[i]->isdecoder=isdecoder;
    snets[i]->isencoder=isencoder;
  }

  if ((isencoder)&&(isdecoder))
    prepare_recurrent_enc_dec(tin, tout, inl, outl, xt, xtd, yt, tinr,toutr);
  else if (isdecoder)
      prepare_recurrent_dec(tin, tout, inl, outl, xt, xtd, yt, tinr,toutr);
  else
    prepare_recurrent_enc(tin, tout, inl, outl, xt, xtd, yt, tinr,toutr);
}

void Net::fit_recurrent(vtensor tin, vtensor tout, int batch, int epochs) {
  int i, j, k, n;

  // prepare data for unroll net
  vtensor xt;
  vtensor xtd;
  vtensor yt;

  vtensor toutr;
  vtensor tinr;

  int inl;
  int outl;

  prepare_recurrent(tin,tout,inl,outl,xt,xtd,yt,tinr,toutr);

  build_rnet(inl,outl);

  rnet->fit(tinr,toutr,batch,epochs);

  if (snets[0]->dev!=DEV_CPU) rnet->sync_weights();

  for(i=0;i<tinr.size();i++) delete(tinr[i]);
  for(i=0;i<toutr.size();i++) delete(toutr[i]);


  for(i=0;i<xt.size();i++)
    delete xt[i];
  xt.clear();

  for(i=0;i<yt.size();i++)
    delete yt[i];
  yt.clear();

}

// TODO:  train_batch_recurrent
/////////////////////////////////////////
void Net::train_batch_recurrent(vtensor tin, vtensor tout,vind sind, int eval) {
  int i, j, k, n;

  // prepare data for unroll net
  vtensor xt;
  vtensor xtd;
  vtensor yt;

  vtensor toutr;
  vtensor tinr;

  int inl;
  int outl;

  prepare_recurrent(tin,tout,inl,outl,xt,xtd,yt,tinr,toutr);

  build_rnet(inl,outl);

  rnet->train_batch(tinr,toutr,sind,eval);

  if (snets[0]->dev!=DEV_CPU) rnet->sync_weights();

  for(i=0;i<tinr.size();i++) delete(tinr[i]);
  for(i=0;i<toutr.size();i++) delete(toutr[i]);


  for(i=0;i<xt.size();i++)
    delete xt[i];
  xt.clear();

  for(i=0;i<yt.size();i++)
    delete yt[i];
  yt.clear();

}


void Net::train_batch(vtensor X, vtensor Y, vind sind, int eval) {

  if (isrecurrent) {
    verboserec=0;
    train_batch_recurrent(X,Y,sind,eval);
  }
  else{

  if (batch_size!=sind.size()) resize(sind.size());

  int comp=snets.size();

  if (batch_size<comp) {
    msg("batch_size lower than computing service parallelism","compute_loss");
  }

  int thread_batch_size=batch_size / comp;

  if (eval) setmode(TSMODE);
  else setmode(TRMODE);

  // Check indices
  if (sind.size() == 0) msg("error void index","Net::train_batch");
  // Split data for each network
  for (int i = 0; i < comp; i++) {
    int start = i * thread_batch_size;
    int end = start + Xs[i][0]->shape[0];

    // Copy samples
    for (int j = 0; j < X.size(); j++) {
      Tensor::select(X[j], Xs[i][j], sind, start, end);
      Tensor::copy(Xs[i][j], snets[i]->lin[j]->input);
    }

    // Copy targets
    for (int j = 0; j < Y.size(); j++) {
      Tensor::select(Y[j], Ys[i][j], sind, start, end);
      snets[i]->lout[j]->check_target();
      Tensor::copy(Ys[i][j], snets[i]->lout[j]->target);
    }
  }

  if (eval)
  run_snets(eval_batch_t);
  else
  run_snets(train_batch_t);

  // If training (eval==0), apply gradients
  if (!eval) {
    // In case of multiple GPUS or FPGA synchronize params
    if ((snets[0]->dev != DEV_CPU) && (comp > 1) && (tr_batches%cs->lsb==0)) {
      sync_weights();
    }
  }

  compute_loss();
}
}



///////////////////////////////////////////
void Net::evaluate(vtensor tin, vtensor tout, int bs) {

    int i, j, k, n;
    int id;
    

    if (isrecurrent) {
        evaluate_recurrent(tin, tout, bs);
    } else {
        if (is_mpi_distributed()) {
            id = get_id_distributed();
        } else {
            id = 0;
        }
        if (id == 0) { // MPI distributed. Only process 0 evaluates
            // Check list shape
            if (tin.size() != lin.size())
                msg("input tensor list does not match with defined input layers", "Net.evaluate");
            if (tout.size() != lout.size())
                msg("output tensor list does not match with defined output layers", "Net.evaluate");

            // Check data consistency
            n = tin[0]->shape[0];

            for (i = 1; i < tin.size(); i++)
                if (tin[i]->shape[0] != n)
                    msg("different number of samples in input tensor", "Net.evaluate");

            for (i = 1; i < tout.size(); i++)
                if (tout[i]->shape[0] != n)
                    msg("different number of samples in output tensor", "Net.evaluate");

            if (bs != -1) resize(bs);
            else if (!isresized) resize(10); // to avoid some issues when no previous fit is performed, TODO


            printf("Evaluate with batch size %d\n", batch_size);

            // Create internal variables
            vind sind;
            for (k = 0; k < batch_size; k++)
                sind.push_back(0);

            // Start eval
            setmode(TSMODE);
            reset_loss();
            for (j = 0; j < n / batch_size; j++) {
                for (k = 0; k < batch_size; k++) {
                    sind [k] = (j * batch_size) + k;
                    //printf("%5d ",sind[k]);
                }
                train_batch(tin, tout, sind, 1);
                print_loss(j + 1, n / batch_size);
                fprintf(stdout, "\r");
                fflush(stdout);
            }
            fprintf(stdout, "\n");
        }
    }
}


///////////////////////////////////////////
void Net::evaluate_recurrent(vtensor tin, vtensor tout, int bs) {

  int i;

  // prepare data for unroll net
  vtensor xt;
  vtensor xtd;
  vtensor yt;

  vtensor toutr;
  vtensor tinr;

  int inl;
  int outl;

  prepare_recurrent(tin,tout,inl,outl,xt,xtd,yt,tinr,toutr);

  build_rnet(inl,outl);

  rnet->evaluate(tinr,toutr,bs);

  for(i=0;i<tinr.size();i++) delete(tinr[i]);
  for(i=0;i<toutr.size();i++) delete(toutr[i]);

  for(i=0;i<xt.size();i++)
      delete xt[i];
  xt.clear();

  for(i=0;i<yt.size();i++)
      delete yt[i];
  yt.clear();


}

///////////////////////////////////////////

void Net::evaluate_distr(vtensor tin, vtensor tout, int bs) {

    int i, j, k, n;
    int id, n_procs;
    int batches_per_proc;
    //int batches;

    if (!is_mpi_distributed()) {
        msg("not running with MPI", "Net.evaluate_distr");
    }

    if (isrecurrent) {
        evaluate_recurrent_distr(tin, tout, bs);
    } else {
        n_procs = get_n_procs_distributed();
        id = get_id_distributed();
        // local batch_size is already set
        mpi_id0(fprintf(stderr, "[DISTR] evaluate\n"));

        // Check list shape
        if (tin.size() != lin.size())
            msg("input tensor list does not match with defined input layers", "Net.evaluate");
        if (tout.size() != lout.size())
            msg("output tensor list does not match with defined output layers", "Net.evaluate");

        // Check data consistency
        n = tin[0]->shape[0];

        for (i = 1; i < tin.size(); i++)
            if (tin[i]->shape[0] != n)
                msg("different number of samples in input tensor", "Net.evaluate");

        for (i = 1; i < tout.size(); i++)
            if (tout[i]->shape[0] != n)
                msg("different number of samples in output tensor", "Net.evaluate");

        if (bs != -1) resize(bs);
        else if (!isresized) resize(10); // to avoid some issues when no previous fit is performed, TODO


        mpi_id0(printf("Evaluate with batch size %d\n", batch_size));

        // Create internal variables
        vind sind;
        for (k = 0; k < batch_size; k++)
            sind.push_back(0);


        // Set some parameters
        int num_batches = n / batch_size;
        batches_per_proc = num_batches / n_procs;
        //batches = 0;

        // Start eval
        setmode(TSMODE);
        reset_loss();
        //for (j = 0; j < n / batch_size; j++) {
        for (j = 0; j < batches_per_proc; j++) {
            //batches = batches + n_procs;
            for (k = 0; k < batch_size; k++) {
                //sind [k] = (j * batch_size) + k;
                sind [k] = (((id * batches_per_proc) + j) * batch_size) + k;
                //printf("%5d ",sind[k]);
            }
            train_batch(tin, tout, sind, 1);
            //         print_loss(batches, num_batches);
            print_loss(j + 1, batches_per_proc, true);
            //print_loss(j + 1, n / batch_size);
            mpi_id0(fprintf(stdout, "\r"));
            mpi_id0(fflush(stdout));
        }
        mpi_id0(fprintf(stdout, "\n"));
    }
}


///////////////////////////////////////////
void Net::evaluate_recurrent_distr(vtensor tin, vtensor tout, int bs) {

  int i;

  // prepare data for unroll net
  vtensor xt;
  vtensor xtd;
  vtensor yt;

  vtensor toutr;
  vtensor tinr;

  int inl;
  int outl;

  prepare_recurrent(tin,tout,inl,outl,xt,xtd,yt,tinr,toutr);

  build_rnet(inl,outl);

  rnet->evaluate_distr(tinr,toutr,bs);

  for(i=0;i<tinr.size();i++) delete(tinr[i]);
  for(i=0;i<toutr.size();i++) delete(toutr[i]);

  for(i=0;i<xt.size();i++)
      delete xt[i];
  xt.clear();

  for(i=0;i<yt.size();i++)
      delete yt[i];
  yt.clear();


}


///////////////////////////////////////////
vtensor Net::predict_recurrent(vtensor tin) {
  vtensor out;

  // prepare data for unroll net
  vtensor xt;
  vtensor xtd;
  vtensor yt;

  vtensor toutr;
  vtensor tinr;
  vtensor tout;

  int inl;
  int outl;

  prepare_recurrent(tin,tout,inl,outl,xt,xtd,yt,tinr,toutr);

  build_rnet(inl,outl);

  out=rnet->predict(tinr);

  for(int i=0;i<tinr.size();i++) delete(tinr[i]);
  for(int i=0;i<toutr.size();i++) delete(toutr[i]);

  for(int i=0;i<xt.size();i++)
    delete xt[i];
  xt.clear();

  for(int i=0;i<yt.size();i++)
    delete yt[i];
  yt.clear();

  return out;
}

vtensor Net::predict(vtensor tin) {
  vtensor out;

  if (isrecurrent) {
    verboserec=0;
    return predict_recurrent(tin);
  }
  else {
    // cout<<"Predict "<<tin[0]->shape[0]<<" samples\n";

    setmode(TSMODE);

    forward(tin);

    for (int i = 0; i < lout.size(); i++) {
      collectTensor(lout[i],"output");
      out.push_back(lout[i]->output->clone());
    }
    return out;
  }

}


bool Net::compare_outputs(Net *net1, Net *net2, bool verbose, float atol, float rtol, bool equal_nan) {
    bool equivalent_nets = true;

    // Check if both layers are the same
    if(net1==net2){
        if(verbose){
            cout << "Both nets point to the same object"  << " [Net::compare_outputs]" << endl;
        }
        return true;
    }

    // Compare the number of layers
    if (net1->layers.size() != net2->layers.size()){
        if(verbose){
            cout << "Nets have a different number of layers"  << " [Net::compare_outputs]" << endl;
        }
        return false;
    }

    // Compare the output of each layer
    for(int i=0; i<net1->layers.size(); i++){
        // Collect Tensors from Device to CPU
        collectTensor(net1->layers[i], "output");
        collectTensor(net2->layers[i], "output");

        // Get tensors
        Tensor *output1 = net1->layers[i]->output;
        Tensor *output2 = net2->layers[i]->output;

        // Check if both outputs are equivalent
        bool equal = Tensor::equivalent(output1, output2, atol, rtol, equal_nan);
        if(equal) {
            if(verbose){
                cout << "[OKAY] The outputs from layers #" << i << " (" << net1->layers[i]->name << " AND " <<
                     net2->layers[i]->name << ") do match" << " [Net::compare_outputs]" << endl;
            }
        }else{
            if(verbose) {
                cout << "[FAIL] The outputs from layers #" << i << " (" << net1->layers[i]->name << " AND " <<
                net2->layers[i]->name << ") do not match" << " [Net::compare_outputs]" << endl;
            }
            equivalent_nets = false;
        }
    }
    return equivalent_nets;
}

bool Net::compare_params(Net *net1, Net *net2, bool verbose, float atol, float rtol, bool equal_nan) {
    bool equivalent_nets = true;

    // Check if both layers are the same
    if(net1==net2){
        if(verbose){
            cout << "Both nets point to the same object"  << " [Net::compare_params]" << endl;
        }
        return true;
    }

    // Compare the number of layers
    if (net1->layers.size() != net2->layers.size()){
        if(verbose){
            cout << "Nets have a different number of layers"  << " [Net::compare_params]" << endl;
        }
        return false;
    }


    // Compare the output of each layer
    for(int i=0; i<net1->layers.size(); i++){

        // Check if both layers have the same number of parameters
        if(net1->layers[i]->params.size() != net2->layers[i]->params.size()){
            if(verbose){
                cout << "The parameters in from layers #" << i << " (" << net1->layers[i]->name << " AND " <<
                     net2->layers[i]->name << ") do not match" << " [Net::compare_params]" << endl;
            }
            return false;
        }

        // Check params of this layer
        for(int j=0; j<net1->layers[j]->params.size(); j++){
            // Collect Tensors from Device to CPU
            collectTensor(net1->layers[i], "param", j);
            collectTensor(net2->layers[i], "param", j);

            Tensor* param1 = net1->layers[j]->params[j];
            Tensor* param2 = net2->layers[j]->params[j];

            // Check if both outputs are equivalent
            bool equal = Tensor::equivalent(param1, param2, atol, rtol, equal_nan);
            if(equal) {
                if(verbose){
                    cout << "[OKAY] The params #" << j << " from layers #" << i << " (" << net1->layers[i]->name << " AND " <<
                         net2->layers[i]->name << ") do match" << " [Net::compare_params]" << endl;
                }
            }else{
                if(verbose) {
                    cout << "[FAIL] The params #" << j << " from layers #" << i << " (" << net1->layers[i]->name << " AND " <<
                         net2->layers[i]->name << ") do not match" << " [Net::compare_params]" << endl;
                }
                equivalent_nets = false;
            }

        }

    }
    return equivalent_nets;
}

//////
