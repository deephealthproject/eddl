/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.1
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <thread>
#include "net.h"
#include <pthread.h>
#include "../utils.h"
#include "../random.h"
#include "../layers/core/layer_core.h"

#define VERBOSE 0

using namespace std;
using namespace std::chrono;

/////////////////////////////////////////////////////////////////
///// NET LEVEL FUNCS
/////////////////////////////////////////////////////////////////

void Net::forward() {
    for (int i = 0; i < vfts.size(); i++) {
        vfts[i]->forward();
        if (VERBOSE) {
          cout << vfts[i]->name << "\n";
          fprintf(stdout, "  %s In:%f\n", vfts[i]->name.c_str(), vfts[i]->input->sum());
          fprintf(stdout, "  %s Out:%f\n", vfts[i]->name.c_str(), vfts[i]->output->sum());
        }
    }
}

void Net::backward() {
    for (int i = 0; i < vbts.size(); i++) {
        vbts[i]->backward();
        if (VERBOSE) cout<<"BACK: "<<vbts[i]->name<<"delta:"<<vbts[i]->delta->sum()<<"\n";
      }

}

void Net::delta() {
    for (int i = 0; i < lout.size(); i++)
        losses[i]->delta(lout[i]->target, lout[i]->output, lout[i]->delta);

}

void Net::calcloss() {
    int p = 0;
    for (int i = 0; i < lout.size(); i++, p += 2) {
        // loss value
        fiterr[p] = losses[i]->value(lout[i]->target, lout[i]->output);
        // metric value
        fiterr[p + 1] = metrics[i]->value(lout[i]->target, lout[i]->output);
    }
}

void Net::applygrads() {
    optimizer->applygrads(batch_size);
}


void Net::reset_loss()
{
  // Reset errors
  int p=0;
  for (int j = 0; j < lout.size(); j++,p+=2){
      total_loss[j] = 0.0;
      total_metric[j] = 0.0;
      fiterr[p] = fiterr[p + 1] = 0.0;
  }
  inferenced_samples=0;
}

void Net::print_loss(int b)
{
  int p = 0;

  for (int k = 0; k < lout.size(); k++, p += 2) {
      total_loss[k] += fiterr[p];  // loss
      total_metric[k] += fiterr[p + 1];  // metric
      fiterr[p] = fiterr[p + 1] = 0.0;

      fprintf(stdout, "%s(%s=%1.3f,%s=%1.3f) ", lout[k]->name.c_str(),
              losses[k]->name.c_str(), total_loss[k] / inferenced_samples,
              metrics[k]->name.c_str(), total_metric[k] / inferenced_samples);

      if ((flog_tr!=nullptr)&&(trmode))
        fprintf(flog_tr, "%s %1.3f %s %1.3f ", losses[k]->name.c_str(), total_loss[k] / inferenced_samples,
                metrics[k]->name.c_str(), total_metric[k] / inferenced_samples);

      if ((flog_ts!=nullptr)&&(!trmode))
        fprintf(flog_ts, "%s %1.3f %s %1.3f ", losses[k]->name.c_str(), total_loss[k] / inferenced_samples,
                metrics[k]->name.c_str(), total_metric[k] / inferenced_samples);

  }
  fflush(stdout);

  if ((flog_tr!=nullptr)&&(trmode)) {
    fprintf(flog_tr, "\n");
    fflush(flog_tr);
  }

  if ((flog_ts!=nullptr)&&(!trmode)) {
    fprintf(flog_ts, "\n");
    fflush(flog_ts);
  }

}








//////
