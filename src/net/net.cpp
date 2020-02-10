/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.3
* copyright (c) 2019, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: October 2019
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/


#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <chrono>
#include <thread>
#include "net.h"
#include <pthread.h>
#include "../utils.h"
#include "../random.h"

#include "../layers/core/layer_core.h"



ostream &operator<<(ostream &os, const vector<int> shape) {
    int i;
    os << "(";
    for (i = 0; i < shape.size() - 1; ++i) {
        os << shape[i];
        os << "x";
    }
    os << shape[i] << ")";

    return os;
}

/////////////////////////////////////////
int isIn(Layer *l, vlayer vl, int &ind) {
    for (int i = 0; i < vl.size(); i++)
        if (l == vl[i]) {
            ind = i;
            return 1;
        }

    return 0;
}

/////////////////////////////////////////
int isInorig(Layer *l, vlayer vl, int &ind) {
    for (int i = 0; i < vl.size(); i++)
        if (l == vl[i]->orig) {
            ind = i;
            return 1;
        }

    return 0;
}


////////////////////////////////////
///// NET CLASS
////////////////////////////////////

Net::Net(vlayer in, vlayer out) {
    // Set input/outlayer
    lin = in;
    lout = out;
    batch_size=1;
    // Default optimizer
    optimizer = nullptr;
    name="model";
    tr_batches=0;
    flog_tr=nullptr;
    flog_ts=nullptr;
    unsigned int verbosity_level = 0;

    // Walk through the pointers of all layers, to get a plain
    // vector with all the layers
    for (int i = 0; i < lin.size(); i++) {
        walk(lin[i]);
    }

    for (int i = 0; i < lout.size(); i++) {
        walk_back(lout[i]);
    }

    for (int i = 0; i < lout.size(); i++) {
        total_loss.push_back(0.0);
        total_metric.push_back(0.0);
        fiterr.push_back(0.0);
        fiterr.push_back(0.0);
    }

    build_randn_table();
}

Net::~Net()
{
  for(int i=0;i<snets.size();i++)
    for(int j=0;j<snets[i]->layers.size();j++) {
      delete snets[i]->layers[j];
    }

  for (int i = 0; i < lout.size(); i++) {
    delete losses[i];
    delete metrics[i];
  }
  delete optimizer;

}

/////////////////////////////////////////
int Net::inNet(Layer *l) {
    // Check if the layer l is in the network
    for (int i = 0; i < layers.size(); i++)
        if (l == layers[i]) return 1;
    return 0;
}


/////////////////////////////////////////
void Net::walk(Layer *l) {
    // If this layer is not in the network, add it, as well as all its children (recursively)
    if (!inNet(l)) {
      if (l->orig!=nullptr) l->net=l->orig->net;
      else l->net=this;

      layers.push_back(l);
      for (int i = 0; i < l->child.size(); i++)
          walk(l->child[i]);
    }
}
/////////////////////////////////////////
void Net::walk_back(Layer *l) {
    // If this layer is not in the network, add it, as well as all its children (recursively)

    if (!inNet(l)) {
      //cout<<l->name<<"  BACK\n";
      if (l->orig!=nullptr) l->net=l->orig->net;
      else l->net=this;

      layers.push_back(l);
    }
    for (int i = 0; i < l->parent.size(); i++)
        walk_back(l->parent[i]);

}


/////////////////////////////////////////
string Net::summary() {
    std::stringstream ss;
    ss << "---------------------------------------------" << endl;
    for (auto & vft : vfts) {
        // Get input/output shapes
        vector<int> ishape(vft->input->shape);
        vector<int> oshape(vft->output->shape);

        // Remove batch (if has)
        if (ishape.size() > 1) { ishape.erase(ishape.begin(), ishape.begin() + 1); }
        if (oshape.size() > 1) { oshape.erase(oshape.begin(), oshape.begin() + 1); }

        // Prepare strings to print
        string istr = "(" + printVector(ishape) + ")";
        string ostr = "(" + printVector(oshape) + ")";

        ss << setw(30) << left << vft->name << "|  ";
        ss << setw(10) << left << istr;
        ss << setw(8) << left << "=>";
        ss << setw(10) << left << ostr;
        ss << endl;
    }
    ss << "---------------------------------------------" << endl;

    return ss.str();
}

void Net::plot(string fname,string mode) {
    ofstream out("tmp.dot");
    int ind;
    string type = fname.substr(fname.find('.') + 1);
    string cmd;


    out << "digraph Model {\n";
    out << "rankdir="<<mode<<";\n";

    // plot layers
    for (int i = 0; i != layers.size(); i++)
        if ((!isIn(layers[i], lin, ind)) && (!isIn(layers[i], lout, ind)))
            out << layers[i]->plot(0) << "\n";

    // Input Layers
    for (int i = 0; i != lin.size(); i++)
        out << lin[i]->plot(1) << "\n";

    // Output Layers
    for (int i = 0; i != lout.size(); i++)
        out << lout[i]->plot(1) << "\n";

    //plot links
    for (int i = 0; i != layers.size(); i++)
        for (int j = 0; j < layers[i]->child.size(); j++)
              out << layers[i]->name << "->" << layers[i]->child[j]->name << "\n";

    out << "}\n";

    out.close();

    cmd = "dot -T " + type + " ./tmp.dot >" + "./" + fname;

    system(cmd.c_str());

}

/////////////////////////////////////////
void Net::setlogfile(string fname)
{
  string str=fname+"_tr.log";
  string sts=fname+"_ts.log";

  flog_tr=fopen(str.c_str(),"wt");
  if (flog_tr==nullptr) msg("error creating tr log file","Net.setlogfile");

  flog_ts=fopen(sts.c_str(),"wt");
  if (flog_ts==nullptr) msg("error creating ts log file","Net.setlogfile");
}


void Net::save(const string& filename, string format){
    // Open file stream
    std::ofstream ofs(filename, std::ios::out | std::ios::binary);

    // Copy from CS devices to layers
    if (snets[0]->dev!=DEV_CPU) 
      sync_weights();


    for (int i = 0; i != layers.size(); i++){
        layers[i]->save(ofs, format);
    }

    // Close file stream
    ofs.close();
}

void Net::load(const string& filename, string format){
    // Open file stream
    std::ifstream ifs(filename, std::ios::in | std::ios::binary);

    for (int i = 0; i != layers.size(); i++){
        layers[i]->load(ifs, format);
    }


    // Copy to CS devices layers
    if (snets[0]->dev!=DEV_CPU) {
      for(int i=0; i!=snets.size(); i++)
        for(int j=0;j<layers.size();j++)
          layers[j]->copy(snets[i]->layers[j]);
    }

    // Close file stream
    ifs.close();
}

void Net::reset_accumulated_gradients(){
	for(Layer* l : layers){
		l->reset_accumulated_gradients();
	}
}

void Net::apply_accumulated_gradients(){
	for(Layer * l : layers){
		l->apply_accumulated_gradients();
	}
}



//////
