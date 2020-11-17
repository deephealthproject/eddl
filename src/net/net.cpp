/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.8
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: November 2020
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
#include "eddl/net/net.h"
#include "eddl/utils.h"
#include "eddl/random.h"

#include "eddl/layers/core/layer_core.h"



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
    for (int i = 0; i < vl.size(); i++) {
        if (l == vl[i]->orig) {
            ind = i;
            return 1;
        }
  }
    return 0;
}


////////////////////////////////////
///// NET CLASS
////////////////////////////////////

Net::Net() {
    batch_size=1;
    optimizer = nullptr;
    cs = nullptr;
    name="model";
    tr_batches=0;
    flog_tr=nullptr;
    flog_ts=nullptr;
    rnet=nullptr;
    isbuild=false;
    isdecoder=false;
    isencoder=false;
    isrecurrent=false;
    decsize=1;
}

Net::Net(vlayer in, vlayer out):Net() {
    // Set input/outlayer
    lin = in;
    lout = out;

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

Net::Net(vector <Net *> vnets):Net()
{
  int vsize=vnets.size();
  int ind;

  if (vsize<2) {
    msg("Use at least two networks to concatenate","Net::Net");
  }

  for(int i=0;i<vnets[0]->lin.size();i++)
    lin.push_back(vnets[0]->lin[i]);

  for(int i=0;i<vnets[0]->layers.size();i++)
    layers.push_back(vnets[0]->layers[i]);

///
  for(int i=0;i<vnets.size()-1;i++) {
    if (vnets[i]->lout.size()!=vnets[i+1]->lin.size())
      msg("out layers does not match in layers","Net");
    for(int j=0;j<vnets[i+1]->layers.size();j++)
        layers.push_back(vnets[i+1]->layers[j]);

    for(int j=0;j<vnets[i]->lout.size();j++) {
        vnets[i]->lout[j]->addchild(vnets[i+1]->lin[j]);
        vnets[i+1]->lin[j]->addparent(vnets[i]->lout[j]);
    }
  }


  for(int i=0;i<vnets[vsize-1]->lout.size();i++)
    lout.push_back(vnets[vsize-1]->lout[i]);


  for (int i = 0; i < lout.size(); i++) {
    total_loss.push_back(0.0);
    total_metric.push_back(0.0);
    fiterr.push_back(0.0);
    fiterr.push_back(0.0);
  }

  isrecurrent=false;
  rnet=nullptr;

  for(int i=0;i<vnets.size();i++)
    mnets.push_back(vnets[i]);

  build_randn_table();


}




Net::~Net(){

    if (mnets.size()) return;


    // IF CPU : net = snets[0]   snets.push_back(this)

   // IF GPU: net , snets[0]= clone en GPU

    for(int i=0;i<snets.size();i++){
      for(int j=0;j<snets[i]->layers.size();j++) {
        if (snets[i]->layers[j]!=nullptr) {
          delete snets[i]->layers[j];
          snets[i]->layers[j] = nullptr;
        }
      }
    }

    //TODO:
    /*
    if (GPU){
      for(int j=0;j<layers.size();j++)
         delete layers[j];
    }
    */

    if (rnet!=nullptr) {delete rnet; rnet = nullptr;}
}


/////////////////////////////////////////
int Net::inNet(Layer *l) {
    // Check if the layer l is in the network
    for (int i = 0; i < layers.size(); i++){
        if (l == layers[i]) return 1;
    }
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
    ss << "-------------------------------------------------------------------------------" << endl;
    ss << name << endl;
    ss << "-------------------------------------------------------------------------------" << endl;

    int tot_size=0;
    for (auto & l : vfts) {
        // Get input/output shapes
        vector<int> ishape(l->input->shape);
        vector<int> oshape(l->output->shape);

        // Remove batch (if has)
        if (ishape.size() > 1) { ishape.erase(ishape.begin(), ishape.begin() + 1); }
        if (oshape.size() > 1) { oshape.erase(oshape.begin(), oshape.begin() + 1); }

        // Prepare strings to print
        string istr = "(" + printVector(ishape) + ")";
        string ostr = "(" + printVector(oshape) + ")";

        int size=0;
        for(auto &p:l->params)
          size+=p->size;
        tot_size+=size;

        ss << setw(20) << left << l->name << "|  ";
        ss << setw(20) << left << istr;
        ss << setw(5) << left << "=>";
        ss << setw(20) << left << ostr;
        ss << setw(10) << left << size;
        ss << endl;
    }
    ss << "-------------------------------------------------------------------------------" << endl;
    ss << "Params: "<<tot_size<<endl;
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
    for (int i = 0; i != layers.size(); i++) {
       if (layers[i]->isrecurrent)
         out << layers[i]->name << "->" << layers[i]->name << "\n";

       for (int j = 0; j < layers[i]->child.size(); j++)
        out << layers[i]->name << "->" << layers[i]->child[j]->name << "\n";
    }

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
    if (!ifs.good()){
        throw std::runtime_error(std::string("File not found. Check the file name and try again (Net::load)"));
    }

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
