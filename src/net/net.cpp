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
#include <iomanip>
#include <fstream>
#include <stdexcept>
#include <string>
#include <chrono>
#include "eddl/net/net.h"
#include "eddl/utils.h"
#include "eddl/random.h"

#include "eddl/layers/core/layer_core.h"



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
    has_to_close_flog_tr = false;
    flog_ts=nullptr;
    has_to_close_flog_ts = false;
    trmode = TRMODE;
    rnet=nullptr;
    isbuild=false;
    isdecoder=false;
    isencoder=false;
    isrecurrent=false;
    isresized=false;
    decoder_teacher_training=true;
    decsize=1;
    do_compserv_delete = true;
    do_optimizer_delete = true;
}

Net::Net(vlayer in, vlayer out):Net() {
    // Set input/outlayer
    //lin = in;
    //lout = out;
    for (auto l : in) lin.push_back(l);
    for (auto l : out) lout.push_back(l);

    // Walk through the pointers of all layers, to get a plain
    // vector with all the layers
    for (int i = 0; i < lin.size(); i++) {
        walk(lin[i],lout);
    }

    for (int i = 0; i < lout.size(); i++) {
        walk_back(lout[i]);
    }

    for(auto l:layersf) layers.push_back(l);
    for(auto l:layersb) if (!inNet(l)) layers.push_back(l);

    for (int i = 0; i < lout.size(); i++) {
        total_loss.push_back(0.0);
        total_metric.push_back(0.0);
        fiterr.push_back(0.0);
        fiterr.push_back(0.0);
    }


    build_randn_table();

    // It is important that layers vector keep the forward sort
    fts();
    while (layers.size()) layers.pop_back();
    for(auto l: vfts ) {
        l->increase_reference_counter();
        layers.push_back(l);
    }
    while (vfts.size()) vfts.pop_back();
}


Net::~Net(){
    // IF CPU : net = snets[0]
    // IF GPU: net , snets[0]= clone on GPU

    if (this->has_to_close_flog_tr && this->flog_tr != nullptr) {
        fclose(this->flog_tr);
        this->flog_tr = nullptr;
        this->has_to_close_flog_tr = false;
    }
    if (this->has_to_close_flog_ts && this->flog_ts != nullptr) {
        fclose(this->flog_ts);
        this->flog_ts = nullptr;
        this->has_to_close_flog_ts = false;
    }

    // not necessary in theory, but valgrid reports "still reachable blocks"
    this->total_loss.clear();
    this->total_metric.clear();
    this->fiterr.clear();
    this->lin.clear();
    this->lout.clear();

    // Clean inputs
    for(int i=0; i<Xs->size(); i++) {
        for(int j=0;j<Xs[i].size();j++)
            delete Xs[i][j];
        Xs[i].clear();
    }

    // Clean targets
    for(int i=0; i<Ys->size(); i++) {
        for(int j=0;j<Ys[i].size();j++)
            delete Ys[i][j];
        Ys[i].clear();
    }

    // delete optimizer
    for(int i=0;i<snets.size();i++){
        if (snets[i]->optimizer!=nullptr && snets[i]->do_optimizer_delete){
            delete snets[i]->optimizer;
        }
    }

    if (snets.size() == 0 || snets[0] != this){
        if (this->optimizer != nullptr && this->do_optimizer_delete){
            delete this->optimizer;
        }
    }

    // clean metrics and losses
    for (auto m : this->metrics) delete m;
    this->metrics.clear();
    for (auto l : this->losses) delete l;
    this->losses.clear();

    // clean device mem
    for(int i=0;i<snets.size();i++){
        for(int j=0;j<snets[i]->layers.size();j++) {
            if (snets[i]->layers[j]!=nullptr) {
                if (snets[i]->layers[j]->decrease_and_get_reference_counter() == 0) {
                    delete snets[i]->layers[j];
                }
                snets[i]->layers[j] = nullptr;
            }
        }
    }

    // net running on device != CPU
    // clean also CPU mem
    if (snets.size() == 0 || snets[0]!=this){
        for(int j=0;j<layers.size();j++) {
            if (layers[j]->decrease_and_get_reference_counter() == 0) {
                delete layers[j];
            }
            layers[j] = nullptr;
        }
    }

    if (rnet!=nullptr) { delete rnet; rnet = nullptr;}

    if (this->do_compserv_delete && this->cs != nullptr) {
        delete this->cs;
        this->cs = nullptr;
    }
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
int Net::inNetF(Layer *l) {
    // Check if the layer l is in the network
    for (int i = 0; i < layersf.size(); i++){
        if (l == layersf[i]) return 1;
    }
    return 0;
}
/////////////////////////////////////////
int Net::inNetB(Layer *l) {
    // Check if the layer l is in the network
    for (int i = 0; i < layersb.size(); i++){
        if (l == layersb[i]) return 1;
    }
    return 0;
}
/////////////////////////////////////////
void Net::walk(Layer *l,vlayer lout) {
    int ind;

    if (!inNetF(l)) {
        l->net=this;
        layersf.push_back(l);
    }
    else return;

    if (isIn(l,lout,ind)) return; // cut recursivity for out layers

    for (int i = 0; i < l->child.size(); i++)
       walk(l->child[i],lout);

}

/////////////////////////////////////////
void Net::walk_back(Layer *l) {

    if (!inNetB(l)) {
        layersb.push_back(l);
        l->net=this;
    }
    else return;

    for (int i = 0; i < l->parent.size(); i++)
        walk_back(l->parent[i]);
}


/////////////////////////////////////////
string Net::summary(bool print_stdout) {
    // Force topological order sort (if vfts is empty
    if(!this->layers.empty() && this->vfts.empty()){
        this->fts();
    }

    // Print stuff
    std::stringstream ss;
    ss << "-------------------------------------------------------------------------------" << std::endl;
    ss << name << std::endl;
    ss << "-------------------------------------------------------------------------------" << std::endl;

    int maxl=0;
    for (auto & l : vfts)
      if (l->name.length() > maxl) maxl = (int)l->name.length();

    int trainable_params_acc=0;
    int nontrainable_params_acc=0;
    for (auto &l : vfts) {
        // Get input/output shapes
        vector<int> ishape(l->input->shape);
        vector<int> oshape(l->output->shape);

        // Remove batch (if has)
        if (ishape.size() > 1) { ishape.erase(ishape.begin(), ishape.begin() + 1); }
        if (oshape.size() > 1) { oshape.erase(oshape.begin(), oshape.begin() + 1); }

        // Prepare strings to print
        string istr = "(" + printVector(ishape) + ")";
        string ostr = "(" + printVector(oshape) + ")";

        int tr_params = 0;
        int no_tr_params = 0;
        for(int j=0; j< l->params.size(); j++){

            // Check if it is frozen
            if (l->trainable){

                // Check layer type
                if(l->get_name_id() == "batchnorm"){
                    // 2 params: NTR(mean, variance)
                    // 4 params: TR(bn_g, bn_b), NTR(mean, variance)
                    if((l->params.size()==2) || (l->params.size()==4 && j >= 2)){
                        no_tr_params += (int)l->params[j]->size;
                    }else{
                        tr_params += (int)l->params[j]->size;
                    }

                }else{ // General case
                    tr_params += (int)l->params[j]->size;
                }

            }else{  // Frozen layers
                no_tr_params += (int)l->params[j]->size;
            }
        }
        trainable_params_acc += tr_params;
        nontrainable_params_acc += no_tr_params;

        ss << setw(maxl) << left << l->name << "|  ";
        ss << setw(20) << left << istr;
        ss << setw(5) << left << "=>";
        ss << setw(20) << left << ostr;
        ss << setw(10) << left << tr_params+no_tr_params;
        ss << endl;
    }
    ss << "-------------------------------------------------------------------------------" << std::endl;
    ss << "Total params: " << trainable_params_acc+nontrainable_params_acc << std::endl;
    ss << "Trainable params: " << trainable_params_acc << std::endl;
    ss << "Non-trainable params: " << nontrainable_params_acc << std::endl;

    // Print to the standard output
    if(print_stdout){
        std::cout << ss.str() << std::endl;
    }

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

    out << "}" << std::endl;
    out.close();

    cmd = "dot -T " + type + " ./tmp.dot >" + "./" + fname;

    int rc = system(cmd.c_str());
    if (rc != EXIT_SUCCESS) {
        std::cerr << "[PLOT] Unable to run the following command:" << std::endl;
        std::cerr << "\t=> " << cmd << std::endl;
    }
}

/////////////////////////////////////////
void Net::setlogfile(string fname)
{
    string str=fname+"_tr.log";
    string sts=fname+"_ts.log";

    this->flog_tr = fopen(str.c_str(),"wt");
    if (this->flog_tr == nullptr) {
        msg("error creating tr log file","Net.setlogfile");
    } else {
        this->has_to_close_flog_tr = true;
    }

    this->flog_ts = fopen(sts.c_str(),"wt");
    if (this->flog_ts == nullptr) {
        msg("error creating ts log file","Net.setlogfile");
    } else {
        this->has_to_close_flog_ts = true;
    }
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
