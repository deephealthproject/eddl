/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.5
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <iterator>
#include <stdio.h>
#include <string.h>

#include "eddl/apis/eddl.h"


using namespace eddl;

//////////////////////////////////
// aclImdb	preprocessing
//////////////////////////////////

void build_dict(map<string,int> &dict,string fname)
{
  ifstream file;

  file.open(fname);
  string str;

  int i=dict.size();
  while (getline(file, str, ' ')) {
      if (!dict.count(str)) {
        dict[str]=i;
        i++;
      }
  }

  file.close();
}

int convert(map<string,int>  &dict,vector<int> &out, string fname, int length)
{
  ifstream file;
  vector<string> words;

  file.open(fname);
  string str;

  int unk=0;
  int i=0;
  while ((getline(file, str, ' '))&&(i<length)) {
      if (!dict.count(str)) {
        unk++;
        out.push_back(0); // 0 is for unknown
        i++;
      }
      else {
         out.push_back(dict[str]);
         i++;
      }

  }

  // padding if needed
  for(int j=i;j<length;j++) out.push_back(0); // 0 for padding

  return unk;

}

void convert(map<string,int>  &dict, string list_fname, int numlines, int length, string tensor_name)
{
  int i;
  string str;
  ifstream file;

  cout<<"Converting "<<list_fname<<endl;
  file.open(list_fname);

  Tensor* xtr=new Tensor({numlines,length});
  Tensor* ytr=new Tensor({numlines,2});
  ytr->fill_(0.0);

  float *xptr=xtr->ptr;
  float *yptr=ytr->ptr;

  int unk=0;
  i=0;
  while (getline(file, str, '\n')) {
     if (i%2==0) {
        vector<int> out;
        cout<<"FILE:"<<str<<"\r";
        unk+=convert(dict,out,str,length);
        for(int j=0;j<out.size();j++){
          *xptr=out[j];
          xptr++;
        }
      }
      else {
        int c=stoi(str);
        if (c==0) *yptr=1;
        else {
          yptr++;
          *yptr=1;
          yptr++;
        }
      }
    i++;
  }

  cout<<endl;

  cout<<unk<<" unknown words in "<<numlines<<" files --> "<<(float)unk/(float)numlines<<" per file\n";

  xtr->save("x"+tensor_name+".bin");
  ytr->save("y"+tensor_name+".bin");

  delete xtr;
  delete ytr;



}


int main(int argc, char **argv) {
    // build dict
    map<string,int> dict;
    int length=100;
    ifstream file;
    string str,str2;


    file.open("list_tr.txt");
    int i=0;


    int numfiles=0;
    // special word for padding and unknown
    dict["UNK"]=0; // 0 allows dropout in the input
    while (getline(file, str, '\n')) {
      if (i%2==0) {
          build_dict(dict,str);
          cout<<"Dict size:"<<dict.size()<<"\r";
          numfiles++;
      }
      i++;
    }
    file.close();

    cout<<"Vocab:"<<dict.size()<<" size\n"; //0 for padding
    ofstream of;
    of.open("vocab.txt");

    i=0;
    for(map<string,int>::iterator it = dict.begin(); it != dict.end(); ++it) {
      of << it->second<< ":"<<it->first<<endl;
      i++;
    }
    of.close();


    convert(dict, "list_tr.txt", numfiles, length, "train");
    convert(dict, "list_ts.txt", numfiles, length, "test");

  }
