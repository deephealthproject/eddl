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
#include <map>
#include <string>
#include <iterator>
#include <stdio.h>
#include <string.h>

#include "eddl/apis/eddl.h"
#include "eddl/apis/eddlT.h"

using namespace eddl;

//////////////////////////////////
// aclImdb	preprocessing
//////////////////////////////////

void build_dict(map<string,int> &dict,string fname)
{
  ifstream file;
  char buffer[256];

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

void convert(map<string,int>  &dict,vector<int> &out, string fname, int length)
{
  ifstream file;
  vector<string> words;

  file.open(fname);
  string str;

  int i=0;
  while ((getline(file, str, ' '))&&(i<length)) {
      if (!dict.count(str)) {
        cout<<"Out of vocabulary: "<<str<<endl;
        exit(1);
      }
      else {
         out.push_back(dict[str]);
      }
      i++;
  }

  for(int j=i;j<length;j++) out.push_back(0);

}

int main(int argc, char **argv) {
    // build dict
    map<string,int> dict;
    int length=80;
    ifstream file;
    string str,str2;


    file.open("list_tr.txt");
    int i=0;

    int c=0;
    while (getline(file, str, '\n')) {
      if (i%2==0) {
          build_dict(dict,str);
          cout<<"Dict size:"<<dict.size()<<"\r";
          c++;
      }
      i++;
    }
    file.close();

    cout<<"Vocab:"<<dict.size()<<" size\n";
    ofstream of;
    of.open("vocab.txt");

    i=0;
    for(map<string,int>::iterator it = dict.begin(); it != dict.end(); ++it) {
      of << it->second<< ":"<<it->first<<endl;
      i++;
    }
    of.close();


    cout<<"Converting...\n";
    file.open("list_tr.txt");

    tensor xtr=new Tensor({c,length});
    tensor ytr=new Tensor({c,2});
    ytr->fill_(0.0);

    float *xptr=xtr->ptr;
    float *yptr=ytr->ptr;
    i=0;
    while (getline(file, str, '\n')) {
       if (i%2==0) {
          vector<int> out;
          cout<<"FILE:"<<str<<"\r";
          convert(dict,out,str,length);
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

    xtr->save("xtrain.bin");
    ytr->save("ytrain.bin");


}

























////////
