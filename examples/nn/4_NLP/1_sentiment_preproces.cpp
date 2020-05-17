/*
* EDDL Library - European Distributed Deep Learning Library.
* Version: 0.6
* copyright (c) 2020, Universidad Politécnica de Valencia (UPV), PRHLT Research Centre
* Date: April 2020
* Author: PRHLT Research Centre, UPV, (rparedes@prhlt.upv.es), (jon@prhlt.upv.es)
* All rights reserved
*/

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <iterator>
#include <stdio.h>
#include <string.h>

#include "eddl/apis/eddl.h"


using namespace eddl;

//////////////////////////////////
// aclImdb	preprocessing
//////////////////////////////////

/// IMDB files.txt have been preprocessed in this way:
/*
cat name.txt | tr '[:upper:]' '[:lower:]' |\
sed 's/[[:punct:]]/ /g' |\
sed 's/ br / /g' |\
sed -E 's/[0-9]+/#num /g'|\
sed -E 's/ +/ /g'|\
sed -e ':loop' -e 's/\([[:alpha:]]\)\1/\1/g' -e 't loop' > name.txt-preprocessed
*/

// utility comparator function to pass to the sort() module
bool sortByVal(const pair<string, int> &a,
               const pair<string, int> &b)
{
    return (a.second > b.second);
}

map<string,int> sort_dict(map<string,int> dict,int max_size) {
	// create a empty vector of pairs
	vector<pair<string, int>> vec;

	// copy key-value pairs from the map to the vector
  map<string, int> :: iterator it;
  for (it=dict.begin(); it!=dict.end(); it++)
  {
    vec.push_back(make_pair(it->first, it->second));
  }

	// // sort the vector by increasing order of its pair's second value
  sort(vec.begin(), vec.end(), sortByVal);

  if (max_size<0) max_size=vec.size();

  map<string,int> sdict;
  for(int i=0;i<max_size;i++) {
     sdict[vec[i].first]=i;
   }

  return sdict;
}


void build_dict(map<string,int> &dict,string fname)
{
  ifstream file;

  file.open(fname);
  string str;

  while (getline(file, str, ' ')) dict[str]++;

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
      }
      else {
        out.push_back(dict[str]+1); // +1 to leave 0 for padding
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
    while (getline(file, str, '\n')) {
      if (i%2==0) {
          build_dict(dict,str);
          cout<<"Dict size:"<<dict.size()<<"\r";
          numfiles++;
      }
      i++;
    }
    file.close();


    // 1000 most frequent words
    //map<string,int> sdict=sort_dict(dict,1000);

    // all words
    map<string,int> sdict=sort_dict(dict,-1);

    cout<<"Vocab:"<<sdict.size()<<" size\n"; //0 for padding
    ofstream of;
    of.open("vocab.txt");

    i=0;
    for(map<string,int>::iterator it = sdict.begin(); it != sdict.end(); ++it) {
      of << it->second<< ":"<<it->first<<endl;
      i++;
    }
    of.close();


    convert(sdict, "list_tr.txt", numfiles, length, "train");
    convert(sdict, "list_ts.txt", numfiles, length, "test");

  }
