#ifndef EDDL_TOPOLOGY_H
#define EDDL_TOPOLOGY_H

#include <map>
#include "eddl/layers/layer.h"
#include "eddl/net/net.h"
#include "eddl/apis/eddl.h"

Net *import_net_topology(string path);

tuple<Layer *,string> create_layer(string params, string path);

vector<string> parse_vector_str(string str_vector, vector<string> out, char separator);

vector<int> parse_vector(string str_vector, vector<int> out, char separator);

#endif