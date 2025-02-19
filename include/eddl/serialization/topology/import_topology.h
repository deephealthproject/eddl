#include <map>
#include "eddl/layers/layer.h"
#include "eddl/net/net.h"
#include "eddl/apis/eddl.h"

Net *import_net_topology(string path);

tuple<Layer *,string> create_layer(string params);

vector<string> parse_vector_str(string str_vector, vector<string> out, char separator);

vector<int> parse_vector(string str_vector, vector<int> out, char separator);

map<int, string> map_da_modes = {
    {0, "constant"},
    {1, "reflect"},
    {2, "nearest"},
    {3, "mirror"},
    {4, "wrap"},
    {5, "original"}
};

map<int, string> map_coord_trans = {
    {0, "half_pixel"},
    {1, "pytorch_half_pixel"},
    {2, "align_corners"},
    {3, "asymmetric"},
    {4, "tf_crop_and_resize"},
};

map<string, Layer *> map_layers;