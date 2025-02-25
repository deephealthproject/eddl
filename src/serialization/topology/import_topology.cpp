#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream> 
#include <map>
#include <sstream>
#include <tuple>
#include "eddl/serialization/topology/import_topology.h"

using namespace eddl;

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

tuple<Layer *,string> create_layer(string params, string file_path){

    Layer *out_layer;
    string layer_name;
    string layer_type;
    string layer_role;

    std::stringstream ss_layer(params);
    
    getline( ss_layer, layer_type, ' ' );

    if (layer_type == "Input"){
        std::vector<int> dimensions;
        string input_dims;

        getline( ss_layer, input_dims, ' ' );

        dimensions = parse_vector(input_dims.substr(input_dims.find("{") + 1, input_dims.find("}")-1), dimensions, ',');
        dimensions.erase(dimensions.begin());

        getline( ss_layer, layer_name, ' ' );

        out_layer = Input(dimensions, layer_name);

        map_layers[layer_name] = out_layer;

    } else if (layer_type == "Dense"){

        //Dense input1 1024 dense1
        string parent_name;
        getline( ss_layer, parent_name, ' ' );
        layer parent_layer = map_layers[parent_name];

        string dense_dim;
        getline( ss_layer, dense_dim, ' ' );

        getline( ss_layer, layer_name, ' ' );

        out_layer = Dense(parent_layer, stoi(dense_dim), true, layer_name);
        map_layers[layer_name] = out_layer;

    } else if (layer_type == "LeakyRelu"){

        // LeakyRelu dense1 leaky_relu1
        string parent_name;
        getline( ss_layer, parent_name, ' ' );
        layer parent_layer = map_layers[parent_name];

        getline( ss_layer, layer_name, ' ' );

        out_layer = LeakyReLu(parent_layer, 0.01F, layer_name);
        map_layers[layer_name] = out_layer;

    } else if (layer_type == "Softmax"){
        // Softmax dense4 softmax4
        string parent_name;
        string aux;
        int axis;

        getline( ss_layer, parent_name, ' ' );
        layer parent_layer = map_layers[parent_name];

        getline( ss_layer, aux, ' ' );
        axis = stof(aux);

        getline( ss_layer, layer_name, ' ' );

        out_layer = Softmax(parent_layer, axis, layer_name);
        map_layers[layer_name] = out_layer;
 
    } else if (layer_type == "LConv"){
        layer parent_layer;
        int filters;
        std::vector<int> kernel;
        std::vector<int> strides;
        string padding;
        std::vector<int> pads;
        int groups;
        std::vector<int> dilation_rate;
        bool use_bias;
        int device;

        string aux;
        getline( ss_layer, aux, ' ' );
        
        parent_layer = map_layers[aux];


        getline( ss_layer, aux, ' ' );
        
        filters = stoi(aux);

        getline( ss_layer, aux, ' ' );
        
        kernel = parse_vector(aux, kernel, ',');

        getline( ss_layer, aux, ' ' );
        
        strides = parse_vector(aux, strides, ',');

        getline( ss_layer, padding, ' ' );

        getline( ss_layer, aux, ' ' );
        
        pads = parse_vector(aux, pads, ',');

        getline( ss_layer, aux, ' ' );
        
        groups = stoi(aux);

        getline( ss_layer, aux, ' ' );
        
        dilation_rate = parse_vector(aux, dilation_rate, ',');

        getline( ss_layer, aux, ' ' );
        
        use_bias = stoi(aux);

        getline( ss_layer, layer_name, ' ' );
        

        getline( ss_layer, aux, ' ' );
        
        device = stoi(aux);

        out_layer = new LConv(parent_layer, filters, kernel, strides, padding, pads, groups, dilation_rate, use_bias, layer_name, device, 0);
        map_layers[layer_name] = out_layer;

    } else if (layer_type == "Sigmoid"){
        layer parent_layer;
        string aux;
        getline( ss_layer, aux, ' ' );
        

        parent_layer = map_layers[aux];

        getline( ss_layer, layer_name, ' ' );
        

        out_layer = Sigmoid(parent_layer, layer_name);
        map_layers[layer_name] = out_layer;
    } else if(layer_type == "LMult"){
        string mult_type;
        string aux;
        getline( ss_layer, mult_type, ' ' );
        if(mult_type == "1"){
            layer parent_1;
            layer parent_2;

            getline( ss_layer, aux, ' ' );
            parent_1 = map_layers[aux];

            getline( ss_layer, aux, ' ' );
            parent_2 = map_layers[aux];

            getline( ss_layer, layer_name, ' ' );

            out_layer = Mult(parent_1, parent_2);
            out_layer->name = layer_name;
            map_layers[layer_name] = out_layer;
        } else if(mult_type == "2"){
            layer parent_layer;
            Tensor* const_tensor;

            getline( ss_layer, aux, ' ' );
            parent_layer = map_layers[aux];

            getline( ss_layer, aux, ' ' );
            const_tensor = Tensor::load(file_path + "/" + aux);


            getline( ss_layer, layer_name, ' ' );

            out_layer = new LMult(parent_layer, const_tensor, layer_name, DEV_CPU, 0);
            map_layers[layer_name] = out_layer;
        }

    } else if(layer_type == "LSelect"){
        layer parent_layer;
        string aux;
        vector<string> indices;
        int device;

        getline( ss_layer, aux, ' ' );
        
        parent_layer = map_layers[aux];

        getline( ss_layer, aux, ' ' );
        
        indices = parse_vector_str(aux, indices, ',');

        getline( ss_layer, layer_name, ' ' );
        

        getline( ss_layer, aux, ' ' );
        
        device = stoi(aux);

        out_layer = new LSelect(parent_layer, indices, layer_name, device, 0);
        map_layers[layer_name] = out_layer;

    } else if(layer_type == "Add"){
        string add_type;
        string aux;
        getline( ss_layer, add_type, ' ' );
        if(add_type == "1"){
            layer parent_1;
            layer parent_2;

            getline( ss_layer, aux, ' ' );
            parent_1 = map_layers[aux];

            getline( ss_layer, aux, ' ' );
            parent_2 = map_layers[aux];

            getline( ss_layer, layer_name, ' ' );

            out_layer = Add(parent_1, parent_2);
            out_layer->name = layer_name;
            map_layers[layer_name] = out_layer;
        }

    } else if(layer_type == "Concat"){
        string aux;
        vector<string> layer_str;
        vector<layer> layer_vec;
        unsigned int axis;

        getline( ss_layer, aux, ' ' );
        
        layer_str = parse_vector_str(aux.substr(aux.find("{") + 1, aux.find("}")-1), layer_str, ',');
        for(int i=0; i<layer_str.size(); i++){
            layer_vec.push_back(map_layers[layer_str[i]]);
        }

        getline( ss_layer, aux, ' ' );
        
        axis = stoi(aux);

        getline( ss_layer, layer_name, ' ' );
        

        out_layer = Concat(layer_vec, axis, layer_name);
        map_layers[layer_name] = out_layer;

    } else if (layer_type == "MaxPool"){
        string aux;
        layer parent_layer;
        vector<int> ksize;
        vector<int> strides;
        vector<int> padding;

        getline( ss_layer, aux, ' ' );
        
        parent_layer = map_layers[aux];

        getline( ss_layer, aux, ' ' );
        
        ksize = parse_vector(aux, ksize, ',');

        getline( ss_layer, aux, ' ' );
        
        strides = parse_vector(aux, strides, ',');

        getline( ss_layer, aux, ' ' );
        
        padding = parse_vector(aux, padding, ',');

        getline( ss_layer, layer_name, ' ' );
        

        out_layer = new LMaxPool(parent_layer, ksize, strides, padding, layer_name, DEV_CPU, 0);
        map_layers[layer_name] = out_layer;

    } else if (layer_type == "Resize"){
        string aux;
        layer parent_layer;
        vector<int> new_shape;
        bool reshape;
        string da_mode;
        float constant;
        string coord_mode;

        getline( ss_layer, aux, ' ' );
        
        parent_layer = map_layers[aux];

        getline( ss_layer, aux, ' ' );
        
        new_shape = parse_vector(aux, new_shape, ',');

        getline( ss_layer, aux, ' ' );
        
        reshape = stoi(aux);

        getline( ss_layer, aux, ' ' );
        
        da_mode = map_da_modes[stoi(aux)];

        getline( ss_layer, aux, ' ' );
        
        constant = stof(aux);

        getline( ss_layer, aux, ' ' );
        
        coord_mode = map_coord_trans[stoi(aux)];
        
        getline( ss_layer, layer_name, ' ' );
        

        out_layer = Resize(parent_layer, new_shape, reshape, da_mode, constant, coord_mode, layer_name);
        map_layers[layer_name] = out_layer;

    } else if (layer_type == "MergeAdd"){
        string aux;
        vector<string> layer_str;
        vector<layer> layer_vec;

        getline( ss_layer, aux, ' ' );
        
        layer_str = parse_vector_str(aux.substr(aux.find("{") + 1, aux.find("}")-1), layer_str, ',');
        for(int i=0; i<layer_str.size(); i++){
            layer_vec.push_back(map_layers[layer_str[i]]);
        }

        getline( ss_layer, layer_name, ' ' );
        

        out_layer = Add(layer_vec, layer_name);
        map_layers[layer_name] = out_layer;

    } else if (layer_type == "Reshape"){
        string aux;
        layer parent_layer;
        vector<int> shape_dim;

        getline( ss_layer, aux, ' ' );
        
        parent_layer = map_layers[aux];

        getline( ss_layer, aux, ' ' );
        
        shape_dim = parse_vector(aux, shape_dim, ',');
        shape_dim.erase(shape_dim.begin());

        getline( ss_layer, layer_name, ' ' );
        

        out_layer = Reshape(parent_layer, shape_dim, layer_name);
        map_layers[layer_name] = out_layer;

    } else if (layer_type == "Permute"){
        string aux;
        layer parent_layer;
        vector<int> axis;

        getline( ss_layer, aux, ' ' );
        
        parent_layer = map_layers[aux];

        getline( ss_layer, aux, ' ' );
        
        axis = parse_vector(aux, axis, ',');

        getline( ss_layer, layer_name, ' ' );
        

        out_layer = Permute(parent_layer, axis, layer_name);
        map_layers[layer_name] = out_layer;
    
    } else if (layer_type == "ConstOfTensor"){
        string aux;
        Tensor* constant_tensor;

        getline( ss_layer, aux, ' ' );
        
        constant_tensor = Tensor::load(file_path + "/" + aux);

        getline( ss_layer, layer_name, ' ' );
        

        out_layer = ConstOfTensor(constant_tensor, layer_name);
        map_layers[layer_name] = out_layer;

    } else if (layer_type == "Sub"){
        string aux;
        string sub_type;

        getline( ss_layer, sub_type, ' ' );
        if (sub_type == "1"){
            layer parent1;
            layer parent2;

            getline( ss_layer, aux, ' ' );
            
            parent1 = map_layers[aux];

            getline( ss_layer, aux, ' ' );
            
            parent2 = map_layers[aux];

            getline( ss_layer, layer_name, ' ' );
            

            out_layer = Sub(parent1, parent2);
            out_layer->name = layer_name;
            map_layers[layer_name] = out_layer;
        } else if (sub_type=="2"){
            Tensor* const_tensor;
            layer parent_layer;

            getline( ss_layer, aux, ' ' );
            
            const_tensor = Tensor::load(file_path + "/" + aux);

            getline( ss_layer, aux, ' ' );
            
            parent_layer = map_layers[aux];

            getline( ss_layer, layer_name, ' ' );
            

            out_layer = new LDiff(const_tensor, parent_layer, layer_name, DEV_CPU, 0);
            map_layers[layer_name] = out_layer;
        }
        
    } else if(layer_type == "Div"){
        string aux;
        string div_type;

        getline( ss_layer, div_type, ' ' );
        if (div_type == "1"){
            layer parent1;
            layer parent2;

            getline( ss_layer, aux, ' ' );
            
            parent1 = map_layers[aux];

            getline( ss_layer, aux, ' ' );
            
            parent2 = map_layers[aux];

            getline( ss_layer, layer_name, ' ' );
            

            out_layer = Div(parent1, parent2);
            out_layer->name = layer_name;
            map_layers[layer_name] = out_layer;
        } else if (div_type=="2"){
            layer parent_layer;
            float const_val;

            getline( ss_layer, aux, ' ' );
            
            parent_layer = map_layers[aux];
            
            getline( ss_layer, aux, ' ' );
            
            const_val = stof(aux);

            getline( ss_layer, layer_name, ' ' );
            

            out_layer = Div(parent_layer, const_val);
            out_layer->name = layer_name;
            map_layers[layer_name] = out_layer;

        } else if (div_type=="3"){
            float const_val;
            layer parent_layer;

            getline( ss_layer, aux, ' ' );
            
            const_val = stof(aux);

            getline( ss_layer, aux, ' ' );
            
            parent_layer = map_layers[aux];
            
            getline( ss_layer, layer_name, ' ' );
            

            out_layer = Div(const_val, parent_layer);
            out_layer->name = layer_name;
            map_layers[layer_name] = out_layer;

        }
    }
    
    getline( ss_layer, layer_role, ' ' );

    return make_tuple(out_layer,layer_role);
}

Net *import_net_topology(string path){
    string file_text;
    vector<Layer *> model_input;
    vector<Layer *> model_output;

    std::size_t botDirPos = path.find_last_of("/");
    std::string dir = path.substr(0, botDirPos);

    ifstream MyReadFile(path);

    
    while (getline (MyReadFile, file_text)) {
        tuple<Layer *,string> parsed_layer = create_layer(file_text, dir);
        if(std::get<1>(parsed_layer) == "input"){
            model_input.push_back(std::get<0>(parsed_layer));
        } else if(std::get<1>(parsed_layer) == "output"){
            model_output.push_back(std::get<0>(parsed_layer));
        }
    }

    MyReadFile.close();

    Net *imported_net = new Net(model_input, model_output);

    return imported_net;

}

vector<int> parse_vector(string str_vector, vector<int> out, char separator){
    std::stringstream ss(str_vector);
    while( ss.good() )
        {
            string substr;
            getline( ss, substr, separator);
            out.push_back(stoi(substr));
        }
        
    return out;
}

vector<string> parse_vector_str(string str_vector, vector<string> out, char separator){
    std::stringstream ss(str_vector);
    while( ss.good() )
        {
            string substr;
            getline( ss, substr, separator);
            out.push_back(substr);
        }
        
    return out;
}
