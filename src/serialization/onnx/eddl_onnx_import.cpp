#include "eddl/serialization/onnx/eddl_onnx.h"
#include <queue>
#include <fstream>
#include "eddl/layers/core/layer_core.h"
#include "eddl/layers/conv/layer_conv.h"
#include "eddl/layers/normalization/layer_normalization.h"
#include "eddl/layers/pool/layer_pool.h"
#include <map>
#include <set>
#include <algorithm>

#define NEW_FROM_VECTOR_PTR(v) (copy((v)->begin(), (v)->end(), new float[(v)->size()]) - (v)->size())
std::vector<int> vf2vi(const std::vector<float>& vf)
{
    std::vector<int> vi;
    vi.reserve(vf.size());
    for (const auto& x : vf) {
        vi.emplace_back(static_cast<int>(x));
    }
    return vi;
}

#if defined(cPROTO)
#include "onnx.pb.h"
#endif

#if defined(cPROTO)
	Net* build_net_onnx(onnx::ModelProto model, int mem);
#endif

#if defined(cPROTO)
	map<string, vector<Tensor*> > get_tensors_from_onnx(onnx::ModelProto model);
#endif


using namespace std;

#ifdef cPROTO 
	enum ONNX_LAYERS{
		//TODO Comment in which section belongs each layer
		BATCHNORM,			// implemented
		CONV,				// implemented
		DENSE,				// implemented
		DROP,               // implemented
		//EMBEDDING,  		// Onnx doesn't support this
		RESHAPE,            // implemented
		FLATTEN,            // implemented
		TRANSPOSE,          // implementing
		TRANSPOSED_CONV,	// not implemented in eddl
		UPSAMPLING,         // deprecated in ONNX, but works for EDDL
		MAXPOOL,			// implemented
		AVGPOOL,            // needs testing
		GLOBAVGPOOL,        // implemented
		GLOBMAXPOOL,        // implemented
		PERMUTE,            // implemented
		// Activation layers
		RELU, 				// implemented
		SOFTMAX,			// implemented
		SIGMOID,            // implemented
		HARD_SIGMOID,       // implemented
		TANH,               // implemented
		LINEAR,             // implemented
		EXPONENTIAL,        // implemented
		LEAKY_RELU,         // implemented
		THRESHOLDED_RELU,   // implemented
		ELU,                // implemented
		SELU,               // implemented
		SOFTPLUS,           // implemented
		SOFTSIGN,           // implemented
		// Merge layers
		CONCAT,				// implemented
		ADD,                // implemented
		AVERAGE,            // implemented
		MAT_MUL,            // implemented
		MAX,				// implemented
		MIN,                // implemented
		SUB,                 // implemented
		CONSTANT            // Not a layer, but needed for detecting this type of node
		


	};


	int verbose=0;
	//Gets the initializers from the onnx layer graph
	vector<onnx::TensorProto> get_initializers(onnx::GraphProto graph) {
		vector<onnx::TensorProto> initializers;
		for(int i = 0; i < graph.initializer_size(); i++) initializers.push_back(graph.initializer(i));
		return initializers;
	}

	//Creates a map containing the name of the node as a key, and the value is a vector containing the nodes that have this node as a input.
	map<string, vector<onnx::NodeProto*>> initialize_input_node_map(vector<onnx::NodeProto> &nodes){
		map<string, vector<onnx::NodeProto*>> input_node_map;
		for( int j = 0; j < nodes.size(); j++){
			onnx::NodeProto node = nodes[j];
			for ( int i = 0; i < node.input_size(); i++ ) {
				string input_name = node.input(i);
				if ( input_node_map.count(input_name) == 0 ) {
					vector<onnx::NodeProto*> v;
					input_node_map[input_name] = v;
				}
				vector<onnx::NodeProto*> v = input_node_map[input_name];
				v.push_back(&(nodes[j]));
				input_node_map[input_name] = v;
			}
		}
		return input_node_map;

	}

	//Creates a map with the name of the constant node as key and the node container from onnx as value.
	map<string, onnx::NodeProto*>  initialize_constant_nodes( vector<onnx::NodeProto> &nodes){
		map<string, onnx::NodeProto*> constant_node_map;
		for(int i = 0; i < nodes.size(); i++){
			if(!nodes[i].op_type().compare("Constant")){
				for(int j = 0; j < nodes[i].output_size(); j++){
					constant_node_map[nodes[i].output(j)] = &(nodes[i]);
				}
			}
		}
		return constant_node_map;
	}

	//Initializes the queue that we are going to explore to construct the eddl graph of the net. The first values are the input layers.
	queue<onnx::NodeProto*> process_inputs(vector<Layer*>* inputs, vector<onnx::ValueInfoProto>* inputs_onnx , map<string, vector<onnx::NodeProto*>>* input_node_map, map<string, Layer*> *output_node_map){
		queue<onnx::NodeProto*> nodeQueue;
		for(int i = 0; i<inputs->size(); i++){
			onnx::ValueInfoProto nameContainer = (*inputs_onnx)[i];
			string name = nameContainer.name();
			(*output_node_map)[name] = (*inputs)[i];
			vector<onnx::NodeProto*> v = (*input_node_map)[name];
			for(onnx::NodeProto* layer: (*input_node_map)[name]){
				nodeQueue.push(layer);
			}
		}
		return nodeQueue;
	}


	//Creates a map where the key is the onnx name for the layer type and the value is the constant value in the enumeration for onnx layer type.
	map<string, ONNX_LAYERS> create_enum_map(){
		map<string, ONNX_LAYERS> map_layers;
		map_layers["BatchNormalization"] = ONNX_LAYERS::BATCHNORM;
		map_layers["Conv"] = ONNX_LAYERS::CONV;
		map_layers["Gemm"] = ONNX_LAYERS::DENSE;
		map_layers["Dropout"] = ONNX_LAYERS::DROP;
		map_layers["Reshape"] = ONNX_LAYERS::RESHAPE;
		map_layers["Flatten"] = ONNX_LAYERS::FLATTEN;
		map_layers["Transpose"] = ONNX_LAYERS::TRANSPOSE;
		map_layers["ConvTranspose"] = ONNX_LAYERS::TRANSPOSED_CONV;
		map_layers["Upsample"] = ONNX_LAYERS::UPSAMPLING;
		map_layers["Softmax"] = ONNX_LAYERS::SOFTMAX;
		map_layers["MaxPool"] = ONNX_LAYERS::MAXPOOL;
		map_layers["AveragePool"] = ONNX_LAYERS::AVGPOOL;
		map_layers["GlobalMaxPool"] = ONNX_LAYERS::GLOBMAXPOOL;
		map_layers["GlobalAveragePool"] = ONNX_LAYERS::GLOBAVGPOOL;
		map_layers["Transpose"] = ONNX_LAYERS::PERMUTE;
		// Activation layers
		map_layers["Relu"] = ONNX_LAYERS::RELU;
		map_layers["Sigmoid"] = ONNX_LAYERS::SIGMOID;
		map_layers["HardSigmoid"] = ONNX_LAYERS::HARD_SIGMOID;
		map_layers["Tanh"] = ONNX_LAYERS::TANH;
		map_layers["Linear"] = ONNX_LAYERS::LINEAR;
		map_layers["Exp"] = ONNX_LAYERS::EXPONENTIAL;
		map_layers["LeakyRelu"] = ONNX_LAYERS::LEAKY_RELU;
		map_layers["ThresholdedRelu"] = ONNX_LAYERS::THRESHOLDED_RELU;
		map_layers["Elu"] = ONNX_LAYERS::ELU;
		map_layers["Selu"] = ONNX_LAYERS::SELU;
		map_layers["Softsign"] = ONNX_LAYERS::SOFTSIGN;
		map_layers["Softplus"] = ONNX_LAYERS::SOFTPLUS;
		// Merge Layers
		map_layers["Concat"] = ONNX_LAYERS::CONCAT;
		map_layers["Add"] = ONNX_LAYERS::ADD;
		map_layers["Sub"] = ONNX_LAYERS::SUB;
		map_layers["Average"] = ONNX_LAYERS::AVERAGE;
		map_layers["MatMul"] = ONNX_LAYERS::MAT_MUL;
		map_layers["Max"] = ONNX_LAYERS::MAX;
		map_layers["Min"] = ONNX_LAYERS::MIN;
		map_layers["Constant"] = ONNX_LAYERS::CONSTANT;

		return map_layers;
	}


	//Converts a raw onnx value tensor and writes it to a vector of that value type.
	template <class T>
	bool TryConvertingTensorRawValues( const onnx::TensorProto& onnx_tensor, vector<T> &field) {
		if (!onnx_tensor.has_raw_data()) {
			return false;
		}
		size_t raw_size = onnx_tensor.raw_data().size();
		if(raw_size % sizeof(T) != 0){
			return false;
		}

		size_t num_elements = raw_size / sizeof(T);
		printf("Converting tensor values\n");
		const void* src_ptr = static_cast<const void*>(onnx_tensor.raw_data().data());
		field.resize(num_elements, 0);
		void* target_ptr = static_cast<void*>(field.data());
		memcpy(target_ptr, src_ptr, raw_size);
		printf("Tensor values converted succesfully\n");
		return true;
	}

	//Parses the values of the onnx tensor to a c++ vector of that type
	vector<float> parseTensorValues(onnx::TensorProto t){
		int data_type = t.data_type(); //Only works for non raw data for now
		vector<float> values;
		switch(data_type){
			case onnx::TensorProto::UNDEFINED:
				//TODO: Make this
				break;
			case onnx::TensorProto::FLOAT:
				if(t.has_raw_data()){
					TryConvertingTensorRawValues(t, values);
				}
				else{
					for(int i = 0; i < t.float_data_size(); i++){
						values.push_back(t.float_data(i));
					}
				}
				break;
			case onnx::TensorProto::UINT8:
				//Se puede
				for(int i = 0; i < t.int32_data_size(); i++){
					values.push_back(t.int32_data(i));
				}
				break;
			case onnx::TensorProto::INT8:
				//Se puede
				for(int i = 0; i < t.int32_data_size(); i++){
					values.push_back(t.int32_data(i));
				}
				break;
			case onnx::TensorProto::UINT16:
				//Se puede
				for(int i = 0; i < t.int32_data_size(); i++){
					values.push_back(t.int32_data(i));
				}
				break;
			case onnx::TensorProto::INT16:
				//Se puede
				for(int i = 0; i < t.int32_data_size(); i++){
					values.push_back(t.int32_data(i));
				}
				break;
			case onnx::TensorProto::INT32:
				//Se puede
				for(int i = 0; i < t.int32_data_size(); i++){
					values.push_back(t.int32_data(i));
				}
				break;
			case onnx::TensorProto::INT64:
				//Se puede
				for(int i = 0; i < t.int64_data_size(); i++){
					values.push_back(t.int64_data(i)); } break;
			case onnx::TensorProto::STRING: //TODO: Make this
				break;
			case onnx::TensorProto::BOOL:
				for(int i = 0; i < t.int32_data_size(); i++){
					values.push_back(t.int32_data(i));
				}
				break;
			case onnx::TensorProto::FLOAT16:
				break;
			case onnx::TensorProto::DOUBLE:
				for(int i = 0; i < t.double_data_size(); i++){
					values.push_back(t.double_data(i));
				}
				break;
			case onnx::TensorProto::UINT32:
				for(int i = 0; i < t.uint64_data_size(); i++){
					values.push_back(t.uint64_data(i));
				}
				break;
			case onnx::TensorProto::UINT64:
				for(int i = 0; i < t.uint64_data_size(); i++){
					values.push_back(t.uint64_data(i));
				}
				break;
			case onnx::TensorProto::COMPLEX64:
				//TODO: Make this
				break;
			case onnx::TensorProto::COMPLEX128:
				//TODO: Make this
				break;
			case onnx::TensorProto::BFLOAT16:
				//TODO: Make this
				break;

			default:
				//TODO: Show an error because the type is not recognized
				cerr << "Vector type not recognized" << endl;
				break;
		}
		return values;

	}

	//Creates two maps. Both have the name of the initializer node as key. The values are a vector containing the weights and a vector containing the shape of the vector, respectively.
	void get_initializers_maps(vector<onnx::TensorProto> tensors, map<string, vector<float> > &values_map, map<string, vector<int> > &dims_map) {

		for(onnx::TensorProto tensor : tensors){
			vector<int> dims;
			for(int i = 0; i < tensor.dims_size(); i++) {
				dims.push_back(tensor.dims(i));
			}
			string tensorName = tensor.name();
			vector<float> values = parseTensorValues(tensor);


			values_map[tensorName] = values;
			dims_map[tensorName] = dims;


		}
		return;
	}

	//Parses one TensorProto pointer (Input or output) to eddl Tensor pointer
	vector<int> parse_IO_tensor(onnx::TypeProto::Tensor tensor) {
		onnx::TensorShapeProto tensorShape = tensor.shape();
		vector<int> shape;
		shape.push_back(1); //TODO check this is required

		for(int i = 1; i < tensorShape.dim_size(); i++){
			shape.push_back(tensorShape.dim(i).dim_value());
		}

		return shape;
	}

	//Converts one vector of TensorProto pointers (Input or output)
	//to one vector of eddl Tensor pointers.
	vector<Layer*> parse_IO_tensors(vector<onnx::ValueInfoProto> io_onnx, int mem) {
		vector<Layer*> io;
		onnx::TypeProto::Tensor tensor;
		int dev = DEV_CPU;

		for(onnx::ValueInfoProto infoProto : io_onnx) {
			tensor = infoProto.type().tensor_type();
			string name = infoProto.name();
			io.push_back(new LInput(new Tensor(parse_IO_tensor(tensor)), name, dev, mem) );
		}

		for(Layer* layer : io){
			Tensor* outputTensor = layer->output;
			vector<int> checkShape = outputTensor->shape;
			for( int dim : checkShape){
			}
		}
		return io;
	}

	//Returns a vector with the input names of the net
	vector<onnx::ValueInfoProto> get_inputs(onnx::GraphProto graph){
		set<string> input_names;
		set<string> initializer_names;//We make the substraction of both sets to find the true inputs

		for(int i = 0; i < graph.input_size(); i++){ //Construct set of input names
			input_names.insert(graph.input(i).name());
		}

		vector<onnx::TensorProto> initializers = get_initializers(graph); for(int i = 0; i < initializers.size(); i++){ //Construct set of initializer names
			if(initializers[i].has_name())
				initializer_names.insert(initializers[i].name());
		}

		vector<string> true_inputs(100);
		std::set_difference(input_names.begin(), input_names.end(), initializer_names.begin(), initializer_names.end(), true_inputs.begin());
		true_inputs.shrink_to_fit();

		vector<onnx::ValueInfoProto> returnVector; // This is for returning the tensor, but we need the names
		for(int i = 0; i < graph.input_size(); i++){
			onnx::ValueInfoProto auxInfoProto = graph.input(i);
			if(count(true_inputs.begin(), true_inputs.end(), auxInfoProto.name())){ //If the name is a true input
				returnVector.push_back(auxInfoProto); //Push it to input vector
			}
		}
		return returnVector;
	}


	//Returns a vector containing the output names of the net
	vector<string> get_outputs(onnx::GraphProto graph){
		vector<string> output_names;

		for(int i = 0; i < graph.output_size(); i++){ //Construct set of output names
			output_names.push_back(graph.output(i).name());
		}
		return output_names;
	}

	//Returns a vector containing all nodes of the graph in onnx containers.
	vector<onnx::NodeProto> get_graph_nodes(onnx::GraphProto graph) {
		vector<onnx::NodeProto> nodes;
		for( int i = 0; i < graph.node_size(); i++) {
			onnx::NodeProto node = graph.node(i);
			nodes.push_back(node);
		}

		return nodes;
	}


	//Imports a net stored in a onnx file
	Net* import_net_from_onnx_file(std::string path, int mem) {
		// Verify that the version of the library that we linked against is
		// compatible with the version of the headers we compiled against.
		GOOGLE_PROTOBUF_VERIFY_VERSION;
		onnx::ModelProto model;

		{
			// Read the existing net.
			fstream input(path , ios::in | ios::binary);
			if (!model.ParseFromIstream(&input)) {
				cerr << "Failed to parse model." << endl;
				//return;
			}
		}
		return build_net_onnx(model, mem);
	}

	//Imports a net from a pointer passed as argument
	Net* import_net_from_onnx_pointer(void* serialized_model, size_t size, int mem){
		// Verify that the version of the library that we linked against is
		// compatible with the version of the headers we compiled against.
		GOOGLE_PROTOBUF_VERIFY_VERSION;
		onnx::ModelProto model;
		{
			if(!model.ParseFromArray(serialized_model, size)){
				cerr << "Failed to parse model." << endl;
			}
			else if (verbose >= 2) cout << "Model parsed succesfuly" << endl;
		}
		return build_net_onnx(model, mem);
	}

	//Imports a net from a c++ string passed as argument.
	Net* import_net_from_onnx_string(string* model_string, int mem){
		// Verify that the version of the library that we linked against is
		// compatible with the version of the headers we compiled against.
		GOOGLE_PROTOBUF_VERIFY_VERSION;
		onnx::ModelProto model;
		{
			if(!model.ParseFromString(*model_string)){
				cerr << "Failed to parse model." << endl;
			}
			else if (verbose >= 2) cout << "Model parsed succesfuly" << endl;
		}
		return build_net_onnx(model, mem);
	}



	//Builds a eddl Net from an instance of the onnx container for model
	Net* build_net_onnx(onnx::ModelProto model, int mem){

		long long int ir_version = model.ir_version();
		// We have to check if the imported net has the
		// version we created this importer for.
		if(ir_version != 0x00000006) {
			cerr << "Ir_version < 6" << endl;
			cerr << "Warning: Version is inferior to the one supported. ONNX module will try to load it anyway" << endl;
		}

		// We omit the OperatorSetIdProto, since it doesn't do anything for EDDL
		cout << "Ir_version = " << ir_version << endl;
		for(int i = 0; i < model.opset_import_size() ; i++){
			cout << "Operator domain  = " << model.opset_import(i).domain() << endl;
			cout << "Operator version = " << model.opset_import(i).version() << endl;
		}
		cout << "Producer_name: " << model.producer_name() << endl;
		cout << "Producer_version: " << model.producer_version() << endl;
		cout << "Domain: " << model.domain() << endl;
		cout << "Model_version: " << model.model_version() << endl;
		int counter = 0;
		onnx::GraphProto graph = model.graph(); //Get the graph of the model.
		//Model needs input in the constructor, so we start with that.

		vector<onnx::ValueInfoProto> inputs_onnx = get_inputs(graph); //Get the inputs

		vector<Layer*> inputs =  parse_IO_tensors(inputs_onnx, mem); //Parse ONNX inputs to EDDL inputs

		vector<onnx::TensorProto> initializers = get_initializers(graph); // Retrieves the initializers from the graph.
																		  // The weight for the layers can be found in the initializers.

		map<string, vector<float>> map_init_values;
		map<string, vector<int>>   map_init_dims;
		get_initializers_maps(initializers, map_init_values, map_init_dims);// Creates 2 maps
																			//  Key: Input Name . Value: Weights
																			//  Key: Input Name . Value: Dims
		vector<onnx::NodeProto> nodes = get_graph_nodes(graph);
		//The methodology is the following:
		//We create three maps:
		//map <string input, vector<onnx::NodeProto *> > input_node_map. The input will point towards the nodes that have this input
		//map <string output, Layer* parent > output_node_map. To know from which (parent) node comes each input (The input is the output of the parent node)
		//map <string input/output, bool> input_active_map     	The outputs will be put inside a bool, where we will see if it is active or not.
		//
		//The algorithm is the following:
		//	1-We check the inputs of each node.
		//		For each input we insert the input string as a key and the Node(s) that use it as a value in the input_node_map.
		//		If that input is already on use, we will append the node to the existing vector
		//	2-We check the outputs of each node. //NOT REQUIRED
		//		For each output we insert the output string as a key and the Node that generates it as a value in the outputs_map //NOT REQUIRED
		//	3-When we add an input/output to the map, we also add it to the input_active_map as key, and the value will be false by default. If it is already there, we do nothing. //NOT REQUIRED
		//	4-Once we have constructed these maps, we create an empty queue of NodeProto
		//	5-For the input nodes in the graph, we create the EDDL layer and add the nodes that use its output(s) to the queue
		//	6-While the queue is not empty:
		//		For each node:
		//		6.1-Check if all its inputs (not the ones in 'initializers') exist in output_node_map
		//		6.2-If they are not  --> continue
		//		6.3-Else:
		//				Create the EDDL layer
		//				Add the nodes that use its outputs to the queue
		//To create each EDDL layer:
		//	1-Get its parent(s) by accessing to output_node_map using this node's input(s) as key(s)
		//	2-Get its weights from 'initializers'
		//	3-Create layer
		//
		//  We need another map for storing the constant nodes, who are always active
		//  We design this map as map<string, onnx::NodeProto> and called constant_node_map

		map<string, Layer*> 			output_node_map;

		//1 2 and 3: Initialize maps

		map<string, vector<onnx::NodeProto*>> 	input_node_map = initialize_input_node_map(nodes);

		//4 and 5: Create queue of NodeProto

		map<string, onnx::NodeProto*> constant_node_map =  initialize_constant_nodes( nodes);

		queue<onnx::NodeProto *> nodeQueue = process_inputs(&inputs, &inputs_onnx, &input_node_map, &output_node_map);

		for(int i = 0; i < nodes.size(); i++){ //Check if any node only has initializers and constant nodes as parameters, so we can process it right away
			onnx::NodeProto *node = &nodes[i];
			bool avaliable = true;
			for(int j = 0; j < node->input_size(); j++){
				string input = node->input(j);
				if(map_init_values.count(input)){
					continue;
				}
				if(constant_node_map.count(input)){
					continue;
				}
				avaliable = false;
				break;
			}
			if(avaliable){
				//cout << "Node " << node->name() << " is avaliable" << endl;
				if(node->op_type() == "Constant" ) continue;
				nodeQueue.push(node);
			}
		}

		map<string, ONNX_LAYERS> map_layers = create_enum_map();
		//6 - While the queue is not empty:
		while(!nodeQueue.empty()){
			counter = 0;
			onnx::NodeProto* node= nodeQueue.front();
			//6.1: Check all inputs are avaliable
			bool avaliable = true;
			for(int i = 0; i < node->input_size(); i++){
				string input = node->input(i);
				if(map_init_values.count(input)){
					continue;
				}
				if(output_node_map.count(input)){
					continue;
				}
				if(constant_node_map.count(input)){
					continue;
				}
				avaliable = false;
				break;
			}
			string output_name = node->output(0);
			if(output_node_map.count(output_name)){
				nodeQueue.pop();
				continue; //This means this node was already created
			}

			//6.2
			if(!avaliable){
				nodeQueue.pop();
				continue;
			}
			//6.3
			//Lets assume the maximum quantity of layer inputs a layer can have is 2
			//vector<Layer *> parents; //Not required because inputs are ordered.
			vector<float> weights;
			vector<int> dims;
			// We have to know which layer to create. For it, I suggest
			// a map <String-Enumeration> for creating a switch, where
			// we call the constructor of that layer
			string layer_type_name = node->op_type();
			ONNX_LAYERS layer_type = map_layers[layer_type_name];
			string name = node->name();
			int dev = DEV_CPU;//TODO: Check what device to use
			Layer *actual_layer;

			switch (layer_type) { //Every case should create the corresponding layer and asign it to "actual_layer" variable

				case ONNX_LAYERS::BATCHNORM:
					{
						double epsilon = 1e-05; //Default value
						double momentum = 0.9;  //Default value
						for ( int j = 0; j < node->attribute_size(); j++ ) { //Set the attributes
							onnx::AttributeProto attribute = node->attribute(j);
							string attr_name = attribute.name();
							if(!attr_name.compare("epsilon")) epsilon = attribute.f();
							if(!attr_name.compare("momentum")) momentum = attribute.f();
						}
						
						string parent_name = node->input(0); //Get parent
						Layer* parent = output_node_map[parent_name];
						vector<int> parent_shape = parent->output->shape;

						string scale_name = node->input(1); // Scale parameter
						vector<float>* scale_weights = &(map_init_values[scale_name]);
						vector<int> scale_dims = map_init_dims[scale_name];

						string bias_name = node->input(2); // Bias parameter
						vector<float>* bias_weights = &(map_init_values[bias_name]);
						vector<int> bias_dims = map_init_dims[bias_name];

						string mean_name = node->input(3); //Get weights and dims
						vector<float>* mean_weights = &(map_init_values[mean_name]);
						vector<int> mean_dims = map_init_dims[mean_name];

						string variance_name = node->input(4); //Get weights and dims
						vector<float>* variance_weights = &(map_init_values[variance_name]);
						vector<int> variance_dims = map_init_dims[variance_name];

						string name = node->name();

						bool affine = true; // The ONNX operator description does not have an "affine" attribute. We have to assume that this will be allways true.

						actual_layer = new LBatchNorm(parent, momentum, epsilon, affine, name, dev, mem);

						Tensor* scale_tensor = new Tensor(scale_dims, NEW_FROM_VECTOR_PTR(scale_weights), dev);
						Tensor::copy(scale_tensor, ((LBatchNorm *)(actual_layer))->bn_g);
						delete scale_tensor;

						Tensor* bias_tensor = new Tensor(bias_dims, NEW_FROM_VECTOR_PTR(bias_weights), dev);
						Tensor::copy(bias_tensor, ((LBatchNorm *)(actual_layer))->bn_b);
						delete bias_tensor;

						Tensor* mean_tensor = new Tensor(mean_dims, NEW_FROM_VECTOR_PTR(mean_weights), dev);
						Tensor::copy(mean_tensor, ((LBatchNorm *)(actual_layer))->mean);
						delete mean_tensor;

						Tensor* variance_tensor = new Tensor(variance_dims, NEW_FROM_VECTOR_PTR(variance_weights), dev);
						Tensor::copy(variance_tensor, ((LBatchNorm *)(actual_layer))->variance);
						delete variance_tensor;

					}
					break;

				case ONNX_LAYERS::CONV:
					{
						int filters;
						vector<int> kernel_shape;
						vector<int> strides;
						vector<int> pads;
						//bool explicit_padding;
						string auto_pad_option = "";
						bool auto_pad = false;
						vector<float> *bias;

						for ( int j = 0; j < node->attribute_size(); j++ ) { //Set the attributes
							onnx::AttributeProto attribute = node->attribute(j);
							string attr_name = attribute.name();
							if (!attr_name.compare("auto_pad")) {
								auto_pad = true;
								if(!attribute.s().compare("NOTSET")){
									auto_pad = false;
									continue;
								}
								else if(!attribute.s().compare("VALID"))
									auto_pad_option = "none";
								else if(!attribute.s().compare("SAME_UPPER"))
									auto_pad_option = "same";
							}
							else if (!attr_name.compare("dilations")) { //It isn't implemented in eddl

							}
							else if (!attr_name.compare("group")) { //We don't know if this exists in eddl

							}
							else if (!attr_name.compare("kernel_shape")) { //
								for( int h = 0; h<attribute.ints_size(); h++){
									kernel_shape.push_back(attribute.ints(h));
								}
							}
							else if (!attr_name.compare("pads")) { //
								for(int h = 0; h < 4; h++){
									pads.push_back(attribute.ints(h));
								}
							}
							else if (!attr_name.compare("strides")) { //
								for(int h = 0; h < attribute.ints_size(); h++){
									strides.push_back(attribute.ints(h));
								}
							}
						}

						/*if(!explicit_padding){ //We have to add 0 padding to the conv descriptor
							pads.resize(4,0);
							pads[0] = 0;
							pads[1] = 0;
							pads[2] = 0;
							pads[3] = 0;
						}*/

						string parent_name = node->input(0); //Get parent
						Layer* parent = output_node_map[parent_name];
						vector<int> parent_shape = parent->output->shape;

						string weights_name = node->input(1); //Get weights and dims
						vector<float>* weights = &(map_init_values[weights_name]);
						vector<int> dims = map_init_dims[weights_name];



						filters = dims[0];
						string name = node->name();
						ConvolDescriptor* convol_descriptor;
						if(!auto_pad){
							kernel_shape.insert(kernel_shape.begin(), filters); //Add number of filters to kernel shape
							convol_descriptor = new ConvolDescriptor(kernel_shape, strides, pads);
						}
						else convol_descriptor = new ConvolDescriptor(filters, kernel_shape, strides, auto_pad_option, node->input_size() > 2, mem);

						actual_layer = new LConv(parent, convol_descriptor, name, dev, mem);

						if(node->input_size() > 2){
							string bias_name = node->input(2);
							bias = &(map_init_values[bias_name]);
							vector<int> bias_shape;
							bias_shape.push_back(bias->size());
							Tensor* bias_tensor = new Tensor(bias_shape, NEW_FROM_VECTOR_PTR(bias), dev);
							Tensor::copy(bias_tensor , convol_descriptor->bias);
							delete bias_tensor;

						}
						Tensor* weights_tensor = new Tensor(dims, NEW_FROM_VECTOR_PTR(weights), dev);
						Tensor::copy(weights_tensor, convol_descriptor->K);
						delete weights_tensor;
						break;
					}

				case ONNX_LAYERS::DENSE:
					{
						int ndim;
						bool use_bias = false;
						float alpha;
						float beta;
						int transA = 0;
						int transB = 0;
						vector<int> bias_dims;
						vector <float>* bias;
						for ( int j = 0; j < node->attribute_size(); j++ ) {
							onnx::AttributeProto attribute = node->attribute(j);
							string attr_name = attribute.name();
							if (!attr_name.compare("alpha")) {
								alpha = attribute.f();
							}
							else if (!attr_name.compare("beta")) {
								beta = attribute.f();
							}
							else if (!attr_name.compare("transA")) {
								transA = attribute.i();
							}
							else if (!attr_name.compare("transB")) {
								transB = attribute.i();
							}
						}

						string parent_name;
						Layer* parent;
						string weights_name;
						string bias_name;
						vector<float> *weights;
						vector<int> dims;

						for(int i = 0; i < 2; i++){
							string input = node->input(i);
							if(!map_init_values.count(input)) { // parent
								parent_name = node->input(0);
								parent = output_node_map[input];
							}
							else { // weights
								weights_name = node->input(i);
								weights = &(map_init_values[input]);
								dims = map_init_dims[input];
								ndim = dims.size();
							}
						}
						use_bias=node->input_size() > 2;
						int neuronas = 0;
						if(transB){
							neuronas = dims[0];
						}
						else
							neuronas = dims[1];
						string name = node->name();
						Tensor * input_size = parent->output;
						LDense* dense = new LDense(parent, neuronas, use_bias, name, dev, mem); 

						Tensor* weights_tensor = new Tensor(dims, NEW_FROM_VECTOR_PTR(weights), dev);
						if(transB){
							vector<int> reverse_dims;
							for(int h = dims.size()-1; h >= 0; h--){
								reverse_dims.push_back(dims[h]);
							}
							Tensor* weights_tensor_T = new Tensor(reverse_dims, dev);
							Tensor::transpose(weights_tensor, weights_tensor_T, {1,0});
							Tensor::copy(weights_tensor_T, dense->W );
							delete weights_tensor_T;
						}
						else Tensor::copy(weights_tensor, dense->W );
						delete weights_tensor;
						if(use_bias){
							bias_name = node->input(2);
							bias = &(map_init_values[bias_name]);
							bias_dims = map_init_dims[bias_name];
							Tensor* bias_tensor = new Tensor(bias_dims, NEW_FROM_VECTOR_PTR(bias), dev);
							Tensor::copy(bias_tensor, dense->bias);
							delete bias_tensor;
						}
						actual_layer = dense;
					}
					break;
					
				case ONNX_LAYERS::UPSAMPLING:
					{
						string interpolation_mode;
						float batch_scale;
						float channel_scale;
						float height_scale;
						float width_scale;
						for ( int j = 0; j < node->attribute_size(); j++ ) { //Set the attributes
							onnx::AttributeProto attribute = node->attribute(j);
							string attr_name = attribute.name();
							if(!attr_name.compare("mode")) interpolation_mode = attribute.s();
						}
						
						string parent_name = node->input(0); //Get parent
						Layer* parent = output_node_map[parent_name];
						vector<int> parent_shape = parent->output->shape;
						
						string scales_name = node->input(1); //Get scales and dims
						vector<float>* scales = &(map_init_values[scales_name]);
						vector<int> scales_dims = map_init_dims[scales_name];
						
						if(scales_dims[0] != 4){
							cerr << "Dimensions of upsampling layer in onnx are wrong" << endl;
						}
						batch_scale = scales->at(0);
						channel_scale = scales->at(1);
						height_scale = scales->at(2);
						width_scale = scales->at(3);
						
						string name = node->name();


						vector<int> size_vector;
						size_vector.push_back((int)height_scale);
						size_vector.push_back((int)width_scale);
						actual_layer = new LUpSampling(parent, size_vector, interpolation_mode, name, dev, mem);
						delete scales;

					}
					break;

				case ONNX_LAYERS::DROP:
					{
						int seed=0;
						float ratio=0.5;
						for ( int j = 0; j < node->attribute_size(); j++ ) { //Set the attributes
							onnx::AttributeProto attribute = node->attribute(j);
							string attr_name = attribute.name();
							if(!attr_name.compare("seed")) seed = attribute.i();
							if(!attr_name.compare("ratio")) ratio = attribute.f();
						}
						
						string parent_name = node->input(0); //Get parent
						Layer* parent = output_node_map[parent_name];
						vector<int> parent_shape = parent->output->shape;

						string name = node->name();
						actual_layer = new LDropout(parent, ratio, true, name, dev, mem);

					}
					break;

				case ONNX_LAYERS::AVGPOOL:
					{
						int filters;
						vector<int> kernel_shape;
						vector<int> strides;
						vector<int> pads;
						bool explicit_padding = true;
						int ceil_mode = 0;
						int count_include_pad = 0;
						vector<int> dilations;
						int storage_order = 0;

						for ( int j = 0; j < node->attribute_size(); j++ ) { //Set the attributes
							onnx::AttributeProto attribute = node->attribute(j);
							string attr_name = attribute.name();
							if (!attr_name.compare("auto_pad")) { //We dont know if it is implemented
								if(!attribute.s().compare("NOTSET")) continue;
								if(!attribute.s().compare("VALID"))
									explicit_padding=false;
							}
							else if (!attr_name.compare("ceil_mode")) {

							}
							else if (!attr_name.compare("count_include_pad")) {

							}
							else if (!attr_name.compare("kernel_shape")) {
								for( int h = 0; h<attribute.ints_size(); h++){
									kernel_shape.push_back(attribute.ints(h));
								}
							}
							else if (!attr_name.compare("pads")) {
								for(int h = 0; h < 4; h++){
									pads.push_back(attribute.ints(h));
								}
							}
							else if (!attr_name.compare("strides")) {
								for(int h = 0; h < attribute.ints_size(); h++){
									strides.push_back(attribute.ints(h));
								}
							}
						}


						if(!explicit_padding){ //We have to add 0 padding to the conv descriptor
							pads.resize(4,0);
							pads[0] = 0;
							pads[1] = 0;
							pads[2] = 0;
							pads[3] = 0;
						}

						string parent_name = node->input(0); //Get parent
						Layer* parent = output_node_map[parent_name];
						vector<int> parent_shape = parent->output->shape;

						string name = node->name();


						actual_layer = new LAveragePool(parent, new PoolDescriptor(kernel_shape, strides, pads), name, dev, mem);
					}
					break;

				case ONNX_LAYERS::GLOBAVGPOOL:
					{
						string parent_name = node->input(0); //Get parent
						Layer* parent = output_node_map[parent_name];
						vector<int> parent_shape = parent->output->shape;

						int h=parent_shape[2];
						int w=parent_shape[3];

						actual_layer = new LAveragePool(parent, {h,w},{1,1}, "none", node->name(), dev, mem);
					}
					break;
				case ONNX_LAYERS::RESHAPE:
					{
						string shape_node_name = node->input(1);
						vector<int> shape;
						if(constant_node_map.count(shape_node_name)){
							onnx::NodeProto* shape_node = constant_node_map[shape_node_name];
							onnx::AttributeProto shape_attribute = shape_node->attribute(0);
							if(shape_attribute.name().compare("value")){
								//This means an error ocurred, but don't know how to proceed then.
								printf("An error ocurred when reading the shape of reshape\n");
							}
							onnx::TensorProto shape_tensor = shape_attribute.t();
							shape = vf2vi(parseTensorValues(shape_tensor));
						}
						else{
							shape = vf2vi(map_init_values[shape_node_name]);
						}
						string name = node->name();
						string parent_name = node->input(0);
						if(output_node_map.count(parent_name)){
							//shape.insert(shape.begin(), 1); //Default batch size = 1 //Required for reshape but not for parameters
							shape[0] = 1;
							Layer *parent = output_node_map[parent_name];
							actual_layer= new LReshape(parent, shape, name, dev, mem);
						}
						else if(map_init_values.count(parent_name)){ //This means it is a parameter and not a layer
							for( int i = 0; i < node->output_size(); i++ ) {
								map_init_values[node->output(i)] = map_init_values[parent_name];
								map_init_dims[node->output(i)] = shape; 
								vector<onnx::NodeProto*> child_nodes = input_node_map[node->output(i)];
								for(onnx::NodeProto * child : child_nodes){
									nodeQueue.push(child);
								}
							}
							nodeQueue.pop();
							continue; //We need to do the update of the queue here because we are not creating a true layer
						} else cerr << "Uknown parent type for reshape" << endl;
					}
					break;

				case ONNX_LAYERS::FLATTEN:
					{
						string parent_name = node->input(0); //Get parent
						Layer* parent = output_node_map[parent_name];

						actual_layer = new LReshape(parent, {1,-1}, name, dev, mem);
					}
					break;
				case ONNX_LAYERS::PERMUTE:
					{
						vector<int> dims;
						for ( int j = 0; j < node->attribute_size(); j++ ) { //Set the attributes
							onnx::AttributeProto attribute = node->attribute(j);
							string attr_name = attribute.name();
							if(!attr_name.compare("perm")){
								for(int h = 1; h < attribute.ints_size(); h++)
									dims.push_back(attribute.ints(h));
							}
						}
						string parent_name = node->input(0);
						Layer *parent = output_node_map[parent_name];

						string name = node->name();
						actual_layer = new LPermute(parent, dims, name, dev, mem);
						break;
					}
				case ONNX_LAYERS::RELU:
					{
						string parent_name = node->input(0);
						Layer *parent = output_node_map[parent_name];

						string name = node->name();
						vector<float> param; 
						actual_layer = new LActivation(parent, "relu", param, name, dev, mem);
						break;
					}
				case ONNX_LAYERS::SIGMOID:
					{
						string parent_name = node->input(0);
						Layer *parent = output_node_map[parent_name];

						string name = node->name();
						vector<float> param; 
						actual_layer = new LActivation(parent, "sigmoid", param, name, dev, mem);
						break;
					}
				case ONNX_LAYERS::HARD_SIGMOID:
					{
						string parent_name = node->input(0);
						Layer *parent = output_node_map[parent_name];

						string name = node->name();
						vector<float> param; 
						actual_layer = new LActivation(parent, "hard_sigmoid", param, name, dev, mem);
						break;
					}
				case ONNX_LAYERS::TANH:
					{
						string parent_name = node->input(0);
						Layer *parent = output_node_map[parent_name];

						string name = node->name();
						vector<float> param; 
						actual_layer = new LActivation(parent, "tanh", param, name, dev, mem);
						break;
					}
				case ONNX_LAYERS::EXPONENTIAL:
					{
						string parent_name = node->input(0);
						Layer *parent = output_node_map[parent_name];

						string name = node->name();
						vector<float> param; 
						actual_layer = new LActivation(parent, "exp", param, name, dev, mem);
						break;
					}
				case ONNX_LAYERS::LINEAR:
					{
						float alpha;
						for ( int j = 0; j < node->attribute_size(); j++ ) { //Set the attributes
							onnx::AttributeProto attribute = node->attribute(j);
							string attr_name = attribute.name();
							if(!attr_name.compare("alpha")) alpha = attribute.f();
						}
						string parent_name = node->input(0);
						Layer *parent = output_node_map[parent_name];

						string name = node->name();
						vector<float> param; 
						param.push_back(alpha);
						actual_layer = new LActivation(parent, "linear", param, name, dev, mem);
						break;
					}
				case ONNX_LAYERS::LEAKY_RELU:
					{
						float alpha;
						for ( int j = 0; j < node->attribute_size(); j++ ) { //Set the attributes
							onnx::AttributeProto attribute = node->attribute(j);
							string attr_name = attribute.name();
							if(!attr_name.compare("alpha")) alpha = attribute.f();
						}
						string parent_name = node->input(0);
						Layer *parent = output_node_map[parent_name];

						string name = node->name();
						vector<float> param; 
						param.push_back(alpha);
						actual_layer = new LActivation(parent, "leaky_relu", param, name, dev, mem);
						break;
					}
				case ONNX_LAYERS::THRESHOLDED_RELU:
					{
						float alpha;
						for ( int j = 0; j < node->attribute_size(); j++ ) { //Set the attributes
							onnx::AttributeProto attribute = node->attribute(j);
							string attr_name = attribute.name();
							if(!attr_name.compare("alpha")) alpha = attribute.f();
						}
						string parent_name = node->input(0);
						Layer *parent = output_node_map[parent_name];

						string name = node->name();
						vector<float> param; 
						param.push_back(alpha);
						actual_layer = new LActivation(parent, "thresholded_relu", param, name, dev, mem);
						break;
					}
				case ONNX_LAYERS::ELU:
					{
						float alpha;
						for ( int j = 0; j < node->attribute_size(); j++ ) { //Set the attributes
							onnx::AttributeProto attribute = node->attribute(j);
							string attr_name = attribute.name();
							if(!attr_name.compare("alpha")) alpha = attribute.f();
						}
						string parent_name = node->input(0);
						Layer *parent = output_node_map[parent_name];

						string name = node->name();
						vector<float> param; 
						param.push_back(alpha);
						actual_layer = new LActivation(parent, "elu", param, name, dev, mem);
						break;
					}
				case ONNX_LAYERS::SELU:
					{
						float alpha = 1.0;
						float gamma = 1.0;
						for ( int j = 0; j < node->attribute_size(); j++ ) { //Set the attributes
							onnx::AttributeProto attribute = node->attribute(j);
							string attr_name = attribute.name();
							if(!attr_name.compare("alpha")) alpha = attribute.f();
							if(!attr_name.compare("gamma")) gamma = attribute.f();
						}
						string parent_name = node->input(0);
						Layer *parent = output_node_map[parent_name];

						string name = node->name();
						vector<float> param; 
						param.push_back(alpha);
						param.push_back(gamma);
						actual_layer = new LActivation(parent, "selu", param, name, dev, mem);
						break;
					}
				case ONNX_LAYERS::SOFTSIGN:
					{
						string parent_name = node->input(0);
						Layer *parent = output_node_map[parent_name];

						string name = node->name();
						vector<float> param; 
						actual_layer = new LActivation(parent, "softsign", param, name, dev, mem);
						break;
					}
				case ONNX_LAYERS::SOFTPLUS:
					{
						string parent_name = node->input(0);
						Layer *parent = output_node_map[parent_name];

						string name = node->name();
						vector<float> param; 
						actual_layer = new LActivation(parent, "softplus", param, name, dev, mem);
						break;
					}
				case ONNX_LAYERS::SOFTMAX:
					{
						string parent_name = node->input(0);
						Layer *parent = output_node_map[parent_name];
						int axis = 1;

						for ( int j = 0; j < node->attribute_size(); j++ ) {
							onnx::AttributeProto attribute = node->attribute(j);
							string attr_name = attribute.name();
							if (!attr_name.compare("axis")) {
								axis = attribute.i();//No use for it on eddl because it is not configurable
							}
							else printf("Error with softmax attributes\n");
						}

						string name = node->name();
						vector<float> param; 
						actual_layer = new LActivation(parent, "softmax", param, name, dev, mem);
						break;

					}
				case ONNX_LAYERS::CONCAT:
					{
						int axis = 1;
						for ( int j = 0; j < node->attribute_size(); j++) {
							onnx::AttributeProto attribute = node->attribute(j);
							string attr_name = attribute.name();
							if (!attr_name.compare("axis")) {
								axis = attribute.i();
							}
							else printf("Error with concat attributes. Attribute name is: %s\n", attr_name.c_str());
						}
						vector<Layer *> parents;
						string parent_name;
						for ( int j = 0; j < node->input_size(); j++) {
							parent_name = node->input(j);
							parents.push_back(output_node_map[parent_name]);
						}
						string name = node->name();
						actual_layer = new LConcat(parents, axis, name, dev, mem);

						break;
					}
				case ONNX_LAYERS::ADD:
					{
						vector<Layer *> parents;
						string parent_name;
						bool parameter_input = false;
						int index_parameter = -1; //Possible values 0 and 1, we won't expect parameters in an add with more than two parents
						for ( int j = 0; j < node->input_size(); j++) {
							parent_name = node->input(j);
							if(output_node_map.count(parent_name))
								parents.push_back(output_node_map[parent_name]);
							else if(map_init_values.count(parent_name))
								parameter_input = true;
								index_parameter = j;
						}
						if(parameter_input){
							LConv* conv;
							LDense* dense;
							if((conv = dynamic_cast<LConv*>(parents[0]) )){
								ConvolDescriptor* convol_descriptor = conv->cd;
								string bias_name = node->input(index_parameter);
								vector<float> *bias = &(map_init_values[bias_name]);
								vector<int> bias_shape;
								bias_shape.push_back(bias->size());
								Tensor* bias_tensor = new Tensor(bias_shape, NEW_FROM_VECTOR_PTR(bias), dev);
								if(!convol_descriptor->use_bias){
									convol_descriptor->use_bias = true; //We need to enable the bias
									Tensor::copy(bias_tensor , convol_descriptor->bias);
								}
								else{
									Tensor* auxiliar_tensor = Tensor::add(convol_descriptor->bias, bias_tensor);
									Tensor::copy(auxiliar_tensor , convol_descriptor->bias);
									delete auxiliar_tensor;

								}
								delete bias_tensor;
								actual_layer = conv;
								break;
							}
							else if((dense = dynamic_cast<LDense*>(parents[0]) )){
								string bias_name = node->input(index_parameter);
								vector<float> *bias = &(map_init_values[bias_name]);
								vector<int> bias_dims = map_init_dims[bias_name];
								if(!dense->use_bias){
									dense->use_bias = true;
									dense->bias = new Tensor(bias_dims, NEW_FROM_VECTOR_PTR(bias), dev);
								}
								else{ //If dense already has a bias, we sum it in top of the bias
									Tensor* add_to_bias = new Tensor(bias_dims, NEW_FROM_VECTOR_PTR(bias), dev);
									dense->bias = Tensor::add(dense->bias, add_to_bias);

								}
								//Tensor::copy(bias_tensor, dense->bias);
								actual_layer=dense;
								break;

							} else cerr << "Error, add with a parameter input where the other input is not a dense or a convolutional layer" << endl;
						}
						string name = node->name();
						actual_layer = new LAdd(parents, name, dev, mem);

						break;
					}
				case ONNX_LAYERS::SUB:
					{
						vector<Layer *> parents;
						string parent_name;
						for ( int j = 0; j < node->input_size(); j++) {
							parent_name = node->input(j);
							parents.push_back(output_node_map[parent_name]);
						}
						string name = node->name();
						actual_layer = new LSubtract(parents, name, dev, mem);

						break;
					}
				case ONNX_LAYERS::AVERAGE:
					{
						vector<Layer *> parents;
						string parent_name;
						for ( int j = 0; j < node->input_size(); j++) {
							parent_name = node->input(j);
							parents.push_back(output_node_map[parent_name]);
						}
						string name = node->name();
						actual_layer = new LAverage(parents, name, dev, mem);

						break;
					}
				case ONNX_LAYERS::MAT_MUL:
					{
						vector<Layer *> parents;
						string parent_name;
						bool dense_detected = false;
						int index_parameter = -1;
						for ( int j = 0; j < node->input_size(); j++) {
							parent_name = node->input(j);
							if(map_init_values.count(parent_name)){
								//Dense detected	
								if(dense_detected){
									cerr << "MAT_MUL with two parameters" << endl;
								}
								dense_detected=true;
								index_parameter = j;
							} else
								parents.push_back(output_node_map[parent_name]);
						}
						if(dense_detected){
							string weights_name = node->input(index_parameter);
							vector<float> *weights = &(map_init_values[weights_name]);
							vector<int> dims = map_init_dims[weights_name];
							int ndim = dims.size();
							int neuronas = dims[1];
							Layer *parent = parents[1-index_parameter];
							bool use_bias = false;
							LDense* dense = new LDense(parent, neuronas, use_bias, name, dev, mem); 
							Tensor* weights_tensor = new Tensor(dims, NEW_FROM_VECTOR_PTR(weights), dev);
							Tensor::copy(weights_tensor, dense->W );
							delete weights_tensor;
							actual_layer = dense;
							break;
						}
						string name = node->name();
						actual_layer = new LMatMul(parents, name, dev, mem);

						break;
					}
				case ONNX_LAYERS::MAX:
					{
						vector<Layer *> parents;
						string parent_name;
						for ( int j = 0; j < node->input_size(); j++) {
							parent_name = node->input(j);
							parents.push_back(output_node_map[parent_name]);
						}
						string name = node->name();
						actual_layer = new LMaximum(parents, name, dev, mem);

						break;
					}
				case ONNX_LAYERS::MIN:
					{
						vector<Layer *> parents;
						string parent_name;
						for ( int j = 0; j < node->input_size(); j++) {
							parent_name = node->input(j);
							parents.push_back(output_node_map[parent_name]);
						}
						string name = node->name();
						actual_layer = new LMinimum(parents, name, dev, mem);

						break;
					}
				case ONNX_LAYERS::MAXPOOL:
					{
						int filters;
						vector<int> kernel_shape;
						vector<int> strides;
						vector<int> pads;
						bool explicit_padding = true;
						int ceil_mode = 0;
						vector<int> dilations;
						int storage_order = 0;

						for ( int j = 0; j < node->attribute_size(); j++ ) { //Set the attributes
							onnx::AttributeProto attribute = node->attribute(j);
							string attr_name = attribute.name();
							if (!attr_name.compare("auto_pad")) { //We dont know if it is implemented
								if(!attribute.s().compare("NOTSET")) continue;
								if(!attribute.s().compare("VALID"))
									explicit_padding=false;
							}
							else if (!attr_name.compare("ceil_mode")) {

							}
							else if (!attr_name.compare("dilations")) {

							}
							else if (!attr_name.compare("kernel_shape")) {
								for( int h = 0; h<attribute.ints_size(); h++){
									kernel_shape.push_back(attribute.ints(h));
								}
							}
							else if (!attr_name.compare("pads")) {
								for(int h = 0; h < 4; h++){
									pads.push_back(attribute.ints(h));
								}
							}
							else if (!attr_name.compare("storage_order")) {

							}
							else if (!attr_name.compare("strides")) {
								for(int h = 0; h < attribute.ints_size(); h++){
									strides.push_back(attribute.ints(h));
								}
							}
						}


						if(!explicit_padding){ //We have to add 0 padding to the conv descriptor
							pads.resize(4,0);
							pads[0] = 0;
							pads[1] = 0;
							pads[2] = 0;
							pads[3] = 0;
						}

						string parent_name = node->input(0); //Get parent
						Layer* parent = output_node_map[parent_name];
						vector<int> parent_shape = parent->output->shape;

						string name = node->name();

						actual_layer = new LMaxPool(parent, new PoolDescriptor(kernel_shape, strides, pads), name, dev, mem);
					}
						break;
				case ONNX_LAYERS::GLOBMAXPOOL:
					{
						string parent_name = node->input(0); //Get parent
						Layer* parent = output_node_map[parent_name];
						vector<int> parent_shape = parent->output->shape;

						int h=parent_shape[2];
						int w=parent_shape[3];

						actual_layer = new LMaxPool(parent, {h,w},{1,1}, "none", "gpool", dev, mem);
					}
					break;

				default:
					cerr << "FATAL: LAYER NOT RECOGNIZED WITH TYPE " << layer_type_name <<  endl;
					nodeQueue.pop();
					continue;
					break;
			}

			for( int i = 0; i < node->output_size(); i++ ) {
				output_node_map[node->output(i)] = actual_layer;
				vector<onnx::NodeProto*> child_nodes = input_node_map[node->output(i)];
				for(onnx::NodeProto * child : child_nodes){
					nodeQueue.push(child);
				}
			}
			nodeQueue.pop();

		}
		vector<Layer *> input_layers;
		for( Layer* layer : inputs) input_layers.push_back(layer);

		vector<string> output_names = get_outputs(graph);
		vector<Layer *> output_layers;
		for( int i = 0; i < output_names.size(); i++ ) {
			output_layers.push_back(output_node_map[output_names[i]]);
		}
		cout << "Net imported from ONNX succesfully" << endl;
		return new Net(input_layers, output_layers);
	}

	void set_weights(Net* net, onnx::ModelProto model){

		map<string, vector<Tensor*> > tensors = get_tensors_from_onnx(model);
		LConv* conv;
		LDense* dense;
		for(Layer* l : net->layers){
			if(!tensors.count(l->name)){
				//cout << "Layer with name " << l->name << " is not trainable " << endl;
				continue;
			}
			vector<Tensor*> layer_tensors = tensors[l->name];
			if((conv = dynamic_cast<LConv*>(l))){
				if(layer_tensors.size() > 1)
					conv->update_weights(layer_tensors[0], layer_tensors[1]);
				else{
					cerr << "EDDL has not implemented convolutional without bias " << endl;
					//conv.update_weights(layer_tensors[0]);
				}

			}
			else if((dense = dynamic_cast<LDense*>( l ) )){
				if(layer_tensors.size() > 1)
					dense->update_weights(layer_tensors[0], layer_tensors[1]);
				else
					dense->update_weights(layer_tensors[0]);
			}
			else cerr << "not implemented layer type" << endl;
		}
		//erase the map we used to free the memory
		map<string, vector<Tensor*> >::iterator it;
		vector<Tensor*> delete_tensors;
		for( it = tensors.begin(); it !=tensors.end(); ++it){
			delete_tensors=it->second;
			for(int i = 0; i < delete_tensors.size(); ++i){
				delete delete_tensors[i];
			}
		}

	}

	//Sets the weights of a input Net to the ones stored in the onnx net inside the pointer
	void set_weights_from_onnx_pointer(Net* net, void *ptr_model, size_t model_size )
	{
		onnx::ModelProto model;
		{
			if(!model.ParseFromArray(ptr_model,model_size)){
				cerr << "Failed to parse model." << endl;
			}
			else if (verbose >= 2) cout << "Model parsed succesfuly" << endl;
		}

		set_weights(net, model);

	}

	//Sets the weights of a input Net to the ones stored in the onnx net inside the c++ string
	void set_weights_from_onnx(Net* net, std::string* model_string){
		onnx::ModelProto model;
		{
			if(!model.ParseFromString(*model_string)){
				cerr << "Failed to parse model." << endl;
			}
			else if (verbose >= 2) cout << "Model parsed succesfuly" << endl;
		}
		set_weights(net, model);

    }

	void apply_grads(Net* net, onnx::ModelProto model){

		map<string, vector<Tensor*> > tensors = get_tensors_from_onnx(model);
		LConv* conv;
		LDense* dense;
		LBatchNorm* bn;
		for(Layer* l : net->layers){
			if(!tensors.count(l->name)) continue;
			vector<Tensor*> layer_tensors = tensors[l->name];
			if((conv = dynamic_cast<LConv*>(l) )){
				if(layer_tensors.size() > 1)
					conv->accumulate_accumulated_gradients(layer_tensors[0], layer_tensors[1]);
				else{
					cerr << "EDDL has not implemented convolutional without bias " << endl;
					//conv.update_weights(layer_tensors[0]);
				}

			}
			else if((dense = dynamic_cast<LDense*>( l ) )){
				if(layer_tensors.size() > 1)
					dense->accumulate_accumulated_gradients(layer_tensors[0], layer_tensors[1]);
				else
					dense->accumulate_accumulated_gradients(layer_tensors[0]);
			}else if((bn = dynamic_cast<LBatchNorm*>( l ) )){
				bn->accumulate_accumulated_gradients(layer_tensors[0], layer_tensors[1]);
			}
			else cerr << "not implemented layer type" << endl;
		}
		//erase the map we used to free the memory
		map<string, vector<Tensor*> >::iterator it;
		vector<Tensor*> delete_tensors;
		for( it = tensors.begin(); it !=tensors.end(); ++it){
			delete_tensors=it->second;
			for(int i = 0; i < delete_tensors.size(); ++i){
				delete delete_tensors[i];
			}
		}
        
        //Distribute the new weights with applyed grads
        net->distribute_weights();

	}
	//Accumulates the gradients stored in the pointer to the input net
    void apply_grads_from_onnx_pointer( Net* net, void * ptr_onnx, size_t count ) {
		onnx::ModelProto model;
		{
			if(!model.ParseFromArray(ptr_onnx,count)){
				cerr << "Failed to parse model." << endl;
			}
			else if (verbose >= 2) cout << "Model parsed succesfuly" << endl;
		}
		apply_grads(net, model);

	}

	//Accumulates the gradients stored in the c++ string to the input net
    void apply_grads_from_onnx(Net* net, std::string* model_string){
		onnx::ModelProto model;
		{
			if(!model.ParseFromString(*model_string)){
				cerr << "Failed to parse model." << endl;
			}
			else if (verbose >= 2) cout << "Model parsed succesfuly" << endl;
		}

		apply_grads(net, model);
    }


	//Returns a map containing the name of the layer as key and a tensor with the values of the model as value
	map<string, vector<Tensor*> > get_tensors_from_onnx(onnx::ModelProto model){

		map<string, vector<Tensor*> > tensors;

		onnx::GraphProto graph = model.graph(); //Get the graph of the model.
		//Model needs input and output in the constructor, so we start with that.

		vector<onnx::TensorProto> initializers = get_initializers(graph); // Retrieves the initializers from the graph.
																		  // The weight for the layers can be found in the initializers.
		map<string, vector<float>> map_init_values;
		map<string, vector<int>>   map_init_dims;
		get_initializers_maps(initializers, map_init_values, map_init_dims); // Creates 2 maps
																			//  Key: Input Name . Value: Weights
																			//  Key: Input Name . Value: Dims
		vector<onnx::NodeProto> nodes = get_graph_nodes(graph);

		map<string, ONNX_LAYERS> map_layers = create_enum_map();
		int dev = DEV_CPU;//TODO: Check what device to use

		for(onnx::NodeProto node : nodes){
			string layer_type_name = node.op_type();
			ONNX_LAYERS layer_type = map_layers[layer_type_name];
			string name = node.name();

			switch (layer_type) {
				case ONNX_LAYERS::CONV:
					{

						vector<Tensor*> conv_tensors;


						string weights_name = node.input(1); //Get weights and dims
						vector<float>* weights = &(map_init_values[weights_name]);
						vector<int> dims = map_init_dims[weights_name];

						conv_tensors.push_back(new Tensor(dims, NEW_FROM_VECTOR_PTR(weights), dev));

						if(node.input_size() > 2){ //This means we also have a bias
							string bias_name = node.input(2);
							vector<float>* bias = &(map_init_values[bias_name]);
							vector<int> bias_shape;
							bias_shape.push_back(bias->size());
							conv_tensors.push_back(new Tensor(bias_shape, NEW_FROM_VECTOR_PTR(bias), dev));
						}

						tensors[name] = conv_tensors;
						break;
					}

				case ONNX_LAYERS::DENSE:
					{

						vector<Tensor*> dense_tensors;

						string weights_name = node.input(1); //Get weights and dims
						vector<float>* weights = &(map_init_values[weights_name]);
						vector<int> dims = map_init_dims[weights_name];

						dense_tensors.push_back(new Tensor(dims, NEW_FROM_VECTOR_PTR(weights), dev));

						if(node.input_size() > 2){
							string bias_name = node.input(2);
							vector<float>* bias = &(map_init_values[bias_name]);
							vector<int> bias_dims = map_init_dims[bias_name];
							dense_tensors.push_back(new Tensor(bias_dims, NEW_FROM_VECTOR_PTR(bias), dev));
						}

						tensors[name] = dense_tensors;

						break;
					}
				case ONNX_LAYERS::BATCHNORM:
					{

						vector<Tensor*> batchnorm_tensors;

						string weights_name = node.input(1); //Get weights and dims
						vector<float>* weights = &(map_init_values[weights_name]);
						vector<int> dims = map_init_dims[weights_name];

						batchnorm_tensors.push_back(new Tensor(dims, NEW_FROM_VECTOR_PTR(weights), dev));

						string bias_name = node.input(2);
						vector<float>* bias = &(map_init_values[bias_name]);
						vector<int> bias_dims = map_init_dims[bias_name];
						batchnorm_tensors.push_back(new Tensor(bias_dims, NEW_FROM_VECTOR_PTR(bias), dev));

						tensors[name] = batchnorm_tensors;

						break;
					}


				default:
					//cout << "The layer with type " << layer_type_name << " has no trainable parameters " << endl;
					continue;
					break;
			}
		}

		return tensors;
	}
#else

	Net* import_net_from_onnx_file(std::string path){
		cerr << "Not compiled for ONNX. Missing protobuf. Returning nullptr" << endl;
		return nullptr;
	}

	Net* import_net_from_onnx_pointer(void* serialized_model, size_t model_size){
		cerr << "Not compiled for ONNX. Missing protobuf. Returning nullptr" << endl;
		return nullptr;
	}

	Net* import_net_from_onnx_string(std::string* model_string){
		cerr << "Not compiled for ONNX. Missing protobuf. Returning nullptr" << endl;
		return nullptr;
	}

#endif //cPROTO

