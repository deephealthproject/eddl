#ifndef EDDL_EDDL_ONNX_H
#define EDDL_EDDL_ONNX_H


/* 2020-01-10
#undef model
#undef layer
#undef initializer
*/
#include "../../net/net.h"
#include <string>
#include <vector>
#include "onnx.pb.h"
#include "../../layers/layer.h"
#include <map>
#include "../../layers/core/layer_core.h"
#include "../../layers/conv/layer_conv.h"
#include "../../layers/pool/layer_pool.h"
#include "../../layers/merge/layer_merge.h"
#include "../../layers/normalization/layer_normalization.h"


namespace eddl{

	//Importing module
	//------------------------------------------------------------------------------

	Net* import_net_from_onnx_file(std::string path);

	Net* import_net_from_onnx_pointer(void* serialized_model, size_t model_size); 
	
	Net* import_net_from_onnx_string(std::string* model_string);

	Net* build_net_onnx(onnx::ModelProto model);

	// Exporting module
	//----------------------------------------------------------------------------------------

	// Saves a model with the onnx format in the file path provided
	void save_net_to_onnx_file( Net *net, string path );

	// Returns a pointer to the serialized model in Onnx
	size_t serialize_net_to_onnx_pointer( Net *net, void * & serialized_model, bool gradients=false );

	// Returns a string containing the serialized model in Onnx 
	std::string* serialize_net_to_onnx_string(Net* net, bool gradients=false);

	// Builds the onnx model from the net
	onnx::ModelProto build_onnx_model( Net *net, bool gradients );

	// Builds the graph of the ModelProto from the net
	void set_graph( onnx::ModelProto *model, Net *net, bool gradients );

	// Builds a node in the onnx graph from the layer of eddl
	void build_node_from_layer( Layer *layer, onnx::GraphProto *graph, bool gradients );

	// Node builders
	//----------------------------------------------------------------------------------------

	void build_conv_node( LConv *layer, onnx::GraphProto *graph, bool gradients );

	void build_gemm_node( LDense *layer, onnx::GraphProto *graph, bool gradients );

	void build_maxpool_node( LMaxPool *layer, onnx::GraphProto *graph );

	void build_reshape_node( LReshape *layer, onnx::GraphProto *graph );

	void build_relu_node( LActivation *layer, onnx::GraphProto *graph );

	void build_softmax_node( LActivation *layer, onnx::GraphProto *graph );

	void build_concat_node( LConcat *layer, onnx::GraphProto *graph );

	void build_batchnorm_node( LBatchNorm *layer, onnx::GraphProto *graph );

	// Distributed Module
	// ---------------------------------------------------------------------------------------
	
	void set_weights_from_onnx(Net* net, std::string* model_string);
	void set_weights_from_onnx_pointer(Net* net, void *ptr_model, size_t model_size );

	void apply_grads_from_onnx(Net* net, std::string* model_string);
    void apply_grads_from_onnx_pointer( Net* net, void * ptr_onnx, size_t count );

	map<string, vector<Tensor*> > get_tensors_from_onnx(onnx::ModelProto model);
}


/* 2020-01-10
#define model	Net*
#define layer	Layer*
#define initializer Initializer*
*/

#endif //EDDL_EDDL_ONNX_H
