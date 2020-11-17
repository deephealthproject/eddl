#ifndef EDDL_EDDL_ONNX_H
#define EDDL_EDDL_ONNX_H


#include <string>
#include <vector>
#include <map>
#include "eddl/net/net.h"
#include "eddl/layers/layer.h"
#include "eddl/layers/core/layer_core.h"
#include "eddl/layers/conv/layer_conv.h"
#include "eddl/layers/recurrent/layer_recurrent.h"
#include "eddl/layers/pool/layer_pool.h"
#include "eddl/layers/merge/layer_merge.h"
#include "eddl/layers/operators/layer_operators.h"
#include "eddl/layers/normalization/layer_normalization.h"

//#if defined(cPROTO)
//#   include "serialization/onnx/onnx.pb.h"
//#endif
	enum LOG_LEVEL{
		TRACE = 0,
		DEBUG = 1,
		INFO = 2,
		WARN = 3,
		ERROR = 4,
		NO_LOGS = 5
	};



	//Importing module
	//------------------------------------------------------------------------------

	Net* import_net_from_onnx_file(std::string path, int mem=0, int log_level= LOG_LEVEL::INFO );

	Net* import_net_from_onnx_pointer(void* serialized_model, size_t model_size, int mem=0); 
	
	Net* import_net_from_onnx_string(std::string* model_string, int mem=0);

//#if defined(cPROTO)
//	Net* build_net_onnx(onnx::ModelProto model, int mem);
//#endif

	// Exporting module
	//----------------------------------------------------------------------------------------

	// Saves a model with the onnx format in the file path provided
	void save_net_to_onnx_file( Net *net, string path );

	// Returns a pointer to the serialized model in Onnx
	size_t serialize_net_to_onnx_pointer( Net *net, void * & serialized_model, bool gradients=false );

	// Returns a string containing the serialized model in Onnx 
	std::string* serialize_net_to_onnx_string(Net* net, bool gradients=false);

	// Builds the onnx model from the net
//#if defined(cPROTO)
//	onnx::ModelProto build_onnx_model( Net *net, bool gradients );
//
//	// Builds the graph of the ModelProto from the net
//	void set_graph( onnx::ModelProto *model, Net *net, bool gradients );
//
//	// Builds a node in the onnx graph from the layer of eddl
//	void build_node_from_layer( Layer *layer, onnx::GraphProto *graph, bool gradients );
//
//	// Node builders
//	//----------------------------------------------------------------------------------------
//
//	void build_conv_node( LConv *layer, onnx::GraphProto *graph, bool gradients );
//
//	void build_gemm_node( LDense *layer, onnx::GraphProto *graph, bool gradients );
//
//	void build_maxpool_node( LMaxPool *layer, onnx::GraphProto *graph );
//
//	void build_averagepool_node( LAveragePool *layer, onnx::GraphProto *graph );
//
//	void build_reshape_node( LReshape *layer, onnx::GraphProto *graph );
//
//	void build_permute_node( LPermute *layer, onnx::GraphProto *graph );
//
//	void build_relu_node( LActivation *layer, onnx::GraphProto *graph );
//
//	void build_sigmoid_node( LActivation *layer, onnx::GraphProto *graph );
//
//	void build_hard_sigmoid_node( LActivation *layer, onnx::GraphProto *graph );
//
//	void build_tanh_node( LActivation *layer, onnx::GraphProto *graph );
//
//	void build_exp_node( LActivation *layer, onnx::GraphProto *graph );
//
//	void build_linear_node( LActivation *layer, onnx::GraphProto *graph );
//
//	void build_leaky_relu_node( LActivation *layer, onnx::GraphProto *graph );
//
//	void build_thresholded_relu_node( LActivation *layer, onnx::GraphProto *graph );
//
//	void build_elu_node( LActivation *layer, onnx::GraphProto *graph );
//
//	void build_selu_node( LActivation *layer, onnx::GraphProto *graph );
//
//	void build_softmax_node( LActivation *layer, onnx::GraphProto *graph );
//
//	void build_softsign_node( LActivation *layer, onnx::GraphProto *graph );
//
//	void build_softplus_node( LActivation *layer, onnx::GraphProto *graph );
//
//	void build_concat_node( LConcat *layer, onnx::GraphProto *graph );
//
//	void build_add_node( LAdd *layer, onnx::GraphProto *graph );
//
//	void build_sub_node( LSubtract *layer, onnx::GraphProto *graph );
//
//	void build_average_node( LAverage *layer, onnx::GraphProto *graph );
//
//	void build_matmul_node( LMatMul *layer, onnx::GraphProto *graph );
//
//	void build_max_node( LMaximum *layer, onnx::GraphProto *graph );
//
//	void build_min_node( LMinimum *layer, onnx::GraphProto *graph );
//
//	void build_batchnorm_node( LBatchNorm *layer, onnx::GraphProto *graph );
//
//	void build_dropout_node( LDropout *layer, onnx::GraphProto *graph );
//
//	void build_upsample_node( LUpSampling *layer, onnx::GraphProto *graph );
//#endif

	// Distributed Module
	// ---------------------------------------------------------------------------------------
	
	void set_weights_from_onnx(Net* net, std::string* model_string);
	void set_weights_from_onnx_pointer(Net* net, void *ptr_model, size_t model_size );

	void apply_grads_from_onnx(Net* net, std::string* model_string);
    void apply_grads_from_onnx_pointer( Net* net, void * ptr_onnx, size_t count );

//#if defined(cPROTO)
//	map<string, vector<Tensor*> > get_tensors_from_onnx(onnx::ModelProto model);
//#endif


#endif //EDDL_EDDL_ONNX_H
