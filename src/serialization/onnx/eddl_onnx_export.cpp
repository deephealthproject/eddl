#include <cstdio>
#include <fstream>
#include "eddl/serialization/onnx/eddl_onnx.h"

using namespace std;

namespace eddl {

#ifdef cPROTO

	void save_net_to_onnx_file( Net *net, string path ) {
		// Builds all the model in onnx from the Net object
		bool export_gradients = false; // We always store weights to file
		onnx::ModelProto model = build_onnx_model( net , export_gradients );
		// Create the file stream and save the serialization of the onnx model in it
		fstream ofs( path, ios::out | ios::binary );
    	if ( !model.SerializeToOstream( &ofs ) ) { // The serialization is automated by the protobuf library
			cerr << "Failed to write the model in onnx." << endl;
    	}
	}

	size_t serialize_net_to_onnx_pointer( Net *net, void * & serialized_model, bool gradients ) {
		// Builds all the model in onnx from the Net object
		onnx::ModelProto model = build_onnx_model( net , gradients );
		// Serialization of the model to an array of bytes
		size_t size = model.ByteSizeLong(); // Get the size of the serialized model
		serialized_model = new char [ size ];
		memset( serialized_model, 0 , size );
		if ( ! model.SerializeToArray( serialized_model, size ) ) {
			cerr << "Failed to serialize the model in onnx into the buffer." << endl;
		}
		return size;
	}

	string* serialize_net_to_onnx_string( Net *net, bool gradients) {
		// Builds all the model in onnx from the Net object
		onnx::ModelProto model = build_onnx_model( net , gradients );
		// Serialization of the model to an array of bytes
		string * model_string = new string();
		if ( ! model.SerializeToString(model_string) ) {
			cerr << "Failed to serialize the model in onnx into a string ." << endl;
		}
		return model_string;
	}
	
	onnx::ModelProto build_onnx_model( Net *net , bool gradients ) {
		string producer_name ( "EDDL" ); 
		string producer_version ( "0.1" ); // ????????????????
		// Create the empty Model in onnx
		onnx::ModelProto model;
		model.set_ir_version( onnx::Version::IR_VERSION );
		model.set_producer_name( producer_name );	
		model.set_producer_version( producer_version );

		// Builds all the graph of the model
		set_graph( &model, net, gradients );

		// Return the finished model
		return model;
	}

	void set_graph( onnx::ModelProto *model, Net *net, bool gradients ) {
		// Add a new empty graph to the model
		onnx::GraphProto* graph = model->mutable_graph();
		graph->set_name( "Computational Graph" );
		onnx::OperatorSetIdProto* opset = model->add_opset_import();
		opset->set_version( 11 );

		// Set the inputs shapes of the graph
		for( Layer* input : net->lin ) {
			onnx::ValueInfoProto* input_info = graph->add_input();
			input_info->set_name( input->name ); 
			onnx::TypeProto* input_type = input_info->mutable_type();
			onnx::TypeProto::Tensor* input_type_tensor = input_type->mutable_tensor_type();
			input_type_tensor->set_elem_type( onnx::TensorProto::FLOAT );
			onnx::TensorShapeProto* input_type_tensor_shape = input_type_tensor->mutable_shape();
			onnx::TensorShapeProto::Dimension* input_type_tensor_dim;
			for ( int i : input->input->getShape() ) {
				input_type_tensor_dim = input_type_tensor_shape->add_dim();
				input_type_tensor_dim->set_dim_value( i );
			}
		}
		
		// Set the outputs shapes of the graph
		for( Layer* aux_output : net->lout ) {
			onnx::ValueInfoProto* output_info = graph->add_output();
			output_info->set_name( aux_output->name );
			onnx::TypeProto* output_type = output_info->mutable_type();
			onnx::TypeProto::Tensor* output_type_tensor = output_type->mutable_tensor_type();
			output_type_tensor->set_elem_type(onnx::TensorProto::FLOAT);
			onnx::TensorShapeProto* output_type_tensor_shape = output_type_tensor->mutable_shape();
			onnx::TensorShapeProto::Dimension* output_type_tensor_dim;
			for ( int i : aux_output->output->getShape() ) {
				output_type_tensor_dim = output_type_tensor_shape->add_dim();
				output_type_tensor_dim->set_dim_value( i );
			}
		}

		// Computational graph
		for( Layer* aux_layer : net->layers ) {
			// Builds a node of the graph from the layer in EDDL
			build_node_from_layer( aux_layer, graph, gradients );
		}

	}

	void build_node_from_layer( Layer *layer, onnx::GraphProto *graph, bool gradients ) {
		// Check the class of the layer to call the corresponding function to build the node
		if ( LInput* t = dynamic_cast<LInput*>( layer ) ) 
		{
	    	return; //Skip the input layers
	    } 
		else if ( LConv* t = dynamic_cast<LConv*>( layer ) ) 
		{
	    	build_conv_node( (LConv*)(LinLayer*)layer, graph, gradients );
	    } 
		else if ( LDense *t = dynamic_cast<LDense*>( layer ) ) 
		{
	    	build_gemm_node( (LDense*)(LinLayer*)layer, graph, gradients );
	    } 
		else if ( LMaxPool *t = dynamic_cast<LMaxPool*>( layer ) ) 
		{
	    	build_maxpool_node( (LMaxPool*)(LinLayer*)layer, graph );
	    } 
	    else if ( LAveragePool *t = dynamic_cast<LAveragePool*>( layer ) ) 
		{
	    	build_averagepool_node( (LAveragePool*)(LinLayer*)layer, graph );
	    } 
		else if ( LReshape *t = dynamic_cast<LReshape*>( layer ) ) 
		{
	    	build_reshape_node( (LReshape*)(LinLayer*)layer, graph );
	    } 
	    else if ( LPermute *t = dynamic_cast<LPermute*>( layer ) ) 
		{
	    	build_permute_node( (LPermute*)(OperatorLayer*)layer, graph );
	    } 
	    else if ( LUpSampling *t = dynamic_cast<LUpSampling*>( layer ) ) 
		{
	    	build_upsample_node( (LUpSampling*)(LinLayer*)layer, graph );
	    }  
		else if ( LActivation *t = dynamic_cast<LActivation*>( layer ) ) 
		{
	    	// Check the type of activation layer
	    	if ( !((LActivation *)(layer))->act.compare( "relu" ) ) 
			{
	    		build_relu_node( (LActivation*)(LinLayer*)layer, graph );
	    	} 
	    	else if ( !((LActivation *)(layer))->act.compare( "sigmoid" ) ) 
			{
	    		build_sigmoid_node( (LActivation*)(LinLayer*)layer, graph );
	    	} 
	    	else if ( !((LActivation *)(layer))->act.compare( "hard_sigmoid" ) ) 
			{
	    		build_hard_sigmoid_node( (LActivation*)(LinLayer*)layer, graph );
	    	} 
	    	else if ( !((LActivation *)(layer))->act.compare( "tanh" ) ) 
			{
	    		build_tanh_node( (LActivation*)(LinLayer*)layer, graph );
	    	} 
	    	else if ( !((LActivation *)(layer))->act.compare( "exp" ) ) 
			{
	    		build_exp_node( (LActivation*)(LinLayer*)layer, graph );
	    	} 
	    	else if ( !((LActivation *)(layer))->act.compare( "linear" ) ) 
			{
	    		build_linear_node( (LActivation*)(LinLayer*)layer, graph );
	    	} 
	    	else if ( !((LActivation *)(layer))->act.compare( "leaky_relu" ) ) 
			{
	    		build_leaky_relu_node( (LActivation*)(LinLayer*)layer, graph );
	    	} 
	    	else if ( !((LActivation *)(layer))->act.compare( "thresholded_relu" ) ) 
			{
	    		build_thresholded_relu_node( (LActivation*)(LinLayer*)layer, graph );
	    	} 
	    	else if ( !((LActivation *)(layer))->act.compare( "elu" ) ) 
			{
	    		build_elu_node( (LActivation*)(LinLayer*)layer, graph );
	    	} 
	    	else if ( !((LActivation *)(layer))->act.compare( "selu" ) ) 
			{
	    		build_selu_node( (LActivation*)(LinLayer*)layer, graph );
	    	} 
			else if ( !((LActivation *)(layer))->act.compare( "softmax" ) ) 
			{
	    		build_softmax_node( (LActivation*)(LinLayer*)layer, graph );
	    	} 
	    	else if ( !((LActivation *)(layer))->act.compare( "softsign" ) ) 
			{
	    		build_softsign_node( (LActivation*)(LinLayer*)layer, graph );
	    	} 
	    	else if ( !((LActivation *)(layer))->act.compare( "softplus" ) ) 
			{
	    		build_softplus_node( (LActivation*)(LinLayer*)layer, graph );
	    	} 
			else 
			{
	    		cout  << "The activation layer " << layer->name << "has no valid type to export." << endl;
	    		return;
	    	}

	    } 
		else if ( LConcat *t = dynamic_cast<LConcat*>( layer ) ) 
		{
	    	build_concat_node( (LConcat*)(MLayer*)layer, graph );
	    } 
	    else if ( LAdd *t = dynamic_cast<LAdd*>( layer ) ) 
		{
	    	build_add_node( (LAdd*)(MLayer*)layer, graph );
	    } 
	    else if ( LSubtract *t = dynamic_cast<LSubtract*>( layer ) ) 
		{
	    	build_sub_node( (LSubtract*)(MLayer*)layer, graph );
	    } 
	    else if ( LAverage *t = dynamic_cast<LAverage*>( layer ) ) 
		{
	    	build_average_node( (LAverage*)(MLayer*)layer, graph );
	    } 
	    else if ( LMatMul *t = dynamic_cast<LMatMul*>( layer ) ) 
		{
	    	build_matmul_node( (LMatMul*)(MLayer*)layer, graph );
	    } 
	    else if ( LMaximum *t = dynamic_cast<LMaximum*>( layer ) ) 
		{
	    	build_max_node( (LMaximum*)(MLayer*)layer, graph );
	    } 
	    else if ( LMinimum *t = dynamic_cast<LMinimum*>( layer ) ) 
		{
	    	build_min_node( (LMinimum*)(MLayer*)layer, graph );
	    } 
		else if ( LBatchNorm *t = dynamic_cast<LBatchNorm*>( layer ) ) 
		{
	    	build_batchnorm_node( (LBatchNorm*)(LinLayer*)layer, graph );
	    } 
	    else if ( LDropout *t = dynamic_cast<LDropout*>( layer ) ) 
		{
	    	build_dropout_node( (LDropout*)(LinLayer*)layer, graph );
	    } 
		else 
		{
	    	cout << "The layer " << layer->name << "has no OpType in Onnx." << endl;
	    	return;
	    }
	}

	// Node builders
	//----------------------------------------------------------------------------------------

	void build_conv_node( LConv *layer, onnx::GraphProto *graph, bool gradients ) {
		// Add an empty node to the graph
		onnx::NodeProto* node = graph->add_node();
		node->set_op_type( "Conv" );
		node->set_name( layer->name );
		// Set the inputs of the node from the parents of the layer
		for ( Layer* parentl : layer->parent ) {
			node->add_input( parentl->name );
		}
		// Set the input params names of the conv op
		node->add_input( layer->name + "_W" );
		node->add_input( layer->name + "_b" );
		// Set the name of the output of the node to link with other nodes
		node->add_output( layer->name );

		////////////////////////// Attributes of the Conv operation //////////////////////////////////
		// Attr dilations
		onnx::AttributeProto* conv_dilations = node->add_attribute();
		conv_dilations->set_name( "dilations" );
		conv_dilations->set_type( onnx::AttributeProto::INTS );
		vector<int> vdilations {1, 1}; // ?? Tenemos esto en EDDL
		for ( int i : vdilations ) {
			conv_dilations->add_ints( i );
		}
		//Attr group
		onnx::AttributeProto* conv_group = node->add_attribute();
		conv_group->set_name( "group" );
		conv_group->set_type( onnx::AttributeProto::INT );
		conv_group->set_i( 1 ); // ????????????????????????????????????????????????????
		// Attr kernel_shape
		onnx::AttributeProto* conv_kernel_shape = node->add_attribute();
		conv_kernel_shape->set_name( "kernel_shape" );
		conv_kernel_shape->set_type( onnx::AttributeProto::INTS );
		conv_kernel_shape->add_ints( layer->cd->kr );
		conv_kernel_shape->add_ints( layer->cd->kc );
		// Attr pads
		onnx::AttributeProto* conv_pads = node->add_attribute();
		conv_pads->set_name( "pads" );
		conv_pads->set_type( onnx::AttributeProto::INTS );
		conv_pads->add_ints( layer->cd->padrt );
		conv_pads->add_ints( layer->cd->padcl );
		conv_pads->add_ints( layer->cd->padrb );
		conv_pads->add_ints( layer->cd->padcr );
		// Attr strides
		onnx::AttributeProto* conv_strides = node->add_attribute();
		conv_strides->set_name( "strides" );
		conv_strides->set_type( onnx::AttributeProto::INTS );
		conv_strides->add_ints( layer->cd->sr );
		conv_strides->add_ints( layer->cd->sc );

		// Check if we are exporting weights or accumulated gradients 
		if ( !gradients ) {
			// Weights input
			onnx::TensorProto* conv_w = graph->add_initializer();
			conv_w->set_name( layer->name + "_W" );
			conv_w->set_data_type( onnx::TensorProto::FLOAT );	
			conv_w->mutable_dims()->Add( layer->cd->K->shape.begin(), layer->cd->K->shape.end() ); // Set the shape of the weights
			conv_w->mutable_float_data()->Add( layer->cd->K->ptr, layer->cd->K->ptr + layer->cd->K->size ); // Set the weights values
			//conv_w->mutable_raw_data()->assign( reinterpret_cast<const char*>(layer->cd->K->ptr), sizeof(float) * layer->cd->K->size );
			// Bias input
			onnx::TensorProto* conv_b = graph->add_initializer();
			conv_b->set_name( layer->name + "_b" );
			conv_b->set_data_type( onnx::TensorProto::FLOAT );	
			conv_b->mutable_dims()->Add( layer->cd->bias->shape.begin(), layer->cd->bias->shape.end() ); // Set the shape of the bias
			conv_b->mutable_float_data()->Add( layer->cd->bias->ptr, layer->cd->bias->ptr + layer->cd->bias->size); // Set the bias values
			//conv_b->mutable_raw_data()->assign( reinterpret_cast<const char*>(layer->cd->bias->ptr), sizeof(float) * layer->cd->bias->size );
		} else {
			// Accumulated gradients (Weights) input
			onnx::TensorProto* conv_w = graph->add_initializer();
			conv_w->set_name( layer->name + "_W" );
			conv_w->set_data_type( onnx::TensorProto::FLOAT );	
			conv_w->mutable_dims()->Add( layer->cd->acc_gK->shape.begin(), layer->cd->acc_gK->shape.end() ); // Set the accumulated gradiens shape (weights)
			conv_w->mutable_float_data()->Add( layer->cd->acc_gK->ptr, layer->cd->acc_gK->ptr + layer->cd->acc_gK->size ); // Set the accumulated gradients values (weights) 
			//conv_w->mutable_raw_data()->assign( reinterpret_cast<const char*>(layer->cd->acc_gK->ptr), sizeof(float) * layer->cd->acc_gK->size );
			// Accumulated gradients (bias) input
			onnx::TensorProto* conv_b = graph->add_initializer();
			conv_b->set_name( layer->name + "_b" );
			conv_b->set_data_type( onnx::TensorProto::FLOAT );	
			conv_b->mutable_dims()->Add( layer->cd->acc_gbias->shape.begin(), layer->cd->acc_gbias->shape.end() ); // Set the accumulated gradients shape (bias)
			conv_b->mutable_float_data()->Add( layer->cd->acc_gbias->ptr, layer->cd->acc_gbias->ptr + layer->cd->acc_gbias->size); // Set the accumulated gradients values (bias)
			//conv_b->mutable_raw_data()->assign( reinterpret_cast<const char*>(layer->cd->acc_gbias->ptr), sizeof(float) * layer->cd->acc_gbias->size );
		}
	}
	
	void build_gemm_node( LDense *layer, onnx::GraphProto *graph, bool gradients) {
		// Add an empty node to the graph
		onnx::NodeProto* node = graph->add_node();
		node->set_op_type( "Gemm" );
		node->set_name( layer->name );
		// Set the inputs of the node from the parents of the layer
		for ( Layer* parentl : layer->parent ) {
			node->add_input( parentl->name );
		}
		// Set the input params names of the Gemm(Dense) op
		node->add_input( layer->name + "_W" );
		node->add_input( layer->name + "_b" );
		// Set the name of the output of the node to link with other nodes
		node->add_output( layer->name );

		// Attr alpha
		onnx::AttributeProto* dense_alpha = node->add_attribute();
		dense_alpha->set_name( "alpha" );
		dense_alpha->set_type( onnx::AttributeProto::FLOAT );
		dense_alpha->set_f( 1 );
		// Attr beta
		onnx::AttributeProto* dense_beta = node->add_attribute();
		dense_beta->set_name( "beta" );
		dense_beta->set_type( onnx::AttributeProto::FLOAT );
		dense_beta->set_f( layer->use_bias );
		// Attr transA
		onnx::AttributeProto* dense_transA = node->add_attribute();
		dense_transA->set_name( "transA" );
		dense_transA->set_type( onnx::AttributeProto::INT );
		dense_transA->set_i( 0 ); 
		// Attr transB
		onnx::AttributeProto* dense_transB = node->add_attribute();
		dense_transB->set_name( "transB" );
		dense_transB->set_type( onnx::AttributeProto::INT );
		dense_transB->set_i( 0 );

		// Check if we are exporting weights or accumulated gradients 
		if ( !gradients ) {
			// Weights input
			onnx::TensorProto* weight = graph->add_initializer();
			weight->set_name( layer->name + "_W" );
			weight->set_data_type( onnx::TensorProto::FLOAT );	
			weight->mutable_dims()->Add( layer->W->shape.begin(), layer->W->shape.end() ); // Set the shape of the weights
			weight->mutable_float_data()->Add( layer->W->ptr, layer->W->ptr + layer->W->size ); // Set the weights values
			//weight->mutable_raw_data()->assign( reinterpret_cast<const char*>(layer->W->ptr), sizeof(float) * layer->W->size );
			if ( layer->use_bias ) {
				// Bias input
				onnx::TensorProto* bias = graph->add_initializer();
				bias->set_name( layer->name + "_b" );
				bias->set_data_type( onnx::TensorProto::FLOAT );	
				bias->mutable_dims()->Add( layer->bias->shape.begin(), layer->bias->shape.end() ); // Set the bias shape
				bias->mutable_float_data()->Add( layer->bias->ptr, layer->bias->ptr + layer->bias->size ); // Set the bias values
				//bias->mutable_raw_data()->assign( reinterpret_cast<const char*>(layer->bias->ptr), sizeof(float) * layer->bias->size );
			}		
		} else {
			// Accumulated gradients (Weights) input
			onnx::TensorProto* weight = graph->add_initializer();
			weight->set_name( layer->name + "_W" );
			weight->set_data_type( onnx::TensorProto::FLOAT );	
			weight->mutable_dims()->Add( layer->acc_gW->shape.begin(), layer->acc_gW->shape.end() ); // Set the accumulated gradients shape (weights)
			weight->mutable_float_data()->Add( layer->acc_gW->ptr, layer->acc_gW->ptr + layer->acc_gW->size ); // Set the accumulated gradients values (weights)
			//weight->mutable_raw_data()->assign( reinterpret_cast<const char*>(layer->acc_gW->ptr), sizeof(float) * layer->acc_gW->size );

			// Check if we are using bias 
			if ( layer->use_bias ) {
				// Accumulated gradients (bias) input
				onnx::TensorProto* bias = graph->add_initializer();
				bias->set_name( layer->name + "_b" );
				bias->set_data_type( onnx::TensorProto::FLOAT );	
				bias->mutable_dims()->Add( layer->acc_gbias->shape.begin(), layer->acc_gbias->shape.end() ); // Set the accumulated gradients shape (bias)
				bias->mutable_float_data()->Add( layer->acc_gbias->ptr, layer->acc_gbias->ptr + layer->acc_gbias->size ); // Set the accumulated gradients values (bias)
				//bias->mutable_raw_data()->assign( reinterpret_cast<const char*>(layer->acc_gbias->ptr), sizeof(float) * layer->acc_gbias->size );
			}
		}
	}

	void build_maxpool_node( LMaxPool *layer, onnx::GraphProto *graph ) {
		// Add an empty node to the graph
		onnx::NodeProto* node = graph->add_node();
		node->set_op_type( "MaxPool" );
		node->set_name( layer->name );
		// Set the inputs of the node from the parents of the layer
		for ( Layer* parentl : layer->parent ) {
			node->add_input( parentl->name );
		}
		// Set the name of the output of the node to link with other nodes
		node->add_output( layer->name );

		// Attr kernel_shape
		onnx::AttributeProto* max_pool_ks = node->add_attribute();
		max_pool_ks->set_name( "kernel_shape" );
		max_pool_ks->set_type( onnx::AttributeProto::INTS );
		for ( int i : layer->pd->ksize ) {
			max_pool_ks->add_ints( i );
		}
		// Attr pads
		onnx::AttributeProto* max_pool_pads = node->add_attribute();
		max_pool_pads->set_name( "pads" );
		max_pool_pads->set_type( onnx::AttributeProto::INTS );
		max_pool_pads->add_ints( layer->pd->padrt );
		max_pool_pads->add_ints( layer->pd->padcl );
		max_pool_pads->add_ints( layer->pd->padrb );
		max_pool_pads->add_ints( layer->pd->padcr );
		// Attr strides
		onnx::AttributeProto* max_pool_strides = node->add_attribute();
		max_pool_strides->set_name( "strides" );
		max_pool_strides->set_type( onnx::AttributeProto::INTS );
		max_pool_strides->add_ints( layer->pd->sr );
		max_pool_strides->add_ints( layer->pd->sc );
	}

	void build_averagepool_node( LAveragePool *layer, onnx::GraphProto *graph ) {
		// Add an empty node to the graph
		onnx::NodeProto* node = graph->add_node();
		node->set_op_type( "AveragePool" );
		node->set_name( layer->name );
		// Set the inputs of the node from the parents of the layer
		for ( Layer* parentl : layer->parent ) {
			node->add_input( parentl->name );
		}
		// Set the name of the output of the node to link with other nodes
		node->add_output( layer->name );

		// Attr kernel_shape
		onnx::AttributeProto* max_pool_ks = node->add_attribute();
		max_pool_ks->set_name( "kernel_shape" );
		max_pool_ks->set_type( onnx::AttributeProto::INTS );
		for ( int i : layer->pd->ksize ) {
			max_pool_ks->add_ints( i );
		}
		// Attr pads
		onnx::AttributeProto* max_pool_pads = node->add_attribute();
		max_pool_pads->set_name( "pads" );
		max_pool_pads->set_type( onnx::AttributeProto::INTS );
		max_pool_pads->add_ints( layer->pd->padrt );
		max_pool_pads->add_ints( layer->pd->padcl );
		max_pool_pads->add_ints( layer->pd->padrb );
		max_pool_pads->add_ints( layer->pd->padcr );
		// Attr strides
		onnx::AttributeProto* max_pool_strides = node->add_attribute();
		max_pool_strides->set_name( "strides" );
		max_pool_strides->set_type( onnx::AttributeProto::INTS );
		max_pool_strides->add_ints( layer->pd->sr );
		max_pool_strides->add_ints( layer->pd->sc );
	}

	void build_reshape_node( LReshape *layer, onnx::GraphProto *graph ) {
		// Add an empty node to the graph
		onnx::NodeProto* node = graph->add_node();
		node->set_op_type( "Reshape" );
		node->set_name( layer->name );
		// Set the inputs names of the node from the parents of the layer
		for ( Layer* parentl : layer->parent ) {
			node->add_input( parentl->name );
		}
		// Set the input with the target shape of the op
		node->add_input( layer->name + "_target_shape" );
		// Set the name of the output of the node to link with other nodes
		node->add_output( layer->name );

		// Constant node input to the reshape node: shape
		onnx::NodeProto* shape_const_node = graph->add_node();
		shape_const_node->add_output( layer->name + "_target_shape" );
		shape_const_node->set_op_type( "Constant" );
		onnx::AttributeProto* shape_attr = shape_const_node->add_attribute();
		shape_attr->set_name( "value" );
		shape_attr->set_type( onnx::AttributeProto::TENSOR );
		onnx::TensorProto* target_shape_tensor = shape_attr->mutable_t();
		target_shape_tensor->set_name( "const_tensor" );
		target_shape_tensor->set_data_type( onnx::TensorProto::INT64 );
		target_shape_tensor->add_dims( layer->ls.size() );
		// Set the target shape
		for ( int i : layer->ls ) {
			target_shape_tensor->add_int64_data( i );
		}
	}

	void build_permute_node( LPermute *layer, onnx::GraphProto *graph ) {
		// Add an empty node to the graph
		onnx::NodeProto* node = graph->add_node();
		node->set_op_type( "Transpose" );
		node->set_name( layer->name );
		// Set the inputs names of the node from the parents of the layer
		for ( Layer* parentl : layer->parent ) {
			node->add_input( parentl->name );
		}
		// Set the name of the output of the node to link with other nodes
		node->add_output( layer->name );

		// Attr perm
		onnx::AttributeProto* alpha_attr = node->add_attribute();
		alpha_attr->set_name( "perm" );
		alpha_attr->set_type( onnx::AttributeProto::INTS );
		for ( int i : layer->sd->dims ) {
			alpha_attr->add_ints( i );
		}
	}

	void build_relu_node( LActivation *layer, onnx::GraphProto *graph ) {
		// Add an empty node to the graph
		onnx::NodeProto* node = graph->add_node();
		node->set_op_type( "Relu" );
		node->set_name( layer->name );
		// Set the inputs names of the node from the parents of the layer
		for ( Layer* parentl : layer->parent ) {
			node->add_input( parentl->name );
		}
		// Set the name of the output of the node to link with other nodes
		node->add_output( layer->name );
	}

	void build_sigmoid_node( LActivation *layer, onnx::GraphProto *graph ) {
		// Add an empty node to the graph
		onnx::NodeProto* node = graph->add_node();
		node->set_op_type( "Sigmoid" );
		node->set_name( layer->name );
		// Set the inputs names of the node from the parents of the layer
		for ( Layer* parentl : layer->parent ) {
			node->add_input( parentl->name );
		}
		// Set the name of the output of the node to link with other nodes
		node->add_output( layer->name );
	}

	void build_hard_sigmoid_node( LActivation *layer, onnx::GraphProto *graph ) {
		// Add an empty node to the graph
		onnx::NodeProto* node = graph->add_node();
		node->set_op_type( "HardSigmoid" );
		node->set_name( layer->name );
		// Set the inputs names of the node from the parents of the layer
		for ( Layer* parentl : layer->parent ) {
			node->add_input( parentl->name );
		}
		// Set the name of the output of the node to link with other nodes
		node->add_output( layer->name );
	}

	void build_tanh_node( LActivation *layer, onnx::GraphProto *graph ) {
		// Add an empty node to the graph
		onnx::NodeProto* node = graph->add_node();
		node->set_op_type( "Tanh" );
		node->set_name( layer->name );
		// Set the inputs names of the node from the parents of the layer
		for ( Layer* parentl : layer->parent ) {
			node->add_input( parentl->name );
		}
		// Set the name of the output of the node to link with other nodes
		node->add_output( layer->name );
	}

	void build_exp_node( LActivation *layer, onnx::GraphProto *graph ) {
		// Add an empty node to the graph
		onnx::NodeProto* node = graph->add_node();
		node->set_op_type( "Exp" );
		node->set_name( layer->name );
		// Set the inputs names of the node from the parents of the layer
		for ( Layer* parentl : layer->parent ) {
			node->add_input( parentl->name );
		}
		// Set the name of the output of the node to link with other nodes
		node->add_output( layer->name );
	}

	void build_linear_node( LActivation *layer, onnx::GraphProto *graph ) {
		// Add an empty node to the graph
		onnx::NodeProto* node = graph->add_node();
		node->set_op_type( "Linear" );
		node->set_name( layer->name );
		// Set the inputs names of the node from the parents of the layer
		for ( Layer* parentl : layer->parent ) {
			node->add_input( parentl->name );
		}
		// Set the name of the output of the node to link with other nodes
		node->add_output( layer->name );

		// Attr alpha
		onnx::AttributeProto* alpha_attr = node->add_attribute();
		alpha_attr->set_name( "alpha" );
		alpha_attr->set_type( onnx::AttributeProto::FLOAT );
		alpha_attr->set_f( layer->params[0] );
	}

	void build_leaky_relu_node( LActivation *layer, onnx::GraphProto *graph ) {
		// Add an empty node to the graph
		onnx::NodeProto* node = graph->add_node();
		node->set_op_type( "LeakyRelu" );
		node->set_name( layer->name );
		// Set the inputs names of the node from the parents of the layer
		for ( Layer* parentl : layer->parent ) {
			node->add_input( parentl->name );
		}
		// Set the name of the output of the node to link with other nodes
		node->add_output( layer->name );

		// Attr alpha
		onnx::AttributeProto* alpha_attr = node->add_attribute();
		alpha_attr->set_name( "alpha" );
		alpha_attr->set_type( onnx::AttributeProto::FLOAT );
		alpha_attr->set_f( layer->params[0] );
	}

	void build_thresholded_relu_node( LActivation *layer, onnx::GraphProto *graph ) {
		// Add an empty node to the graph
		onnx::NodeProto* node = graph->add_node();
		node->set_op_type( "ThresholdedRelu" );
		node->set_name( layer->name );
		// Set the inputs names of the node from the parents of the layer
		for ( Layer* parentl : layer->parent ) {
			node->add_input( parentl->name );
		}
		// Set the name of the output of the node to link with other nodes
		node->add_output( layer->name );

		// Attr alpha
		onnx::AttributeProto* alpha_attr = node->add_attribute();
		alpha_attr->set_name( "alpha" );
		alpha_attr->set_type( onnx::AttributeProto::FLOAT );
		alpha_attr->set_f( layer->params[0] );
	}

	void build_elu_node( LActivation *layer, onnx::GraphProto *graph ) {
		// Add an empty node to the graph
		onnx::NodeProto* node = graph->add_node();
		node->set_op_type( "Elu" );
		node->set_name( layer->name );
		// Set the inputs names of the node from the parents of the layer
		for ( Layer* parentl : layer->parent ) {
			node->add_input( parentl->name );
		}
		// Set the name of the output of the node to link with other nodes
		node->add_output( layer->name );

		// Attr alpha
		onnx::AttributeProto* alpha_attr = node->add_attribute();
		alpha_attr->set_name( "alpha" );
		alpha_attr->set_type( onnx::AttributeProto::FLOAT );
		alpha_attr->set_f( layer->params[0] );
	}

	void build_selu_node( LActivation *layer, onnx::GraphProto *graph ) {
		// Add an empty node to the graph
		onnx::NodeProto* node = graph->add_node();
		node->set_op_type( "Selu" );
		node->set_name( layer->name );
		// Set the inputs names of the node from the parents of the layer
		for ( Layer* parentl : layer->parent ) {
			node->add_input( parentl->name );
		}
		// Set the name of the output of the node to link with other nodes
		node->add_output( layer->name );

		// Attr alpha
		onnx::AttributeProto* alpha_attr = node->add_attribute();
		alpha_attr->set_name( "alpha" );
		alpha_attr->set_type( onnx::AttributeProto::FLOAT );
		alpha_attr->set_f( layer->params[0] );

		// Attr gamma
		onnx::AttributeProto* gamma_attr = node->add_attribute();
		gamma_attr->set_name( "gamma" );
		gamma_attr->set_type( onnx::AttributeProto::FLOAT );
		gamma_attr->set_f( layer->params[1] );
	}

	void build_softmax_node( LActivation *layer, onnx::GraphProto *graph ) {
		// Add an empty node to the graph
		onnx::NodeProto* node = graph->add_node();
		node->set_op_type( "Softmax" );
		node->set_name( layer->name );
		// Set the inputs names of the node from the parents of the layer
		for ( Layer* parentl : layer->parent ) {
			node->add_input( parentl->name );
		}
		// Set the name of the output of the node to link with other nodes
		node->add_output( layer->name );
	}

	void build_softsign_node( LActivation *layer, onnx::GraphProto *graph ) {
		// Add an empty node to the graph
		onnx::NodeProto* node = graph->add_node();
		node->set_op_type( "Softsign" );
		node->set_name( layer->name );
		// Set the inputs names of the node from the parents of the layer
		for ( Layer* parentl : layer->parent ) {
			node->add_input( parentl->name );
		}
		// Set the name of the output of the node to link with other nodes
		node->add_output( layer->name );
	}

	void build_softplus_node( LActivation *layer, onnx::GraphProto *graph ) {
		// Add an empty node to the graph
		onnx::NodeProto* node = graph->add_node();
		node->set_op_type( "Softplus" );
		node->set_name( layer->name );
		// Set the inputs names of the node from the parents of the layer
		for ( Layer* parentl : layer->parent ) {
			node->add_input( parentl->name );
		}
		// Set the name of the output of the node to link with other nodes
		node->add_output( layer->name );
	}

	void build_concat_node( LConcat *layer, onnx::GraphProto *graph ) {
		// Add an empty node to the graph
		onnx::NodeProto* node = graph->add_node();
		node->set_op_type( "Concat" );
		node->set_name( layer->name );
		// Set the inputs names of the node from the parents of the layer
		for ( Layer* parentl : layer->parent ) {
			node->add_input( parentl->name );
		}
		// Set the name of the output of the node to link with other nodes
		node->add_output( layer->name );

		// Attr axis
		onnx::AttributeProto* concat_axis = node->add_attribute();
		concat_axis->set_name( "axis" );
		concat_axis->set_type( onnx::AttributeProto::INT );
		concat_axis->set_i( 1 );
	}

	void build_add_node( LAdd *layer, onnx::GraphProto *graph ) {
		// Add an empty node to the graph
		onnx::NodeProto* node = graph->add_node();
		node->set_op_type( "Add" );
		node->set_name( layer->name );
		// Set the inputs names of the node from the parents of the layer
		for ( Layer* parentl : layer->parent ) {
			node->add_input( parentl->name );
		}
		// Set the name of the output of the node to link with other nodes
		node->add_output( layer->name );
	}

	void build_sub_node( LSubtract *layer, onnx::GraphProto *graph ) {
		// Add an empty node to the graph
		onnx::NodeProto* node = graph->add_node();
		node->set_op_type( "Sub" );
		node->set_name( layer->name );
		// Set the inputs names of the node from the parents of the layer
		for ( Layer* parentl : layer->parent ) {
			node->add_input( parentl->name );
		}
		// Set the name of the output of the node to link with other nodes
		node->add_output( layer->name );
	}

	void build_average_node( LAverage *layer, onnx::GraphProto *graph ) {
		// Add an empty node to the graph
		onnx::NodeProto* node = graph->add_node();
		node->set_op_type( "Average" );
		node->set_name( layer->name );
		// Set the inputs names of the node from the parents of the layer
		for ( Layer* parentl : layer->parent ) {
			node->add_input( parentl->name );
		}
		// Set the name of the output of the node to link with other nodes
		node->add_output( layer->name );
	}

	void build_matmul_node( LMatMul *layer, onnx::GraphProto *graph ) {
		// Add an empty node to the graph
		onnx::NodeProto* node = graph->add_node();
		node->set_op_type( "MatMul" );
		node->set_name( layer->name );
		// Set the inputs names of the node from the parents of the layer
		for ( Layer* parentl : layer->parent ) {
			node->add_input( parentl->name );
		}
		// Set the name of the output of the node to link with other nodes
		node->add_output( layer->name );
	}

	void build_max_node( LMaximum *layer, onnx::GraphProto *graph ) {
		// Add an empty node to the graph
		onnx::NodeProto* node = graph->add_node();
		node->set_op_type( "Max" );
		node->set_name( layer->name );
		// Set the inputs names of the node from the parents of the layer
		for ( Layer* parentl : layer->parent ) {
			node->add_input( parentl->name );
		}
		// Set the name of the output of the node to link with other nodes
		node->add_output( layer->name );
	}

	void build_min_node( LMinimum *layer, onnx::GraphProto *graph ) {
		// Add an empty node to the graph
		onnx::NodeProto* node = graph->add_node();
		node->set_op_type( "Min" );
		node->set_name( layer->name );
		// Set the inputs names of the node from the parents of the layer
		for ( Layer* parentl : layer->parent ) {
			node->add_input( parentl->name );
		}
		// Set the name of the output of the node to link with other nodes
		node->add_output( layer->name );
	}

	void build_batchnorm_node( LBatchNorm *layer, onnx::GraphProto *graph ) {
		// Add an empty node to the graph
		onnx::NodeProto* node = graph->add_node();
		node->set_op_type( "BatchNormalization" );
		node->set_name( layer->name );
		// Set the inputs of the node from the parents of the layer
		for ( Layer* parentl : layer->parent ) {
			node->add_input( parentl->name );
		}
		node->add_input( layer->name + "_scale" );
		node->add_input( layer->name + "_bias" );
		node->add_input( layer->name + "_mean"  );
		node->add_input( layer->name + "_variance" );
		// Set the name of the output of the node to link with other nodes
		node->add_output( layer->name );
		// Attr epsilon
		onnx::AttributeProto* epsilon_attr = node->add_attribute();
		epsilon_attr->set_name( "epsilon" );
		epsilon_attr->set_type( onnx::AttributeProto::FLOAT );
		epsilon_attr->set_f( layer->epsilon );
		// Attr momentum
		onnx::AttributeProto* momentum_attr = node->add_attribute();
		momentum_attr->set_name( "momentum" );
		momentum_attr->set_type( onnx::AttributeProto::FLOAT );
		momentum_attr->set_f( layer->momentum );

		int n_features = layer->input->getShape()[1];

		// Scale input
		onnx::TensorProto* scale = graph->add_initializer();
		scale->set_name( layer->name + "_scale" );
		scale->set_data_type( onnx::TensorProto::FLOAT );	
		scale->add_dims( n_features );	

		// Bias input
		onnx::TensorProto* bias = graph->add_initializer();
		bias->set_name( layer->name + "_bias" );
		bias->set_data_type( onnx::TensorProto::FLOAT );	
		bias->add_dims( n_features );

		// Check if the layer has trainable parameters
		if ( layer->affine ) 
		{
			for( int i = 0; i < n_features; ++i ) {
				scale->add_float_data( layer->bn_g->ptr[i] );
				bias->add_float_data( layer->bn_b->ptr[i] );
			}
		} 
		else 
		{
			for( int i = 0; i < n_features; ++i ) {
				// Set the scale values to 1 (1 is the default value in case of not having trainable parameters)
				scale->add_float_data( 1 );
				// Set the bias values to 0 (0 is the default value in case of not having trainable parameters)
				bias->add_float_data( 0 );
			}
		}

		// Mean input
		onnx::TensorProto* mean = graph->add_initializer();
		mean->set_name( layer->name + "_mean" );
		mean->set_data_type( onnx::TensorProto::FLOAT );	
		mean->add_dims( n_features );
		mean->mutable_float_data()->Add( layer->mean->ptr, layer->mean->ptr + layer->mean->size ); // Set the mean values

		// variance input
		onnx::TensorProto* variance = graph->add_initializer();
		variance->set_name( layer->name + "_variance" );
		variance->set_data_type( onnx::TensorProto::FLOAT );	
		variance->add_dims( n_features );
		variance->mutable_float_data()->Add( layer->variance->ptr, layer->variance->ptr + layer->variance->size ); // Set the mean values
	}

	void build_dropout_node( LDropout *layer, onnx::GraphProto *graph ) {
		// Add an empty node to the graph
		onnx::NodeProto* node = graph->add_node();
		node->set_op_type( "Dropout" );
		node->set_name( layer->name );
		// Set the inputs of the node from the parents of the layer
		for ( Layer* parentl : layer->parent ) {
			node->add_input( parentl->name );
		}
		// Set the name of the output of the node to link with other nodes
		node->add_output( layer->name );

		// Attr ratio
		onnx::AttributeProto* momentum_attr = node->add_attribute();
		momentum_attr->set_name( "ratio" );
		momentum_attr->set_type( onnx::AttributeProto::FLOAT );
		momentum_attr->set_f( layer->df );
	}

	void build_upsample_node( LUpSampling *layer, onnx::GraphProto *graph ) {
		// Add an empty node to the graph
		onnx::NodeProto* node = graph->add_node();
		node->set_op_type( "Upsample" );
		node->set_name( layer->name );
		// Set the inputs of the node from the parents of the layer
		for ( Layer* parentl : layer->parent ) {
			node->add_input( parentl->name );
		}
		// Set the input with the scale values
		node->add_input( layer->name + "_scales" );	
		// Set the name of the output of the node to link with other nodes
		node->add_output( layer->name );

		// Attr mode
		onnx::AttributeProto* mode_attr = node->add_attribute();
		mode_attr->set_name( "mode" );
		mode_attr->set_type( onnx::AttributeProto::STRING );
		mode_attr->set_s( layer->interpolation );

		// Scales input
		onnx::TensorProto* scales = graph->add_initializer();
		scales->set_name( layer->name + "_scales" );
		scales->set_data_type( onnx::TensorProto::FLOAT );	
		scales->add_dims( 2 + layer->size.size() ); // (batch_size, channels, height, width)

		// Add the scale factor for the first two dimensions
		for( int i = 0; i < 2; ++i ) {
			scales->add_float_data( 1 );
		}	

		for( int i = 0; i < layer->size.size(); ++i) {
			scales->add_float_data( layer->size[i] );
		}
	}
	// End: Node builders
	//----------------------------------------------------------------------------------------

	// End: Exporting Module
	//----------------------------------------------------------------------------------------
#else
	void save_net_to_onnx_file( Net *net, string path ){
		cerr << "Not compiled for ONNX. Missing Protobuf" << endl;
	}

	size_t serialize_net_to_onnx_pointer( Net *net, void * & serialized_model, bool gradients ){
		cerr << "Not compiled for ONNX. Missing Protobuf. Returning -1" << endl;
		return -1;
	}

	std::string* serialize_net_to_onnx_string(Net* net, bool gradients){
		cerr << "Not compiled for ONNX. Missing Protobuf. Returning nullptr" << endl;
		return nullptr;
	}

#endif //cPROTO

}
