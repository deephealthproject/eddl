#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/merge/matmul_onnx.h"
#include "eddl/layers/core/layer_core.h"

// ONNX import
Layer* build_matmul_layer(onnx::NodeProto *node,
                          map<string, vector<float>> &map_init_values,
                          map<string, vector<int>> &map_init_dims,
                          map<string, Layer *> &output_node_map,
                          int dev,
                          int mem)
{
  vector<Layer *> parents;
  string parent_name;
  bool dense_detected = false;
  int index_parameter = -1;
  for (int j = 0; j < node->input_size(); j++)
  {
    parent_name = node->input(j);
    if (map_init_values.count(parent_name))
    {
      // Dense detected
      if (dense_detected)
      {
        cerr << "MAT_MUL with two parameters" << endl;
      }
      dense_detected = true;
      index_parameter = j;
    }
    else
      parents.push_back(output_node_map[parent_name]);
  }
  if (dense_detected)
  {
    string weights_name = node->input(index_parameter);
    vector<float> *weights = &(map_init_values[weights_name]);
    vector<int> dims = map_init_dims[weights_name];
    int neuronas = dims[1];
    Layer *parent = parents[1 - index_parameter];
    bool use_bias = false;
    LDense *dense = new LDense(parent, neuronas, use_bias, node->name(), dev, mem);
    Tensor *weights_tensor = new Tensor(dims, nullptr, dev);
    COPY_FROM_VECTOR_PTR_TO_TENSOR(weights, weights_tensor);
    Tensor::copy(weights_tensor, dense->W);
    delete weights_tensor;
    return dense;
  }
  cout << "#################################################################" << endl;
  cout << "[DEBUG] Mult layer " << node->name() << " parents:" << endl;
  // Auxiliary function to print a vector
  auto print_shape = [](vector<int> shape){
    cout << "  - {";
    for (int i : shape)
      cout << i << ", ";
    cout << "}" << endl;
  };
  // Print the shape of each parent to check if they are compatible
  for (auto& p : parents) {
      cout << "  - " << p->name << endl;
      print_shape(p->getShape());
  }

  // Go through the list of parent to check if the shapes are compatible
  for (int p = 0; 0 < parents.size() - 1; ++p)
  {
    // Compare by adjacent pairs of parents (parent p and p + 1)
    if (!Tensor::sameShape(parents[p]->output, parents[p + 1]->output))
    {
      cout << "[DEBUG] Different shape detected in node " << node->name() << endl;
      cout << "[DEBUG] Going to create the broadcast layer" << endl;
      LBroadcast *l_broadcast = new LBroadcast(parents[p], parents[p + 1], node->name() + "_broadcast", dev, mem);
      print_shape(l_broadcast->output->getShape()); // DEBUG
      cout << "[DEBUG] Broadcast layer created!" << endl;
      // Note: The broadcast layer can apply the broadcast operation to the first or second layer
      //       passed as argument depending on the shape. It detects which layer need the broadcast operation
      //
      // Check if the parent that has the broadcast layer applied is p + 1
      if (l_broadcast->shapes_swapped)
      {
        cout << "[DEBUG] Going to change parent b" << endl;
        parents[p + 1] = l_broadcast; // The new parent is the output of the broadcast
        cout << "[DEBUG] Parent b changed!" << endl;
      }
      else // The parent p is the one that need the broadcast operation
      {
        cout << "[DEBUG] Going to change parent a" << endl;
        parents[p] = l_broadcast; // The new parent is the output of the broadcast
        cout << "[DEBUG] Parent a changed!" << endl;
      }
    }
  }

  // Print the shape of the parents after broadcast
  cout << "[DEBUG] New parent shapes:" << endl;
  for (auto& p : parents) {
      cout << "  - " << p->name << endl;
      print_shape(p->getShape());
  }
  cout << "#################################################################" << endl;
  return new LMatMul(parents, node->name(), dev, mem);
}

/*
 * DISTRIBUTED TRAINING
 */

vector<Tensor *> get_matmul_tensors(onnx::NodeProto &node,
                                    map<string, vector<float>> &map_init_values,
                                    map<string, vector<int>> &map_init_dims)
{
  vector<Tensor *> dense_tensors;

  bool dense_detected = false;
  int index_parameter = -1;
  for (int j = 0; j < node.input_size(); j++)
  {
    string parent_name = node.input(j);
    if (map_init_values.count(parent_name))
    {
      // Dense detected
      dense_detected = true;
      index_parameter = j;
      break;
    }
  }
  if (dense_detected)
  {
    string weights_name = node.input(index_parameter);
    vector<float> *weights = &(map_init_values[weights_name]);
    vector<int> dims = map_init_dims[weights_name];
    Tensor *weights_tensor = new Tensor(dims, nullptr, DEV_CPU);
    COPY_FROM_VECTOR_PTR_TO_TENSOR(weights, weights_tensor);
    dense_tensors.push_back(weights_tensor);
  }

  return dense_tensors;
}

#endif // defined(cPROTO)
