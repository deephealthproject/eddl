#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/core/split_onnx.h"
#include "eddl/serialization/onnx/utils_onnx.h"

// ONNX import
Layer* build_split_layer(onnx::NodeProto *node,
                         map<string, vector<float>> &map_init_values,
                         map<string, Layer *> &output_node_map,
                         int dev,
                         int mem)
{
  int axis = 0; // Dimension to split
  vector<int> splits_sizes; // To store the splits_sizes sizes
  bool op_version_13 = true; // To detect the version of the operator
  for (int j = 0; j < node->attribute_size(); j++)
  { // Set the attributes
    onnx::AttributeProto attribute = node->attribute(j);
    string attr_name = attribute.name();
    if (!attr_name.compare("axis"))
    {
      axis = attribute.i();
    }
    else if (!attr_name.compare("split"))
    {
      op_version_13 = false;
      for (int h = 0; h < attribute.ints_size(); h++)
        splits_sizes.push_back(attribute.ints(h));
    }
    else
      msg("Found an unexpected attribute (" + attr_name +") in Split operator", "ONNX::ImportNet");
  }

  // In op version 13 the splits_sizes are defined with a node input tensor (not attribute)
  if (op_version_13 && node->input_size() > 1)
  {
    // Get starts indexes
    string starts_node_name = node->input(1);
      splits_sizes = vf2vi(map_init_values[starts_node_name]);
  }

  // Get the input layer
  Layer *parent = output_node_map[node->input(0)];
  vector<int> in_shape = parent->output->getShape();

  // Prepare axis attribute
  if (axis == 0)
    msg("Error: EDDL doesn't support a Split layer that splits_sizes the batch dimension.", "ONNX::ImportNet");
  else if (axis < 0)
      axis = in_shape.size() + axis; // Convert to positive index
  axis -= 1; // Remove batch dimension

  vector<int> splits_idx; // The final vector to pass to the LSplit constructor
  bool drop_last = false; // In case of not selecting the full axis with the splits we have to drop the reminder elements
  if (splits_sizes.size() == 0)
    // If the splits_sizes sizes are not provided we split the input in two equal sized tensors
    splits_idx.push_back(in_shape[axis + 1] / 2);
  else if (splits_sizes.size() == 1)
    msg("Error: In Split node " + node->name() + ". Expected 2 or more values in split attribute, got 1.", "ONNX::ImportNet");
  else
  {
    // Get the positions to make the splits_sizes
    //  - Note: ONNX "split" attribute gives a list with the sizes of each split. But EDDL Split layer
    //          wants a list with the positions along the target axis to make the splits_sizes.
    int aux_index = 0;
    for (int i = 0; i < splits_sizes.size() - 1/*avoid the last value*/; ++i)
    {
      aux_index += splits_sizes[i];
      splits_idx.push_back(aux_index);
    }
    aux_index += splits_sizes[splits_sizes.size()-1]; // Get the full size of all the splits together
    if (aux_index != in_shape[axis + 1])
      msg("Error: Invalid splits_sizes in Split layer " + node->name() + ". The sum of the splits_sizes lengths must be equal to the target axis size.", "ONNX::ImportNet");
  }

  LSplit *split_layer = new LSplit(parent, splits_idx, axis, false, node->name(), dev, mem);
  vlayer select_layers = split_layer->split_layers; // Ge the list of internal Select layers of the Split

  if (select_layers.size() != node->output_size())
    msg("Error in Split layer " + node->name() + ". The number of outputs is not the same than the number of splits.", "ONNX::ImportNet");

  // Add the Select layers to the output_node_map manually
  for (int i = 0; i < node->output_size(); ++i)
    output_node_map[node->output(i)] = select_layers[i];

  return nullptr; // We don't return any layer because we already added the Select layers of the Split
}

#endif // defined(cPROTO)
