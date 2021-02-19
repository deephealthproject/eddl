#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/core/squeeze_onnx.h"

void build_squeeze_node(string node_name, string input, string output, vector<int> axes, onnx::GraphProto *graph)
{
  onnx::NodeProto *node_sq = graph->add_node();
  node_sq->set_op_type("Squeeze");
  node_sq->set_name(node_name);
  node_sq->add_input(input);
  onnx::AttributeProto *axes_attr = node_sq->add_attribute();
  axes_attr->set_name("axes");
  axes_attr->set_type(onnx::AttributeProto::INTS);
  for (int ax : axes)
    axes_attr->add_ints(ax);
  node_sq->add_output(output);
}

#endif // defined(cPROTO)
