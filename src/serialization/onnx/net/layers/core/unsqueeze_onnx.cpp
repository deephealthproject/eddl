#if defined(cPROTO)
#include "eddl/serialization/onnx/layers/core/unsqueeze_onnx.h"

void build_unsqueeze_node(string node_name, string input, string output, vector<int> axes, onnx::GraphProto *graph)
{
  onnx::NodeProto *node_usq = graph->add_node();
  node_usq->set_op_type("Unsqueeze");
  node_usq->set_name(node_name);
  node_usq->add_input(input);
  onnx::AttributeProto *axes_attr = node_usq->add_attribute();
  axes_attr->set_name("axes");
  axes_attr->set_type(onnx::AttributeProto::INTS);
  for (int ax : axes)
    axes_attr->add_ints(ax);
  node_usq->add_output(output);
}

#endif // defined(cPROTO)
