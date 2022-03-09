import argparse

import onnx
import onnxruntime


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='ONNX RUNTIME MNIST Inference')

    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-f', '--onnx-file', type=str, default="onnx_models/trained_model.onnx",
                        help='File path to the onnx file with the pretrained model to test')
    parser.add_argument('-m', '--target-metric', type=str, default="",
                        help='Path to a file with a single value with the target metric to achieve')
    args = parser.parse_args()

    # Print ONNX graph
    onnx_model = onnx.load(args.onnx_file)
    print(onnx.helper.printable_graph(onnx_model.graph))

    onnx.checker.check_model(onnx_model, full_check=True)

    # Prepare ONNX runtime
    # Create a session with the onnx model
    session = onnxruntime.InferenceSession(args.onnx_file, None)

    # Perform inference
    # input_name = session.get_inputs()[0].name
    # output_name = session.get_outputs()[0].name
    # result = session.run([output_name], {input_name: data})


if __name__ == '__main__':
    main()
