import argparse

from tensorflow.keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from onnx2keras import onnx_to_keras
import onnx

parser = argparse.ArgumentParser(description='Keras MNIST ONNX import example')
parser.add_argument('--model-path', type=str, default="onnx_models/conv2D_mnist.onnx",
                    help='Path of the onnx file to load')
parser.add_argument('--input-1D', action='store_true', default=False,
                    help='To change the input size to a 784 length vector')
parser.add_argument('--no-channel', action='store_true', default=False,
                    help='If --input-1D is enabled, removes the channel dimension. (bs, 1, 784) -> (bs, 784)')
args = parser.parse_args()

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Make sure images have shape (28, 28, 1)
if args.input_1D:
    if args.no_channel:
        x_train = x_train.reshape((x_train.shape[0], 784))
        x_test = x_test.reshape((x_test.shape[0], 784))
    else:
        x_train = x_train.reshape((x_train.shape[0], 1, 784))
        x_test = x_test.reshape((x_test.shape[0], 1, 784))
else:
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Get one hot encoding from labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("Train data shape:", x_train.shape)
print("Train labels shape:", y_train.shape)
print("Test data shape:", x_test.shape)
print("Test labels shape:", y_test.shape)

# Load ONNX model
onnx_model = onnx.load(args.model_path)

# Call the converter (input - is the main model input name, can be different for your model)
model = onnx_to_keras(onnx_model, ['input1'])

model.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['accuracy'])

model.summary()

# Evaluation
acc = model.evaluate(x_test, y_test)
print("Evaluation result: Loss:", acc[0], " Accuracy:", acc[1])
