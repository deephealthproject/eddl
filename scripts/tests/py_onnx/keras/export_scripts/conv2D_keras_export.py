import numpy as np
import argparse
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D 
from tensorflow.keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import keras2onnx

# Training settings
parser = argparse.ArgumentParser(description='Keras Conv2D MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 5)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--output-path', type=str, default="onnx_models/conv2D_mnist.onnx",
                    help='Output path to store the onnx file')
parser.add_argument('--output-metric', type=str, default="",
                    help='Output file path to store the metric value obtained in test set')
args = parser.parse_args()

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Make sure images have shape (28, 28, 1)
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Get one hot encoding from labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("Train data shape:", x_train.shape)
print("Train labels shape:", y_train.shape)
print("Test data shape:", x_test.shape)
print("Test labels shape:", y_test.shape)

model = Sequential()
model.add(Input(shape=(28, 28, 1), name="linput"))
model.add(Conv2D(16, 5, activation="relu"))
model.add(BatchNormalization(scale=True, center=True))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(16, 4, activation="relu"))
model.add(BatchNormalization(scale=True, center=True))
model.add(AveragePooling2D(2, 2))
model.add(Conv2D(16, 3, activation="relu", padding="same"))
model.add(BatchNormalization(scale=True, center=True))
model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(10, activation = 'softmax'))

model.build(input_shape=(28, 28, 1))  # For keras2onnx

model.compile(loss = 'categorical_crossentropy',
        optimizer = "adam",
        metrics = ['accuracy'])

model.summary()

# Training
model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs)

# Evaluation
res = model.evaluate(x_test, y_test)
print("Evaluation result: Loss:", res[0], " Accuracy:", res[1])

# In case of providing output metric file, store the test accuracy value
if args.output_metric != "":
    with open(args.output_metric, 'w') as ofile:
        ofile.write(str(res[1]))

# Convert to ONNX
onnx_model = keras2onnx.convert_keras(model, "conv2D_mnist", debug_mode=1)
# Save ONNX to file
keras2onnx.save_model(onnx_model, args.output_path)
