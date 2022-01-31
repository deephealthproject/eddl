import numpy as np
import argparse
import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv3D, Flatten, MaxPooling3D, Input
from tensorflow.keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import onnx
import tf2onnx

# Training settings
parser = argparse.ArgumentParser(description='Keras Conv3D synthetic example')
parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                    help='input batch size for training (default: 2)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 1)')
parser.add_argument('--output-path', type=str, default="onnx_models/conv3D_synthetic.onnx",
                    help='Output path to store the onnx file')
parser.add_argument('--output-metric', type=str, default="",
                    help='Output file path to store the metric value obtained in test set')
args = parser.parse_args()

# Create synthetic data
n_samples = 6
num_classes = 1
# Shape: (n_samples, ch=3, depth=16, height=16, width=16)
x_train = np.linspace(0, 1, n_samples*3*16*16*16).reshape((n_samples, 3, 16, 16, 16)).astype(np.float32)
# (B, C, D, H, W) -> (B, D, H, W, C)
x_train = np.transpose(x_train, (0, 2, 3, 4, 1))  # Set channel last
# Shape: (n_samples, dim=2)
y_train = np.linspace(0, 1, n_samples*num_classes).reshape((n_samples, num_classes)).astype(np.float32)
x_test, y_test = x_train, y_train

print("Train data shape:", x_train.shape)
print("Train labels shape:", y_train.shape)
print("Test data shape:", x_test.shape)
print("Test labels shape:", y_test.shape)

model = Sequential()
model.add(Input(shape=(16, 16, 16, 3), name="linput"))
model.add(Conv3D(5, 3, padding="same"))
model.add(MaxPooling3D())
model.add(Conv3D(10, 3, padding="same"))
model.add(MaxPooling3D())
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(num_classes))

model.build(input_shape=(16, 16, 16, 3)) 

model.compile(loss='mse',
              optimizer="adam")

model.summary()

# Training
model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs)

# Evaluation
mse = model.evaluate(x_test, y_test)
print("Evaluation result: Loss:", mse)

# In case of providing output metric file, store the test mse value
if args.output_metric != "":
    with open(args.output_metric, 'w') as ofile:
        ofile.write(str(mse))

# Convert to ONNX
input_spec = (tf.TensorSpec((args.batch_size, 16, 16, 16, 3), dtype=tf.float32),)
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_spec)
# Save ONNX to file
onnx.save(onnx_model, args.output_path)
