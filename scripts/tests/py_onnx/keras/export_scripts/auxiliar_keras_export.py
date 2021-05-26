import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, GlobalAveragePooling2D 
from tensorflow.keras.layers import ZeroPadding2D, Concatenate
from tensorflow.keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import keras2onnx

# Training settings
parser = argparse.ArgumentParser(description='Keras Conv2D MNIST Example with some auxiliar layers')
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
parser.add_argument('--output-path', type=str, default="onnx_models/auxiliar_mnist.onnx",
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

in_ = Input(shape=(28, 28, 1), name="linput")
x = ZeroPadding2D((2, 2))(in_)
x = Conv2D(30, 5, activation="relu")(x)
x = BatchNormalization(scale=True, center=True)(x)
x = MaxPooling2D(2, 2)(x)
x1, x2, x3 = tf.split(x, 3, 3)

x1 = ZeroPadding2D((1, 1))(x1)
x1 = Conv2D(16, 3, activation="relu")(x1)
x1 = BatchNormalization(scale=True, center=True)(x1)
x1 = AveragePooling2D(2, 2)(x1)

x2 = ZeroPadding2D((2, 2))(x2)
x2 = Conv2D(16, 5, activation="relu")(x2)
x2 = BatchNormalization(scale=True, center=True)(x2)
x2 = AveragePooling2D(2, 2)(x2)

x3 = Conv2D(16, 5, padding="same", activation="relu")(x3)
x3 = BatchNormalization(scale=True, center=True)(x3)
x3 = AveragePooling2D(2, 2)(x3)

x = Concatenate(axis=3)([x1, x2, x3])

x = Conv2D(16, 3, activation="relu", padding="same")(x)
x = BatchNormalization(scale=True, center=True)(x)
x = GlobalAveragePooling2D()(x)
x = Flatten()(x)
out_ = Dense(10, activation = 'softmax')(x)

model = keras.Model(in_, out_)

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
onnx_model = keras2onnx.convert_keras(model, "auxiliar_mnist", debug_mode=1)
# Save ONNX to file
keras2onnx.save_model(onnx_model, args.output_path)
