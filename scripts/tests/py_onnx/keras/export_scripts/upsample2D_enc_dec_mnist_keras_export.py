import argparse

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, MaxPooling2D
from tensorflow.keras.datasets import mnist
import keras2onnx

# Training settings
parser = argparse.ArgumentParser(description='Keras Conv+Upsample encoder decoder MNIST Example')
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
parser.add_argument('--output-path', type=str, default="onnx_models/upsample2D_enc_dec_mnist.onnx",
                    help='Output path to store the onnx file')
parser.add_argument('--output-metric', type=str, default="",
                    help='Output file path to store the metric value obtained in test set')
args = parser.parse_args()

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Prepare images (bs, 28, 28, 1)
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# Use the input images as target outputs
y_train = x_train
y_test = x_test

print("Train data shape:", x_train.shape)
print("Train labels shape:", y_train.shape)
print("Test data shape:", x_test.shape)
print("Test labels shape:", y_test.shape)

# Definer encoder
model = Sequential()
model.add(Input(shape=(28, 28, 1)))
# Encoder
model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPooling2D(2, 2))
# Decoder
model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(1, 1, padding="same", activation="sigmoid"))

model.compile(loss='mse',
              optimizer="adam",
              metrics=[])

model.summary()

# Training
model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs)

# Evaluation
eval_loss = model.evaluate(x_test, y_test)
print("Evaluation result: Loss:", eval_loss)

# In case of providing output metric file, store the test mse value
if args.output_metric != "":
    with open(args.output_metric, 'w') as ofile:
        ofile.write(str(eval_loss))

# Convert to ONNX
onnx_model = keras2onnx.convert_keras(model, "upsample2D_mnist", debug_mode=1)
# Save ONNX to file
keras2onnx.save_model(onnx_model, args.output_path)
