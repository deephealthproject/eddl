import numpy as np
import argparse
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LeakyReLU, LSTM, Softmax
from tensorflow.keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import onnx
import tf2onnx

# Training settings
parser = argparse.ArgumentParser(description='Keras LSTM encoder decoder MNIST Example')
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
parser.add_argument('--output-path', type=str, default="onnx_models/lstm_enc_dec_mnist.onnx",
                    help='Output path to store the onnx file')
parser.add_argument('--output-metric', type=str, default="",
                    help='Output file path to store the metric value obtained in test set')
args = parser.parse_args()

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Prepare sequences (bs, 28, 28)
x_train = x_train.reshape((x_train.shape[0], 28, 28))
x_test = x_test.reshape((x_test.shape[0], 28, 28))
# Prepare shifted input sequences for decoder
x_train_dec = np.pad(x_train, ((0,0), (1,0), (0,0)), 'constant')[:,:-1,:]
x_test_dec = np.pad(x_test, ((0,0), (1,0), (0,0)), 'constant')[:,:-1,:]

# Use the input sequences as target outputs
y_train = x_train
y_test = x_test

print("Train data shape:", x_train.shape)
print("Train labels shape:", y_train.shape)
print("Test data shape:", x_test.shape)
print("Test labels shape:", y_test.shape)

# Definer encoder
encoder_inputs = Input(shape=(28, 28))
encoder = LSTM(64, return_state=True)
encoder_outputs, encoder_h, encoder_c = encoder(encoder_inputs)
encoder_states = [encoder_h, encoder_c]
# Define decoder
decoder_inputs = Input(shape=(28, 28))
decoder = LSTM(64, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(28, activation='sigmoid')
decoder_outputs = decoder_dense(decoder_outputs)
# Create the full model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(loss = 'mse',
        optimizer = "adam",
        metrics = [])

model.summary()

# Training
model.fit([x_train, x_train_dec], y_train, batch_size=args.batch_size, epochs=args.epochs)

# Evaluation
eval_loss = model.evaluate([x_test, x_test_dec], y_test)
print("Evaluation result: Loss:", eval_loss)

# In case of providing output metric file, store the test mse value
if args.output_metric != "":
    with open(args.output_metric, 'w') as ofile:
        ofile.write(str(eval_loss))

# Convert to ONNX
input_spec = (tf.TensorSpec((args.batch_size, 28, 28), dtype=tf.float32),)
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=[input_spec, input_spec])
# Save ONNX to file
onnx.save(onnx_model, args.output_path)
