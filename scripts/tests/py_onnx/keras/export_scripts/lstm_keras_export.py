import os
import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.datasets import imdb
import onnx
import tf2onnx

# Training settings
parser = argparse.ArgumentParser(description='Keras IMDB LSTM Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=7, metavar='N',
                    help='number of epochs to train (default: 5)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--vocab-size', type=int, default=2000,
                    help='Max size of the vocabulary (default: 2000)')
parser.add_argument('--max-len', type=int, default=250,
                    help='Sequence max length (default: 250)')
parser.add_argument('--output-path', type=str, default="onnx_models/lstm_imdb.onnx", 
                    help='Output path to store the onnx file')
parser.add_argument('--output-metric', type=str, default="",
                    help='Output file path to store the metric value obtained in test set')
args = parser.parse_args()

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=args.vocab_size, maxlen=args.max_len)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=args.max_len)
x_test = sequence.pad_sequences(x_test, maxlen=args.max_len)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Input(shape=(x_train.shape[-1])))
model.add(Embedding(args.vocab_size, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=args.batch_size,
          epochs=args.epochs,
          validation_data=(x_test, y_test))

loss, acc = model.evaluate(x_test, y_test,
                            batch_size=args.batch_size)

print("Evaluation result: Loss:", loss, " Accuracy:", acc)

# In case of providing output metric file, store the test accuracy value
if args.output_metric != "":
    with open(args.output_metric, 'w') as ofile:
        ofile.write(str(acc))

# Convert to ONNX
input_spec = (tf.TensorSpec((args.batch_size, x_train.shape[-1]), dtype=tf.float32),)
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_spec)
# Save ONNX to file
onnx.save(onnx_model, args.output_path)
