import os
import onnx
import argparse
import keras2onnx
from onnx2keras import onnx_to_keras
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.datasets import imdb

parser = argparse.ArgumentParser(description='Keras IMDB import ONNX example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size (default: 64)')
parser.add_argument('--model-path', type=str, default="onnx_models/lstm_imdb.onnx", 
                    help='Path of the onnx file to load')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--vocab-size', type=int, default=2000,
                    help='Max size of the vocabulary (default: 2000)')
parser.add_argument('--max-len', type=int, default=250,
                    help='Sequence max length (default: 80)')
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

print('Load ONNX model...')
onnx_model = onnx.load(args.model_path)

#onnx.checker.check_model(onnx_model)

print('Convert ONNX to Keras...')
k_model = onnx_to_keras(onnx_model, ['embedding_input'])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

loss, acc = model.evaluate(x_test, y_test,
                            batch_size=args.batch_size)

print("Evaluation result: Loss:", loss, " Accuracy:", acc)
