import argparse

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv3D, UpSampling3D, MaxPooling3D
import keras2onnx

# Training settings
parser = argparse.ArgumentParser(description='Keras Conv3D+Upsampling encoder decoder with synthetic data Example')
parser.add_argument('--batch-size', type=int, default=2, metavar='N',
                    help='input batch size for training (default: 2)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 5)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--output-path', type=str, default="onnx_models/upsample3D_enc_dec_synthetic.onnx",
                    help='Output path to store the onnx file')
parser.add_argument('--output-metric', type=str, default="",
                    help='Output file path to store the metric value obtained in test set')
args = parser.parse_args()

# Create synthetic data
n_samples = 6
# Shape: (n_samples, ch=3, depth=16, height=16, width=16)
x_train = np.linspace(0, 1, n_samples*3*16*16*16)
x_train = x_train.reshape((n_samples, 3, 16, 16, 16)).astype(np.float32)
# (B, C, D, H, W) -> (B, D, H, W, C)
x_train = np.transpose(x_train, (0, 2, 3, 4, 1))  # Set channel last

print("Train data shape:", x_train.shape)

# Definer encoder
model = Sequential()
model.add(Input(shape=(16, 16, 16, 3)))
# Encoder
model.add(Conv3D(32, 3, padding="same", activation="relu"))
model.add(MaxPooling3D(2, 2))
model.add(Conv3D(64, 3, padding="same", activation="relu"))
model.add(MaxPooling3D(2, 2))
# Decoder
model.add(Conv3D(64, 3, padding="same", activation="relu"))
model.add(UpSampling3D((2, 2, 2)))
model.add(Conv3D(32, 3, padding="same", activation="relu"))
model.add(UpSampling3D((2, 2, 2)))
model.add(Conv3D(3, 1, padding="valid", activation="sigmoid"))

model.compile(loss='mse',
              optimizer="adam",
              metrics=[])

model.summary()

# Training
model.fit(x_train, x_train, batch_size=args.batch_size, epochs=args.epochs)

# Evaluation
eval_loss = model.evaluate(x_train, x_train)
print("Evaluation result: Loss:", eval_loss)

# In case of providing output metric file, store the test mse value
if args.output_metric != "":
    with open(args.output_metric, 'w') as ofile:
        ofile.write(str(eval_loss))

# Convert to ONNX
onnx_model = keras2onnx.convert_keras(model, "upsample3D_synthetic", debug_mode=1)
# Save ONNX to file
keras2onnx.save_model(onnx_model, args.output_path)
