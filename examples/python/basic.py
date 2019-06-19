import numpy as np

import pyeddl
from pyeddl.model import Model
from pyeddl.datasets import mnist
from pyeddl.utils import to_categorical

# Params
batch_size = 1000
num_classes = 10
epochs = 1

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

# Reshape training dataset
x_train = x_train.reshape((len(x_train), -1))
x_test = x_test.reshape((len(x_test), -1))

# Tranform to categorical
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# View model
m = Model.from_model('mlp')
print(m.summary())
m.plot("model.pdf")

# Building params
optim = pyeddl.optim.SGD(0.01, 0.9)
losses = ['soft_crossentropy']
metrics = ['accuracy']

# Build model
m.compile(optimizer=optim, losses=losses, metrics=metrics, device='cpu')

# Training
m.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

# Evaluate
print("Evaluate train:")
m.evaluate(x_test, y_test)
