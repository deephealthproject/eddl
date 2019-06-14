import pyeddl
from pyeddl.model import Model
from pyeddl.datasets import mnist

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

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
m.fit(x_train, y_train, batch_size=1000, epochs=5)

# Evaluate
print("Evaluate train:")
m.evaluate(x_train, y_train)
