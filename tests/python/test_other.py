# import pyeddl
# from pyeddl.layers import *
#
# mnist = pyeddl.datasets.mnist
#
# batch_size = 128
# num_classes = 10
# epochs = 20
#
# # the data, split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# x_train = x_train.reshape(60000, 784)
# x_test = x_test.reshape(10000, 784)
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')
#
# # convert class vectors to binary class matrices
# y_train = pyeddl.utils.to_categorical(y_train, num_classes)
# y_test = pyeddl.utils.to_categorical(y_test, num_classes)
#
# Dense(512, activation='relu', input_shape=(784,))
# model.add(Dropout(0.2))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(num_classes, activation='softmax'))
#
# model.summary()
#
# model.plot()
#
# optim = pyeddl.optim.SGD(lr=0.01, momentum=0.9, nesterov=True)
# loss = pyeddl.losses.categorical_crossentropy
# metric1 = pyeddl.metrics.categorical_accuracy
# model.compile(optimizer=optim, loss=loss, metrics=[metric1])
#
# history = model.fit(x_train, y_train,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1,
#                     validation_data=(x_test, y_test))
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
