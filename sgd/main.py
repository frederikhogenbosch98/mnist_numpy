import numpy as np
from mnist import MNIST
from funcs import forward

len_input = 784
layer_1 = 16
layer_2 = 16
layer_3 = 10
len_output = 10

mndata = MNIST('data')

X_train, y_train = mndata.load_training()
X_test, y_test = mndata.load_testing()

X_train = np.array(X_train, dtype='float64')
y_train = np.array(y_train, dtype='float64')
X_test = np.array(X_test, dtype='float64')
y_test = np.array(y_test, dtype='float64')

w0 = np.random.rand(len_input, layer_1)
w1 = np.random.rand(layer_1, layer_2)
w2 = np.random.rand(layer_2, layer_3)
w3 = np.random.rand(layer_3, len_output)

X_train *= 1.0/X_train.max() 
y_train *= 1.0/y_train.max() 
X_test *= 1.0/X_test.max() 
y_test *= 1.0/y_test.max() 


def train(X_train, y_train):
    print(forward(X_train, w0))



def test(X_test, y_test):
    pass


if __name__ == "__main__":
    train(X_train, y_train)
    test(X_test, y_test)


