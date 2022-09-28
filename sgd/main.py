import numpy as np
from mnist import MNIST
from funcs import forward, normalize

len_input = 784
layer_1 = 16
layer_2 = 16
layer_3 = 10
len_output = 10

mndata = MNIST('../data')

X_train, y_train = mndata.load_training()
X_test, y_test = mndata.load_testing()

X_train = np.array(X_train, dtype='float64')
y_train = np.array(y_train, dtype='int64')
X_test = np.array(X_test, dtype='float64')
y_test = np.array(y_test, dtype='int64')

w0 = np.random.rand(len_input, layer_1)
w1 = np.random.rand(layer_1, layer_2)
w2 = np.random.rand(layer_2, layer_3)
w3 = np.random.rand(layer_3, len_output)

X_train *= 1.0/X_train.max() 
X_test *= 1.0/X_test.max() 


y_train_s = np.zeros((len(y_train) ,len_output))

for j in range(len(y_train)):
    k = y_train[j]
    y_train_s[j][k] = 1


y_test_s = np.zeros((len(y_train_s) ,len_output))

for j in range(len(y_test)):
    k = y_train[j]
    y_test_s[j][k] = 1


y_train_s = np.array(y_train_s, dtype='float64')
y_test_s = np.array(y_test_s, dtype='float64')



def train(X_train, y_train):
    l1 = normalize(forward(X_train[0], w0))
    l2 = normalize(forward(l1, w1))
    l3 = normalize(forward(l2, w2))
    l4 = normalize(forward(l3, w3))
    error = np.sum(np.abs(y_train[0] - l4))
    print(error, y_train[0])


def test(X_test, y_test):
    pass


if __name__ == "__main__":
    train(X_train, y_train)
    test(X_test, y_test)


