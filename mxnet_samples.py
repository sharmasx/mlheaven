import mxnet as mx
from mxnet import nd, autograd
import matplotlib.pyplot as plt
import numpy as np
mx.random.seed(1)


# generate fake data that is linearly separable with a margin epsilon given the data
def getfake(samples, dimensions, epsilon):
    wfake = nd.random_normal(shape=(dimensions))  # fake weight vector for separation
    bfake = nd.random_normal(shape=(1))  # fake bias
    wfake = wfake / nd.norm(wfake)  # rescale to unit length

    # making some linearly separable data, simply by chosing the labels accordingly
    X = nd.zeros(shape=(samples, dimensions))
    Y = nd.zeros(shape=(samples))

    i = 0
    while (i < samples):
        tmp = nd.random_normal(shape=(1, dimensions))
        margin = nd.dot(tmp, wfake) + bfake
        if (nd.norm(tmp).asscalar() < 3) & (abs(margin.asscalar()) > epsilon):
            X[i, :] = tmp
            Y[i] = 2 * (margin > 0) - 1
            i += 1
    return X, Y


# plot the data with colors chosen according to the labels
def plotdata(X, Y):
    for (x, y) in zip(X, Y):
        if (y.asscalar() == 1):
            plt.scatter(x[0].asscalar(), x[1].asscalar(), color='r')
        else:
            plt.scatter(x[0].asscalar(), x[1].asscalar(), color='b')


# plot contour plots on a [-3,3] x [-3,3] grid
def plotscore(w, d):
    xgrid = np.arange(-3, 3, 0.02)
    ygrid = np.arange(-3, 3, 0.02)
    xx, yy = np.meshgrid(xgrid, ygrid)
    zz = nd.zeros(shape=(xgrid.size, ygrid.size, 2))
    zz[:, :, 0] = nd.array(xx)
    zz[:, :, 1] = nd.array(yy)
    vv = nd.dot(zz, w) + b
    CS = plt.contour(xgrid, ygrid, vv.asnumpy())
    plt.clabel(CS, inline=1, fontsize=10)


X, Y = getfake(50, 2, 0.3)
plotdata(X, Y)
#plt.show()

def perceptron(w,b,x,y):
    if (y * (nd.dot(w,x) + b)).asscalar() <= 0:
        w += y * x
        b += y
        return 1
    else:
        return 0

# w = nd.zeros(shape=(2))
# b = nd.zeros(shape=(1))
# for (x,y) in zip(X,Y):
#     res = perceptron(w,b,x,y)
#     if (res == 1):
#         print('Encountered an error and updated parameters')
#         print('data   {}, label {}'.format(x.asnumpy(),y.asscalar()))
#         print('weight {}, bias  {}'.format(w.asnumpy(),b.asscalar()))
#         plotscore(w,b)
#         plotdata(X,Y)
#         plt.scatter(x[0].asscalar(), x[1].asscalar(), color='g')
#         plt.show()

# autograd -


import mxnet as mx
from mxnet import nd, autograd
mx.random.seed(1)



num_inputs = 2
num_outputs = 1
num_examples = 10000

X = nd.random_normal(shape=(num_examples, num_inputs))
y = 2 * X[:, 0] - 3.4 * X[:, 1] + 4.2 + .01 * nd.random_normal(shape=(num_examples,))

batch_size = 4
train_data = mx.io.NDArrayIter(X, y, batch_size, shuffle=True) # stochastic
batch = train_data.next()
print(batch.data[0])
print(batch.label[0])

# end of an epoch
# reset reshuffles the dat


counter = 0
train_data.reset()
for batch in train_data:
    counter += 1
print(counter)

w = nd.random_normal(shape=(num_inputs, num_outputs))
b = nd.random_normal(shape=num_outputs)
params = [w, b]

for param in params:
    param.attach_grad()

def net(X):
    return mx.nd.dot(X, w) + b


def square_loss(yhat, y):
    return nd.mean((yhat - y) ** 2)


def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad


epochs = 2
ctx = mx.cpu()
learning_rate = .001
moving_loss = 0.

for e in range(epochs):
    train_data.reset()
    for i, batch in enumerate(train_data):
        # we only have one input and output
        # that is why we take [0]
        data = batch.data[0].as_in_context(ctx)
        label = batch.label[0].as_in_context(ctx).reshape((-1, 1))
        with autograd.record():
            output = net(data)
            mse = square_loss(output, label)
        mse.backward()
        SGD(params, learning_rate)

        ##########################
        #  Keep a moving average of the losses
        ##########################
        if (i == 0) and (e == 0):
            moving_loss = nd.mean(mse).asscalar()
        else:
            moving_loss = .99 * moving_loss + .01 * nd.mean(mse).asscalar()

        if (i + 1) % 500 == 0:
            print("Epoch %s, batch %s. Moving avg of loss: %s" % (e, i, moving_loss))

# tensor board
