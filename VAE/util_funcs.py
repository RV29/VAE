import numpy as np

# most basic functions
def sigmoid(x):
    """sigmoid function"""
    return 1/(1+np.exp(-x))

def sigmoid_gradient(x):
    """gradient of sigmoid function"""
    return x * (1-x)

def tanh(x):
    """tanh function"""
    return np.tanh(x)
    
def tanh_gradient(x):
    """gradient of tanh function"""
    return 1-np.power(x,2)

def get_Batch(M, trainX, trainy):
    """randomly sample a mini batch of size M from the training data"""
    
    N = trainX.shape[0]
    sample = np.random.choice(N,M)
    
    return trainX[sample], trainy[sample]

