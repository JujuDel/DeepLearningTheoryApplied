# IMPORTS

import numpy as np
import matplotlib.pyplot as plt

# ACTIVATION FUNCTIONS

## Functions

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def ReLU(z):
    return np.maximum(z,0)

def LReLU(z):
    return np.maximum(z,0.01*z)

## Derivatives

def d_sigmoid(Z):
    s = sigmoid(Z)
    return s * (1 - s)

def d_tanh(Z):
    return 1 - np.power(np.tanh(Z), 2)

def d_ReLU(Z):
    dZ = np.array(Z, copy=True)
    dZ[dZ <= 0] = 0
    dZ[dZ > 0] = 1
    
    return dZ

def d_LReLU(Z):
    dZ = np.array(Z, copy=True)
    dZ[dZ > 0] = 1
    dZ[dZ <= 0] = 0.01 * dZ[dZ <= 0]
    
    return dZ

# INITIALIZATION

def displayDimensions(parameters):
    L = len(parameters) // 3
    print("number of Layers: %i"%(L))
    
    for l in range(L):
        print("  %i / W(%i, %i) / b(%i, %i) / %s" %(l+1, parameters['W' + str(l+1)].shape[0], parameters['W' + str(l+1)].shape[1], parameters['b' + str(l+1)].shape[0], parameters['b' + str(l+1)].shape[1], parameters['F' + str(l+1)]))
    print("")

def initialize(nbNeurons, actFunctions):
    assert(len(nbNeurons) == len(actFunctions) + 1)
    parameters = {}
    
    for l in range(1, len(nbNeurons)):
        parameters['W' + str(l)] = np.random.randn(nbNeurons[l], nbNeurons[l-1]) * np.sqrt( 2 / nbNeurons[l-1])
        parameters['b' + str(l)] = np.zeros((nbNeurons[l], 1))
        
        assert(parameters['W' + str(l)].shape == (nbNeurons[l], nbNeurons[l-1]))
        assert(parameters['b' + str(l)].shape == (nbNeurons[l], 1))
        
        parameters['F' + str(l)] = actFunctions[l-1]
    
    displayDimensions(parameters)
    
    return parameters

# FORWARD PROPAGATION

def forwardPropagation(W, b, previousA, actFunction):
    Z = np.dot(W, previousA) + b
    
    assert(Z.shape == (W.shape[0], previousA.shape[1]))
    
    if actFunction == "sigmoid":
        A = sigmoid(Z)
    elif actFunction == "tanh":
        A = tanh(Z)
    elif actFunction == "ReLU":
        A = ReLU(Z)
    elif actFunction =="LReLU":
        A = LReLU(Z)
    
    assert (A.shape == Z.shape)
    return Z, A

def globalForwardPropagation(X, parameters):
    Zs = []
    As = [X]
    A = X
    L = len(parameters) // 3
    
    for l in range(1, L+1):
        A_prev = A
        Z, A = forwardPropagation(parameters['W'+str(l)], parameters['b'+str(l)], A_prev, parameters['F'+str(l)])
        Zs = Zs + [Z]
        As = As + [A]
    
    return Zs, As, A

# BACKWARD PROPAGATION

def backwardPropagation(W_l, Z_l, dA_l, A_l1, actFunction):
    
    if actFunction == "sigmoid":
        dZ_l = dA_l * d_sigmoid(Z_l)
    elif actFunction == "tanh":
        dZ_l = dA_l * d_tanh(Z_l)
    elif actFunction == "ReLU":
        dZ_l = dA_l * d_ReLU(Z_l)
    elif actFunction == "LReLU":
        dZ_l = dA_l * d_LReLU(Z_l)
    
    dW_l = np.dot(dZ_l, A_l1.T) / A_l1.shape[1]
    db_l = np.sum(dZ_l, axis = 1, keepdims = True) / A_l1.shape[1]
    dA_l1 = np.dot(W_l.T, dZ_l)
    
    assert(dA_l1.shape == A_l1.shape)
    assert(dW_l.shape == W_l.shape)
    
    return dA_l1, dW_l, db_l
 
def globalBackwardPropagation(Y, Zs, As, parameters):
    grads = {}
    L = len(Zs)
    
    # Zs from 1 to L and As from 0 to L
    assert(len(Zs) + 1 == len(As))
    
    # Initializing the backpropagation
    AL = As[L]
    Y = Y.reshape((AL.shape))
    dA_l1 = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    assert(dA_l1.shape == AL.shape)
     
    # From l=L-1 to l=0, outbox included
    for l in reversed(range(L)):
        dA_l1, dW_l, dB_l = backwardPropagation(parameters["W" + str(l+1)], Zs[l], dA_l1, As[l], parameters["F" + str(l+1)])
        grads["dW" + str(l + 1)] = dW_l
        grads["db" + str(l + 1)] = dB_l

    return grads

# UPDATES OF THE PARAMETERS

def updateParameters(parameters, grads, learning_rate):
    
    L = len(parameters) // 3
    
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters

# COST CALCULUS

def computeCost(AL, Y):
    
    cost = -np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL)) / Y.shape[1]
    
    return np.squeeze(cost)

# PREDICTION CALCULUS

def predict(parameters, X, decision_treshold = 0.5):
    _, _, A = globalForwardPropagation(X, parameters)

    P = (A > decision_treshold)
    
    return P

# MODEL CONSTRUCTION and TRAINING

def model(X, Y, X_test, Y_test, layers_dims, actFunctions, learning_rate = 0.0008, num_iterations = 5000, print_cost = False):
    # Initialize the weights and the bias
    parameters = initialize(layers_dims, actFunctions)
    
    costs = []
    
    for i in range(0, num_iterations):

        Zs, As, AL = globalForwardPropagation(X, parameters)
        
        epoch_cost = computeCost(AL, Y)
        
        grads = globalBackwardPropagation(Y, Zs, As, parameters)
        
        parameters = updateParameters(parameters, grads, learning_rate)
                
        # Print the cost every 100 training example
        if print_cost:
            if i % 100 == 0:
                print ("Cost after epoch %i: %f" %(i, epoch_cost))
            if i % 5:
                costs.append(epoch_cost)
    
    if print_cost:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate = " + str(learning_rate))
        plt.show()
    
    # Test the model
    Y_prediction = predict(parameters, X)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction - Y)) * 100))
    
    Y_prediction_test = predict(parameters, X_test)
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    
    return parameters