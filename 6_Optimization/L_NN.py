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

def initialize(nbNeurons, actFunctions, printDimensions = False):
    assert(len(nbNeurons) == len(actFunctions) + 1)
    parameters = {}
    
    for l in range(1, len(nbNeurons)):
        parameters['W' + str(l)] = np.random.randn(nbNeurons[l], nbNeurons[l-1]) * np.sqrt(2 / nbNeurons[l-1])
        parameters['b' + str(l)] = np.zeros((nbNeurons[l], 1))
        
        assert(parameters['W' + str(l)].shape == (nbNeurons[l], nbNeurons[l-1]))
        assert(parameters['b' + str(l)].shape == (nbNeurons[l], 1))
        
        parameters['F' + str(l)] = actFunctions[l-1]
    
    if printDimensions:
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

def forwardPropagationRegularized(W, b, previousA, actFunction, keep_prob):
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
    
    Keeper = np.random.rand(A.shape[0], A.shape[1])
    Keeper = (Keeper < keep_prob)
    A = A * Keeper
    A = A / keep_prob
    
    assert (A.shape == Z.shape)
    return Z, A, Keeper

def globalForwardPropagationRegularized(X, parameters, keep_probs):
    Zs = []
    As = [X]
    Keeps = [np.ones((X.shape[0], 1))]
    A = X
    L = len(parameters) // 3
    
    for l in range(L):
        A_prev = A
        Z, A, Keeper = forwardPropagationRegularized(parameters['W'+str(l+1)], parameters['b'+str(l+1)], A_prev, parameters['F'+str(l+1)], keep_probs[l+1])
        Zs = Zs + [Z]
        As = As + [A]
        Keeps = Keeps + [Keeper]
    
    assert(len(Keeps) == len(As))
    
    return Zs, As, Keeps, A

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
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    assert(dAL.shape == AL.shape)
    grads["dA" + str(L)] = dAL

    # From l=L-1 to l=0, outbox included
    for l in reversed(range(L)):
        dA_l1, dW_l, dB_l = backwardPropagation(parameters["W" + str(l+1)], Zs[l], grads["dA" + str(l+1)], As[l], parameters["F" + str(l+1)])
        grads["dA" + str(l)] = dA_l1
        grads["dW" + str(l + 1)] = dW_l
        grads["db" + str(l + 1)] = dB_l
    return grads

def globalBackwardPropagationRegularized(Y, Zs, As, parameters, lambd, Keeps, keep_probs):
    grads = {}
    L = len(Zs)
    
    # Zs from 1 to L and As from 0 to L
    assert(len(Zs) + 1 == len(As))
    
    # Dropout applied only on the hidden units and sometimes on the output units
    assert(len(As) == len(Keeps))
    
    # Initializing the backpropagation
    AL = As[L]
    Y = Y.reshape((AL.shape))
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    assert(dAL.shape == AL.shape)
    grads["dA" + str(L)] = dAL

    # From l=L-1 to l=0, outbox included
    for l in reversed(range(L)):
        dA_l1, dW_l, dB_l = backwardPropagationRegularized(parameters["W" + str(l+1)], Zs[l], grads["dA" + str(l+1)], As[l], parameters["F" + str(l+1)], lambd)
        
        grads["dW" + str(l + 1)] = dW_l
        grads["db" + str(l + 1)] = dB_l
        
        dA_l1 = dA_l1 * Keeps[l]
        dA_l1 = dA_l1 / keep_probs[l]
        grads["dA" + str(l)] = dA_l1
        
    return grads

# UPDATES OF THE PARAMETERS

def updateParameters(parameters, grads, learning_rate):
    
    L = len(parameters) // 3
    
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - grads["dW" + str(l+1)] * learning_rate
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - grads["db" + str(l+1)] * learning_rate
        
    return parameters

# COST CALCULUS

def computeCost(AL, Y):
    cost = -np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL)) / Y.shape[1]
    
    return np.squeeze(cost)

# PREDICTION CALCULUS

def predict(parameters, X, decision_treshold = 0.5):
    prediction = np.zeros( (1,X.shape[1]) )
    
    _, _, A = globalForwardPropagation(X, parameters)

    for i in range(0, A.shape[1]):
        if A[0][i] > decision_treshold:
            prediction[0][i] = 1
        else:
            prediction[0][i] = 0
    
    return prediction

# MODEL CONSTRUCTION and TRAINING

def model(X, Y, layers_dims, actFunctions, learning_rate = 0.0075, num_iterations = 3000, print_cost = False):
    # Initialize the weights and the bias
    parameters = initialize(layers_dims, actFunctions)
    
    for i in range(0, num_iterations):

        Zs, As, AL = globalForwardPropagation(X, parameters)
        
        cost = computeCost(AL, Y)
        
        grads = globalBackwardPropagation(Y, Zs, As, parameters)
        
        parameters = updateParameters(parameters, grads, learning_rate)
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    return parameters

def modelRegularized(X, Y, layers_dims, actFunctions, keep_probs, learning_rate = 0.0075, num_iterations = 3000, lambd = 0.7, print_cost = False):
    # Initialize the weights and the bias
    parameters = initialize(layers_dims, actFunctions, print_cost)
    
    assert(len(layers_dims) == len(keep_probs))
    
    for i in range(0, num_iterations):

        Zs, As, Keeps, AL = globalForwardPropagationRegularized(X, parameters, keep_probs)
        
        cost = computeCost(AL, Y)
        
        grads = globalBackwardPropagationRegularized(Y, Zs, As, parameters, lambd, Keeps, keep_probs)
        
        parameters = updateParameters(parameters, grads, learning_rate)
                
        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    return parameters

# DISPLAY OF A SET OF PREDICTION

def visualize_decision(parameters, X, Y, display_img = True):

    X_prep = X.reshape(X.shape[0], -1).T / 255.
    prediction = predict(parameters, X_prep)
    
    if display_img:
        indexes = np.random.randint(prediction.shape[1], size = 15)

        plt.figure(figsize=(20, 32))
        for i, index in enumerate(indexes):
            plt.subplot(6, 3, i+1)

            if (prediction[0][index]):
                plt.title("YES: Cat picture.")
            else:
                plt.title("NO: Non-cat picture.")
            plt.imshow(X[index])
    
    return np.sum( (prediction == Y) / X_prep.shape[1] )