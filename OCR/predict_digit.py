import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1./(1+np.exp(-z))

def predict(inpt):
    inpt = (inpt-np.mean(inpt, axis=1)[:,np.newaxis])/255
    inpt = np.hstack((np.ones((inpt.shape[0], 1)), inpt))
    num_cases = inpt.shape[0]
    thetaVec = np.load('weights.npy')
    t1 = np.reshape(thetaVec[0:(25*401)], (25, 401))
    t2 = np.reshape(thetaVec[(25*401):], (10,26))
    a1 = sigmoid(inpt.dot(t1.T))
    a1 = np.hstack((np.ones((a1.shape[0],1)), a1))
    a2 = sigmoid(a1.dot(t2.T))
    for i, case in enumerate(a2):
        plt.subplot(1, num_cases, i+1)
        plt.axis('off')
        prediction = np.argmax(case)
        plt.title(prediction)
        plt.imshow(np.reshape(inpt[i][1:], (20,20)).T, cmap='gray')
    plt.show()

