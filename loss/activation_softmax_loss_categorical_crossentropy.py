import numpy as np
import categorical_cross_entropy as Loss_CategoricalCrossentropy
from activation_func import softmax as Activation_Softmax


class Activation_Softmax_Loss_CategoricalCrossentropy():

    def __init__(self):
        self.activation=Activation_Softmax()
        self.loss=Loss_CategoricalCrossentropy()

    
    def forward(self,inputs,y_true):
        self.activation.forward(inputs)

        self.output=self.activation.output

        return self.loss.calculate(self.output,y_true)
    
    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)

        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
          y_true = np.argmax(y_true, axis=1)
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples