import numpy as np


class Activation_Sigmoid:

    def forward(self,inputs):
        self.inputs=inputs
        self.output=1/(1+np.exp(-inputs))

    def backward(self,dvalues):
        self.dinputs=dvalues*(1-self.output)*self.output


