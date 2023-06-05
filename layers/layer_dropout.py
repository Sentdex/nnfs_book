import numpy as np



class Layer_Dropout:


    #init
    def __init__(self,rate):
        self.rate=1-rate
    
    def forward(self,inputs):
        self.inputs=inputs

        self.binary_mask=np.random.binomial(1,self.rate,size=inputs.shape)/self.rate

        self.output=inputs*self.binary_mask

    
    def backward(self,dvalues):
        #gradients inputs*binary_mask dy/dinputs=binary_mask
        self.dinputs=dvalues*self.binary_mask




