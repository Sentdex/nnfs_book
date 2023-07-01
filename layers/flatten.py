import numpy as np 


""" flatten layer is just to be able to reshape the tensors to 2d array in forward pass and in backward 
to have a state to return the old tensor with same shape"""
class Flatten:
    
    #flatten the inputs to be shape [N,-1] 
    def forward(self,inputs,training):
        self.inputs=inputs
        (self.m,self.n_h,self.n_w,self.n_c)=inputs.shape
        self.output=inputs.reshape(self.m,-1)

    #flatten from [N,-1] to [m,n_h,n_w,_nc] previous shape || dvalues is to not brake the code in model class  
    def backward(self,dvalues):
        self.dinputs=self.inputs.reshape(self.m,self.n_h,self.n_w,self.n_c)        



