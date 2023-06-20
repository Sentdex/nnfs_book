import numpy as np

class Conv3D:
    
    #[f,f,n_c_prev,nc] 
    def __init__(self,weights,biases,padding=0,stride=1):
        #add here numpy init for weights/filters and biases  
        self.weights=weights
        self.biases=biases
        self.padding=padding
        self.stride=stride
    
    #zero padding across [h,w] dimensions [h+(2*padding)] out dimension
    #same convolution can be done when use padding=(f-1)/2 to have the sam out.shape and input.shape          
    def zero_padding(self,inputs):
        inputs_padded = np.pad(inputs, ((0,0), (self.padding,self.padding), (self.padding,self.padding),
                                   (0,0)), mode='constant', constant_values = (0,0))
        return inputs_padded
    
    def conv_single_step(self,a_slice_prev,weights,biases):
        s=np.sum(np.multiply(a_slice_prev,weights))
        output=s+np.float64(biases)
        return output
    
    def forward(self,inputs):
        #inputs shape [batch,height,width,number of channels]
        (m,n_h_prev,n_w_prev,n_c_prev)=inputs.shape
        



        pass


    def backward(self,dinputs):
        pass

