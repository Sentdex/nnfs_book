import numpy as np

class Conv3D:
    
    """    
    weights shape=[f_h,f_w,n_c_prev,n_c]
    f_h,h_w,n_c_prev == input shapes   
    nc ==channel depth/number of filters/n_c prime  
      """
    def __init__(self,w_shape,b_shape,padding=0,stride=1):
        self.weights=np.random.randn(*w_shape) * 0.1
        self.biases=np.random.randn(*b_shape) * 0.1
        self.padding=padding
        self.stride=stride
    
    #zero padding across [h,w] dimensions [h+(2*padding)] out dimension
    #same convolution can be done when use padding=(f-1)/2 to have the sam out.shape and input.shape          
    def zero_padding(self,inputs):
        inputs_padded = np.pad(inputs, ((0,0), (self.padding,self.padding), (self.padding,self.padding),
                                   (0,0)), mode='constant', constant_values = (0,0))
        return inputs_padded
      
    def forward(self,inputs):
        #inputs shape [batch,height,width,number of channels]
        (m,n_h_prev,n_w_prev,n_c_prev)=inputs.shape
        #weights shape[height,width,number of channels previous,number of channels output]
        (f_h,f_w,f_n_c_prev,n_c)=self.weights.shape
        
        #define output shape H,W=[N-F+2P/S]+1 
        n_h=int((n_h_prev-f_h+2*self.padding)/self.stride)+1
        n_w=int((n_w_prev-f_w+2*self.padding)/self.stride)+1

        #init output 
        output=np.zeros((m,n_h,n_w,n_c))
        
        #zero padding
        inputs_padded=self.zero_padding(inputs)

        for i in range(m):
            inputi=inputs_padded[i]
            for h in range(n_h):
                for w in range(n_w):
                    for c in range(n_c):
                        #vertical/horizontal start-end 
                        vert_start=h*self.stride
                        vert_end=vert_start+f_h
                        horiz_start=w*self.stride
                        horiz_end=horiz_start+f_w
                        #get input slice which matches shapes of weights
                        input_slice=inputi[vert_start:vert_end,horiz_start:horiz_end,:]
                        #todo:convolution
                        output[i,h,w,c]=np.sum(np.multiply(input_slice,self.weights[...,c]))+self.biases[...,c]
        

        return  output
    
    def backward(self,dinputs):
        pass

