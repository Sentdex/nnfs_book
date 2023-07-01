import numpy as np

class Conv3D:
    
    """    
    weights shape=[f_h,f_w,n_c_prev,n_c]
    f_h,h_w,n_c_prev == input shapes   
    nc ==channel depth/number of filters/n_c prime  
      """
    def __init__(self,w_shape,b_shape,padding=0,stride=1):
        self.weights=np.random.randn(*w_shape) * 0.1
        self.biases=np.random.randn(b_shape) * 0.1
        self.padding=padding
        self.stride=stride
        #TODO:add l1 and l2 regularization   
        self.weight_regularizer_l1 = 0
        self.weight_regularizer_l2 = 0
        self.bias_regularizer_l1 = 0
        self.bias_regularizer_l2 = 0


    #for comparing results in tests with pytorch  
    def set_params(self,weights,biases):
        self.weights=weights
        self.biases=biases
    
    #zero padding across [h,w] dimensions [h+(2*padding)] out dimension
    #same convolution can be done when use padding=(f-1)/2 to have the sam out.shape and input.shape          
    def zero_padding(self,inputs):
        inputs_padded = np.pad(inputs, ((0,0), (self.padding,self.padding), (self.padding,self.padding),
                                   (0,0)), mode='constant', constant_values = (0,0))
        return inputs_padded
      
    def forward(self,inputs,training):
        self.inputs=inputs
        #inputs shape [batch,height,width,c_in]
        (m,n_h_prev,n_w_prev,n_c_prev)=inputs.shape
        #weights shape[height,width,c_in,c_out]
        (f_h,f_w,f_n_c_prev,n_c)=self.weights.shape
        
        #define output shape H,W=[N-F+2P/S]+1 
        self.n_h=int((n_h_prev-f_h+2*self.padding)/self.stride)+1
        self.n_w=int((n_w_prev-f_w+2*self.padding)/self.stride)+1

        #init output 
        self.output=np.zeros((m,self.n_h,self.n_w,n_c))
        
        #zero padding
        self.inputs_pad=self.zero_padding(inputs)

        for i in range(m):
            inputi=self.inputs_pad[i]
            for h in range(self.n_h):
                for w in range(self.n_w):
                    for c in range(n_c):
                        #vertical/horizontal start-end 
                        vert_start=h*self.stride
                        vert_end=vert_start+f_h
                        horiz_start=w*self.stride
                        horiz_end=horiz_start+f_w
                        #get input slice which matches shapes of weights
                        input_slice=inputi[vert_start:vert_end,horiz_start:horiz_end,:]
                        #todo:convolution
                        self.output[i,h,w,c]=np.sum(np.multiply(input_slice,self.weights[...,c]))+self.biases[c]
        

    def backward(self,dvalues):
        #init gradients with zeros
        self.dweights=np.zeros_like(self.weights)
        self.dbiases=np.zeros_like(self.biases)
        
        #get input shape
        (m,n_h_prev,n_w_prev,n_c_prev)=self.inputs.shape
        (f_h,f_w,f_c_prev,n_c)=self.weights.shape

        #initalize input gradients with zeroes 
        self.dinputs=np.zeros((m,n_h_prev,n_w_prev,n_c_prev))
        dinputs_pad=self.zero_padding(self.dinputs)
    
        #chain rule 
        for i in range(m):
            inputi=self.inputs_pad[i]
            dinputs_padi=dinputs_pad[i]
            for h in range(self.n_h):
                for w in range(self.n_w):
                    for c in range(n_c):
                        vert_start=h *self.stride
                        vert_end=vert_start+f_h
                        horiz_start=w*self.stride
                        horiz_end=horiz_start+f_w
                        #get input  slice
                        input_slice=inputi[vert_start:vert_end,horiz_start:horiz_end,:]
                        #dvalues/dinputs
                        dinputs_padi[vert_start:vert_end,horiz_start:horiz_end,:]+=self.weights[:,:,:,c]*dvalues[i,h,w,c]
                        #dvalues/dweights 
                        self.dweights[:,:,:,c]+=np.multiply(input_slice,dvalues[i,h,w,c])
            # Set the ith training example's dinputs to the unpaded dinputs_padi
            self.dinputs[i,:,:,:]=dinputs_padi[self.padding:-self.padding,self.padding:-self.padding,:]        
        
        #dvalues/dbiases
        self.dbiases=np.sum(dvalues,axis=(0,1,2))
