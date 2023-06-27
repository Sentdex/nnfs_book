import numpy as np



class MaxPooling2D:

    def __init__(self,f,stride):
        self.f=f
        self.stride=stride

    #forward pass
    def forward(self,inputs):
        #get inputs shape [batch,n_h,n_w]
        (m,n_h_prev,n_w_prev)=inputs.shape
        #define output shape    
        n_h=int((n_h_prev-self.f)/self.stride)+1
        n_w=int((n_w_prev-self.f)/self.stride)+1

        #init output
        self.output=np.zeros_like(m,n_h,n_w)

        for i in range(m):
            inputi=inputs[i]
            for h in range(n_h):
                for w in range(n_w):
                    #define vert/horiz start-end 
                    vert_start=h*self.stride
                    vert_end=vert_start+self.f
                    horiz_start=w*self.stride
                    horiz_end=horiz_start+self.f
                    #get input_slice to match shape [f,f]
                    input_slice=inputi[vert_start:vert_end,horiz_start:horiz_end]
                    #get max value of the input_slice,store in output  
                    self.output[i,h,w]=np.max(input_slice)

    #backward pass 
