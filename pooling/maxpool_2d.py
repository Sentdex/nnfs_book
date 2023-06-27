import numpy as np



class MaxPooling2D:

    def __init__(self,f,stride):
        self.f=f
        self.stride=stride

    #forward pass
    def forward(self,inputs):
        self.inputs=inputs
        #get inputs shape [batch,n_h,n_w]
        (m,n_h_prev,n_w_prev)=self.inputs.shape
        #define output shape    
        n_h=int((n_h_prev-self.f)/self.stride)+1
        n_w=int((n_w_prev-self.f)/self.stride)+1

        #init output
        self.output=np.zeros((m,n_h,n_w))

        for i in range(m):
            inputi=self.inputs[i]
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
    def backward(self,dvalues):
        (m,n_h_prev,n_w_prev)=self.inputs.shape
        (m_out,n_h,n_w)=self.output.shape
        #init the dinputs        
        self.dinputs=np.zeros_like(self.inputs)

        for i in range(m):
            inputi=self.inputs[i]
            for h in range(n_h):
                for w in range(n_w):
                    #vert/horiz start-end 
                    vert_start=h*self.stride
                    vert_end=vert_start+self.f
                    horiz_start=w*self.stride
                    horiz_end=horiz_start+self.f
                    #get input slice with shape [f,f]
                    input_slice=inputi[vert_start:vert_end,horiz_start:horiz_end]
                    # Find the indexes of the maximum value in the input_slice
                    max_index=np.unravel_index(np.argmax(input_slice), input_slice.shape)
                    #set max value
                    self.dinputs[i,vert_start:vert_end,horiz_start:horiz_end][max_index]+=1
                    #chain rule
                    self.dinputs[i,vert_start:vert_end,horiz_start:horiz_end]=np.multiply(self.dinputs[i,vert_start:vert_end,horiz_start:horiz_end],dvalues[i,h,w])
