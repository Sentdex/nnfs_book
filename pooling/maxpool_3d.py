import numpy as np



class MaxPooling3D:

    def __init__(self,f,stride):
        self.f=f
        self.stride=stride

    def forward(self,inputs,training):
        self.inputs=inputs
        #get inputs shape
        (m,n_h_prev,n_w_prev,n_c_prev)=self.inputs.shape

        #define output shape
        n_h=int((n_h_prev-self.f)/self.stride)+1
        n_w=int((n_w_prev-self.f)/self.stride)+1

        #init output
        self.output=np.zeros((m,n_h,n_w,n_c_prev))
        
        for i in range(m):
            inputi=self.inputs[i]
            for h in range(n_h):
                for w in range(n_w):
                    for c in range(n_c_prev):
                        #define vert/horiz start-end
                        vert_start=h*self.stride
                        vert_end=vert_start+self.f
                        horiz_start=w*self.stride
                        horiz_end=horiz_start+self.f
                        #get image slice
                        inputi_slice=inputi[vert_start:vert_end,horiz_start:horiz_end,c]
                        #get max value of inputi_slice 
                        self.output[i,h,w,c]=np.max(inputi_slice)
        

    def backward(self,dvalues):
        #get inp/out shapes
        (m,n_h_prev,n_w_prev,n_c_prev)=self.inputs.shape
        (m,n_h,n_w,n_c_prev)=self.output.shape
        
        self.dinputs=np.zeros_like(self.inputs)

        for i in range(m):
            inputi=self.inputs[i]
            for h in range(n_h):
                for  w in range(n_w):
                    for c in range(n_c_prev):
                        vert_start=h*self.stride
                        vert_end=vert_start+self.f
                        horiz_start=w*self.stride
                        horiz_end=horiz_start+self.f
                        #get input slice shape[fh,fw,n_c_prev]
                        input_slice=inputi[vert_start:vert_end,horiz_start:horiz_end,c]
                        
                        max_index=np.unravel_index(np.argmax(input_slice), input_slice.shape)
                        # Find the indexes of the maximum value in the input_slice
                        self.dinputs[i,vert_start:vert_end,horiz_start:horiz_end,c][max_index]+=1
                        #chain rule 
                        self.dinputs[i,vert_start:vert_end,horiz_start:horiz_end,c]=np.multiply(self.dinputs[i,vert_start:vert_end,horiz_start:horiz_end,c],dvalues[i,h,w,c])




