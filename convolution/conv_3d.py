import numpy as np



#TODO:review re-write 
class Conv3D:
        
    def __init__(self,inputs,weights,biases,padding,stride):
        self.inputs=inputs
        self.weights=weights
        self.biases=biases
        self.padding=padding
        self.stride=stride
        
    
    def zero_padding(self):
        #zero padding 3D over height,width 
        inputs_padded = np.pad(self.inputs, ((0,0), (self.padding,self.padding), (self.padding,self.padding),
                                   (0,0)), mode='constant', constant_values = (0,0))
        return inputs_padded
    
    def conv_single_step(self,a_slice_prev,weights,biases):
        s=np.sum(a_slice_prev*weights)
        output=s+np.float64(biases)
        return output
        
    def conv_forward(self):
        #get shapes of inputs and weights 
        (m, n_H_prev, n_W_prev, n_C_prev) = self.inputs.shape
        (f, f, n_C_prev, n_C) = self.weights.shape
        
        #compute the dims of output volume formula: [(n-f+2p/stride)+1]
        n_H=int((n_H_prev-f+2*self.padding)/self.stride)+1
        n_W=int((n_W_prev-f+2*self.padding)/self.stride)+1
        
        #init the output volume with 0
        Z=np.zeros((m,n_H,n_W,n_C))
        
        #zero padding
        inputs_padded=self.zero_padding()
        
        for i in range(m):
            a_prev_pad=inputs_padded[i]
            for h in range(n_H):
                for w in range(n_W):
                    for c in range(n_C):
                        vert_start=h*self.stride
                        vert_end=vert_start+f
                        horiz_start=w*self.stride
                        horiz_end=horiz_start+f
                        
                        a_slice_prev=a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                        
                        Z[i,h,w,c]=self.conv_single_step(a_slice_prev,self.weights[...,c],self.biases[...,c])

                        return Z
        
        #todo
        def conv_backward(self):
            pass
    
    