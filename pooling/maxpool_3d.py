import numpy as np


class MaxPooling3D:

    def __init__(self, inputs, f, stride,mode):
        self.inputs = inputs
        self.f = f
        self.stride = stride
        self.mode=mode

    def forward(self):
        (m, n_H_prev, n_W_prev, n_C_prev) = self.inputs.shape

        # dimens of the output
        n_H = int(1 + (n_H_prev - self.f) / self.stride)
        n_W = int(1 + (n_W_prev - self.f) / self.stride)
        n_C = n_C_prev

        #init output matrix
        output=np.zeros((m,n_H,n_W,n_C))
        for i in range(m):
            for h in range(n_H):
                for w in range(n_W):
                    for c in range(n_C):
                        vert_start =h*self.stride
                        vert_end = vert_start+self.f
                        horiz_start = w*self.stride
                        horiz_end = horiz_start+self.f
                        a_prev_slice=self.inputs[i,vert_start:vert_end,horiz_start:horiz_end,c]

                        if self.mode=='max':
                            output[i,h,w,c]=np.max(a_prev_slice)
                        elif self.mode=='average':
                            output[i,h,w,c,]=np.mean(a_prev_slice)

                        return output
                    

    #TODO
    def backward(self):
        pass