import unittest
from convolution.conv_3d import Conv3D
from convolution.conv_2d import Conv2D

import numpy as np
import torch 
import torch.nn as nn
import helpers.compare as compare


class TestConv3D(unittest.TestCase):
    
    #todo:write tests to check not just the shape
    #zero padding shape should match [n+2p]
    def test_zero_pad_3d(self):
        input=np.random.randn(3,6,6,3)
        w_shape=(3,3,3,2);b_shape=(1,1,1,2);pad=3;stride=1
        conv3d=Conv3D(w_shape,b_shape,pad,stride)
        #conv3d=Conv3D(w,b,padding,stride=1)
        out=conv3d.zero_padding(input)
        self.assertEqual(out.shape[1],input.shape[1]+(2*pad),'shapes in 3d pad are wrong')

    #zero padding shape should match [n+2p]
    def test_zero_pad_2d(self):
        inputs=np.random.randn(3,6,6)
        w_shape=(3,3,2);b_shape=2;pad=3;stride=1
        conv2d=Conv2D(w_shape,b_shape,pad,stride)
        out=conv2d.zero_padding(inputs)
        self.assertEqual(out.shape[1],inputs.shape[1]+2*pad,'shapes in 2d pad are work')



    def test_conv2d(self):
        #inputs shapes [n,c_in,h_in,w_in]
        #m.weights shape is [out_c, in_c, kernel_height, kernel_width]
        input=torch.rand(1,1,3,3)
        m=nn.Conv2d(1,2,[2,2],bias=True,padding=2,
                    padding_mode='zeros',stride=2)
        m.requires_grad_=True
        out=m(input)
        #sum cause t.backward works only on scalar values i think 
        t=out.sum()
        t.backward()
        #permute values to match the shapes of conv2d impl. 
        t_w=m.weight.grad.permute(2,3,1,0).numpy()[:,:,0,:]
        t_b=m.bias.grad
        t_out=out.permute(0,2,3,1).detach().numpy()
        
        #init conv2d,we gonna use the weights and biases from pytorch   
        conv2d=Conv2D([2,2],2,padding=2,stride=2)
        inputs=input.permute(0,2,3,1).numpy()[:,:,:,0]
        weights=m.weight.permute(2,3,1,0).detach().numpy()[:,:,0,:]
        biases=m.bias.detach().numpy()
        #set params from pytorch 
        conv2d.set_params(weights,biases)
        #forward pass 
        conv2d.forward(inputs)
        #backward pass 
        dinputs=np.ones_like(conv2d.output)
        conv2d.backward(dinputs)

        #all close because of numerical instability to not check equals  
        self.assertEqual(np.allclose(t_out,conv2d.output),1,'output values are close')
        self.assertEqual(np.allclose(t_w,conv2d.dweights),1,'weights gradients are close')
        self.assertEqual(np.allclose(t_b,conv2d.dbiases),1,'bias gradients are close')




if __name__=='__main__':
    unittest.main()









