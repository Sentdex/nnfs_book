import unittest
from convolution.conv_3d import Conv3D
from convolution.conv_2d import Conv2D
from pooling.maxpool_2d import MaxPooling2D
from pooling.maxpool_3d import MaxPooling3D
import numpy as np
import torch 
import torch.nn as nn

class TestConv3D(unittest.TestCase):
    
    #todo:write tests to check not just the shape
    #zero padding shape should match [n+2p]
    def test_zero_pad_3d(self):
        input=np.random.randn(3,6,6,3)
        w_shape=(3,3,3,2);b_shape=2;pad=3;stride=1
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
        self.assertEqual(out.shape[1],inputs.shape[1]+2*pad,'shapes in 2d pad are wrong')

    def test_conv2d(self):
        #inputs shapes [n,c_in,h_in,w_in]
        #m.weights shape is [out_c, in_c, kernel_height, kernel_width]
        input=torch.rand(1,1,3,3)
        m=nn.Conv2d(1,2,[2,2],bias=True,padding=2,
                padding_mode='zeros',stride=2)
        #set params to require gradients 
        m.requires_grad_=True
        input.requires_grad=True

        out=m(input)
        #sum cause t.backward works only on scalar values i think 
        t=out.sum()
        t.backward()
        #permute values to match the shapes of conv2d impl. 
        t_input_grad=input.grad.permute(0,2,3,1).detach().numpy()[:,:,:,0]
        t_w=m.weight.grad.permute(2,3,1,0).numpy()[:,:,0,:]
        t_b=m.bias.grad
        t_out=out.permute(0,2,3,1).detach().numpy()

        #init conv2d,we gonna use the weights and biases from pytorch   
        conv2d=Conv2D([2,2],2,padding=2,stride=2)
        inputs=input.permute(0,2,3,1).detach().numpy()[:,:,:,0]
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
        self.assertEqual(np.allclose(t_out,conv2d.output),1,'output values are not close')
        self.assertEqual(np.allclose(t_input_grad,conv2d.dinputs),1,'input gradients are not close')
        self.assertEqual(np.allclose(t_w,conv2d.dweights),1,'weights gradients are not close')
        self.assertEqual(np.allclose(t_b,conv2d.dbiases),1,'bias gradients are not close')

    # 
    def test_max_pool2d(self):
        #inputs shapes [n,c_in,h_in,w_in]
        input = torch.randn(1, 1, 3, 3)
        #filter size =[2,2]
        m = nn.MaxPool2d(2, stride=1)
        input.requires_grad=True
        out=m(input)
        t=torch.sum(out)
        t.backward()
        in_grad=input.grad
        #permute the tensors too match the shapes of maxpool2d 
        input_permuted=input.permute(0,2,3,1).detach().numpy()[:,:,:,0]
        input_grad=input.grad.permute(0,2,3,1).detach().numpy()[:,:,:,0]
        out_perm=out.permute(0,2,3,1).detach().numpy()[:,:,:,0]
        
        pool2d=MaxPooling2D(2,1)
        dvalues=np.ones_like(input_permuted)
        pool2d.forward(input_permuted)
        pool2d.backward(dvalues)
        #forward check
        self.assertEqual(np.allclose(pool2d.output,out_perm),1,'output values are not close')
        #backward check 
        self.assertEqual(np.allclose(pool2d.dinputs,input_grad),1,'gradients values are not close')
        
    def test_conv3d(self):
        input=torch.rand(5,3,10,10)
        m=nn.Conv2d(3,2,[3,3],bias=True,padding=1,
            padding_mode='zeros',stride=1)
        #set params to require gradients 
        m.requires_grad_=True
        input.requires_grad=True

        out=m(input)
        #sum cause t.backward works only on scalar values i think 
        t=out.sum()
        t.backward()
        #permute values to match the shapes of conv2d impl. 
        t_input_grad=input.grad.permute(0,2,3,1).detach().numpy()
        t_w=m.weight.grad.permute(2,3,1,0).numpy()
        t_b=m.bias.grad
        t_out=out.permute(0,2,3,1).detach().numpy()

        inputs=input.permute(0,2,3,1).detach().numpy()
        weights=m.weight.permute(2,3,1,0).detach().numpy()
        biases=m.bias.detach().numpy()

        conv3d=Conv3D([3,3,3,6],6,padding=1,stride=1)
        conv3d.set_params(weights,biases)
        conv3d.forward(inputs)
        dinputs=np.ones_like(conv3d.output)
        conv3d.backward(dinputs)
        #all close because of numerical instability to not check equals || round to 3 decimals    
        self.assertEqual(np.allclose(conv3d.output.round(3),t_out.round(3)),1,'output values are not close')
        self.assertEqual(np.allclose(conv3d.dweights,t_w),1,'weights gradients  are not close')
        self.assertEqual(np.allclose(conv3d.dbiases,t_b),1,'bias gradients  are not close')
        self.assertEqual(np.allclose(conv3d.dinputs,t_input_grad),1,'input gradients  are not close')

    def test_max_pool3d(self):
        input = torch.randn(1, 1, 3, 3)
        #filter size =[2,2]
        m = nn.MaxPool2d(2, stride=1)
        input.requires_grad=True
        out=m(input)
        t=torch.sum(out)
        t.backward()
        in_grad=input.grad
        #permute the tensors too match the shapes of maxpool2d 
        input_permuted=input.permute(0,2,3,1).detach().numpy()
        input_grad=input.grad.permute(0,2,3,1).detach().numpy()
        out_perm=out.permute(0,2,3,1).detach().numpy()

        #init maxpool3D forward backward pass   
        pool3d=MaxPooling3D(2,stride=1)
        pool3d.forward(input_permuted)
        dvalues=np.ones_like(input_permuted)
        pool3d.backward(dvalues)
        
        self.assertEqual(np.allclose(pool3d.output,out_perm),1,'output values are not close')
        self.assertEqual(np.allclose(pool3d.dinputs,input_grad),1,'output values are not close')





if __name__=='__main__':
    unittest.main()









