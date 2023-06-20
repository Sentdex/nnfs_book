import unittest
from convolution.conv_3d import Conv3D
import numpy as np


class TestConv3D(unittest.TestCase):
    
    #zero padding shape should match [n+2p]
    def test_zero_padding(self):
        input=np.random.randn(3,6,6,3)
        padding=3
        conv3d=Conv3D(0,0,padding,stride=1)
        out=conv3d.zero_padding(input)
        self.assertEqual(out.shape[1],input.shape[1]+(2*padding),'shapes are wrong')

    #todo:add for other functions  




if __name__=='__main__':
    unittest.main()









