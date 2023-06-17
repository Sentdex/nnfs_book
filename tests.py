import unittest
from convolution.conv_3d import Conv3D
import numpy as np


class TestConv3D(unittest.TestCase):
    
    #unit test for zero padding
    def test_zero_padding(self):
        images=np.random.randn(3,6,6,3)
        padding=3
        conv3d=Conv3D(images,0,0,padding=3,stride=0)
        out=conv3d.zero_padding()
        self.assertEqual(out.shape[1],images.shape[1]+(2*padding),'shapes are wrong')

    #todo:add for other functions  

if __name__=='__main__':
    unittest.main()









