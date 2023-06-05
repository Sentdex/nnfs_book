import numpy as np
from activation_func import  relu
from activation_func import softmax


test=np.random.randn(5,3)
relu = relu.Activation_ReLU()
relu.forward(test)
print(relu.output)
print(relu.output.shape)










