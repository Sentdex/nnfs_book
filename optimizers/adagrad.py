import numpy as np


class Optimizer_Adagrad:

    # Initialize optimizer - set settings
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    # Call once before any parameter updates,SGD with decay
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations)) 

    #update params using adaptive gradient
    def update_params(self,layer):
        
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache=np.zeros.like(layer.weights)
            layer.bias_cache=np.zeros.like(layer.biases)
        
        #squared current gradients 
        layer.weight_cache+=layer.dweights**2
        layer.bias_cache+=layer.dbiases**2
        
        #SGD params update +normalization 
        #with square root cache
        layer.weights+=-self.current_learning_rate*layer.dweights/(np.sqrt(layer.weight_cache)+self.epsilon)
        layer.biases+=-self.current_learning_rate*layer.dbiases/(np.sqrt(layer.bias_cache)+self.epsilon)

    #call once after update params
    def post_update_params(self):
        self.iterations+=1

        


