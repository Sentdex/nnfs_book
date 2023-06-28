import numpy as np

class Optimizer_SGD:

    #default learning rate is 0.1
    def __init__(self,learning_rate=0.01,decay=0.,momentum=0.9):
        self.learning_rate=learning_rate
        self.current_learning_rate=learning_rate
        self.decay=decay
        self.iterations=0
        self.momentum=momentum

    #call once before update params,learning rate decay
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate=self.learning_rate*(1. /(1.+self.decay*self.iterations)) 

    #update params
    def update_params(self,layer):
        #with sgd momentum 
        if self.momentum:

            if not hasattr(layer,'weight_momentums'):
                layer.weight_momentums=np.zeros_like(layer.weights) 
                layer.bias_momentums=np.zeros_like(layer.biases)
            
            #weights updates with momentum
            weight_updates=self.momentum *layer.weight_momentums-self.current_learning_rate*layer.dweights
            layer.weight_momentums=weight_updates

            bias_updates=self.momentum*layer.bias_momentums-self.current_learning_rate*layer.biases
            layer.bias_momentums=bias_updates

        #without momentum regular SGD with decay 
        else:
            weight_updates=-self.current_learning_rate*layer.dweights
            bias_updates=-self.current_learning_rate*layer.biases

        #update w/b
        layer.weights+=weight_updates
        layer.biases+=bias_updates

    #call once after update params
    def post_update_params(self):
        self.iterations+=1

