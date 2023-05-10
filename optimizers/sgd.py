s

class Optimizer_SGD:

    #default learning rate is 0.1
    def __init__(self,learning_rate=0.01):
        self.learning_rate=learning_rate

    def update_params(self,layer):
        layer.weights+=-self.learning_rate*layer.dweights
        layer.biases+=-self.learning_rate*layer.biases

