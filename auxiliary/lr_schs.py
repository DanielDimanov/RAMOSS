## Defined as a class to save parameters as attributes
class LR_polynomial_decay:
    def __init__(self, epochs=50, initial_learning_rate=0.0001, power=1.0):
        # store the maximum number of epochs, base learning rate,
        # and power of the polynomial
        self.epochs = epochs
        self.initial_learning_rate = initial_learning_rate
        self.power = power
        
    def __call__(self, epoch):
        # compute the new learning rate based on polynomial decay
        decay = (1 - (epoch / float(self.epochs))) ** self.power
        updated_eta = self.initial_learning_rate * decay + 1e-21
        # return the new learning rate
        return float(updated_eta)