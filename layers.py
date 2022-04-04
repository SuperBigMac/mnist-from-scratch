import numpy as np
from math import exp

def weighted_sum(arr1, arr2):
    assert(len(arr1) == len(arr2))
    return np.multiply(arr1, arr2) #1D arrays Only

#converts 2D arrays into 1D arrays
def one_dimensional_array(array):
    arr = []
    for i in range(len(array)):
        for k in array[i]:
            arr.append(k)
    return arr

def reLU_array(array):
    for i in range(len(array)):
        array[i] = reLU(array[i])

def reLU(num):
    if num < 0:
        return 0
    else:
        return num

def sigmoid_array(array):
    for i in range(len(array)):
        array[i] = sigmoid(array[i])

def sigmoid(num):
    return 1/(1+exp(-1 * num))

class Layer:
    def __init__(self, flag='', size = 0): #, layer_size=1, init_Weight=1, init_Bias=0):
        #self.nodes = [[init_Weight, init_Bias] for i in range(layer_size)]
        self.TYPE = "layer"
        self.FLAG = flag
        self.SIZE = size

    def evaluate(array):
        raise NotImplementedError("No evaluation function implemented.")

    def addFlag(flag):
        self.FLAG = flag

class Convolutional_Layer(Layer):
    def __init__(self, layer_size=1, init_Bias=0, init_Weight=1, previous_size=1, random_init=True, flag = ''):
        super().__init__(flag = flag, size = layer_size)
        self.nodes = [[[init_Weight for i in range(previous_size)], init_Bias] for n in range(layer_size)]
        if random_init:
            for i in range(len(self.nodes)):
                for k in range(len(self.nodes[i][0])):
                    self.nodes[i][0][k] = np.random.uniform(-1, 1)
                self.nodes[i][1] = np.random.uniform(-1, 1)
        self.TYPE = "convolutional"

    def adjust_weights(self, deltas):
        assert(len(deltas) == len(self.nodes))
        for i in range(len(deltas)):
            assert(len(self.nodes[i]) == len(deltas[i]))
            for k in range(len(self.nodes[i])):
                self.nodes[i][0][k] += deltas[i][k]

    def adjust_bias(self, deltas):
        assert(len(deltas) == len(self.nodes))
        for i in range(len(deltas)):
            assert(len(self.nodes[i]) == len(deltas[i]))
            for k in range(len(self.nodes[i])):
                self.nodes[i][1][k] += deltas[i][k]

    def set_weights(self, new_weights):
        assert(len(self.nodes) == len(new_weights))
        for i in range(len(self.nodes)):
            assert(len(self.nodes[i]) == len(new_weights[i]))
            self.nodes[i][0] = new_weights[i]

    def set_bias(self, new_biases):
        assert(len(self.nodes) == len(new_biases))
        for i in range(len(self.nodes)):
            self.nodes[i][1] = new_biases[i]

    def evaluate(self, input_array):
        assert(len(input_array) == len(self.nodes[0][0]))
        out = []
        for i in range(len(self.nodes)):
            out.append(np.dot(self.nodes[i][0], input_array) + self.nodes[i][1])
        #return np.add(weighted_sum(input_array, [weight[0] for weight in self.nodes]), [bias[1] for bias in self.nodes])
        return out


class Input_Layer(Layer):
    def __init__(self, input_dimensions=1, layer_size=1, flag = ''):
        super().__init__(flag = flag, size = layer_size)
        self.TYPE = "input"
        self.input_dimension = input_dimensions

    def evaluate(self, input_array):
        if self.input_dimension == 1:
            return input_array
        elif self.input_dimension == 2:
            return one_dimensional_array(input_array)
        else:
            raise Exception("Too many input dimensions. Current: " + self.input_dimension)

class Activation_Layer(Layer):
    def __init__(self, activationMode, flag = ''):
        super().__init__(flag = flag, size = layer_size)
        self.mode = activationMode
        self.TYPE = "activation"

    def set_mode(self, newMode):
        self.mode = newMode

    def evaluate(self, input_array):
        arr = input_array[:]

        if self.mode == 'reLU':
            reLU_array(arr)
        elif self.mode == 'sigmoid':
            sigmoid_array(arr)
        else:
            raise Exception("Invalid activation function. Current: " + self.mode)

        return arr

def main():
    print("main method of layers.py")
    arr1 = [[-1,0],[1,4]]
    arr2 = [-4, 2, -2.4, 1.75]

    layer = Convolutional_Layer(layer_size = 2, previous_size=2)
    layer.set_weights(arr1)
    layer.adjust_weights(arr1)
    print(layer.nodes)

    """IN = Input_Layer(2)
    arr1 = IN.evaluate(arr1)
    AL1 = Activation_Layer('reLU')

    arr1 = AL1.evaluate(arr1)
    #reLU_array(arr1)

    print(arr1)
    conv = Convolutional_Layer(4)
    conv.set_weights(arr1)
    conv.set_bias(arr2)
    arr3 = conv.evaluate(arr1)
    #arr3 = weighted_sum(arr1, arr2)
    print(arr3)

    AL2 = Activation_Layer('sigmoid')
    arr3 = AL2.evaluate(arr3)
    print(arr3)"""

if __name__ == "__main__":
    main()
