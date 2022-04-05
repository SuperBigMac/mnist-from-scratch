import layers
from math import pow
import numpy as np

def L2_cost(estimated_val, actual):
    return math.pow(estimated_val - actual, 2)

def reLU_deriv(val):
    if val > 0:
        return 1
    else:
        return 0

def sigmoid_deriv(sigmoid_output):
    return sigmoid_output * (1-sigmoid_output)

class NN:
    def __init__(self, training_data):
        self.layers = []
        self.training_data = training_data
        self.outputs = []
        self.delC_delA = []

    def add_input_layer(self, input_size, input_dimensions = 1, flag = ''):
        self.layers.append(layers.Input_Layer(input_dimensions, flag = flag))
        self.layers[0].SIZE = input_size

        self.outputs = [[] for i in range(len(self.layers))]

    def add_convolutional_layer(self, layer_size, activation_mode, flag = ''):
        if len(self.layers) == 1:
            self.layers.append(layers.Convolutional_Layer(layer_size = layer_size, previous_size = self.layers[-1].SIZE, flag = flag))
        elif len(self.layers) > 1:
            self.layers.append(layers.Convolutional_Layer(layer_size = layer_size, previous_size = self.layers[-2].SIZE, flag = flag))
        else:
            raise Exception('No input layer added!')

        self.layers.append(layers.Activation_Layer(activation_mode, size = self.layers[-1].SIZE, flag = flag))

        self.outputs = [[] for i in range(len(self.layers))]

    def evaluate(self, array):
        arr = array[:]
        for i in range(len(self.layers)):
            arr = self.layers[i].evaluate(arr)
            self.outputs[i] = arr
            #print(arr) #for debug, large text output!

        return arr

    def cost(self, data): #data is a list of tuples where first item is input, second is label
        assert(len(data.shape) == 2)

        cost = 0
        for i in range(len(data)):
            cost += L2_cost(self.evaluate(data[i][0]),data[i][1])

        return cost / len(data)

    def stochastic_GD(self, batch_size = 10, epochs = 1):
        pass

    #layer_index = j
    def weight_gradient(self, layer_index, weight_index):
        prev_val = self.outputs[layer_index-1][weight_index]
        activation_deriv = sigmoid_deriv(self.outputs[layer_index+1][weight_index])

        gradient = prev_val * activation_deriv * next_weight_gradient(layer_index, weight_index)

        return sum(prev_val * activation_deriv * next_weight_gradient(next_layer, weights))

    def next_weight_gradient(layer_index, prev_index, curr_index):
        #current layer = self.layers[layer_index]
        if self.layers[layer_index].FLAG == 'output':
            return 2 * (self.outputs[-1][curr_index] - self.training_data[curr_index][1])

        weight = self.layers[layer_index].nodes[curr_index][0][prev_index]
        unnormalized_output = self.outputs[layer_index][curr_index]

        """figure out how to iterate through the tree recursively"""
        return weight * unnormalized_output * next_weight_gradient(next_convolutional_layer(layer_index + 1), curr_index, curr_index)


    def populate_delC_delA(self, data_index = 0):
        self.delC_delA = [[1 for i in range(len(self.outputs[k]))] for k in range(len(self.outputs))] #clearing delC_delA array
        for i in reversed(range(len(self.layers))):
            if self.layers[i].TYPE == 'convolutional':
                if self.layers[i].FLAG == 'output':
                    for k in range(len(self.outputs[i])):
                        self.delC_delA[i][k] = 2 * (self.outputs[-1][k] - self.training_data[data_index][1][k])
                else:
                    for k in range(len(self.outputs[i])):
                        count = 0
                        for j in range(len(self.outputs[i+2])):
                            count += self.layers[i+2].nodes[j][0][k] * sigmoid_deriv(self.outputs[i+1][k]) * self.delC_delA[i+2][j]
                        self.delC_delA[i][k] = count

    #searches for next convolutional layer including the starting index
    def next_convolutional_layer(self, start_index = 0):
        i = start_index
        while i < len(self.layers) - 1:
            if self.layers[i].TYPE == 'convolutional':
                return i

        return -1 #no convolutional layer was found


    def bias_gradient(self, layer_index, bias_index):
        pass

def main():
    print("main method of network.py")
    arr1 = [[-1,0],[1,4]]
    arr2 = [-4, 2, -2.4, 1.75]

    temp_data = [[0,[0 for i in range(10)]],0]
    temp_data[0][1][1] = 1
    network = NN(temp_data)

    network.add_input_layer(784)
    network.add_convolutional_layer(100, 'sigmoid')
    network.add_convolutional_layer(50, 'sigmoid')
    network.add_convolutional_layer(10, 'sigmoid', flag = "output")

    #print(network.layers) #chonky text output

    #chonky text output
    """for i in network.layers:
        if i.TYPE == "activation":
            print([i.nodes[index][0] for index in range(len(i.nodes))])"""

    network.evaluate([i/783 for i in range(784)])
    #out = np.array(network.outputs)
    network.populate_delC_delA()
    #print([len(network.delC_delA[i]) for i in range(len(network.delC_delA))])
    print(network.delC_delA)
    #print([np.array(out[i]).shape for i in range(len(out))])
    print(network.outputs[-1])

if __name__ == "__main__":
    main()
