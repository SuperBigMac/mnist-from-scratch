import math
import numpy as np
import random

def sigmoid(num):
    return 1/(1+math.exp(-1 * num))

def sigmoid_deriv(num):
    return sigmoid(num) * (1 - sigmoid(num))

class Node:
    def __init__(self, weights = [], biases = []):
        assert(len(weights) == len(biases))
        self.w = weights
        self.b = biases
    
    def size(self):
        return len(self.w)
    
    def populateRandom(self, size):
        self.w = [random.uniform(-1,1) for i in range(size)]
        self.b = [random.uniform(-1,1) for i in range(size)]

    def eval(self, input):
        assert(len(input) == len(self.w))
        sum = 0
        for i in range(self.size()):
            sum += sigmoid((input[i] * self.w[i]) + self.b[i])
        
        return sum
    
    def pDeriv(self, input, index): #returns partial derivative of output wrt input at index
        return sigmoid_deriv(input[index] * self.w[index] + self.b[index]) * self.w[index]
    
    def pDerivWeight(self, input, index): #returns pDeriv of output wrt weight at certain index
        return sigmoid_deriv(input[index] * self.w[index] + self.b[index]) * input[index]
    
    def pDerivBias(self, input, index): #returns pDeriv of output wrt bias at certain index
        return sigmoid_deriv(input[index] * self.w[index] + self.b[index])

class Layer:
    def __init__(self, size, prevLayerSize):
        self.nodes = [Node() for i in range(size)]
        for n in self.nodes:
            n.populateRandom(prevLayerSize)
        self.output = [0 for i in range(size)]
        self.prevLayerSize = prevLayerSize
        self.size = size
    
    def eval(self, input):
        assert(len(input) == self.prevLayerSize)
        for i in range(self.size):
            self.output[i] = self.nodes[i].eval(input)
        
        return self.output

class NN:
    def __init__(self, dimensions, learning_rate = 1): #dimensions: an array [input size, layer 1 size, layer 2 size, ..., output size]
        #layers: 1D array of layers (no input layer). Nodes accessible by .nodes
        self.layers = [Layer(dimensions[i+1],dimensions[i]) for i in range(len(dimensions)-1)]
        
        #values: 2D array of outputted values (incl. input and output)
        self.values = [[0 for j in range(dimensions[i])] for i in range(len(dimensions))]
        
        #nodes: 1D array of nodes
        self.nodes = []
        for layer in self.layers:
            self.nodes.append(layer.nodes)
        #cost: cost calculated from last training example
        self.cost = 0

        #expected: expected value(s) for traning example (input is self.values[0])
        self.expected = None

        #learning rate: hyperparam 
        self.learning_rate = learning_rate
    
    def eval(self, input):
        assert(len(self.values[0]) == len(input))
        self.values[0] = input
        for l in range(len(self.layers)):
            for n in range(len(self.layers[l].nodes)):
                self.values[l+1][n] = self.layers[l].nodes[n].eval(self.values[l])
        
        return self.values[-1]
    
    def setExpected(self, expected): #for supervised learning
        assert(len(expected) == len(self.layers[-1].nodes))
        self.expected = expected

    def L2_cost(self):
        assert(len(self.expected) == len(self.values[-1]))
        self.cost = sum([0.5 * math.pow((self.values[-1][i] - self.expected[i]),2) for i in range(len(self.expected))])
        return self.cost

    def cost_gradient(self, index):
        assert(index < len(self.values[-1]))
        return self.values[-1][index] - self.expected[index]

    def gradient(self, layerIndex, nodeIndex, prevNodeIndex = None): #recursively calculates partial derivative of cost wrt node's output
        assert(nodeIndex < len(self.layers[layerIndex].nodes))

        #self_gradient: node's output wrt node's input
        self_gradient = 1
        if prevNodeIndex != None:
            self_gradient = self.layers[layerIndex].nodes[nodeIndex].pDeriv(self.values[layerIndex],prevNodeIndex)
 
        if layerIndex == len(self.layers) - 1:
            return self_gradient * self.cost_gradient(nodeIndex)
        else:
            return self_gradient * sum([self.gradient(layerIndex + 1, i, nodeIndex) for i in range(len(self.layers[layerIndex + 1].nodes))])

    def calc_gradient(self, layerIndex, nodeIndex): #returns an array of tuples of gradients (w,b)
        assert(nodeIndex < len(self.layers[layerIndex].nodes))
        current_node = self.layers[layerIndex].nodes[nodeIndex]
        node_gradient = self.gradient(layerIndex, nodeIndex)
        return [(current_node.pDerivWeight(self.values[layerIndex],i) * node_gradient, current_node.pDerivBias(self.values[layerIndex],i) * node_gradient) for i in range(len(current_node.w))]

    def train_node(self, layerIndex, nodeIndex): #adjusts node by self.learning_rate
        adj = self.calc_gradient(layerIndex, nodeIndex)
        node = self.layers[layerIndex].nodes[nodeIndex]

        for i in range(len(node.w)):
            node.w[i] -= self.learning_rate * adj[i][0]
            node.b[i] -= self.learning_rate * adj[i][1]
    
    def GD(self):
        self.L2_cost()
        for i in range(len(self.layers)):
            for n in range(len(self.layers[i].nodes)):
                self.train_node(i,n)
        

def main():
    network = NN([1,1])
    for i in range(3000):
        network.setExpected([0.3])
        network.eval([0])
        network.GD()

        
        network.setExpected([0.4])
        network.eval([1])
        network.GD()

        network.setExpected([0.7])
        network.eval([2])
        network.GD()
        
        if i % 100 == 0:
            print("i = " + str(i))
            #print("Out: " + str(network.values[-1]))
            print("Cost: " + str(network.cost))
            print("----")
    
    print("Out: " + str(network.values[-1]))
    print("Cost: " + str(network.cost))
    #print(adj)
    print("----")

    print(network.eval([0]))
    print(network.eval([1]))
    print(network.eval([2]))
        


if __name__ == "__main__":
    main()
