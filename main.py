import pylab
import numpy
import matplotlib as plt
from MLP import *


def readFile(file,amountOfData):
    try:
        table = []
        with open(file, "r") as f:
            lista_linii = [line.rstrip("\n") for line in f]
            for linia in lista_linii:
                for pole in linia.split(","):
                    table.append(pole)
        ir = [[]]
        counter_2 = 0
        counter = 0
        while counter < len(table):
            ir.append([])
            for x in range(amountOfData):
                ir[counter_2].append(table[counter])
                counter += 1
            counter_2 += 1
        return ir


    except FileNotFoundError:
        print("File not found")


#Execuate Code
ir = readFile("Iris.txt",5)

def main():
    pattern = [
        [[1,0,0,0], [1,0,0,0]],
        [[0,1,0,0], [0,1,0,0]],
        [[0,0,1,0], [0,0,1,0]],
        [[0,0,0,1], [0,0,0,1]]
    ]

    # create a network with two input, two hidden, and one output nodes
    neuralNetwork = NeuralNetwork(4, 5, 4)
    # train it with some patterns
    neuralNetwork.train(pattern)
    # test it
    neuralNetwork.test(pattern)
    neuralNetwork.weights()


if __name__ == '__main__':
    main()
