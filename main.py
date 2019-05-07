import pylab
import numpy as np
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


def createTrainData(testAmount= 30):
    random.seed()
    ir = readFile("Iris.txt",5)
    table = []
    #Setos ->      [1,0,0,0]
    #Versicolor -> [0,1,0,0]
    #Verginica ->  [0,0,1,0]
    i = 0
    while i<testAmount:
        i += 1
        ran = random.randint(1,149)
        b = []
        d = []
        b.append(float(ir[ran][0]))
        b.append(float(ir[ran][1]))
        b.append(float(ir[ran][2]))
        b.append(float(ir[ran][3]))
        d.append(b)
        if ir[ran][4] == "Iris-setosa":
            d.append([1,0,0,0])
            print("Setosa")
        elif ir[ran][4] == "Iris-versicolor":
            d.append([0,1,0,0])
            print("Versicolor")
        else:
            d.append([0,0,1,0])
            print("Verginica")
        table.append(d)
    return table

def createTestData():
    ir = readFile("Iris.txt",5)
    i = 0
    d = []
    while i<150:
        b = []
        b.append(float(ir[i][0]))
        b.append(float(ir[i][1]))
        b.append(float(ir[i][2]))
        b.append(float(ir[i][3]))
        d.append(b)
        i += 1
    print(len(d))
    return d


def main():

    pattern = createTrainData()
    testData = createTestData()
    neuralNetwork = NeuralNetwork(4, 3, 4)
    # train it with some patterns
    neuralNetwork.train(pattern)
    # test it
    neuralNetwork.test(testData)

    #neuralNetwork.test()






if __name__ == '__main__':
    main()
