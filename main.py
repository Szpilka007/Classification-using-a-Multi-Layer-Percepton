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

def printChapterError(table):

    pylab.plot(table)
    pylab.grid(True)
    pylab.xlabel("Epochs X 100")
    pylab.ylabel("Error")
    pylab.show()
    pylab.title("Errors of training")

def printChapterOfClasification(table1,table2,table3):
    tableX = []
    tableX1 = []
    tableX2 = []
    for x in table1:
        tableX.append(x[2])
    for x in table2:
        tableX1.append(x[2])
    for x in table3:
        tableX2.append(x[2])
    tableY = []
    tableY1 = []
    tableY2 = []
    for x in table1:
        tableY.append(x[3])
    for x in table2:
        tableY1.append(x[3])
    for x in table3:
        tableY2.append(x[3])
    pylab.plot(tableX, tableY,'ro', color='green')
    pylab.plot(tableX1,tableY1,'ro', color = 'blue')
    pylab.plot(tableX2,tableY2, 'ro', color = 'red')
    pylab.grid(True)
    pylab.show()

def createTrainData(testAmount):
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
        elif ir[ran][4] == "Iris-versicolor":
            d.append([0,1,0,0])
        else:
            d.append([0,0,1,0])
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
    return d


def main():

    #dates nedeed to program
    amountOfTestData = 30
    iterations = 10000
    learingRate = 0.01
    momentumFactor = 0.1
    # Dates needed to creating MLP
    inputNodes = 4
    hiddenNeurons = 4
    outputNodes = 4


    pattern = createTrainData(amountOfTestData)
    testData = createTestData()

    neuralNetwork = NeuralNetwork(inputNodes, hiddenNeurons, outputNodes)
    erors = neuralNetwork.train(pattern,iterations,learingRate,momentumFactor)
    neuralNetwork.testAndClassfication(testData)
    printChapterError(erors)


if __name__ == '__main__':
    main()
