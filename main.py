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

def wczytanie():
    try:
        table = []
        with open("Iris.txt", "r") as f:
            lista_linii = [line.rstrip("\n") for line in f]
            for linia in lista_linii:
                for pole in linia.split(","):
                    table.append(pole)

        print(len(table))

        ir = [[]]
        licznik = 0
        licznik_2 = 0

        while licznik < len(table):
            if licznik % 5 != 4:
                ir[licznik_2].append(float(table[licznik]))
                licznik += 1
            else:
                if licznik + 1 != len(table):
                    ir[licznik_2].append(table[licznik])
                    ir.append([])
                    licznik += 1
                    licznik_2 += 1
                else:
                    ir[licznik_2].append(table[licznik])
                    licznik += 1
                    licznik_2 += 1

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

def podziel_na(ir):
    table_setosa = []
    table_versicolor = []
    table_verginica = []

    licznik = 0
    for x in range(len(ir)):
        if ir[licznik][4] == "Iris-setosa":
            table_setosa.append(ir[licznik])
            licznik += 1
        elif ir[licznik][4] == "Iris-versicolor":
            table_versicolor.append(ir[licznik])
            licznik += 1
        else:
            table_verginica.append(ir[licznik])
            licznik += 1

    return table_setosa,table_versicolor,table_verginica

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

    oryginal = wczytanie()
    setosa,versicolor,vergenica = podziel_na(oryginal)

    tableX11 = []
    tableX12 = []
    tableX13 = []
    for x in range(len(setosa)):
        tableX11.append(setosa[x][2])
    for x in range(len(vergenica)):
        tableX12.append(vergenica[x][2])
    for x in range(len(versicolor)):
        tableX13.append(versicolor[x][2])
    tableY11 = []
    tableY12 = []
    tableY13 = []
    for x in range(len(setosa)):
        tableY11.append(setosa[x][3])
    for x in range(len(vergenica)):
        tableY12.append(vergenica[x][3])
    for x in range(len(versicolor)):
        tableY13.append(versicolor[x][3])

    pylab.plot(tableX, tableY,'bs', color='green')
    pylab.plot(tableX1,tableY1,'bs', color = 'blue')
    pylab.plot(tableX2,tableY2, 'bs', color = 'red')
    pylab.plot(tableX11, tableY11,'+', color='green')
    pylab.plot(tableX12,tableY12,'+', color = 'blue')
    pylab.plot(tableX13,tableY13, '+', color = 'red')
    pylab.grid(True)
    pylab.show()

def createTrainData(testAmount):
    random.seed()
    ir = readFile("Iris.txt",5)
    table = []
    #Setos ->      [1,0,0,0]
    #Versicolor -> [0,1,0,0]
    #Verginica ->  [0,0,1,0]
    c = 0
    for j in range(3):
        for i in range(10):
            ran = random.randint(c,c+49)
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
        c += 50
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
    amountOfTestData = 100
    iterations = 10000
    learingRate = 0.01
    momentumFactor = 0.1
    # Dates needed to creating MLP
    inputNodes = 4
    hiddenNeurons = 4
    outputNodes = 4
    #------------------------------

    pattern = createTrainData(amountOfTestData)
    testData = createTestData()

    neuralNetwork = NeuralNetwork(inputNodes, hiddenNeurons, outputNodes)
    erors = neuralNetwork.train(pattern,iterations,learingRate,momentumFactor)
    neuralNetwork.testAndClassfication(testData)
    printChapterError(erors)


if __name__ == '__main__':
    main()
