import pylab
import numpy
import matplotlib as plt


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
        for x in ir:
             print(x)
        return ir


    except FileNotFoundError:
        print("File not found")


#Execuate Code
readFile("Iris.txt",5)
