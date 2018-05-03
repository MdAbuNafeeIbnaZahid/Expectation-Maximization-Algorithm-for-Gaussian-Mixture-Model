import numpy as np

import definitions


def getDataSet(fileName):
    pathToFile = definitions.ROOT_DIR + "/" + fileName;
    openedFile = open(pathToFile, "rb")

    loadedText = np.loadtxt(openedFile, delimiter=",", skiprows=1, dtype=str)
    return loadedText



def getTrimmedDataSet(dataSet, columnsToThrow):

    dataSet = np.array( dataSet )
    row, column = np.shape(dataSet)

    fullColumnList = list(range(column) )
    keepColumnList = [ x for x in fullColumnList if x not in columnsToThrow ]

    trimmedDataSet = dataSet[:,keepColumnList]
    return trimmedDataSet


def runEMAlgo(dataSet, distCnt):
    row, column = np.shape(dataSet)


def getRandMu(featureCnt):



fileName = "Iris.csv"
dataSet = getDataSet(fileName)
trimmedDataSet = getTrimmedDataSet(dataSet, [0, 5])
print(trimmedDataSet)
