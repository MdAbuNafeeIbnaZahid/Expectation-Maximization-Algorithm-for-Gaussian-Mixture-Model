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

    exampleCnt = row
    featureCnt = column

    

# This will return a 1 X featureCnt dimensional array
def getRandMu(featureCnt):
    randMu = np.random.rand( 1, featureCnt )
    return randMu

def getRandSigma(featureCnt):
    randSqAr = np.random.rand(featureCnt, featureCnt)
    randSqArTrans = randSqAr.T
    randSigma = randSqAr.dot(randSqArTrans)
    return randSigma



fileName = "Iris.csv"
dataSet = getDataSet(fileName)
trimmedDataSet = getTrimmedDataSet(dataSet, [0, 5])
runEMAlgo(trimmedDataSet, 3)
