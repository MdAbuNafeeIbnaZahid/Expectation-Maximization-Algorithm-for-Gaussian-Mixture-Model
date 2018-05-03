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

    kRandMu = getKRandMu(featureCnt, distCnt)
    kRandSigma = getKRandSigma(featureCnt, distCnt)

    print( kRandSigma )



# This will return a 1 X featureCnt dimensional array
def getRandMu(featureCnt):
    randMu = np.random.rand( 1, featureCnt )
    return randMu

def getKRandMu(featureCnt, k):
    ret = np.zeros( (k, featureCnt) )

    for i in range(k):
        ret[i, :] = getRandMu(featureCnt)

    return ret;

# This will return a featureCnt X featureCnt dimensional array
def getRandSigma(featureCnt):
    randSqAr = np.random.rand(featureCnt, featureCnt)
    randSqArTrans = randSqAr.T
    randSigma = randSqAr.dot(randSqArTrans)
    return randSigma

# This will return a k X featureCnt X featureCnt dimensional array
def getKRandSigma(featureCnt, k):
    ret = np.zeros( ( k, featureCnt, featureCnt ) )

    for i in range(k):
        ret[i, :, :] = getRandSigma(featureCnt)

    return ret;


fileName = "Iris.csv"
dataSet = getDataSet(fileName)
trimmedDataSet = getTrimmedDataSet(dataSet, [0, 5])
runEMAlgo(trimmedDataSet, 3)
