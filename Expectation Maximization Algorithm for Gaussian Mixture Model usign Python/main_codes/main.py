import numpy as np
from scipy.stats import multivariate_normal

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


def runEMAlgo(dataSet, distCnt, roundCnt):
    row, column = np.shape(dataSet)

    exampleCnt = row
    featureCnt = column


    muAr = getKRandMu(featureCnt, distCnt)  # This is a k * 1 * featureCnt dimensional array
    sigmaAr = getKRandSigma(featureCnt, distCnt) # This is a k * featureCnt * featureCnt dimensional array

    wAr = np.random.rand(distCnt, 1)
    wAr = wAr / np.sum(wAr)
    print(wAr)


    # nAr is a distCnt * exampleCnt dimensional Array

    for i in range(roundCnt):
        nAr = getNAr(dataSet=dataSet, muAr=muAr, sigmaAr=sigmaAr)
        nwAr = np.multiply(wAr, nAr)
        print(nwAr.shape)
        print(nwAr)

# This will return a 1 X featureCnt dimensional array
def getRandMu(featureCnt):
    randMu = np.random.rand( 1, featureCnt )
    return randMu

# This will return a k * featureCnt dimensional array
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



# This will return a   distCnt * exampleCnt dimensional array
def getNAr(dataSet, muAr, sigmaAr):
    # dataSet is a exampleCnt * featureCnt dimensional array
    # muAr is  a distCnt * featureCnt dimensional array
    # sigmaAr is a distCnt * featureCnt * featureCnt dimensional array

    exampleCnt, featureCnt = dataSet.shape
    distCnt = sigmaAr.shape[0]

    nAr = np.zeros( (distCnt, exampleCnt) )

    for i in range(distCnt):
        sigma = sigmaAr[i, :, :]
        mu = muAr[i, :]
        print( mu.shape )
        for j in range(exampleCnt):
            example = dataSet[j,:];
            nAr[i,j] = multivariate_normal.pdf(x=example, mean=mu, cov=sigma)

    return nAr



fileName = "Iris.csv"
dataSet = getDataSet(fileName)
trimmedDataSet = getTrimmedDataSet(dataSet, [0, 5])
runEMAlgo(trimmedDataSet, 3, 1)
