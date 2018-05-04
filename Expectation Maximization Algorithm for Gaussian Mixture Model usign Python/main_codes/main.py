import numpy as np
from scipy.stats import multivariate_normal
import copy

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
    trimmedDataSet = trimmedDataSet.astype(np.float)
    return trimmedDataSet


def runEMAlgo(dataSet, distCnt, roundCnt):
    row, column = np.shape(dataSet)

    exampleCnt = row
    featureCnt = column

    # singleMu = np.mean( dataSet, axis=0 )
    # singleSigma = np.dot(dataSet.T, dataSet) / exampleCnt ;
    # print( singleSigma )




    muAr = getKRandMu(dataSet, distCnt)  # This is a k * 1 * featureCnt dimensional array
    print( "printing muAr "   )
    print(muAr)

    sigmaAr = getKRandSigma(dataSet, distCnt) # This is a k * featureCnt * featureCnt dimensional array
    print("printing sigmaAr" )
    print( sigmaAr )

    wAr = (1.0 / distCnt) * np.ones( [distCnt, 1] );


    # nAr is a distCnt * exampleCnt dimensional Array

    for i in range(roundCnt):
        nAr = getNAr(dataSet=dataSet, muAr=muAr, sigmaAr=sigmaAr)
        WMulN = np.multiply(wAr, nAr)  # nwAr has size distCnt * exampleCnt
        wnArDistSum = np.sum(WMulN, axis=0)[np.newaxis]
        pAr = WMulN / wnArDistSum;

        mStep(dataSet=dataSet, oldMuAr=muAr, oldSigmaAr=sigmaAr, pAr=pAr, oldWAr=wAr)

        logLikelihood = np.sum(np.log( wnArDistSum ) )  # log likelihood is a scalar value


def getRandomScaled(ar):
    randomAr = np.random.rand(*ar.shape)
    scaleAr = randomAr * 2

    ret = np.multiply(ar, scaleAr)
    return ret;



# This will return a 1 X featureCnt dimensional array
def getRandMu(dataSet):
    randDS = getRandomScaled(dataSet)
    randMu = np.mean(randDS, axis=0)
    return randMu;

# This will return a k * featureCnt dimensional array
def getKRandMu(dataSet, k):
    ret = np.zeros( (k, dataSet.shape[1]) )

    for i in range(k):
        ret[i, :] = getRandMu(dataSet)

    return ret;

# This will return a featureCnt X featureCnt dimensional array
def getRandSigma(dataSet):
    exampleCnt = dataSet.shape[0]

    randDS = getRandomScaled(dataSet)
    randSigma = np.dot(randDS.T, randDS) / exampleCnt;

    return randSigma

# This will return a k X featureCnt X featureCnt dimensional array
def getKRandSigma(dataSet, k):
    featureCnt = dataSet.shape[1]

    ret = np.zeros( ( k, featureCnt, featureCnt ) )

    for i in range(k):
        ret[i, :, :] = getRandSigma(dataSet=dataSet)

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
        for j in range(exampleCnt):
            example = dataSet[j,:];
            nAr[i,j] = multivariate_normal.pdf(x=example, mean=mu, cov=sigma)

    return nAr




def mStep(dataSet, pAr, oldMuAr, oldSigmaAr, oldWAr):

    # pAr has size  distCnt * exampleCnt

    pArExampleSum = np.sum(pAr, axis=1)[np.newaxis]  # pArExampleSum is a   1 * distCnt matrix
    print( pArExampleSum )
    disCnt = pArExampleSum.shape[1]

    newMuAr = copy.deepcopy(oldMuAr)
    newSigmaAr = copy.deepcopy(oldSigmaAr)
    newWAr = copy.deepcopy(oldWAr)

    for i in range(disCnt):
        pass





fileName = "Iris.csv"
dataSet = getDataSet(fileName)
trimmedDataSet = getTrimmedDataSet(dataSet, [0, 5])
runEMAlgo(trimmedDataSet, 3, 1)
