import numpy as np
from scipy.stats import multivariate_normal
import copy
import tkinter
from matplotlib import pyplot as plt




import definitions



def getRandomWMuSigmaAr(distCnt, featureCnt):
    retDic = {}

    trueWAr = np.full((1, distCnt), 1.0 / distCnt)
    trueMuAr = np.random.rand(distCnt, featureCnt)
    trueSigmaAr = getRandomSigmaAr(distCnt=distCnt, featureCnt=featureCnt)

    retDic['w'] = trueWAr
    retDic['mu'] = trueMuAr
    retDic['sigma'] = trueSigmaAr

    return retDic


# This will generate some examples each having two features
def generateDataSet(trueWAr, trueMuAr, trueSigmaAr, exampleCnt):
    print("trueWAr")
    print(trueWAr)

    print("Inside generateDataSet")
    print("exampleCnt")
    print(exampleCnt)

    featureCnt = trueMuAr.shape[1];
    print("featureCnt")
    print(featureCnt)

    distCnt = trueWAr.shape[1];
    print("distCnt")
    print(distCnt)

    dataSet = np.zeros((exampleCnt, featureCnt) )


    for j in range(exampleCnt):
        zj = np.random.choice(a=np.arange(distCnt), p=trueWAr[0])

        currentMu = trueMuAr[zj]
        # print( "currentMu" )
        # print( currentMu )

        currentSigma = trueSigmaAr[zj]
        # print('currentSigma')
        # print( currentSigma )

        dataSet[j] = np.random.multivariate_normal(mean=currentMu, cov=currentSigma)


    print("dataSet.shape ")
    print( dataSet.shape )
    return dataSet

def getRandomSigmaAr(distCnt, featureCnt):
    randSigmaAr = np.random.rand( distCnt, featureCnt, featureCnt )
    for i in range(distCnt):
        randSigmaAr[i] = getRandSigma(featureCnt)
    return randSigmaAr

def getRandSigma(featureCnt):
    randSqAr = np.random.rand(featureCnt, featureCnt)
    randSigma = randSqAr * randSqAr.T;
    return randSigma


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

    exampleCnt, featureCnt = dataSet.shape

    wAr = (1.0 / distCnt) * np.ones([distCnt, 1]);
    muAr = np.zeros( (distCnt, featureCnt) )
    sigmaAr = np.zeros( (distCnt, featureCnt, featureCnt) )

    for i in range(distCnt):
        randDS = getRandomScaled(dataSet)
        muAr[i] = np.mean(randDS, axis=0)
        sigmaAr[i] = np.dot(randDS.T, randDS) / exampleCnt;


    # muAr = getKRandMu(dataSet, distCnt)  # This is a k * 1 * featureCnt dimensional array
    # sigmaAr = getKRandSigma(dataSet, distCnt) # This is a k * featureCnt * featureCnt dimensional array

    print( "shape of initial muAr" )
    print( muAr.shape )

    print( "shape of initial sigmaAR " )
    print( sigmaAr.shape )



    # nAr is a distCnt * exampleCnt dimensional Array

    for i in range(roundCnt):
        nAr = getNAr(dataSet=dataSet, muAr=muAr, sigmaAr=sigmaAr)
        WMulN = np.multiply(wAr, nAr)  # nwAr has size distCnt * exampleCnt
        wnArDistSum = np.sum(WMulN, axis=0)[np.newaxis]
        pAr = WMulN / wnArDistSum;

        logLikelihood = np.sum(np.log(wnArDistSum))  # log likelihood is a scalar value

        # print( i )
        # print( logLikelihood )


        newValues = mStep(dataSet=dataSet, oldMuAr=muAr, oldSigmaAr=sigmaAr, pAr=pAr, oldWAr=wAr)

        muAr = newValues['newMuAr']
        wAr = newValues['newWAr']
        sigmaAr = newValues['newSigmaAr']


    print("Printing muAr")
    print( muAr )

    print("printing wAR")
    print( wAr )

    print( "printing sigmaAr" )
    print( sigmaAr )



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
def getRandSigmaFromDataSet(dataSet):
    exampleCnt = dataSet.shape[0]

    randDS = getRandomScaled(dataSet)
    randSigma = np.dot(randDS.T, randDS) / exampleCnt;

    return randSigma

# This will return a k X featureCnt X featureCnt dimensional array
def getKRandSigma(dataSet, k):
    featureCnt = dataSet.shape[1]

    ret = np.zeros( ( k, featureCnt, featureCnt ) )

    for i in range(k):
        ret[i, :, :] = getRandSigmaFromDataSet(dataSet=dataSet)

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

    exampleCnt = dataSet.shape[0]

    # pAr has size  distCnt * exampleCnt
    pArExampleSum = np.sum(pAr, axis=1)[np.newaxis]
    pArExampleSum = pArExampleSum.T # pArExampleSum is a   distCnt * 1 matrix

    disCnt = pArExampleSum.shape[0]
    # print( disCnt )


    newSigmaAr = copy.deepcopy(oldSigmaAr)


    newWAr = pArExampleSum / exampleCnt

    newMuAr = np.dot( pAr, dataSet ) / pArExampleSum;
    # print( newMuAr )


    for i in range(disCnt):
        muForThisDist = oldMuAr[i]
        # print( muForThisDist )

        xMinMu = dataSet - muForThisDist  # This is a exampleCnt * featureCnt dimensional array

        pArForThisDist = pAr[i,:][np.newaxis]  # this is a 1 * 150 dimensional array
        pArForThisDist = pArForThisDist.T  # this is a 150 * 1 dimensiona array
        # print( pArForThisDist.shape )
        # print( pArForThisDist )


        xMinMuMulPArForThisDist = np.multiply( xMinMu, pArForThisDist ) # this is a exampleCnt *
                                                            # featureCnt dimensinal array
        # print( xMinMuMulPArForThisDist.shape )
        newSigmaAr[i] = np.dot(xMinMuMulPArForThisDist.T, xMinMu) / np.sum(pArForThisDist)


    return {'newWAr' : newWAr, 'newMuAr' : newMuAr, 'newSigmaAr' : newSigmaAr}








# fileName = "Iris.csv"
# dataSet = getDataSet(fileName)
# trimmedDataSet = getTrimmedDataSet(dataSet, [0, 5])

featureCnt = 2
distCnt = 3


trueWMuSigmaAr = getRandomWMuSigmaAr(distCnt=distCnt, featureCnt=featureCnt)

trueWAr = trueWMuSigmaAr['w']
trueMuAr = trueWMuSigmaAr['mu']
trueSigmaAr = trueWMuSigmaAr['sigma']

print("trueMuAr")
print(trueMuAr)


dataSet = generateDataSet(trueWAr=trueWAr, trueMuAr=trueMuAr, trueSigmaAr=trueSigmaAr,
                          exampleCnt=150)

print(trueMuAr)
plt.scatter(trueMuAr[:,0], trueMuAr[:,1] )
plt.show()


plt.scatter(dataSet[:,0], dataSet[:,1] )
plt.show()

# print(dataSet)
runEMAlgo(dataSet, distCnt=3, roundCnt=100)



