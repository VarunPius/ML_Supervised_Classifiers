from bs4 import BeautifulSoup
import urllib2
import numpy as np
from numpy.linalg import inv
import pandas as pd
import sympy as sp
from scipy import linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA


def Scrape():
    site= "https://en.wikipedia.org/wiki/Iris_flower_data_set"
    hdr = {'User-Agent': 'Mozilla/5.0'}
    req = urllib2.Request(site, headers=hdr)
    page = urllib2.urlopen(req)
    soup = BeautifulSoup(page, "lxml")

    SepalLen = 0
    SepalWidth = 0
    PetalLen = 0
    PetalWidth = 0
    Species = ""
    table = soup.find("table", { "class" : "wikitable sortable" })

    fl = open('Data.csv', 'w')
    write_to_file = "SepalLength" + "," + "SepalWidth" + "," + "PetalLength" + "," + "PetalWidth" + "," + "Species" + "\n"
    fl.write(write_to_file)

    for row in table.findAll("tr"):
        cells = row.findAll("td")
        #For each "tr", assign each "td" to a variable.
        if len(cells) == 5:
            SepalLen = cells[0].find(text=True)
            SepalWidth = cells[1].find(text=True)
            PetalLen = cells[2].find(text=True)
            PetalWidth = cells[3].find(text=True)
            Species = cells[4].find(text=True)
            write_to_file = SepalLen + "," + SepalWidth + "," + PetalLen + "," + PetalWidth + "," + Species[3:] + "\n"
            write_unic = write_to_file.encode("utf8")
            fl.write(write_unic)

    fl.close()

def MatrixDef():
    no_of_datasets = 150
    dataset_per_class = no_of_datasets/3
    for percent in (10, 30, 50):
        df = pd.read_csv("Data.csv")
        Xi = df.as_matrix()     #Intermediate X to convert data into Matrix

        Xc = Xi[:,:-1]
        Xc = np.c_[Xc, np.ones(no_of_datasets)]     #Adding column of 1's for unit matrix

        A = Xc[0:50]
        B = Xc[50:100]
        C = Xc[100:150]

        #Concatenated Matrices
        cA = np.c_[A, np.ones(dataset_per_class)]   #Adding column of 1's
        cA = np.c_[cA, np.zeros(dataset_per_class)] #for unit matrix and
        cA = np.c_[cA, np.zeros(dataset_per_class)] #column of 0's for others

        cB = np.c_[B, np.zeros(dataset_per_class)]
        cB = np.c_[cB, np.ones(dataset_per_class)]
        cB = np.c_[cB, np.zeros(dataset_per_class)]

        cC = np.c_[C, np.zeros(dataset_per_class)]
        cC = np.c_[cC, np.zeros(dataset_per_class)]
        cC = np.c_[cC, np.ones(dataset_per_class)]

        samples_per_class = dataset_per_class * percent /100

        matrownosA = np.random.choice(dataset_per_class, samples_per_class, replace=False)  #random number gen
        matrownosA = matrownosA.tolist()
        rA = cA[matrownosA]
        rTestA = np.delete(cA,matrownosA,0)

        matrownosB = np.random.choice(dataset_per_class, samples_per_class, replace=False)
        matrownosB = matrownosB.tolist()
        rB = cB[matrownosB]
        rTestB = np.delete(cB,matrownosB,0)

        matrownosC = np.random.choice(dataset_per_class, samples_per_class, replace=False)
        matrownosC = matrownosC.tolist()
        rC = cC[matrownosC]
        rTestC = np.delete(cC,matrownosC,0)

        randomMatrix = np.append(rA, rB, axis=0)
        randomMatrix = np.append(randomMatrix, rC, axis=0)

        randomTestMatrix = np.append(rTestA, rTestB, axis=0)
        randomTestMatrix = np.append(randomTestMatrix, rTestC, axis=0)

        X = randomMatrix[:,0:5]     #X training Matrix
        Xtest = randomTestMatrix[:,0:5]

        Y = randomMatrix[:,5:]
        Ytest = randomTestMatrix[:,5:]

        Xt = X.transpose()

        lmbd = 10
        posD = (np.dot(Xt,X))
        posDef = sp.Matrix(posD)
        idenScalar = (lmbd*np.identity(5))
        posDef = posDef + idenScalar
        posInv = posDef.inv()
        thetaHat = posInv * Xt * Y
        thetaHatnp =  np.array(thetaHat.tolist()).astype(np.float64)    #convert to Numpy from Sympy

        Yhat = np.dot(X,thetaHatnp)
        rsltTrain = np.argmax(Yhat, axis= 1)
        countError = 0
        YhatRow = Yhat.shape[0]
        for i in range(YhatRow):
            if ((i<samples_per_class and rsltTrain[i]!=0)
                or  ( i>=samples_per_class and i<(samples_per_class*2) and rsltTrain[i]!=1)
                or ( i>=(samples_per_class*2) and i<(samples_per_class*3) and rsltTrain[i]!=2)):
                countError += 1

        misclassErrorTrain = countError/(YhatRow + 0.0)
        print "Nos of Training errors for "+str(percent) + "% data:" + str(countError)
        print "Misclassification error for "+str(percent)+ "% Training data:" + str(misclassErrorTrain)

        YhatTest = np.dot(Xtest,thetaHatnp)
        rsltTest = np.argmax(YhatTest, axis= 1)
        countErrorTest = 0
        YhatTestRow = no_of_datasets - YhatRow
        TestDataPerClass = dataset_per_class - samples_per_class
        for i in range(YhatTestRow):
            if ((i<TestDataPerClass and rsltTest[i]!=0) or ( i>=TestDataPerClass and i<(TestDataPerClass*2) and rsltTest[i]!=1) or ( i>=(TestDataPerClass*2) and i<(TestDataPerClass*3) and rsltTest[i]!=2)):
                countErrorTest += 1

        misclassErrorTest = countErrorTest/(YhatTestRow + 0.0)
        print "Nos of Test errors for "+str(percent)+"% data:" + str(countErrorTest)
        print "Misclassification error for "+str(percent)+"% Test data:" + str(misclassErrorTest)

Scrape()
MatrixDef()