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
    print soup

    SepalLen = 0
    SepalWidth = 0
    PetalLen = 0
    PetalWidth = 0
    Species = ""
    table = soup.find("table", { "class" : "wikitable sortable" })
    print table

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
            #print write_unic
            fl.write(write_unic)

    fl.close()


def MatrixDef():
    no_of_datasets = 150

    for percent in (10, 30, 50):
        #print "I is: " + str(i)
        df = pd.read_csv("Data.csv")
        #X = np.loadtxt(open("Data.csv","rb"),delimiter=",",skiprows=1)
        Xi = df.as_matrix()
        #Xz = np.zeros((150, 8))
        #Xz = Xi[:,:-1]
        #X = Xi.resize()
        #print "Xz is : "
        #print Xz
        #print "Row count is : " + str(Xz.shape)

        Xc = Xi[:,:-1]
        print "Xc Row count is : " + str(Xc.shape)
        print Xc

        dataset_per_class = no_of_datasets/3
        A = Xc[0:50]
        print A
        print "Row count of A is : " + str(A.shape)
        B = Xc[50:100]
        print B
        print "Row count of B is : " + str(B.shape)
        C = Xc[100:150]
        print C
        print "Row count of C is : " + str(C.shape)

        cA = np.c_[A, np.ones(dataset_per_class)]
        cA = np.c_[cA, np.ones(dataset_per_class)]
        cA = np.c_[cA, np.zeros(dataset_per_class)]
        cA = np.c_[cA, np.zeros(dataset_per_class)]
        print cA
        cB = np.c_[B, np.ones(dataset_per_class)]
        cB = np.c_[cB, np.zeros(dataset_per_class)]
        cB = np.c_[cB, np.ones(dataset_per_class)]
        cB = np.c_[cB, np.zeros(dataset_per_class)]
        print cB
        cC = np.c_[C, np.ones(dataset_per_class)]
        cC = np.c_[cC, np.zeros(dataset_per_class)]
        cC = np.c_[cC, np.zeros(dataset_per_class)]
        cC = np.c_[cC, np.ones(dataset_per_class)]
        print cC
        print "Dimension of concatentated matriices : " + str(cA.shape) + " | " + str(cB.shape) + " | " + str(cC.shape)
        #z = np.ones((50,1))
        #print z
        #cA = np.append(A, z, axis=1)
        #cB = np.append(B, z, axis=1)
        #print cB
        #cC = np.append(C, z, axis=1)
        #print cC

        #iA = np.random.randint(0,cA.shape[0],5) # for non-unique you may select this
        samples_per_class = dataset_per_class * percent /100
        matrownosA = np.random.choice(dataset_per_class, samples_per_class, replace=False) #value generated will be of the form [38 35 43 48 47] which are nothing but row numbers
        print "Lines for A:" + str(matrownosA)
        #print tuple(matrownosA)
        matrownosA = matrownosA.tolist()
        print matrownosA
        rA = cA[matrownosA]
        print "Random Matrix A: " + str(rA)
        print "Dimension of Matrix A: " + str(rA.shape)
        rTestA = np.delete(cA,matrownosA,0)
        print "rTestA is: " + str(rTestA)
        print "Dimension of RandomTest A: " + str(rTestA.shape)
        print " "
        matrownosB = np.random.choice(dataset_per_class, samples_per_class, replace=False)
        print "Lines for B:" + str(matrownosB)
        matrownosB = matrownosB.tolist()
        rB = cB[matrownosB]
        print "Random Matrix B: " + str(rB)
        print "Dimension of Matrix B: " + str(rB.shape)
        rTestB = np.delete(cB,matrownosB,0)
        print "rTestB is: " + str(rTestB)
        print "Dimension of RandomTest B: " + str(rTestB.shape)
        print " "
        matrownosC = np.random.choice(dataset_per_class, samples_per_class, replace=False)
        print "Lines for C:" + str(matrownosC)
        matrownosC = matrownosC.tolist()
        rC = cC[matrownosC]
        print "Random Matrix C: " + str(rC)
        print "Dimension of Matrix C: " + str(rC.shape)
        rTestC = np.delete(cC,matrownosC,0)
        print "rTestC is: " + str(rTestC)
        print "Dimension of RandomTest C: " + str(rTestC.shape)

        randomMatrix = np.append(rA, rB, axis=0)
        randomMatrix = np.append(randomMatrix, rC, axis=0)
        print randomMatrix

        randomTestMatrix = np.append(rTestA, rTestB, axis=0)
        randomTestMatrix = np.append(randomTestMatrix, rTestC, axis=0)
        print randomTestMatrix
        print "Dimension of Random Test Matrix is: " + str(randomTestMatrix.shape)

        X = randomMatrix[:,0:5]
        print X
        Xtest = randomTestMatrix[:,0:5]
        print "Dimension of X Test : " + str(Xtest.shape)
        Y = randomMatrix[:,5:]
        print Y
        Ytest = randomTestMatrix[:,5:]
        print "Dimension of Y Test : " + str(Ytest.shape)

        Xt = X.transpose()
        print Xt

        lmbd = 1
        posD = (np.dot(Xt,X))
        print posD
        posDef = sp.Matrix(posD)
        idenScalar = (lmbd*np.identity(5))
        print posDef
        print idenScalar
        print "Size of matrix is: " + str(posDef.shape)
        posDef = posDef + idenScalar
        print posDef
        posInv = posDef.inv()  #np.linalg.inv(posDef)
        print posInv
        print "Size of matrix is: " + str(posInv.shape)

        thetaHat = posInv * Xt * Y
        #thetaHat = thetaHat
        print thetaHat
        print "Dimension of Theta: " + str(thetaHat.shape)

        thetaHatnp =  np.array(thetaHat.tolist()).astype(np.float64)
        print "theta matrix is: "+ str(thetaHat)

        Yhat = np.dot(X,thetaHatnp)
        print Yhat
        print "Dimension of Yhat: " + str(Yhat.shape)

        rsltTrain = np.argmax(Yhat, axis= 1)
        print rsltTrain

        countError = 0
        YhatRow = Yhat.shape[0]
        print "number of rows in Yhat: " + str(Yhat.shape[0])
        for i in range(samples_per_class*3):
            #countError = 0
            if ((i<samples_per_class and rsltTrain[i]!=0) or ( i>=samples_per_class and i<(samples_per_class*2) and rsltTrain[i]!=1) or ( i>=(samples_per_class*2) and i<(samples_per_class*3) and rsltTrain[i]!=2)):
                countError += 1

        print "Nos of errors: " + str(countError)

        misclassErrorTrain = countError/15.0
        print misclassErrorTrain

        YhatTest = np.dot(Xtest,thetaHatnp)
        print YhatTest

        rsltTest = np.argmax(YhatTest, axis= 1)
        print rsltTest
        countErrorTest = 0
        YhatTestRow = no_of_datasets - YhatRow
        TestDataPerClass = dataset_per_class - samples_per_class
        for i in range(YhatTestRow):
            #countError = 0
            if ((i<TestDataPerClass and rsltTest[i]!=0) or ( i>=TestDataPerClass and i<(TestDataPerClass*2) and rsltTest[i]!=1) or ( i>=(TestDataPerClass*2) and i<(TestDataPerClass*3) and rsltTest[i]!=2)):
                countErrorTest += 1

        print "Nos of Test errors: " + str(countErrorTest)


Scrape()
MatrixDef()
