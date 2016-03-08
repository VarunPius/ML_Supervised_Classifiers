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

def LinearLeastSq():
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

        SVM(X, Y, thetaHatnp, Xtest, Ytest)

def SVM(X, Y, thetaHatnp, Xtest, Ytest):
    global matpart12, matpart12
    global matpart13, matpart13
    global matpart23, matpart23
    Xt = X.transpose()

    rowcount = len(X[:,1])
    colcount = len(X[1,:])
    print "nos of rows: " + str(rowcount)

    transprod = np.empty([rowcount,3], dtype=float)
    Z = np.empty([rowcount,1], dtype=float)
    prodMatrix = np.empty([rowcount,1], dtype=float)
    epsilon = 0.001
    Ynew = Y
    Ynew[Ynew == 0] = -1.0
    #print "replaced Y: " +str(Ynew)
    for i in range(0,rowcount):
        transprod[i,:] = np.dot(X[i,:],thetaHatnp)
        prodMatrix[i,:] = np.dot(Ynew[i,:],transprod[i,:].T)
        Z[i,:] = 3.0 - prodMatrix[i,:]
        Z[i,:] = max(epsilon,Z[i,:])

    D = np.identity(5)
    D[4,4] = 0
    C = 10

    setval = rowcount/3
    X1 = X[0:setval]
    X2 = X[setval:2*setval]
    X3 = X[2*setval:3*setval]

    Xnew = np.empty([rowcount,5], dtype=float)
    for i in range(0,rowcount):
        Xnew[i,:] = X[i,:]/Z[i,0]
    Xnew1 = Xnew[0:setval]
    Xnew2 = Xnew[setval:2*setval]
    Xnew3 = Xnew[2*setval:3*setval]

    X12 = np.append(X1, X2, axis=0)
    X13 = np.append(X1, X3, axis=0)
    X23 = np.append(X2, X3, axis=0)

    Xn12 = np.append(Xnew1, Xnew2, axis=0)
    Xn13 = np.append(Xnew1, Xnew3, axis=0)
    Xn23 = np.append(Xnew2, Xnew3, axis=0)

    #for i in range(0,colcount):
    matpart12 = np.dot(X12.T,Xn12)
    matpart13 = np.dot(X13.T,Xn13)
    matpart23 = np.dot(X23.T,Xn23)
    #print "matpart12: " + str(matpart12)

    mat12 = D + (C*matpart12)
    mat12 = sp.Matrix(mat12)
    #print mat12
    mat12inv = mat12.inv()

    mat13 = D + (C*matpart13)
    mat13 = sp.Matrix(mat13)
    #print mat13
    mat13inv = mat13.inv()

    mat23 = D + (C*matpart23)
    mat23 = sp.Matrix(mat23)
    #print mat23
    mat23inv = mat23.inv()

    mat12invnp = np.array(mat12inv.tolist()).astype(np.float64)
    mat13invnp = np.array(mat13inv.tolist()).astype(np.float64)
    mat23invnp = np.array(mat23inv.tolist()).astype(np.float64)

    Y12 = np.empty([2*setval,1], dtype=float)
    Y13 = np.empty([2*setval,1], dtype=float)
    Y23 = np.empty([2*setval,1], dtype=float)

    for i in range(0, setval):
        Y12[i,0] = 1
        Y23[i,0] = 1
        Y13[i,0] = 1

    for i in range(setval,2*setval):
        Y12[i,0] = -1
        Y13[i,0] = -1
        Y23[i,0] = -1

    X2new = np.empty([rowcount,5], dtype=float)
    for i in range(0,rowcount):
        X2new[i,:] = ((1+Z[i,0])/(2*Z[i,0]))*X[i,:] #X[i,:]/Z[i,0]

    X2new1 = X2new[0:setval]
    X2new2 = X2new[setval:2*setval]
    X2new3 = X2new[2*setval:3*setval]

    X2n12 = np.append(X2new1, X2new2, axis=0)
    X2n13 = np.append(X2new1, X2new3, axis=0)
    X2n23 = np.append(X2new2, X2new3, axis=0)

    mat2part12 = C*np.dot(X2n12.T,Y12)
    mat2part13 = C*np.dot(X2n13.T,Y13)
    mat2part23 = C*np.dot(X2n23.T,Y23)

    thetaHatNew12 = np.dot(mat12invnp, mat2part12)
    thetaHatNew13 = np.dot(mat13invnp, mat2part13)
    thetaHatNew23 = np.dot(mat23invnp, mat2part23)

    rowcountTest = len(Xtest[:,1])
    setvalTest = rowcountTest/3

    Xtest1 = Xtest[0:setvalTest]
    Xtest2 = Xtest[setvalTest:2*setvalTest]
    Xtest3 = Xtest[2*setvalTest:3*setvalTest]

    Xtest12 = np.append(Xtest1, Xtest2, axis=0)
    Xtest13 = np.append(Xtest1, Xtest3, axis=0)
    Xtest23 = np.append(Xtest2, Xtest3, axis=0)

    YTest12 = np.empty([2*setvalTest,1], dtype=float)
    YTest13 = np.empty([2*setvalTest,1], dtype=float)
    YTest23 = np.empty([2*setvalTest,1], dtype=float)

    for i in range(0, setvalTest):
        YTest12[i,0] = 1
        YTest13[i,0] = 1
        YTest23[i,0] = 1

    for i in range(setvalTest,2*setvalTest):
        YTest12[i,0] = -1
        YTest13[i,0] = -1
        YTest23[i,0] = -1

    thetaProd12 = np.dot(Xtest12, thetaHatNew12)
    print "Total test dataset for class 1 and 2: " + str(2*setvalTest)
    #finalProd12 = np.dot(YTest12.T, thetaProd12)
    #op12 = (2*setvalTest) - finalProd12/2
    finalProd12 = 0
    for i in range(0,2*setvalTest):
        finalProd12 = finalProd12 + max(0,(1 - (YTest12[i,0] * thetaProd12[i,0])))
    print "Misclassification for 1 & 2: " + str(finalProd12)

    thetaProd13 = np.dot(Xtest13, thetaHatNew13)
    print "Total test dataset for class 1 and 3: " + str(2*setvalTest)
    #finalProd13 = np.dot(YTest13.T, thetaProd13)
    #op13 = (2*setvalTest) - finalProd13/2
    finalProd13 = 0
    for i in range(0,2*setvalTest):
        finalProd13 = finalProd13 + max(0,(1 - (YTest13[i,0] * thetaProd13[i,0])))
    print "Misclassification for 1 & 3: " + str(finalProd13)

    thetaProd23 = np.dot(Xtest23, thetaHatNew23)
    print "Total test dataset for class 2 and 3: " + str(2*setvalTest)
    finalProd23 = 0
    for i in range(0,2*setvalTest):
        finalProd23 = finalProd23 + max(0,(1 - (YTest23[i,0] * thetaProd23[i,0])))
    print "Misclassification for 2 & 3: " + str(finalProd23)

#Scrape()
LinearLeastSq()