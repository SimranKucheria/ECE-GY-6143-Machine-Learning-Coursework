from sklearn import svm
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


#Splitting dataset
data = sio.loadmat('dataset.mat')
X_train, X_test, y_train, y_test = train_test_split(
    data.get("X"), data.get("Y"), test_size=0.5, random_state=0)



#Linear kernel
cVals = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
errorVals = []
alphaVals = []
nsvVals = []
for c in cVals:
    model = svm.SVC(kernel="linear",C=c)
    model.fit(X_train,y_train)

    preds = model.predict(X_test)
    err = 0
    for i in range (0,len(preds)):
        if preds[i]!=y_test[i]:
            err+=1
    errorVals.append(err/len(y_test))
    alphaVals.append(np.sum(np.abs(model.dual_coef_)))
    nsvVals.append(len(model.support_vectors_))

#Plots
plt.clf()
plt.plot([1,2,3,4,5,6,7,8,9,10],errorVals)
plt.xticks([1,2,3,4,5,6,7,8,9,10],cVals)
plt.xlabel('CValues')
plt.ylabel('Error')
plt.savefig("CValuesErrorLinearKernel.png")

plt.clf()
plt.plot([1,2,3,4,5,6,7,8,9,10],alphaVals)
plt.xticks([1,2,3,4,5,6,7,8,9,10],cVals)
plt.xlabel('CValues')
plt.ylabel('Alpha')
plt.savefig("CValuesAlphaLinearKernel.png")

plt.clf()
plt.plot([1,2,3,4,5,6,7,8,9,10],nsvVals)
plt.xticks([1,2,3,4,5,6,7,8,9,10],cVals)
plt.xlabel('CValues')
plt.ylabel('NSV')
plt.savefig("CValuesNSVLinearKernel.png")


# Polynomial Kernel
#Varying C for different Degree Values

DegreeVals = [1,2,3,4,5]
for d in DegreeVals:
    errorVals = []
    alphaVals = []
    nsvVals = []
    for c in cVals:
        model = svm.SVC(kernel="poly",C=c, degree=d)
        model.fit(X_train,y_train)

        preds = model.predict(X_test)
        err = 0
        for i in range (0,len(preds)):
            if preds[i]!=y_test[i]:
                err+=1
        errorVals.append(err/len(y_test))
        alphaVals.append(np.sum(np.abs(model.dual_coef_)))
        nsvVals.append(len(model.support_vectors_))

    #Plots for varying CValues for fixed degrees
    plt.clf()
    plt.plot([1,2,3,4,5,6,7,8,9,10],errorVals)
    plt.xticks([1,2,3,4,5,6,7,8,9,10],cVals)
    plt.xlabel('CValues')
    plt.ylabel('Error')
    plt.savefig("CValuesErrorPolyKernelDegree" + str(d)+".png")

    plt.clf()
    plt.plot([1,2,3,4,5,6,7,8,9,10],alphaVals)
    plt.xticks([1,2,3,4,5,6,7,8,9,10],cVals)
    plt.xlabel('CValues')
    plt.ylabel('Alpha')
    plt.savefig("CValuesAlphaPolyKernelDegree" + str(d) +".png")

    plt.clf()
    plt.plot([1,2,3,4,5,6,7,8,9,10],nsvVals)
    plt.xticks([1,2,3,4,5,6,7,8,9,10],cVals)
    plt.xlabel('CValues')
    plt.ylabel('NSV')
    plt.savefig("CValuesNSVPolyKernelDegree" + str(d)+".png")



#Varying Degree for C Values
DegreeVals = [1,2,3,4,5]

for c in [0.01, 0.1,1]:
    errorVals = []
    alphaVals = []
    nsvVals = []
    for d in DegreeVals:
        model = svm.SVC(kernel="poly",C=c, degree=d)
        model.fit(X_train,y_train)

        preds = model.predict(X_test)
        err = 0
        for i in range (0,len(preds)):
            if preds[i]!=y_test[i]:
                err+=1
        errorVals.append(err/len(y_test))
        alphaVals.append(np.sum(np.abs(model.dual_coef_)))
        nsvVals.append(len(model.support_vectors_))

    #Plots for different degrees for fixed C
    plt.clf()
    plt.plot([1,2,3,4,5],errorVals)
    plt.xticks([1,2,3,4,5],DegreeVals)
    plt.xlabel('DegreeValues')
    plt.ylabel('Error')
    plt.savefig("DegreeValuesErrorPolyKernel"+str(c)+".png")

    plt.clf()
    plt.plot([1,2,3,4,5],alphaVals)
    plt.xticks([1,2,3,4,5],DegreeVals)
    plt.xlabel('DegreeValues')
    plt.ylabel('Alpha')
    plt.savefig("DegreeValuesAlphaPolyKernel"+str(c)+".png")

    plt.clf()
    plt.plot([1,2,3,4,5],nsvVals)
    plt.xticks([1,2,3,4,5],DegreeVals)
    plt.xlabel('DegreeValues')
    plt.ylabel('NSV')
    plt.savefig("DegreeValuesNSVPolyKernel"+str(c)+".png")

#RBF Kernel
# Varying C for different Sigma Values

SigmaVals = [0.001,0.01,0.1,1,10]
for s in SigmaVals:
    errorVals = []
    alphaVals = []
    nsvVals = []
    for c in cVals:
        model = svm.SVC(kernel="rbf",C=c,gamma=s)
        model.fit(X_train,y_train)

        preds = model.predict(X_test)
        err = 0
        for i in range (0,len(preds)):
            if preds[i]!=y_test[i]:
                err+=1
        errorVals.append(err/len(y_test))
        alphaVals.append(np.sum(np.abs(model.dual_coef_)))
        nsvVals.append(len(model.support_vectors_))

    #Plots for differenct C values with fixed sigma vals
    plt.clf()
    plt.plot([1,2,3,4,5,6,7,8,9,10],errorVals)
    plt.xticks([1,2,3,4,5,6,7,8,9,10],cVals)
    plt.xlabel('CValues')
    plt.ylabel('Error')
    plt.savefig("CValuesErrorRBFKernelSigma" + str(s)+".png")

    plt.clf()
    plt.plot([1,2,3,4,5,6,7,8,9,10],alphaVals)
    plt.xticks([1,2,3,4,5,6,7,8,9,10],cVals)
    plt.xlabel('CValues')
    plt.ylabel('Alpha')
    plt.savefig("CValuesAlphaRBFKernelSigma" + str(s) +".png")

    plt.clf()
    plt.plot([1,2,3,4,5,6,7,8,9,10],nsvVals)
    plt.xticks([1,2,3,4,5,6,7,8,9,10],cVals)
    plt.xlabel('CValues')
    plt.ylabel('NSV')
    plt.savefig("CValuesNSVRBFKernelSigma" + str(s)+".png")


#Varying Sigma values for fixed C
for c in [0.1,1,100]:
    errorVals = []
    alphaVals = []
    nsvVals = []
    for s in SigmaVals:
        model = svm.SVC(kernel="rbf",C=c, gamma=s)
        model.fit(X_train,y_train)

        preds = model.predict(X_test)
        err = 0
        for i in range (0,len(preds)):
            if preds[i]!=y_test[i]:
                err+=1
        errorVals.append(err/len(y_test))
        alphaVals.append(np.sum(np.abs(model.dual_coef_)))
        nsvVals.append(len(model.support_vectors_))

    #Plots
    plt.clf()
    plt.plot([1,2,3,4,5],errorVals)
    plt.xticks([1,2,3,4,5],SigmaVals)
    plt.xlabel('SigmaValues')
    plt.ylabel('Error')
    plt.savefig("SigmaValuesErrorRBFKernel"+str(c)+".png")

    plt.clf()
    plt.plot([1,2,3,4,5],alphaVals)
    plt.xticks([1,2,3,4,5],SigmaVals)
    plt.xlabel('SigmaValues')
    plt.ylabel('Alpha')
    plt.savefig("SigmaValuesAlphaRBFKernel"+str(c)+".png")

    plt.clf()
    plt.plot([1,2,3,4,5],nsvVals)
    plt.xticks([1,2,3,4,5],SigmaVals)
    plt.xlabel('SigmaValues')
    plt.ylabel('NSV')
    plt.savefig("SigmaValuesNSVRBFKernel"+str(c)+".png")



