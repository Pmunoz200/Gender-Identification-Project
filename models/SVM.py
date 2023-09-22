import numpy as np
import pandas as pd
import os
from tabulate import tabulate
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("MLandPattern"))
import MLandPattern as ML

def load(pathname, vizualization=0):
    df = pd.read_csv(pathname, header=None)
    if vizualization:
        print(df.head())
    attribute = np.array(df.iloc[:, 0: len(df.columns) - 1])
    attribute = attribute.T
    print(attribute)
    label = np.array(df.iloc[:, -1])

    return attribute, label


def split_db(D, L, fraction, seed=0):
    nTrain = int(D.shape[1] * fraction)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]

    return (DTR, LTR), (DTE, LTE)


if __name__ == "__main__":
    path = os.path.abspath("data/Train.txt")
    [full_train_att, full_train_label] = load(path)

    priorProb = ML.vcol(np.ones(2) * 0.5)

    pi = [0.5, 0.3, 0.7]
    Cfn = 1
    Cfp = 1
    initial_C = np.logspace(-6, 6, 15)
    initial_K = 1
    k = 5
    headers = [
    "Model",
    "π=0.5 DCF/minDCF",
    "π=0.3 DCF/minDCF",
    "π=0.7 DCF/minDCF",
    ]
    searchC=float(input("Enter a C value you want to search for: "))

    # ###--------------K-fold----------------------###
    section_size = int(full_train_att.shape[1] / k)
    low = 0
    all_values = np.c_[full_train_att.T, full_train_label]
    all_values = np.random.permutation(all_values)
    attributes = all_values[:, 0:12].T
    labels = all_values[:, -1].astype("int32")
    high = section_size
    if high > attributes.shape[1]:
        high = attributes.shape
    test_att = attributes[:, low:high]
    test_labels = labels[low:high]
    train_att = attributes[:, :low]
    train_label = labels[:low]
    train_att = np.hstack((train_att, attributes[:, high:]))
    train_label = np.hstack((train_label, labels[high:]))

    minDCFvalues5F = []
    minDCFvalues3F = []
    minDCFvalues7F = []

    for p in pi:
        for C in initial_C:
            contrain = C

            k = initial_K * np.power(10, 0)
            [SPost, Predictions, accuracy] = ML.svm(
                train_att, train_label, test_att, test_labels, contrain, K=k
            )
            confusion_matrix = ML.ConfMat(Predictions, test_labels)
            DCF, DCFnorm = ML.Bayes_risk(confusion_matrix, p, Cfn, Cfp)
            (minDCF, FPRlist, FNRlist, _) = ML.minCostBayes(
                SPost, test_labels, p, Cfn, Cfp
            )
            if p == 0.5:
                minDCFvalues5F.append(minDCF)
            if p == 0.3:
                minDCFvalues3F.append(minDCF)
            if p == 0.7:
                minDCFvalues7F.append(minDCF)
                
    tableFull=[]
    cont=0
    for p in pi:
        tableFull.append([f"SVM pi= {p}"])
        for x in pi:
            contrain = searchC
            k = initial_K * np.power(10, 0)
            [SPost, Predictions, accuracy] = ML.svm(
                train_att, train_label, test_att, test_labels, contrain, K=k,pit=p
            )
            confusion_matrix = ML.ConfMat(Predictions, test_labels)
            DCF, DCFnorm = ML.Bayes_risk(confusion_matrix, x, Cfn, Cfp)
            (minDCF, FPRlist, FNRlist, _) = ML.minCostBayes(
                SPost, test_labels, x, Cfn, Cfp
            )
            tableFull[cont].append([DCFnorm,minDCF])
        cont+=1
        
    
    ### --------z-score----------###
    standard_deviation = np.std(full_train_att)
    z_data = ML.center_data(full_train_att) / standard_deviation
    section_size = int(full_train_att.shape[1] / k)
    low = 0
    all_values = np.c_[z_data.T, full_train_label]
    all_values = np.random.permutation(all_values)
    attributes = all_values[:, 0:12].T
    labels = all_values[:, -1].astype("int32")
    high = section_size
    if high > attributes.shape[1]:
        high = attributes.shape
    test_att = attributes[:, low:high]
    test_labels = labels[low:high]
    train_att = attributes[:, :low]
    train_label = labels[:low]
    train_att = np.hstack((train_att, attributes[:, high:]))
    train_label = np.hstack((train_label, labels[high:]))

    minDCFvalues5Z = []
    minDCFvalues3Z = []
    minDCFvalues7Z = []

    for p in pi:
        for C in initial_C:
            contrain = C

            k = initial_K * np.power(10, 0)
            [SPost, Predictions, accuracy] = ML.svm(
                train_att, train_label, test_att, test_labels, contrain, K=k,
            )
            confusion_matrix = ML.ConfMat(Predictions, test_labels)
            DCF, DCFnorm = ML.Bayes_risk(confusion_matrix, p, Cfn, Cfp)
            (minDCF, FPRlist, FNRlist, _) = ML.minCostBayes(
                SPost, test_labels, p, Cfn, Cfp
            )
            if p == 0.5:
                minDCFvalues5Z.append(minDCF)
            if p == 0.3:
                minDCFvalues3Z.append(minDCF)
            if p == 0.7:
                minDCFvalues7Z.append(minDCF)
    tableFullZ=[]
    cont=0
    for p in pi:
        tableFullZ.append([f"SVM pi= {p}"])
        for x in pi:
            contrain = searchC
            k = initial_K * np.power(10, 0)
            [SPost, Predictions, accuracy] = ML.svm(
                train_att, train_label, test_att, test_labels, contrain, K=k,pit=p
            )
            confusion_matrix = ML.ConfMat(Predictions, test_labels)
            DCF, DCFnorm = ML.Bayes_risk(confusion_matrix, x, Cfn, Cfp)
            (minDCF, FPRlist, FNRlist, _) = ML.minCostBayes(
                SPost, test_labels, x, Cfn, Cfp
            )
            tableFullZ[cont].append([DCFnorm,minDCF])
        cont+=1
        
    ## -----------------Polynomial---------###
    section_size = int(full_train_att.shape[1] / k)
    low = 0
    all_values = np.c_[full_train_att.T, full_train_label]
    all_values = np.random.permutation(all_values)
    attributes = all_values[:, 0:12].T
    labels = all_values[:, -1].astype("int32")
    high = section_size
    if high > attributes.shape[1]:
        high = attributes.shape
    test_att = attributes[:, low:high]
    test_labels = labels[low:high]
    train_att = attributes[:, :low]
    train_label = labels[:low]
    train_att = np.hstack((train_att, attributes[:, high:]))
    train_label = np.hstack((train_label, labels[high:]))

    initial_d = 2
    initial_const = 0
    initial_K = 1

    minDCFvalues5Q=[]
    minDCFvalues3Q=[]
    minDCFvalues7Q=[]
    for p in pi:
        for C in initial_C:
            contrain = C
            d = initial_d
            const = initial_const
            k = initial_K * np.power(10, 0)
            [SPost, Predictions, accuracy] = ML.svm(
                train_att,
                train_label,
                test_att,
                test_labels,
                contrain,
                dim=d,
                c=const,
                eps=k**2,
                model="polynomial",
            )
            confusion_matrix = ML.ConfMat(Predictions, test_labels)
            DCF, DCFnorm = ML.Bayes_risk(confusion_matrix, p, Cfn, Cfp)
            (minDCF, FPRlist, FNRlist,_) = ML.minCostBayes(
                SPost, test_labels, p, Cfn, Cfp
            )
            if p== 0.5:
                minDCFvalues5Q.append(minDCF)
            if p== 0.3:
                minDCFvalues3Q.append(minDCF)
            if p== 0.7:
                minDCFvalues7Q.append(minDCF)
    tablePol=[]
    gamma=10**(-3)
    cont=0
    for p in pi:
        tablePol.append([f"PSVM pi= {p}"])
        for x in pi:
            contrain = searchC
            k = initial_K * np.power(10, 0)
            [SPost, Predictions, accuracy] = ML.svm(
                train_att,
                train_label,
                test_att,
                test_labels,
                contrain,
                dim=initial_d,
                c=initial_const,
                eps=k**2,
                model="polynomial",
                pit=p
            )
            confusion_matrix = ML.ConfMat(Predictions, test_labels)
            DCF, DCFnorm = ML.Bayes_risk(confusion_matrix, x, Cfn, Cfp)
            (minDCF, FPRlist, FNRlist, _) = ML.minCostBayes(
                SPost, test_labels, x, Cfn, Cfp
            )
            tablePol[cont].append([DCFnorm,minDCF])
        cont+=1

    ###----------------Pol-z----------------###
    standard_deviation = np.std(full_train_att)
    z_data = ML.center_data(full_train_att) / standard_deviation
    k=5
    section_size = int(full_train_att.shape[1] / k)
    low = 0
    all_values = np.c_[z_data.T, full_train_label]
    all_values = np.random.permutation(all_values)
    attributes = all_values[:, 0:12].T
    labels = all_values[:, -1].astype("int32")
    high = section_size
    if high > attributes.shape[1]:
        high = attributes.shape
    test_att = attributes[:, low:high]
    test_labels = labels[low:high]
    train_att = attributes[:, :low]
    train_label = labels[:low]
    train_att = np.hstack((train_att, attributes[:, high:]))
    train_label = np.hstack((train_label, labels[high:]))
    minDCFvalues5Qz=[]
    minDCFvalues3Qz=[]
    minDCFvalues7Qz=[]
    for p in pi:
        for C in initial_C:
            contrain = C
            d = initial_d
            const = initial_const
            k = initial_K * np.power(10, 0)
            [SPost, Predictions, accuracy] = ML.svm(
                train_att,
                train_label,
                test_att,
                test_labels,
                contrain,
                dim=d,
                c=const,
                eps=k**2,
                model="polynomial",
            )
            confusion_matrix = ML.ConfMat(Predictions, test_labels)
            DCF, DCFnorm = ML.Bayes_risk(confusion_matrix, pi, Cfn, Cfp)
            (minDCF, FPRlist, FNRlist) = ML.minCostBayes(
                SPost, test_labels, pi, Cfn, Cfp
            )
            if p== 0.5:
                minDCFvalues5Qz.append(minDCF)
            if p== 0.3:
                minDCFvalues3Qz.append(minDCF)
            if p== 0.7:
                minDCFvalues7Qz.append(minDCF)

    ## -----------------------Radial--------------------###
    section_size = int(full_train_att.shape[1] / k)
    low = 0
    all_values = np.c_[full_train_att.T, full_train_label]
    all_values = np.random.permutation(all_values)
    attributes = all_values[:, 0:12].T
    labels = all_values[:, -1].astype("int32")
    high = section_size
    if high > attributes.shape[1]:
        high = attributes.shape
    test_att = attributes[:, low:high]
    test_labels = labels[low:high]
    train_att = attributes[:, :low]
    train_label = labels[:low]
    train_att = np.hstack((train_att, attributes[:, high:]))
    train_label = np.hstack((train_label, labels[high:]))

    gamma = [10**-1, 10**-2, 10**-3]
    pi = 0.5
    minDCFvalues1R = []
    minDCFvalues2R = []
    minDCFvalues3R = []
    for g in gamma:
        for C in initial_C:
            contrain = C
            gamma = g
            [SPost, Predictions, accuracy] = ML.svm(
                train_att,
                train_label,
                test_att,
                test_labels,
                contrain,
                gamma=gamma,
                model="radial",
            )
            confusion_matrix = ML.ConfMat(Predictions, test_labels)
            DCF, DCFnorm = ML.Bayes_risk(confusion_matrix, pi, Cfn, Cfp)
            (minDCF, FPRlist, FNRlist, _) = ML.minCostBayes(
                SPost, test_labels, pi, Cfn, Cfp
            )
            if g == 10**-1:
                minDCFvalues1R.append(minDCF)
            if g == 10**-2:
                minDCFvalues2R.append(minDCF)
            if g == 10**-3:
                minDCFvalues3R.append(minDCF)
    tableRad=[]
    gamma=10**(-3)
    cont=0
    for p in pi:
        tableRad.append([f"RBSVM pi= {p}"])
        for x in pi:
            contrain = searchC
            k = initial_K * np.power(10, 0)
            [SPost, Predictions, accuracy] = ML.svm(
                train_att,
                train_label,
                test_att,
                test_labels,
                contrain,
                gamma=gamma,
                model="radial",
                pit=p
            )
            confusion_matrix = ML.ConfMat(Predictions, test_labels)
            DCF, DCFnorm = ML.Bayes_risk(confusion_matrix, x, Cfn, Cfp)
            (minDCF, FPRlist, FNRlist, _) = ML.minCostBayes(
                SPost, test_labels, x, Cfn, Cfp
            )
            tableRad[cont].append([DCFnorm,minDCF])
        cont+=1
    print("SVM with k-fold={k}")
    plt.figure(figsize=(10, 6))
    plt.semilogx(initial_C, minDCFvalues5F, label=f'π= 0.5')
    plt.semilogx(initial_C, minDCFvalues3F, label=f'π= 0.3')
    plt.semilogx(initial_C, minDCFvalues7F, label=f'π= 0.7')
    plt.xlabel('C')
    plt.ylabel('minDCF')
    plt.title('Full SVM regression')
    plt.legend()
    plt.show()

    print("SVM with Z-norm")
    plt.figure(figsize=(10, 6))
    plt.semilogx(initial_C, minDCFvalues5Z, label=f'π= 0.5')
    plt.semilogx(initial_C, minDCFvalues3Z, label=f'π= 0.3')
    plt.semilogx(initial_C, minDCFvalues7Z, label=f'π= 0.7')
    plt.xlabel('C')
    plt.ylabel('minDCF')
    plt.title('Z-Norm SVM regression')
    plt.legend()
    plt.show()

    print("Polynomial kernel degree 2")
    plt.figure(figsize=(10, 6))
    plt.semilogx(initial_C, minDCFvalues5Q, label=f'π= 0.5')
    plt.semilogx(initial_C, minDCFvalues3Q, label=f'π= 0.3')
    plt.semilogx(initial_C, minDCFvalues7Q, label=f'π= 0.7')
    plt.xlabel('C')
    plt.ylabel('minDCF')
    plt.title('Polynomial kernel degree 2')
    plt.legend()
    plt.show()

    print("Polynomial kernel degree 2 Z-norm")
    plt.figure(figsize=(10, 6))
    plt.semilogx(initial_C, minDCFvalues5Qz, label=f'π= 0.5')
    plt.semilogx(initial_C, minDCFvalues3Qz, label=f'π= 0.3')
    plt.semilogx(initial_C, minDCFvalues7Qz, label=f'π= 0.7')
    plt.xlabel('C')
    plt.ylabel('minDCF')
    plt.title('Polynomial kernel degree 2 Z-norm')
    plt.legend()
    plt.show()

    print("Radial SVM")
    plt.figure(figsize=(10, 6))
    plt.semilogx(initial_C, minDCFvalues1R, label=f'log(gamma)= -1')
    plt.semilogx(initial_C, minDCFvalues2R, label=f'log(gamma)= -2')
    plt.semilogx(initial_C, minDCFvalues3R, label=f'log(gamma)= -3')
    plt.xlabel('C')
    plt.ylabel('minDCF')
    plt.title('Full Radial SVM')
    plt.legend()
    plt.show()

    print("Full SVM table:")
    print(tabulate(tableFull, headers))
    print("Full SVM Z-norm table:")
    print(tabulate(tableFullZ, headers))
    print("Full RBSVM table:")
    print(tabulate(tableRad, headers))
    print("Full PSVM table:")
    print(tabulate(tablePol, headers))