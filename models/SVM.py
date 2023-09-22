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
    # print(attribute)
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
    tablePCA = []
    tableKFold = []
    tableZ = []

    headers = [
        "SVM  C=0.1",
        "SVM  C=1",
    ]
    initial_C = np.logspace(-6, 6, 15)
    initial_K = 1

    # ###--------------K-fold----------------------###
    k = 5
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

    cont = 0
    tableKFold.append(["Full"])
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
            # tableKFold[0].append([round(accuracy * 100, 2), DCFnorm, minDCF])

            cont += 1

    ### --------z-score----------###
    standard_deviation = np.std(full_train_att)
    z_data = ML.center_data(full_train_att) / standard_deviation
    k = 5
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

    cont = 0
    tableZ.append(["Full"])
    minDCFvalues5Z = []
    minDCFvalues3Z = []
    minDCFvalues7Z = []

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
                minDCFvalues5Z.append(minDCF)
            if p == 0.3:
                minDCFvalues3Z.append(minDCF)
            if p == 0.7:
                minDCFvalues7Z.append(minDCF)
            # tableZ[0].append([round(accuracy * 100, 2), DCFnorm, minDCF])

    #         cont += 1
    ### -----------------Polynomial---------###
    # k=5
    # section_size = int(full_train_att.shape[1] / k)
    # low = 0
    # all_values = np.c_[full_train_att.T, full_train_label]
    # all_values = np.random.permutation(all_values)
    # attributes = all_values[:, 0:12].T
    # labels = all_values[:, -1].astype("int32")
    # high = section_size
    # if high > attributes.shape[1]:
    #     high = attributes.shape
    # test_att = attributes[:, low:high]
    # test_labels = labels[low:high]
    # train_att = attributes[:, :low]
    # train_label = labels[:low]
    # train_att = np.hstack((train_att, attributes[:, high:]))
    # train_label = np.hstack((train_label, labels[high:]))

    # initial_d = 2
    # initial_const = 0
    # initial_K = 1

    # minDCFvalues5Q=[]
    # minDCFvalues3Q=[]
    # minDCFvalues7Q=[]
    # for p in pi:
    #     for C in initial_C:
    #         contrain = C
    #         d = initial_d
    #         const = initial_const
    #         k = initial_K * np.power(10, 0)
    #         [SPost, Predictions, accuracy] = ML.svm(
    #             train_att,
    #             train_label,
    #             test_att,
    #             test_labels,
    #             contrain,
    #             dim=d,
    #             c=const,
    #             eps=k**2,
    #             model="polynomial",
    #         )
    #         confusion_matrix = ML.ConfMat(Predictions, test_labels)
    #         DCF, DCFnorm = ML.Bayes_risk(confusion_matrix, p, Cfn, Cfp)
    #         (minDCF, FPRlist, FNRlist,_) = ML.minCostBayes(
    #             SPost, test_labels, p, Cfn, Cfp
    #         )
    #         if p== 0.5:
    #             minDCFvalues5Q.append(minDCF)
    #         if p== 0.3:
    #             minDCFvalues3Q.append(minDCF)
    #         if p== 0.7:
    #             minDCFvalues7Q.append(minDCF)
    #         #tablePCA[0].append([round(accuracy * 100, 2), DCFnorm, minDCF])

    #         #cont += 1
    # ###----------------Pol-z----------------###
    # standard_deviation = np.std(full_train_att)
    # z_data = ML.center_data(full_train_att) / standard_deviation
    # k=5
    # section_size = int(full_train_att.shape[1] / k)
    # low = 0
    # all_values = np.c_[z_data.T, full_train_label]
    # all_values = np.random.permutation(all_values)
    # attributes = all_values[:, 0:12].T
    # labels = all_values[:, -1].astype("int32")
    # high = section_size
    # if high > attributes.shape[1]:
    #     high = attributes.shape
    # test_att = attributes[:, low:high]
    # test_labels = labels[low:high]
    # train_att = attributes[:, :low]
    # train_label = labels[:low]
    # train_att = np.hstack((train_att, attributes[:, high:]))
    # train_label = np.hstack((train_label, labels[high:]))
    # minDCFvalues5Qz=[]
    # minDCFvalues3Qz=[]
    # minDCFvalues7Qz=[]
    # for p in pi:
    #     for C in initial_C:
    #         contrain = C
    #         d = initial_d
    #         const = initial_const
    #         k = initial_K * np.power(10, 0)
    #         [SPost, Predictions, accuracy] = ML.svm(
    #             train_att,
    #             train_label,
    #             test_att,
    #             test_labels,
    #             contrain,
    #             dim=d,
    #             c=const,
    #             eps=k**2,
    #             model="polynomial",
    #         )
    #         confusion_matrix = ML.ConfMat(Predictions, test_labels)
    #         DCF, DCFnorm = ML.Bayes_risk(confusion_matrix, pi, Cfn, Cfp)
    #         (minDCF, FPRlist, FNRlist) = ML.minCostBayes(
    #             SPost, test_labels, pi, Cfn, Cfp
    #         )
    #         if p== 0.5:
    #             minDCFvalues5Qz.append(minDCF)
    #         if p== 0.3:
    #             minDCFvalues3Qz.append(minDCF)
    #         if p== 0.7:
    #             minDCFvalues7Qz.append(minDCF)
    #         #tablePCA[0].append([round(accuracy * 100, 2), DCFnorm, minDCF])

    #         cont += 1
    # initial_gamma = 1
    # for ten in range(2):
    #     contrain = initial_C * np.power(10, ten)
    #     for j in range(2):
    #         gamma = initial_gamma * np.power(10, j)
    #         [SPost, Predictions, accuracy] = ML.svm(
    #             train_att,
    #             train_label,
    #             test_att,
    #             test_labels,
    #             contrain,
    #             gamma=gamma,
    #             model="radial",
    #         )
    #         confusion_matrix = ML.ConfMat(Predictions, test_labels)
    #         DCF, DCFnorm = ML.Bayes_risk(confusion_matrix, pi, Cfn, Cfp)
    #         (minDCF, FPRlist, FNRlist) = ML.minCostBayes(
    #             SPost, test_labels, pi, Cfn, Cfp
    #         )
    #         tablePCA[0].append([round(accuracy * 100, 2), DCFnorm, minDCF])

    #         cont += 1
    # cont += 1

    print("SVM with k-fold={k}")
    print(tabulate(tableKFold, headers=headers))
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
    print(tabulate(tableZ, headers=headers))
    plt.figure(figsize=(10, 6))
    plt.semilogx(initial_C, minDCFvalues5Z, label=f'π= 0.5')
    plt.semilogx(initial_C, minDCFvalues3Z, label=f'π= 0.3')
    plt.semilogx(initial_C, minDCFvalues7Z, label=f'π= 0.7')
    plt.xlabel('C')
    plt.ylabel('minDCF')
    plt.title('Z-Norm SVM regression')
    plt.legend()
    plt.show()

    # print("Polynomial kernel degree 2")
    # print(tabulate(tableZ, headers=headers))
    # plt.figure(figsize=(10, 6))
    # plt.semilogx(initial_C, minDCFvalues5Q, label=f'π= 0.5')
    # plt.semilogx(initial_C, minDCFvalues3Q, label=f'π= 0.3')
    # plt.semilogx(initial_C, minDCFvalues7Q, label=f'π= 0.7')
    # plt.xlabel('C')
    # plt.ylabel('minDCF')
    # plt.title('Polynomial kernel degree 2')
    # plt.legend()
    # plt.show()

    # print("Polynomial kernel degree 2 Z-norm")
    # print(tabulate(tableZ, headers=headers))
    # plt.figure(figsize=(10, 6))
    # plt.semilogx(initial_C, minDCFvalues5Qz, label=f'π= 0.5')
    # plt.semilogx(initial_C, minDCFvalues3Qz, label=f'π= 0.3')
    # plt.semilogx(initial_C, minDCFvalues7Qz, label=f'π= 0.7')
    # plt.xlabel('C')
    # plt.ylabel('minDCF')
    # plt.title('Polynomial kernel degree 2 Z-norm')
    # plt.legend()
    # plt.show()
