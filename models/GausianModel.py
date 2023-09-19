import numpy as np
import pandas as pd
import os
from tabulate import tabulate
import sys

sys.path.append(os.path.abspath("MLandPattern"))
import MLandPattern as ML



def load(pathname, vizualization=0):
    df = pd.read_csv(pathname, header=None)
    if vizualization:
        print(df.head())
    attribute = np.array(df.iloc[:, 0 : len(df.columns) - 1])
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


def gaussian_train(attributes, labels, pi=0.5, Cfn=1, Cfp=1):
    ### Parameter definition ###
    tableKFold = []
    headers = ["MVG", "Naive", "Tied Gaussian", "Tied Naive"]
    ####
    tableKFold.append(["Full"])
    # k_fold_value = int(input("Value for k partitions: "))
    k_fold_value = 10
    for model in headers:
        [SPost, Predictions, accuracy, DCFnorm, minDCF] = ML.k_fold(
            k_fold_value, attributes, labels, priorProb, model=model
        )
        tableKFold[0].append([DCFnorm, minDCF])

    cont = 1
    for i in reversed(range(9,13)):

        tableKFold.append([f"PCA {i}"])
        for model in headers:
            [SPost, Predictions, accuracy, DCFnorm, minDCF] = ML.k_fold(
                k_fold_value,
                attributes,
                labels,
                priorProb,
                model=model,
                PCA_m=i,
            )
            tableKFold[cont].append([DCFnorm, minDCF])

        cont += 1

    newHeaders = []
    print("PCA with k-fold")
    for i in headers:
        newHeaders.append(i + " DCF/MinDCF")
    print(tabulate(tableKFold, headers=newHeaders))


if __name__ == "__main__":
    path = os.path.abspath("data/Train.txt")
    [full_train_att, full_train_label] = load(path)

    priorProb = ML.vcol(np.ones(2) * 0.5)

    standard_deviation = np.std(full_train_att)
    z_data = ML.center_data(full_train_att) / standard_deviation

    print("Full dataset")
    gaussian_train(full_train_att, full_train_label)
    print("Z-Norm dataset")
    gaussian_train(z_data, full_train_label)
