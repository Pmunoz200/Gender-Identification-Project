import numpy as np
import pandas as pd
import scipy
import os
from tabulate import tabulate
import sys
from matplotlib import pyplot as plt

sys.path.append(os.path.abspath("MLandPattern"))
import MLandPattern as ML


pi = 0.5
Cfn = 1
Cfp = 1


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

def call_GMM(attributes, labels, prior_probability, models, k_value=5, m = 0, pi = [0.5]):
    tableGMM = []
    total_iter = len(models)*len(pi)
    perc = 0

    ### ------------- K-FOLD CROSS-VALIDATION AND PCA ---------------------- ####
    tableGMM.append(["Full"])
    result_minDCF = []
    c2 = 1
    for model in models:
        ref = model.split(":")
        mod = ref[0]
        constrains = eval(ref[1])
        tableGMM[0].append([])
        for p in pi:
            if not m:
                [SPost, Predictions, accuracy, _, minDCF] = ML.k_fold(
                k_value, attributes, labels, prior_probability, model=mod, niter=constrains[1],alpha=constrains[0], psi=constrains[2], pi=p
                )
            else:
                [SPost, Predictions, accuracy, _, minDCF] = ML.k_fold(
                k_value, attributes, labels, prior_probability, model=mod, niter=constrains[1],alpha=constrains[0], psi=constrains[2], PCA_m=m, pi=p
                )
            tableGMM[0][c2].append(minDCF)
            perc += 1
            print(f"{round(perc * 100 / total_iter, 2)}%")
        print(tableGMM)
        result_minDCF.append(minDCF)
        c2 += 1
    print()
    cont = 1
    print("minDCF table")
    print(tabulate(tableGMM, headers=models[0:3]))
    return result_minDCF

def graph_data(raw_results, z_results, model, pca = 0):
    attribute = {
    "Raw": raw_results,
    "Z-Score": z_results
    }

    maxVal = max(max(attribute["Raw"]), max(attribute["Z-Score"]))

    x = np.arange(len(raw_results))  # the label locations
    print(x)
    width = 0.25  # the width of the bars
    multiplier = 0

    labels = np.power(2, x)

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in attribute.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('minDCF')
    ax.set_xticks(x + width/2, labels)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, maxVal + 0.1)
    if pca != 0:
        plt.savefig(f"{os.getcwd()}/Image/GMM-{model}-PCA.png")
    else:
        plt.savefig(f"{os.getcwd()}/Image/GMM-{model}.png")
    plt.show()


if __name__ == "__main__":
    path = os.path.abspath("data/Train.txt")
    [full_train_att, full_train_label] = load(path)
    priorProb = ML.vcol(np.ones(2) * 0.5)

    # The headers define which models the code will train. It is a list of all the models to try, and the format is:
    # [model: [alpha, G, psi]], where psi and alpha are hyperparameters, and G corresponds to the amount of clusters,
    # which are defined as 2^G.

    headers = [
    "Diagonal:[0.1, 2, 0.01]",
    ]
    # headers = [
    # "GMM:[0.1, 0, 0.01]",
    # "GMM:[0.1, 1, 0.01]",
    # "GMM:[0.1, 2, 0.01]",
    # "GMM:[0.1, 3, 0.01]",
    # "GMM:[0.1, 4, 0.01]",
    # "GMM:[0.1, 5, 0.01]",
    # ]
    pi = [0.3, 0.5, 0.7]
    standard_deviation = np.std(full_train_att)
    z_data = ML.center_data(full_train_att) / standard_deviation
    print("Full data")
    raw_values = call_GMM(full_train_att, full_train_label, priorProb, m=12, models=headers)
    print("Z-norm data")
    z_values = call_GMM(z_data, full_train_label, priorProb, m=12, models=headers)
    graph_data(raw_values, z_values, headers[0][0:3], pca=1)
