import numpy as np
import pandas as pd
import scipy
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
    attribute = np.array(df.iloc[:, 0 : len(df.columns) - 1])
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
    tableKFold = []
    pi=[0.5,0.3,0.7]
    Cfn = 1
    Cfp = 1
    l_list = np.logspace(-6, 6, 5)
    path = os.path.abspath("data/Train.txt")
    [full_train_att, full_train_label] = load(path)
    priorProb = ML.vcol(np.ones(2) * 0.5)
    q=[0,1]
    k=5
    
    for i in q:       
        minDCFvalues5=[]
        minDCFvalues3=[]
        minDCFvalues7=[]
        for p in pi:
            
            for l in l_list:
                [_, _, accuracy, DCFnorm, minDCF] = ML.k_fold(
                    k,
                    full_train_att,
                    full_train_label,
                    priorProb,
                    "regression",
                    l=l,
                    pi=p,
                    quadratic=i,
    
                )
                if p== 0.5:
                    minDCFvalues5.append(minDCF)
                if p== 0.3:
                    minDCFvalues3.append(minDCF)
                if p== 0.7:
                    minDCFvalues7.append(minDCF)

        if i==1:
            print(f"Quadratic regression with k-fold = {k}")
        if i==0:
            print(f"Logarithmic regression with k-fold = {k}")
        plt.figure(figsize=(10, 6))
        plt.semilogx(l_list, minDCFvalues5, label=f'π= 0.5')
        plt.semilogx(l_list, minDCFvalues3, label=f'π= 0.3')
        plt.semilogx(l_list, minDCFvalues7, label=f'π= 0.7')
        plt.xlabel('lambda')
        plt.ylabel('minDCF')
        if i==0:
         plt.title('Logarithmic regression')
        if i==1:
         plt.title('Quadratic regression')
        plt.legend()
        plt.show()
        
    # #Z-norm
    standard_deviation = np.std(full_train_att)
    z_data = ML.center_data(full_train_att) / standard_deviation      
    for i in q:       
        minDCFvalues5=[]
        minDCFvalues3=[]
        minDCFvalues7=[]
        for p in pi:
            
            for l in l_list:
                [_, _, accuracy, DCFnorm, minDCF] = ML.k_fold(
                    k,
                    z_data,
                    full_train_label,
                    priorProb,
                    "regression",
                    l=l,
                    pi=p,
                    quadratic=i,
    
                )
                if p== 0.5:
                    minDCFvalues5.append(minDCF)
                if p== 0.3:
                    minDCFvalues3.append(minDCF)
                if p== 0.7:
                    minDCFvalues7.append(minDCF)

        if i==1:
            print(f"Quadratic regression Z-norm ")
        if i==0:
            print(f"Logarithmic regression Z-norm ")
        plt.figure(figsize=(10, 6))
        plt.semilogx(l_list, minDCFvalues5, label=f'π= 0.5')
        plt.semilogx(l_list, minDCFvalues3, label=f'π= 0.3')
        plt.semilogx(l_list, minDCFvalues7, label=f'π= 0.7')
        plt.xlabel('lambda')
        plt.ylabel('minDCF')
        if i==0:
         plt.title('Logarithmic regression Z-norm')
        if i==1:
         plt.title('Quadratic regression Z-norm')
        plt.legend()
        plt.show()
    for i in q:       
        minDCFvalues5=[]
        minDCFvalues3=[]
        minDCFvalues7=[]
        for p in pi:
            
            for l in l_list:
                [_, _, accuracy, DCFnorm, minDCF] = ML.k_fold(
                    k,
                    full_train_att,
                    full_train_label,
                    priorProb,
                    "regression",
                    PCA_m=11,
                    l=l,
                    pi=p,
                    quadratic=i,
    
                )
                if p== 0.5:
                    minDCFvalues5.append(minDCF)
                if p== 0.3:
                    minDCFvalues3.append(minDCF)
                if p== 0.7:
                    minDCFvalues7.append(minDCF)

        if i==1:
            print(f"Quadratic regression PCA 11")
        if i==0:
            print(f"Logarithmic regression PCA 11 ")
        plt.figure(figsize=(10, 6))
        plt.semilogx(l_list, minDCFvalues5, label=f'π= 0.5')
        plt.semilogx(l_list, minDCFvalues3, label=f'π= 0.3')
        plt.semilogx(l_list, minDCFvalues7, label=f'π= 0.7')
        plt.xlabel('lambda')
        plt.ylabel('minDCF')
        if i==0:
         plt.title('Logarithmic regression PCA 11')
        if i==1:
         plt.title('Quadratic regression PCA 11')
        plt.legend()
        plt.show()  
    
      
    searchL=float(input("Enter a lambda value you want to search for: "))
    headers = [
    "Model",
    "π=0.5 DCF/minDCF",
    "π=0.3 DCF/minDCF",
    "π=0.7 DCF/minDCF",
    "π=0.5 Z-norm DCF/minDCF",
    "π=0.3 Z-norm DCF/minDCF",
    "π=0.7 Z-norm DCF/minDCF",
    ]
    pit = [0.5,0.3,0.7]
    cont = 0
    for i in q:
        for x in pit:
            if i==0:
                tableKFold.append([f"LR {x}"])
            if i==1:
                tableKFold.append([f"QLR {x}"])
            
            for p in pi:
                [_, _, accuracy, DCFnorm, minDCF] = ML.k_fold(
                        k,
                        full_train_att,
                        full_train_label,
                        priorProb,
                        "regression",
                        l=searchL,
                        pi=p,
                        quadratic=i,
                        pit=x
                    )
                tableKFold[cont].append([DCFnorm, minDCF])
            for p in pi:
                [_, _, accuracy, DCFnorm, minDCF] = ML.k_fold(
                        k,
                        z_data,
                        full_train_label,
                        priorProb,
                        "regression",
                        l=searchL,
                        pi=p,
                        quadratic=i,
                        pit=x
                    )
                tableKFold[cont].append([DCFnorm, minDCF])
            cont += 1
   
    print(tabulate(tableKFold, headers))
    
    [_, _, accuracy, DCFnorm, minDCF] = ML.k_fold(
                        k,
                        full_train_att,
                        full_train_label,
                        priorProb,
                        "regression",
                        l=10**-3,
                        pi=0.5,
                        quadratic=0,
                        pit=0.5
                    )  

