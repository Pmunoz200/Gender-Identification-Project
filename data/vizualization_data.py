import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
sys.path.append(os.path.abspath("MLandPattern"))
import MLandPattern as ML

# Constants
class_label = ["Non-target Languages", "Target Language"]
alpha_val = 0.5

# Helper functions
def load(pathname, visualization=0):
    df = pd.read_csv(pathname, header=None)
    if visualization:
        print(df.head())
    attribute = np.array(df.iloc[:, 0 : len(df.columns) - 1])
    attribute = attribute.T
    label = np.array(df.iloc[:, -1])
    return attribute, label

def histogram_1n(nonTargetL, targetL, x_axis="", y_axis=""):
    plt.xlim((min(min(nonTargetL), min(targetL)) - 1, max(max(nonTargetL), max(targetL)) + 1))
    plt.hist(nonTargetL, color="blue", alpha=alpha_val, label=class_label[0], density=True, bins=np.arange(nonTargetL.min(), nonTargetL.max()+2,1), edgecolor='grey')
    plt.hist(targetL, color="red", alpha=alpha_val, label=class_label[1], density=True, bins=np.arange(targetL.min(), targetL.max()+2,1), edgecolor='grey')
    plt.legend(class_label)

# Main plotting functions
def graficar(attributes):
    attribute_names = [str(i) for i in range(attributes.shape[0])]
    values_histogram = {name: [attributes[i, labels == 0], attributes[i, labels == 1]] for i, name in enumerate(attribute_names)}

    cont = 1

    for xk, xv in values_histogram.items():
        for yk, yv in values_histogram.items():
            if xk == yk:
                histogram_1n(xv[0], xv[1], x_axis=xk)
                plt.savefig(f"{os.getcwd()}/Image/histogram-dim-{cont}.png")
                plt.show()
                cont += 1

def graf_LDA(attributes, lables):
    W, _ = ML.LDA1(attributes, lables, 1)
    LDA_attributes = np.dot(W.T, attributes)
    for i in range(1):
        plt.hist(LDA_attributes[i, labels == 0], color="blue", density=True, alpha=0.7, label='0', bins=45)
        plt.hist(LDA_attributes[i, labels == 1], color="red", density=True, alpha=0.7, label='1', bins=45)
        plt.legend()
        plt.xlabel(i)
    plt.title("LDA Direction")
    plt.savefig(f"{os.getcwd()}/Image/LDA-direction.png")
    plt.show()

def graf_PCA(attributes, lables):
    fractions = []
    total_eig, _ = np.linalg.eigh(ML.covariance(attributes))
    total_eig = np.sum(total_eig)
    for i in range(attributes.shape[0]):
        if i == 0:
            continue
        _, reduces_attributes = ML.PCA(attributes, i)
        PCA_means, _ = np.linalg.eigh(ML.covariance(reduces_attributes))
        PCA_means = np.sum(PCA_means)
        fractions.append(PCA_means/total_eig)
    fractions.append(1)
    plt.plot(range(1, 7), fractions, marker='o')
    plt.plot(range(1, 7), [0.97]*6, '--')
    plt.grid(True)
    plt.ylabel('Fraction of the retained variance')
    plt.xlabel('Number of dimensions')
    plt.title("PCA variability analysis")
    plt.savefig(f"{os.getcwd()}/Image/PCA-Analysis.png")
    plt.show()

def plot_scatter(features, labels, label_names, display_legend=True, plot_title="title", transparency=1):
    num_features = features.shape[0]
    for i in range(num_features):
        for j in range(num_features):
            if i < j:
                plt.figure()
                for label_index in range(len(label_names)):
                    x_values = features[i][labels == label_index]
                    y_values = features[j][labels == label_index]
                    plt.scatter(x_values, y_values, label=label_names[label_index], alpha=transparency)
                if display_legend:
                    plt.legend()
                if plot_title != "title":
                    plt.title(plot_title)
                plt.show()

def graph_corr(attributes, labels):
    corr_attr = np.corrcoef(attributes)
    sns.heatmap(corr_attr, cmap='Greens')
    plt.title("Total attribute correlation")
    plt.savefig(f"{os.getcwd()}/Image/Dataset-correlation.png")
    plt.show()

    nonTargetLs = attributes[:, labels == 0]
    corr_attr = np.corrcoef(nonTargetLs)
    sns.heatmap(corr_attr, cmap='Blues')
    plt.title("non Target Languages attribute correlation")
    plt.savefig(f"{os.getcwd()}/Image/nonTargetL-correlation.png")
    plt.show()

    targetLs = attributes[:, labels == 1]
    corr_attr = np.corrcoef(targetLs)
    sns.heatmap(corr_attr, cmap="Reds")
    plt.title("Target Language attribute correlation")
    plt.savefig(f"{os.getcwd()}/Image/targetL-correlation.png")
    plt.show()

if __name__ == "__main__":
    path = os.path.abspath("data/Train.txt")
    [attributes, labels] = load(path)
    
    for i in range(attributes.shape[0]):
        print(f"max {i}: {max(attributes[i])}", end="\t")
        print(f"min {i}: {min(attributes[i])}")
    
    print(f"Attribute dimensions: {attributes.shape[0]}")
    print(f"Points on the dataset: {attributes.shape[1]}")
    print(f"Distribution of labels (0, 1): {labels[labels==0].shape}, {labels[labels==1].shape}")
    print(f"Possible classes: {class_label[0]}, {class_label[1]}")
    
    pt, _ = ML.PCA(attributes, 6)
    att6 = np.dot(pt, attributes)
    
    plot_scatter(att6, labels, ['1', '2', '3', '4', '5', '6'], display_legend=False)
    graficar(attributes)
    graf_LDA(attributes, labels)
    graf_PCA(attributes, labels)
    graph_corr(attributes, labels)
