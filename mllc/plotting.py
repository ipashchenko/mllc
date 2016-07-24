# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea


def plot_corr(df, size=10):
    """Function plots a graphical correlation matrix for each pair of columns in
    the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot
    """

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)


def plot_corr_matrix(corr_matrix):
    """
    Plot correlation matrix.

    :param corr_matrix:
        Correaltion matrix.
    """
    sea.set(style="white")
    mask = np.zeros_like(corr_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(11, 9))
    sea.heatmap(corr_matrix, mask=mask, square=True, xticklabels=5,
                yticklabels=5, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

