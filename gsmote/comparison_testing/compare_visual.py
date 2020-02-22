import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
import datetime


def visualize_data(X,y,title):
    feat_cols = ['pixel' + str(i) for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))
    X, y = None, None

    # For reproducability of the results
    np.random.seed(42)
    rndperm = np.random.permutation(df.shape[0])

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df[feat_cols].values)
    df['pca-one'] = pca_result[:, 0]
    df['pca-two'] = pca_result[:, 1]

    plt.figure(figsize=(16, 10))
    plt.title(title)
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="y",
        palette=sns.color_palette("hls", 2),
        data=df.loc[rndperm, :],
        legend="full",
        alpha=0.3
    )
    plt.xlim(-6,6)
    plt.ylim(-6,6)

    #plt.show()
    plt.savefig("../../output/compare_" + datetime.datetime.now().strftime("%Y-%m-%d__%H_%M_%S") + ".png", dpi=1200)

def visualize(X):
    feat_cols = ['pixel' + str(i) for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feat_cols)
    df['y'] = 1
    df['label'] = df['y'].apply(lambda i: str(i))

    # For reproducability of the results
    np.random.seed(42)
    rndperm = np.random.permutation(df.shape[0])

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df[feat_cols].values)
    df['pca-one'] = pca_result[:, 0]
    df['pca-two'] = pca_result[:, 1]



    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=5)
    kmeans.fit(pca_result)
    y_kmeans = kmeans.predict(pca_result)
    plt.scatter(pca_result[:,0], pca_result[:, 1], c=y_kmeans, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
    plt.show()

