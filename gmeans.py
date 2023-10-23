import numpy as np
from scipy.stats import anderson
import math
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings("ignore")


class GMeans:
    def __init__(self, significance = 0.0001, k_min = 1, k_max = np.inf, random_state=None):
        """
        Reference:
        Hamerly, G., & Elkan, C.P. (2003). Learning the k in k-means. Neural Information Processing Systems.
        """
        self.k_min = k_min
        self.k_max = k_max
        self.random_state = random_state

        # Significance level for the Anderson-Darling test
        self.significance = significance
    
    def fit_predict(self, X: np.ndarray):
        if self.random_state:
            np.random.seed(self.random_state)

        # Initialize the initial set of centroids C
        labels, centroids = self._initialize_centers(X)
        k = self.k_min
        while k <= self.k_max:
            k_old = k

            # For each label cluster see if it can be split based in the significance level
            for label in range(k):
                # Get the data points for the current cluster
                X_cluster = X[labels == label]

                # If there aren't enough data points, skip
                if X_cluster.shape[0] < 2:
                    continue
                
                # Split cluster into two clusters
                _ , centroids_cluster = self._split_cluster(X_cluster, centroids[label])

                # Project data form cluster onto resulting connection axis
                v = centroids_cluster[1] - centroids_cluster[0]
                X_projected = np.dot(X_cluster, v) / np.linalg.norm(v)

                adjusted_stat, _ , _  = anderson(X_projected, "norm")

                # Test the null hypothesis that a sample is drawn from a normal distribution.
                # H0: The data around the center are smapled from a gaussian distribution
                # H1: The data around the center are not sampled from a gaussian distribution
                p_value = self.anderson_to_pvalue(adjusted_stat, len(X_cluster))
                accept_null_hypothesis = p_value > self.significance  # False - not a gaussian distribution (reject H0)

                if not accept_null_hypothesis:
                    # If the null hypothesis is rejected, the cluster is split
                    # The new cluster centers are the centroids of the two new clusters
                    centroids[label] = centroids_cluster[0]
                    centroids = np.append(centroids, [centroids_cluster[1]], axis=0) # Add the new 2 centroid to the list
                    k += 1

            # If k does not change, stop
            if k_old == k:
                break

            # Prepare for the next iteration
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, init=centroids, n_init=5)
            labels = kmeans.fit_predict(X)
            centroids = kmeans.cluster_centers_

        self.centroids = centroids
        return labels

    def _initialize_centers(self, X: np.ndarray):
        # Initialize the initial set of centroids C
        if self.k_min == 1:
            # The initial center is the centroid of the data
            labels = np.zeros(X.shape[0])
            centroids = np.mean(X, axis=0).reshape(1, -1)
        else:
            kmeans = KMeans(n_clusters=self.k_min, random_state=self.random_state, init='k-means++', n_init=5)
            labels = kmeans.fit_predict(X)
            centroids = kmeans.centroids

        return labels, centroids


    def _split_cluster(self, X_cluster, old_centroid):
        random_centroid = X_cluster[np.random.choice(X_cluster.shape[0])]
        new_centroid = old_centroid - (random_centroid - old_centroid)

        tmp_centroid = np.array([random_centroid, new_centroid])

        kmeans = KMeans(n_clusters=2, random_state=self.random_state, init=tmp_centroid)
        labels = kmeans.fit_predict(X_cluster)
        centroids = kmeans.cluster_centers_
        return labels, centroids
    
    @staticmethod
    def anderson_to_pvalue(estimation, n_points):
        """
        Reference:
        R.B. D’Augostino and M.A. Stephens, Eds., 1986, Goodness-of-Fit Techniques, Marcel Dekker.
        """
        adjusted_stat = estimation * (1 + (.75 / n_points) + 2.25 / (n_points ** 2))
        if adjusted_stat > 0.6:
            p = math.exp(1.2937 - 5.709 * adjusted_stat + 0.0186 * adjusted_stat**2)
        elif 0.34 < adjusted_stat <= 0.6:
            p = math.exp(0.9177 - 4.279 * adjusted_stat - 1.38 * adjusted_stat**2)
        elif 0.2 < adjusted_stat <= 0.34:
            p = 1 - math.exp(-8.318 + 42.796 * adjusted_stat - 59.938 * adjusted_stat**2)
        else:
            p = 1 - math.exp(-13.436 + 101.14 * adjusted_stat - 223.73 * adjusted_stat**2)
        return p



if __name__ == "__main__":
    import seaborn as sbn
    from matplotlib import pyplot as plt
    import pandas as pd
    from sklearn import datasets

    iris = datasets.make_blobs(n_samples=1000,
        n_features=2,
        centers=6,
        cluster_std=0.7)[0]

    gmeans = GMeans(k_min=1)
    labels_gmeans = gmeans.fit_predict(iris)

    plot_data = pd.DataFrame(iris[:, 0:2])
    plot_data.columns = ['x', 'y']
    plot_data['labels_gmeans'] = labels_gmeans
    
    km = KMeans(n_clusters=6, init='k-means++', n_init=10)
    labels_kmeans = km.fit_predict(iris)
    plot_data['labels_km'] = labels_kmeans
    
    sbn.lmplot(x='x', y='y', data=plot_data, hue='labels_gmeans', fit_reg=False)
    sbn.lmplot(x='x', y='y', data=plot_data, hue='labels_km', fit_reg=False)
    plt.show()