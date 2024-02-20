import numpy as np

from cluster import cluster
from random import uniform
from scipy.spatial import KDTree


class KMeans(cluster):
    def __init__(self, k=5, max_iterations=100):
        """
        Set hyperparameters for KMeans

        :param int k: target number of cluster centroids
        :param int max_iterations: maximum number of times to execute the convergence attempt
        """
        super().__init__()
        self.k = k
        self.max_iterations = max_iterations

    def fit(self, X):
        """
        Takes an n x d list of values and creates cluster hypotheses using the KMeans algorithm

        :param list X: n instance by d feature list
        :returns: List of cluster hypotheses (n) for each instance, List (k) of lists (d) of cluster centroids' values
        """
        # Create numpy arrays based on the data for faster operations and convenience
        data = np.array(X)
        data_x = np.array([x for x, _ in X])
        data_y = np.array([y for _, y in X])

        # Initialize hypotheses as -1 for unassigned, random floats between the min of all x and y for the centroids
        hypotheses = [-1 for _ in range(len(X))]
        centroids = [[uniform(data_x.min(), data_x.max()), uniform(data_y.min(), data_y.max())] for _ in range(self.k)]

        i = 0
        while i < self.max_iterations:
            # Uses scipy's K-Dimensional Tree to quickly find the nearest centroid from any point
            kd_tree = KDTree(centroids)
            hypotheses = [kd_tree.query(x)[1] for x in X]

            # Groups the indices of the hypotheses in the same cluster, compute the mean of the points and the cluster
            for k in range(len(centroids)):
                grouped_hypotheses = np.argwhere(np.array(hypotheses) == k).flatten()
                centroids[k] = np.mean(np.append(data[grouped_hypotheses], [centroids[k]], axis=0), axis=0).tolist()

            i += 1

        return hypotheses, centroids


def main():
    k_means = KMeans(2)
    hyp, cen = k_means.fit([[0, 0], [2, 2], [0, 2], [2, 0], [10, 10], [8, 8], [10, 8], [8, 10]])
    print(f'Hypotheses are {hyp}\nCentroids are {cen}')


if __name__ == "__main__":
    main()
