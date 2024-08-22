import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.metrics import davies_bouldin_score, silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler, MinMaxScaler  
from sklearn.metrics.pairwise import euclidean_distances
from kneed import KneeLocator
from scipy.sparse import csgraph
from numpy import linalg as LA
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
import matplotlib.cm as cm

def calculate_centroids(clusters, values):
    res = {}
    for value, cluster in zip(values, clusters):
        key = cluster
        if key in res:
            res[key].append(value)
        else:
            res[key] = [value]

    centroids = []
    for key in res:
        arrays = [np.array(x) for x in res[key]]
        centroids.append([np.mean(k) for k in zip(*arrays)])

    return np.asarray(centroids)

def means_tuning(X, alg, name):
    silhouette_avg = []
    inertia = []
    for n_clusters in range(2,11):
        k_means = alg(n_clusters=n_clusters, random_state=0)
        k_means.fit(X)

        inertia.append(k_means.inertia_)
        silhouette_avg.append(silhouette_score(X, k_means.labels_))

    kl = KneeLocator(range(2, 11), inertia, curve="convex", direction="decreasing")
    kl.plot_knee()

    # Elbow Method
    plt.plot(range(2,11), inertia, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title(f'Elbow Method for {name}')
    plt.savefig(f"graphs/descriptive/profile_elbow.pdf")
    plt.close()

    # # Silhouette Method
    # plt.plot(range(2,11), silhouette_avg, 'bx-')
    # plt.xlabel('k')
    # plt.ylabel('Silhouette')
    # plt.title(f'Silhouette Method for {name}')
    # plt.savefig(f"graphs/descriptive/{name}_silhouette.pdf")

    plt.close()

def spectral_tuning(X):
    affinity_matrix = getAffinityMatrix(X, k = 10)
    k, _,  _ = eigenDecomposition(affinity_matrix, plot=False)
    print(k)

from scipy.spatial.distance import pdist, squareform
def getAffinityMatrix(coordinates, k = 7):
    # calculate euclidian distance matrix
    dists = squareform(pdist(coordinates)) 
    
    # for each row, sort the distances ascendingly and take the index of the 
    #k-th position (nearest neighbour)
    knn_distances = np.sort(dists, axis=0)[k]
    knn_distances = knn_distances[np.newaxis].T
    
    # calculate sigma_i * sigma_j
    local_scale = knn_distances.dot(knn_distances.T)

    affinity_matrix = dists * dists
    affinity_matrix = -affinity_matrix / local_scale
    # divide square distance matrix by local scale
    affinity_matrix[np.where(np.isnan(affinity_matrix))] = 0.0
    # apply exponential
    affinity_matrix = np.exp(affinity_matrix)
    np.fill_diagonal(affinity_matrix, 0)
    return affinity_matrix


def eigenDecomposition(A, plot = False, topK = 5):
    L = csgraph.laplacian(A, normed=True)
    n_components = A.shape[0]
    
    # LM parameter : Eigenvalues with largest magnitude (eigs, eigsh), that is, largest eigenvalues in 
    # the euclidean norm of complex numbers.
#     eigenvalues, eigenvectors = eigsh(L, k=n_components, which="LM", sigma=1.0, maxiter=5000)
    eigenvalues, eigenvectors = LA.eig(L)
    
    if plot:
        plt.title('Largest eigen values of input matrix')
        plt.scatter(np.arange(len(eigenvalues)), eigenvalues)
        plt.grid()
        
    # Identify the optimal number of clusters as the index corresponding
    # to the larger gap between eigen values
    index_largest_gap = np.argmax(np.diff(eigenvalues))
    nb_clusters = index_largest_gap + 1
        
    return nb_clusters, eigenvalues, eigenvectors


def K_means_analysis(X):
    file = open("kmeans.out", "w")
    # for n_clusters in range(2,10):
    for random_state in range(100, 1000, 100):
        k_means = KMeans(n_clusters=4, random_state=random_state)
        labels = k_means.fit_predict(X)
        centroids = k_means.cluster_centers_

        file.write(f"nclusters: {4} random_state: {random_state}\n")

        db_index = davies_bouldin_score(X, labels)
        file.write(f"Davies-Bouldin: {db_index}\n")

        ed_index = euclidean_distances(centroids)
        file.write(f"Avg Euclidian Distance: {ed_index.mean()}\n")
    file.close()

def K_plot(alg, X):
    k_means = alg(n_clusters=3, random_state=50)
    labels = k_means.fit_predict(X)
    centroids = k_means.cluster_centers_

    #Getting unique labels
    u_labels = np.unique(labels)

    #plotting the results:
    for i in u_labels:
        plt.scatter(X[labels == i , 0] , X[labels == i , 1] , label = i)
    plt.scatter(centroids[:,0] , centroids[:,1] , s = 40, color = 'k')

    plt.legend(['Cluster 0', 'Cluster 1','Cluster 2', 'Centroids'])
    # plt.title("Clustering with KMeans for the Average Salary and Unemploymant")
    # plt.xlabel("average salary")
    # plt.ylabel("unemploymant delta")
    plt.title("Clustering with KMeans")
    plt.savefig(f"graphs/descriptive/profile/KMeans_profile_cluster.pdf")

def K_medoids(X):
    file = open("kmedoids.out", "w")
    for n_clusters in range(2,10):
        for random_state in range(100, 1000, 100):
            k_medoids = KMedoids(n_clusters=n_clusters, random_state=random_state)
            labels = k_medoids.fit_predict(X)
            centroids = k_medoids.cluster_centers_

            file.write(f"nclusters: {n_clusters} random_state: {random_state}\n")

            db_index = davies_bouldin_score(X, labels)
            file.write(f"Davies-Bouldin: {db_index}\n")

            ed_index = euclidean_distances(centroids)
            file.write(f"Avg Euclidian Distance: {ed_index.mean()}\n")
    file.close()

def spectral_clustering(X):
    file = open("spectral.out", "w")
    for n_clusters in range(2,10):
        for random_state in range(100, 1000, 100):
            spectral = SpectralClustering(n_clusters=n_clusters, random_state=random_state, eigen_solver="arpack", affinity="nearest_neighbors")
            labels = spectral.fit_predict(X)
            centroids = calculate_centroids(labels, X)

            file.write(f"nclusters: {n_clusters} random_state: {random_state}\n")

            db_index = davies_bouldin_score(X, labels)
            file.write(f"Davies-Bouldin: {db_index}\n")

            ed_index = euclidean_distances(centroids)
            file.write(f"Avg Euclidian Distance: {ed_index.mean()}\n")

    file.close()

def spectral_plot(X):
    spectral = SpectralClustering(n_clusters=4, random_state=50, eigen_solver="arpack", affinity="nearest_neighbors")
    labels = spectral.fit_predict(X)
    centroids = calculate_centroids(labels, X)

    #Getting unique labels
    u_labels = np.unique(labels)

    #plotting the results:
    for i in u_labels:
        plt.scatter(X[labels == i , 0] , X[labels == i , 1] , label = i)
    plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')

    plt.legend(['Cluster 0', 'Cluster 1', 'Cluster 2','Cluster 3', 'Centroids'])
    plt.savefig(f"graphs/descriptive/full_cluster_spectral.pdf")


def silhouette(X):
    range_n_clusters = [2]
    silhouette_avg_n_clusters = []

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = SpectralClustering(n_clusters=n_clusters, random_state=42, eigen_solver="arpack", affinity="nearest_neighbors")
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
            "The average silhouette_score is :", silhouette_avg)

        silhouette_avg_n_clusters.append(silhouette_avg)
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)


        print("Silhouette Score:", silhouette_avg)
        print("Davies Bouldin Sccore:",davies_bouldin_score(X, cluster_labels))
        print("Average Euclidian Distance:",euclidean_distances(calculate_centroids(cluster_labels, X)).mean())

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = calculate_centroids(cluster_labels, X)
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        # ax2.set_xlabel("Average Salary")
        # ax2.set_ylabel("Ratio of Urban Inhabitans")

        plt.suptitle(("Silhouette analysis for Spectral Clustering on sample data "
                    "with n_clusters = %d" % n_clusters),
                    fontsize=14, fontweight='bold')

        plt.savefig(f"graphs/descriptive/profile-silhouette-{n_clusters}.pdf")
        plt.close()

    plt.plot(range_n_clusters, silhouette_avg_n_clusters, 'bx-')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("silhouette score")
    plt.savefig(f"graphs/descriptive/profile-silhouette.pdf")

    return KMeans(n_clusters=2, random_state=42).fit_predict(X)



def preprocess():
    X = pd.read_csv('data/processed/data_3it.csv')
    y = X['status']
    X.drop(['status'], inplace=True, axis=1)

    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    pca = PCA(2)
    X = pca.fit_transform(X)

    return X

columns = ["balance_mean", "mean_trans_profit", "trans_mth", "duration", "payments", "status", "owner_age_on_loan", "owner_gender", "no. of municipalities with inhabitants < 499 ",
     "no. of municipalities with inhabitants 500-1999", "no. of municipalities with inhabitants 2000-9999 ", "no. of municipalities with inhabitants >10000 ", "no. of cities ", "ratio of urban inhabitants ",
     "average salary ", "ratio entrepeneurs", "crime_delta", "unemploymant_delta"]

def preprocess_profile():

    X = pd.read_csv('data/processed/data_3it.csv')

    #X = X[columns]
    X = X[["average salary ", "balance_mean"]]

    df = X


    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    pca = PCA(2)
    X = pca.fit_transform(X)

    return X, df

def centroids_attributes():
    X, df = preprocess_profile()

    cluster0 = []
    cluster1 = []
    for col in columns:
        Y = df[[col]]

        scaler = StandardScaler()
        scaler.fit(Y)
        Y = scaler.transform(Y)

        clusterer = KMeans(n_clusters=2, random_state=42).fit(Y)
        cluster_labels = clusterer.labels_
        centroids = clusterer.cluster_centers_

        cluster0.append(centroids[0][0])
        cluster1.append(centroids[1][0])

    x = list(range(0, len(columns)))

    fig,ax = plt.subplots(figsize=(30, 5))
    ax.plot(x, cluster0, label="Cluster 0")
    ax.plot(x, cluster1, label="Cluster 1")
    ax.set_xticks(x)
    ax.set_xticklabels(columns, rotation=45, ha='right')

    plt.xlabel('Attributes')
    plt.ylabel('Centroid Coordinates')
    plt.title('Centroid Values for Each Attribute')
    plt.legend()

    plt.savefig(f"graphs/descriptive/profile-centroids.pdf", bbox_inches='tight')


X, df = preprocess_profile()

labels = silhouette(X)



# labels = KMeans(n_clusters=2, random_state=42).fit_predict(X)
# df.loc[:,"cluster"] = labels

# df1 = df.loc[df.cluster == 0].mean().round(3)
# df2 = df.loc[df.cluster == 1].mean().round(3)


# df3 = pd.concat({"Cluster 0": df1, "Cluster 1" : df2}, axis=1)
# df3.drop(['cluster'], inplace=True)
# df3.reset_index(inplace=True)
# df3.columns.values[0] = 'Attributes'

# print(df3)

# import dataframe_image as dfi
# dfi.export(df3.style.hide(axis='index'), "table.png")


