#LIBRAIRIES
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



#METHODS
def tfidf_vectorizer(max_features):
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features,
                                   use_idf=True,
                                   stop_words='english',
                                   tokenizer=nltk.word_tokenize)
    return tfidf_vectorizer


def compute_cluster_metrics(model, labels, X):
    """"Function takes as parameters:
    the model, , the true labels and vectorized data using Tfidf """
  
    #model = KMeans(n_clusters=clusters).fit(X)
    cluster_label = model.labels_
    inertia = model.inertia_
    #Compute a tuple for homogeneity, v_measure and completeness
    homog, compl, v_meas = metrics.homogeneity_completeness_v_measure(labels,
                                                                    cluster_label)

    adj_score = metrics.adjusted_rand_score(labels, cluster_label)
    silhouette = metrics.silhouette_score(X, cluster_label)
    scores = {"Homogeneity" : homog, "Completeness": compl,
                                "V_measure" : v_meas, 
                                "Adjuted_Rank_score" : adj_score,
                                "Silhouette_score" : silhouette, 
                                "Inertia" :inertia}
    return scores

def plot_metrics(model, labels, X):
    k = range(2,17,1)
    scores = []
    for i in k:
        model.set_params(n_clusters= i).fit(X) # set the number of clusters to i
        #compute the current scores
        scores.append(compute_cluster_metrics(model, labels, X))
        #group scores values in numpy array for every score
    total_score = {key: np.asarray([score[key] for score in scores])
                  for key in scores[0]}
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3,
                                                         figsize=(10,10))
    # display one figure for each score from a range from 2 to 16
    ax1.plot(k, total_score["Adjuted_Rank_score"])
    ax1.set_title("Adjuted_Rank_score")
    ax2.plot(k, total_score["Completeness"])
    ax2.set_title("Completeness")
    ax3.plot(k, total_score["Homogeneity"])
    ax3.set_title("Homogeneity")
    ax4.plot(k, total_score["Silhouette_score"])
    ax4.set_title("Silhouette_score")
    ax5.plot(k, total_score["V_measure"])
    ax5.set_title("V_measure")
    ax6.plot(k, total_score["Inertia"])
    ax6.set_title("Inertia")
    for ax in fig.get_axes():
        ax.label_outer()
    return total_score

def cluster_visualization(model,n_clusters, X):
    print('Dispylaying the clusters...')
    dist = 1 - cosine_similarity(X)
    """Model is a KMeans clustering with n_clusters and X_vec is the Tfidf 
    matrix as input"""
    # Multidimensional scaling to convert the dist matrix into a 2-dimensional array 
    MDS()
    # n_components=2 to plot results in a two-dimensional plane
    # "precomputed" because the  distance matrix dist is already computed
    # `random_state` set to 1 so that the plot is reproducible.
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    pos = mds.fit_transform(dist)  # shape (n_components, n_samples)
    xs, ys = pos[:, 0], pos[:, 1]

    #set up colors per clusters using a dict
    # #1b9e77 (green) #d95f02 (orange) #7570b3 (purple) #e7298a (pink)
    cluster_colors = {0: '#f0140c', 1: '#ad7144', 2: '#f5b92f', 3: '#e8f007', 4: '#88e014', \
                  5:"#0eedb2", 6:"#0dafdb",\
                  7:"#1330ed", 8:"#9a09e8", 9:"#e605b1", \
                  10:"#c4a29d", 
                  11:"#695232", 12:"#f7f088", 13:"#7e8778",
                  14:"#7dada2", 15:"#628cf5"}
    cluster_colors = { k:cluster_colors[k] for k in range(n_clusters)}
    #set up cluster names using a dict
    #create data frame that has the result of the MDS plus the cluster numbers and titles
    tmp_df = pd.DataFrame(dict(x=xs, y=ys, label=model.labels_))

    #group by cluster
    groups = tmp_df.groupby('label')

    # set up plot
    fig, ax = plt.subplots(figsize=(17, 9)) # set size
    ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

    #iterate through groups to layer the plot
    #note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
    for name, group in groups:
        ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
        #    label=cluster_names[name], 
            color=cluster_colors[name], 
            mec='none')
        ax.set_aspect('auto')
        ax.tick_params(axis= 'x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
              top=False,         # ticks along the top edge are off
            labelbottom=False)
        ax.tick_params(axis= 'y',         # changes apply to the y-axis
            which='both',      # both major and minor ticks are affected
            left=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelleft=False)
    
     #ax.legend(numpoints=1)  #show legend with only 1 point
    plt.show() #show the plot
    print('Done')
    
def top_terms_by_cluster(model, true_k, vectorizer):

    # get the number of clusters
    #true_k = np.unique(labels_list).shape[0]
    print("Top terms per cluster:")

        # get the cluster center of each cluster 

    # argsort() return the index of each dimension in the cluster center and sort them in increasing value order
    # [:, ::-1] reverts the argsort() list to place the indices with highest value first (decreasing order)
    
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]

    # terms maps a vectorizer index to the corresponding token
    terms = vectorizer.get_feature_names_out()

    # for each cluster
    for i in range(true_k):
        print("Cluster %d:" % i, end='')
        # print out the token of the centroid (order by decreasing tf-idf value)
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print('\n')
        




