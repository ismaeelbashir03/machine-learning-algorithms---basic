# K Means Clustering
# data is scattered on a graph, randomly centroids are placed (k amount - how many groups)
# a straight line is drawn between centroids, in the middle of that line a point is placed
# that point then creates a line 90 degrees to the previous line
# this new line is used to show what data goes to what centroid depending on which centroid
# it has in its area. the centroids are then set to the middle of all there points, this can
# be done by getting the avg. afte this is done we repeat our previous steps: 
# (get line from each centroid to centroid, 90 degree line is created at the mid point, data i sassigned
# centroids move to middle....etc). 
# we do this until after the centroids are placed the middle of its points and no points change

# this is an unsupervised learning algorithm, this means that the data given to the model
# is not labeled or classified and the model has to classify and label the data itself

from random import random
import numpy as np
import sklearn 
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import PCA

#getting our digits
digits = load_digits()

#scaling our digits data to be between -1 and 1, from greyscale (0 - 255)
data = scale(digits.data)

pca = PCA(2)
 
#Transform the data for graphing
df = pca.fit_transform(digits.data)

#getting our y from digit targets, (used for scoring function)
y = digits.target

# getting how many clusters we need (10)
k = len(np.unique(y))

# getting how many samples we have and how many features each sample has using array shape (24,45 etc)
samples, features = data.shape

# function to score our model
def bench_k_means(estimator, name, data):
    #model is trained
    estimator.fit(data)

    #printing different test results back to us
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))

    # homogeneity: checks if each cluster contains only members of a single class
    # score: 0-1, 1 is the best
    
    # completeness: checks if all members of a given class are assigned to the same cluster 
    # score: 0-1, 1 is the best

    return estimator

# creating our model:
# passing in our num of clusters, 
# our cluster starting pos, 
# how many times we run the algorith (because they are init randomly placed),
model = KMeans(n_clusters = k, init = 'k-means++', n_init = 30)

#using our scoring function
result = bench_k_means(model, 'model 1', data)

#---graphing---#

#results for graphing
label = result.fit_predict(df)

#import matplotlib
import matplotlib.pyplot as plt
 
#Getting the Centroids
centroids = result.cluster_centers_
u_labels = np.unique(label)
 
#plotting the results:
 
for i in u_labels:
    plt.scatter(df[label == i , 0] , df[label == i , 1] , label = i)
plt.scatter(centroids[:,0] , centroids[:,1] , s = 80, color = 'k')
plt.legend()
plt.show()

#from the graph we can see that the data has been split into 10 parts