import numpy

import numpy as np
from scipy.misc import imread
from collections import defaultdict

# data preperation (loading, normalizing, reshaping)
path = 'dog.jpeg'
A = imread(path)
A = A.astype(float) / 255.
img_size = A.shape
X = A.reshape(img_size[0] * img_size[1], img_size[2])
#f = A[1,1,0]
h = X.shape[0]
w = X.shape[1]


# import numpy as np


def init_centroids(X, K):
    """
    Initializes K centroids that are to be used in K-Means on the dataset X.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Samples, where n_samples is the number of samples and n_features is the number of features.
    K : int
        The number of centroids.

    Returns
    -------
    centroids : ndarray, shape (K, n_features)
    """
    if K == 2:
        return np.asarray([[0., 0., 0.],
                           [0.07843137, 0.06666667, 0.09411765]])
    elif K == 4:
        return np.asarray([[0.72156863, 0.64313725, 0.54901961],
                           [0.49019608, 0.41960784, 0.33333333],
                           [0.02745098, 0., 0.],
                           [0.17254902, 0.16862745, 0.18823529]])
    elif K == 8:
        return np.asarray([[0.01568627, 0.01176471, 0.03529412],
                           [0.14509804, 0.12156863, 0.12941176],
                           [0.4745098, 0.40784314, 0.32941176],
                           [0.00784314, 0.00392157, 0.02745098],
                           [0.50588235, 0.43529412, 0.34117647],
                           [0.09411765, 0.09019608, 0.11372549],
                           [0.54509804, 0.45882353, 0.36470588],
                           [0.44705882, 0.37647059, 0.29019608]])
    elif K == 16:
        return np.asarray([[0.61568627, 0.56078431, 0.45882353],
                           [0.4745098, 0.38039216, 0.33333333],
                           [0.65882353, 0.57647059, 0.49411765],
                           [0.08235294, 0.07843137, 0.10196078],
                           [0.06666667, 0.03529412, 0.02352941],
                           [0.08235294, 0.07843137, 0.09803922],
                           [0.0745098, 0.07058824, 0.09411765],
                           [0.01960784, 0.01960784, 0.02745098],
                           [0.00784314, 0.00784314, 0.01568627],
                           [0.8627451, 0.78039216, 0.69803922],
                           [0.60784314, 0.52156863, 0.42745098],
                           [0.01960784, 0.01176471, 0.02352941],
                           [0.78431373, 0.69803922, 0.60392157],
                           [0.30196078, 0.21568627, 0.1254902],
                           [0.30588235, 0.2627451, 0.24705882],
                           [0.65490196, 0.61176471, 0.50196078]])
    else:
        print('This value of K is not supported.')
        return None



def calculateCentroids(numCentroids,k):
    #min = 1000
    sum1 = 0
    sum2 = 0
    sum3 = 0
    average1=0
    average2=0
    average3=0
    n = len(X)
    numOfPixel=0
    dist=[]

    #oneCluster= [[0] * 3 for i in range(n)]
    clusters=[[] * n for i in range(k)]
    for i in range(0, len(X)):
            for t in range(0,k):
                dist.append(numpy.linalg.norm(X[i] - numCentroids[t]) ** 2)
            minimum =min(dist)
            for j in range(0,len(dist)):
               if dist[j]==minimum:
                   clusters[j].append(X[i])
            dist.clear()


    for i in range(0, len(clusters)):
       clusterArr = clusters[i]
       for j in range(0,len(clusterArr)):
          sum1+=clusterArr[j][0]
          sum2 += clusterArr[j][1]
          sum3 += clusterArr[j][2]
       average1 = sum1 / len(clusterArr)
       average2 = sum2 / len(clusterArr)
       average3 = sum3 / len(clusterArr)
       numCentroids[i][0] = numpy.floor(average1 * 100) / 100
       numCentroids[i][1] = numpy.floor(average2 * 100) / 100
       numCentroids[i][2] = numpy.floor(average3 * 100) / 100
       sum1=0
       sum2=0
       sum3=0
    print(numCentroids)


twoCentroids = init_centroids(X, 2)
calculateCentroids(twoCentroids,2)
fourCentroids = init_centroids(X,4)
eightrCentroids = init_centroids(X,8)
sixteenCentroids = init_centroids(X,16)

# begin

#oneCluster = []



# #h = cluster1[0]
# #w = cluster1[1]
# #t=len(cluster1)
# #for i in range(0, len(cluster1)):
#    # for j in range (0,3):
#  #    sum1 = sum1 + cluster1[i][0]
#   #   sum2 = sum2 + cluster1[i][1]
#    #  sum3 = sum3 + cluster1[i][2]
# #average1 = sum1 / len(cluster1)
# average2 = sum2 / len(cluster1)
# average3 = sum3 / len(cluster1)
# twoCentroids[0][0] = numpy.floor(average1 * 100) / 100
# twoCentroids[0][1] = numpy.floor(average2 * 100) / 100
# twoCentroids[0][2] = numpy.floor(average3 * 100) / 100
#
#
# sum1 = 0
# sum2 = 0
# sum3 = 0
# for i in range(0, len(cluster2)):
#     sum1 = sum1 + cluster2[i][0]
#     sum2 = sum2 + cluster2[i][1]
#     sum3 = sum3 + cluster2[i][2]
# average1 = sum1 / len(cluster2)
# average2 = sum2 / len(cluster2)
# average3 = sum3 / len(cluster2)
# twoCentroids[1][0] = numpy.floor(average1 * 100) / 100
# twoCentroids[1][1] = numpy.floor(average2 * 100) / 100
# twoCentroids[1][2] = numpy.floor(average3 * 100) / 100
#
# print(twoCentroids)