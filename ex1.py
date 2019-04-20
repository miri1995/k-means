

import numpy
import numpy as np
import scipy
import scipy.io as sio
import matplotlib.pyplot as plt

from scipy.misc import imread


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
        return np.asarray([[0.        , 0.        , 0.        ],
                            [0.07843137, 0.06666667, 0.09411765]])
    elif K == 4:
        return np.asarray([[0.72156863, 0.64313725, 0.54901961],
                            [0.49019608, 0.41960784, 0.33333333],
                            [0.02745098, 0.        , 0.        ],
                            [0.17254902, 0.16862745, 0.18823529]])
    elif K == 8:
        return np.asarray([[0.01568627, 0.01176471, 0.03529412],
                            [0.14509804, 0.12156863, 0.12941176],
                            [0.4745098 , 0.40784314, 0.32941176],
                            [0.00784314, 0.00392157, 0.02745098],
                            [0.50588235, 0.43529412, 0.34117647],
                            [0.09411765, 0.09019608, 0.11372549],
                            [0.54509804, 0.45882353, 0.36470588],
                            [0.44705882, 0.37647059, 0.29019608]])
    elif K == 16:
        return np.asarray([[0.61568627, 0.56078431, 0.45882353],
                            [0.4745098 , 0.38039216, 0.33333333],
                            [0.65882353, 0.57647059, 0.49411765],
                            [0.08235294, 0.07843137, 0.10196078],
                            [0.06666667, 0.03529412, 0.02352941],
                            [0.08235294, 0.07843137, 0.09803922],
                            [0.0745098 , 0.07058824, 0.09411765],
                            [0.01960784, 0.01960784, 0.02745098],
                            [0.00784314, 0.00784314, 0.01568627],
                            [0.8627451 , 0.78039216, 0.69803922],
                            [0.60784314, 0.52156863, 0.42745098],
                            [0.01960784, 0.01176471, 0.02352941],
                            [0.78431373, 0.69803922, 0.60392157],
                            [0.30196078, 0.21568627, 0.1254902 ],
                            [0.30588235, 0.2627451 , 0.24705882],
                            [0.65490196, 0.61176471, 0.50196078]])
    else:
        print('This value of K is not supported.')
        return None




def calculateCentroide(arrOfCentroids,X,k,it):
    sumR = 0
    sumG = 0
    sumB = 0
    finalImage=[]
    clusters_dict= {new_list: [] for new_list in range(k)}  #init key of dict. number of key=k

    for j in range(0,len(X)):
        dist_dict = {new_list2: [] for new_list2 in range(k)}
        for t in range(0,k):
            dist = numpy.linalg.norm(X[j] - arrOfCentroids[t]) ** 2
            dist_dict[t].append(dist)
        MIN=min(dist_dict, key=lambda x: dist_dict.get(x))

        clusters_dict[MIN].append(X[j])  # num of cluster k and add to him pixel x[j]

        dist_dict.clear()

    for t in range(0, k):
        rows=clusters_dict[t]
        for i in range(0,len(rows)):
            sumR += rows[i][0]
            sumG += rows[i][1]
            sumB += rows[i][2]
        average1 = sumR /len(rows)
        average2 = sumG / len(rows)
        average3 = sumB /len(rows)
        #update centroids
        arrOfCentroids[t][0] = average1
        arrOfCentroids[t][1] = average2
        arrOfCentroids[t][2] = average3
        sumR=0
        sumG=0
        sumB=0

    return arrOfCentroids



def print_cent(cent):
    if type(cent) == list:
        cent = np.asarray(cent)
    if len(cent.shape) == 1:
        return ' '.join(str(np.floor(100*cent)/100).split()).replace('[ ', '[').replace('\n', ' ').replace(' ]',']').replace(' ', ', ')
    else:
        return ' '.join(str(np.floor(100*cent)/100).split()).replace('[ ', '[').replace('\n', ' ').replace(' ]',']').replace(' ', ', ')[1:-1]

def load():
    path = 'dog.jpeg'
    A = imread(path)
    A = A.astype(float) / 255.
    img_size = A.shape
    X = A.reshape(img_size[0] * img_size[1], img_size[2])
    return X

def main():

    X=load()
    twoCentroids = init_centroids(X, 2)
    fourCentroids = init_centroids(X, 4)
    eightCentroids = init_centroids(X, 8)
    sixteenCentroids = init_centroids(X, 16)

    #print the new centroide
    print("k=2:")
    print('iter'+(str(0)).join(' :'),print_cent(twoCentroids))
    for i in range(1,11):
        newcent=calculateCentroide(twoCentroids,X,2,i)
        print('iter'+str(i).join(' :'),print_cent(newcent))
    print("k=4:")
    print('iter'+str(0).join(' :'), print_cent(fourCentroids))
    for i in range(1, 11):
        newcent=calculateCentroide(fourCentroids,X,4,i)
        print('iter'+str(i).join(' :'), print_cent(newcent))
    print("k=8:")
    print('iter'+str(0).join(' :'),print_cent(eightCentroids))
    for i in range(1, 11):
        newcent=calculateCentroide(eightCentroids,X,8,i)
        print('iter'+str(i).join(' :'),print_cent(newcent))
    print("k=16:")
    print('iter'+str(0).join(' :'), print_cent(sixteenCentroids))
    for i in range(1, 11):
        newcent=calculateCentroide(sixteenCentroids,X,16,i)
        print('iter'+str(i).join(' :'), print_cent(newcent))


main()