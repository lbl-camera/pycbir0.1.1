'''
Fisher Vector based on 
Image Classification with the Fisher Vector: Theory and Practice
Jorge Sánchez · Florent Perronnin · Thomas Mensink · Jakob Verbeek
Int Journal Computer Vision (2013)
@author: romuere
'''


"""
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import Normalizer

def gaussian(gmm,data,k):
    D = len(data)
    part_1 = 1/((2*np.pi)**(D/2)*np.linalg.det(gmm.covariances_[k])**(0.5)) 
    part_2 = np.matmul(data-gmm.means_[k],np.linalg.inv(gmm.covariances_[k]))
    part_3 = data-gmm.means_[k]
    part_4 = np.matmul(part_2,part_3)
    part_4 = np.exp(-0.5*part_4)
    
    return part_1*part_4

def gamma(gmm,data,k):
    part_1 = gmm.weights_[k]*gaussian(gmm,data,k)
    part_2 = np.zeros((gmm.n_components))
    
    for id_ in range(gmm.n_components):
        part_2[id_] = gmm.weights_[id_]*gaussian(gmm,data,id_)
    
    return (part_1/np.sum(part_2))        

def gradients(gmm,data):
    T = data.shape[0]
    shape = data.shape[1]
    gradient_alpha = np.zeros(gmm.n_components)    
    gradient_mu = np.zeros((gmm.n_components,shape))
    gradient_sigma = np.zeros((gmm.n_components,shape))
    for id_k in range(gmm.n_components):
        part_1 = 1/np.sqrt(gmm.weights_[id_k])
        g_a = 0
        g_m = np.zeros(shape)
        g_s = np.zeros(shape)
        for id_t in range(T):
            g = gamma(gmm,data[id_t,:],id_k)-gmm.weights_[id_k]
            
            g_a += g
            g_m = np.add(g*((data[id_t,:]-gmm.means_[id_k])/np.diag(gmm.covariances_[id_k])),g_m) 
            g_s = np.add((g*(1/(2**0.5)))*((((data[id_t,:]-gmm.means_[id_k])**2)/
                                            (np.diag(gmm.covariances_[id_k])**2)) - 1),g_s)

        gradient_alpha[id_k] = part_1*g_a
        gradient_mu[id_k,:] = part_1*g_m
        gradient_sigma[id_k,:] = part_1*g_s
    
    return gradient_alpha,gradient_mu,gradient_sigma


def fisher(features_train, features_test, labels_test, n_comp, feature_size):

    
    #n_comp is the number of gaussians to be used
    features_train = features_train.reshape((3*len(features_train),feature_size))    
    
    gmm = GaussianMixture(n_components=n_comp,max_iter = 1000,random_state=0)
    gmm.fit(features_train,labels_test)
    #gmm.predict(features_train)
    print(gmm.n_iter_)
    feature_vector = np.zeros((len(features_test),(feature_size*2 + 1)*n_comp))
    
    for id_ft,ft in enumerate(features_test):
        ft = ft.reshape((len(ft)/feature_size,feature_size))
        a,b,c = gradients(gmm,ft)
        fv = np.concatenate((a,b.reshape(-1),c.reshape(-1)))
        fv = feature_vector.reshape(1,-1)
        norm = Normalizer()
        feature_vector[id_ft,:] = norm.fit_transform(fv)[0]
    
    return feature_vector
"""


import numpy as np
from sklearn.mixture import GMM
from sklearn.mixture import GaussianMixture
import os
import csv
import pickle

def fisher_vector(xx, gmm):
    """Computes the Fisher vector on a set of descriptors.
    Parameters
    ----------
    xx: array_like, shape (N, D) or (D, )
        The set of descriptors
    gmm: instance of sklearn mixture.GMM object
        Gauassian mixture model of the descriptors.
    Returns
    -------
    fv: array_like, shape (K + 2 * D * K, )
        Fisher vector (derivatives with respect to the mixing weights, means
        and variances) of the given descriptors.
    Reference
    ---------
    J. Krapac, J. Verbeek, F. Jurie.  Modeling Spatial Layout with Fisher
    Vectors for Image Categorization.  In ICCV, 2011.
    http://hal.inria.fr/docs/00/61/94/03/PDF/final.r1.pdf
    """
    xx = np.atleast_2d(xx)
    N = xx.shape[0]

    # Compute posterior probabilities.
    Q = gmm.predict_proba(xx)  # NxK

    # Compute the sufficient statistics of descriptors.
    Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
    Q_xx = np.dot(Q.T, xx) / N
    Q_xx_2 = np.dot(Q.T, xx ** 2) / N

    # Compute derivatives with respect to mixing weights, means and variances.
    d_pi = Q_sum.squeeze() - gmm.weights_
    d_mu = Q_xx - Q_sum * gmm.means_
    d_sigma = (
        - Q_xx_2
        - Q_sum * gmm.means_ ** 2
        + Q_sum * gmm.covariances_
        + 2 * Q_xx * gmm.means_)

    # Merge derivatives into a vector.
    return np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))

def fisher(features_test, n_comp, feature_size, file_fisher, file_train = ''):
    
    if not(os.path.isfile(file_fisher)):
        
        reader = csv.reader(open(file_train),delimiter=',')
        x = list(reader)
        features_train = np.array(x).astype(dtype = np.float64)
    
        K = n_comp
        N = feature_size
        features_train = features_train.reshape(((len(features_test[0])/feature_size)*len(features_train),feature_size))

        gmm = GaussianMixture(n_components=K, covariance_type='diag')
        gmm.fit(features_train)

        
        with open(file_fisher, 'wb') as handle:   
            pickle.dump(gmm, handle)
    else:
        with open(file_fisher, 'rb') as handle:
            gmm = pickle.load(handle)
        
    # Short demo.
    
    feature_vector = np.zeros((len(features_test),(feature_size*2 + 1)*n_comp))
    
    for id_ft,ft in enumerate(features_test):
        ft = ft.reshape((len(ft)/feature_size,feature_size))
        fv = fisher_vector(ft, gmm)
        feature_vector[id_ft,:] = fv

    return feature_vector