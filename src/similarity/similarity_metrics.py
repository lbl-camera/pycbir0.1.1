'''
Created on 15 de sep de 2016

@author: romuere
'''
import numpy as np
from scipy.spatial.distance import euclidean,cityblock,chebyshev,cosine
from scipy.stats import pearsonr,chisquare,entropy,ks_2samp
import math

def similarity_metrics(vec1,vec2,med='all'):
    """
    Function that computes the similarity/distance between two vectors
    
    Parameters
    ----------
    vec1 : list numpy array
        the first vector
    vec2 : list numpy array  
        the second vector
    med : string
        the metric that will be computed
        Minkowski and Standard Measures
            Euclidean Distance : 'ED'
            Cityblock Distance : 'CD'
            Infinity Distance : 'ID'
            Cosine Similarity : 'CS'
        Statistical Measures
            Pearson Correlation Coefficient : 'PCC'
            Chi-Square Dissimilarity : 'CSD'
            Kullback-Liebler Divergence : 'KLD'
            Jeffrey Divergence : 'JD'
            Kolmogorov-Smirnov Divergence : 'KSD'
            Cramer-von Mises Divergence : 'CMD'
            
    Returns
    -------
    similarity/distance : float
        the similarity/distance between the two vectors
    """
    distance = 0
    if med == 'ed':
        distance = euclidean(vec1,vec2)
    elif med == 'cd':
        distance = cityblock(vec1, vec2)
    elif med == 'id':
        distance = chebyshev(vec1,vec2)
    elif med == 'cs':
        distance = cosine(vec1, vec2)
    elif med == 'pcc':
        distance = dist_pearson(vec1, vec2)
    elif med == 'csd':
        distance = chisquare(vec1, vec2)[0]
    elif med == 'kld':
        distance = entropy(vec1,vec2)
    elif med == 'jd':
        distance = dist_jeffrey(vec1, vec2)
    elif med == 'ksd':
        distance = ks_2samp(vec1, vec2)[0]
    elif med == 'cmd':
        distance = dist_cvm(vec1, vec2)
    
    return distance

#---------------------------------------------------------------------------------------------------------------#
'''
Above some distance functions
'''
def average(x):
    assert len(x) > 0
    return float(sum(x)) / len(x)

def dist_pearson(x, y):
    assert len(x) == len(y)
    n = len(x)
    assert n > 0
    avg_x = average(x)
    avg_y = average(y)
    diffprod = 0
    xdiff2 = 0
    ydiff2 = 0
    for idx in range(n):
        xdiff = x[idx] - avg_x
        ydiff = y[idx] - avg_y
        diffprod += xdiff * ydiff
        xdiff2 += xdiff * xdiff
        ydiff2 += ydiff * ydiff

    return diffprod / math.sqrt(xdiff2 * ydiff2)

def dist_jeffrey(h1,h2):
    h1 = np.array(h1)
    h2 = np.array(h1)
    d = 0;
    m = (h1+h2)/2;

    for i in range(1,len(h1)):
        if (m[i]==0):
            continue;
        x1 = h1[i]*np.log10(h1[i]/m[i]);
        if (np.isnan(x1) == False):
            d = d + x1;
        x2 = h2[i]*np.log10(h2[i]/m[i]);
        if (np.isnan(x2) == False):
            d = d + x2;
    return d

def dist_cvm(h1,h2):

    y1 = np.cumsum(h1);
    y2 = np.cumsum(h2);

    d = sum((y1-y2)**2);
    return d

