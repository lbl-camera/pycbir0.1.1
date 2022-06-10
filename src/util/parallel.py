'''
Created on 14 de sep de 2016

@author: dani ushizima
'''
import joblib
import os
from os.path import join as pjoin


def apply_parallel(image_collection, collection_filenames, labels, func, kwargs={},
                   output_dir='', postfix=''):
    

    """Segment all images in the specified collection.
    Parameters
    ----------
    image_collection : ImageCollection
        Input images.
    func : callable
        Function to apply to each image in the collection.
    kwargs : dict
        Arguments to pass to func.
    postfix : str
        Desired postfix for processed files.
    """
    """
    def target_files():
        for fn in image_collection.files:
            fname, ext = os.path.splitext(fn)
            fname = os.path.basename(fname)
            yield os.path.relpath(pjoin(output_dir, fname + postfix + ext))

    fname_nr = zip(target_files(), range(len(image_collection)))
    fname_nr = ((fn, nr) for (fn, nr) in fname_nr if not os.path.exists(fn))
    """
    
    res = joblib.Parallel(n_jobs=-1, verbose=5)(joblib.delayed(func)(image_collection[nr], collection_filenames[nr], labels[nr], **kwargs) for nr in range(len(image_collection)))

    return res