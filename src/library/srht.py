#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 10:37:19 2018

Class instantiation of an efficient SRHT sketch.

@author: cdickens
"""

import numpy as np
from hadamard import fastwht


def shift_bit_length(x):
    '''Given int x find next largest power of 2.
    If x is a power of 2 then x is returned '''
    return 1<<(x-1).bit_length()


def srht_transform(input_matrix, sketch_size, seed=None):
    '''Generate a sketching matrix S to reduce the sample count (i.e. sketching
    on the LHS of data) via the Subsampled Hadamard Transform.
    
    Given an input_matrix ``A`` of size ``(n, d)``, compute a matrix ``A'`` of
    size (sketch_size, d) which holds:
    .. math:: ||Ax|| = (1 \pm \epsilon)||A'x||
    with high probability. [1]
    The error is related to the number of rows of the sketch and it is bounded
    
    tbc
    
    
    Parameters
    ----------
    input_matrix: array_like
        Input matrix, of shape ``(n, d)``.
    sketch_size: int
        Number of rows for the sketch.
    seed : None or int or `numpy.random.RandomState` instance, optional
        This parameter defines the ``RandomState`` object to use for drawing
        random variates.
        If None (or ``np.random``), the global ``np.random`` state is used.
        If integer, it is used to seed the local ``RandomState`` instance.
        Default is None.
    Returns
    -------
    S_A : array_like
        Sketch of the input matrix ``A``, of size ``(sketch_size, d)``.
    
    Notes
    -------
    This implementation of the SRHT is fast up to the line which requires the 
    copying of the fastmat PRODUCT type to a NumPy array [2].
    The fastmat library is used to quickly compute D*A and the fht library is 
    used to compute the product H(DA) quickly by exploiting the FFT [3].
        
    References
    -------------
    [1] - https://arxiv.org/abs/1411.4357
    [2] - https://github.com/EMS-TU-Ilmenau/fastmat
    [3] - https://github.com/nbarbey/fht
    
    
    
    '''
    # new_matrix_length == input_matrix.shape[0] when input is power of 2 long.
    '''
    #new_matrix_length = shift_bit_length(input_matrix.shape[0])
    original_input_length = input_matrix.shape[0]
    A = pad_zeros(input_matrix)
    input_length_with_zeros = A.shape[0]
    #D = sparse.diags(np.random.choice([1,-1], num_rows_data),0)
    D = np.diag(np.random.choice([1,-1], input_length_with_zeros),0)
    print("Shape of D: {}".format(D.shape))
    D = fm.Diag(np.random.choice([1,-1], input_length_with_zeros))
    prod = (D*fm.Matrix(A)).reference() # CONVERSION TO NDARRAY
    transform = fht.fht(prod)
    '''
    
    '''
    if shift_bit_length(num_rows_data) == num_rows_data:
        D = sparse.diags(np.random.choice([1,-1], num_rows_data),0)
        D = np.diag(np.random.choice([1,-1], num_rows_data),0)
        D = fm.Diag(np.random.choice([1,-1], num_rows_data))
        prod = (D*fm.Matrix(input_matrix)).reference() # CONVERSION TO NDARRAY
        transform = fht.fht(prod)
    else:
        input_matrix = pad_zeros(input_matrix)
        num_rows_data = input_matrix.shape[0] # Overwrite for new input matrix shape
        #D = sparse.diags(np.random.choice([1,-1], input_matrix.shape[0]),0)
        D = np.diag(np.random.choice([1,-1], input_matrix.shape[0]),0)
    '''
    nrows = input_matrix.shape[0]
    diag = np.random.choice([1,-1], nrows)
    diag = diag[:,None]
    #diag = np.tile(diag, input_matrix.shape[1])
    signed_mat = np.multiply(diag,input_matrix)
    S = shift_bit_length(nrows)*fastwht(signed_mat)
    

    #transform = fht.fht(prod)
    
     # sample sketch size
    sample = np.random.choice(nrows, sketch_size, replace=False) 
    sample.sort()
    #print(sample)
    # number from num_rows_data universe
    
    S = S[sample]
    S = (sketch_size)**(-0.5)*S
    return S

if __name__== "__main__":
    import time
    from tabulate import tabulate

    seed = np.random.seed(1)

    A = np.random.randn(1000, 50)
    x = np.random.randn(A.shape[1])
    true_norm = np.linalg.norm(A@x,ord=2)**2

    
    start = time.time()
    S_A = srht_transform(input_matrix=A, sketch_size=300, seed=seed)
    duration =  time.time() - start
    approx_norm = np.linalg.norm(S_A@x, ord=2)**2
    relative_error = approx_norm / true_norm

    print(tabulate([["SRHT", relative_error, duration]],
                   headers=['Sketch type', 'Relative Error', 'Time'],
                   tablefmt='orgtbl'))


    
