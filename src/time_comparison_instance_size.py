#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 10:17:51 2018

@author: cdickens
Script to test the speed of computing two data summaries with a
given data density across varying dimensionalities.
"""
import os
import json
from pprint import pprint
import numpy as np
from scipy.linalg import clarkson_woodruff_transform
from library.srht import srht_transform
from library.countSketch import countSketch, sort_row_order
from timeit import default_timer as timer
import scipy.sparse
from scipy.sparse import random


# Read parameters from file and initialise control variables

PARAMS_PATH = "~/sparse-iterations/src/model_specs/baseline.json"
file_path = os.path.expanduser(PARAMS_PATH)
with open(file_path) as fi:
    parameters = json.load(fi)
pprint("Using parameters: {}".format(parameters))
n_samples = parameters['n_samples']
max_data_instance = parameters['max_data_instance']
min_data_instance = parameters['min_data_instance']
column_fraction = parameters['column_fraction']
density = parameters['density']
np.random.seed(parameters['rng_seed'])


instance_sizes = [2**ii for ii in range(min_data_instance,max_data_instance)]
dimensionality = [np.int(np.ceil(value/column_fraction)) for value in instance_sizes]
sketch_sizes = [10*np.int(d*np.log(d)) for d in dimensionality]

# filename for the file you want to save
inputs_filename = "input_parameters.npy"
experiment_setup = {"rows" : instance_sizes,
                    "columns" : dimensionality,
                    "sketch size" : sketch_sizes,
                    "density" : density }

countSketch_output = {"sketch time" : np.zeros(len(instance_sizes)),
                      "product time" : np.zeros(len(instance_sizes)),
                      "norms" : np.zeros(len(instance_sizes)),
                      "distortion" : np.zeros(len(instance_sizes))}

srht_output = {"sketch time" : np.zeros(len(instance_sizes)),
               "product time" : np.zeros(len(instance_sizes)),
               "norms" : np.zeros(len(instance_sizes)),
               "distortion" : np.zeros(len(instance_sizes))}

true_output = {"product time" : np.zeros(len(instance_sizes)),
               "norms" : np.zeros(len(instance_sizes))}

OUT_PATH = "~/sparse-iterations/bld/basic_experiments"
OUT_DIR = os.path.expanduser(OUT_PATH)


if not os.path.exists(os.path.join(OUT_DIR, inputs_filename)):
             #os.mkdir(os.path.join(OUT_DIR, inputs_filename))
             np.save(os.path.join(OUT_DIR, inputs_filename), experiment_setup)
else:
        np.save(os.path.join(OUT_DIR, inputs_filename), experiment_setup)






# Dry run for jit
countSketch(np.random.randint(1000,size=1000), np.random.randn(1000), 1000, 10)

for iter_no in range(n_samples):

    for num_rows in instance_sizes:

        dimension = dimensionality[instance_sizes.index(num_rows)]
        reduced_rows = sketch_sizes[instance_sizes.index(num_rows)]
        matrix = random(num_rows, dimension, density)
        random_vector = np.random.randn(dimension)
        idx = instance_sizes.index(num_rows)
        #idy = idx+1  # just doubling up so idx goes 0,2,4 etc and idy fills gaps.
    
        # No sketching for comparison
        start = timer()
        Ax = matrix@random_vector
        end = timer()
        true_norm = np.linalg.norm(Ax, ord=2)**2
        mat_vector_prod_time = end - start
        true_output["product time"][idx] += mat_vector_prod_time
        true_output["norms"][idx] += true_norm
        print("True mat-vec time on ({},{}): {}".format(num_rows,dimension,mat_vector_prod_time))


        # SRHT sketching
        start = timer()
        S_A = srht_transform(input_matrix=matrix.todense(), sketch_size=reduced_rows)
        sketch_time = timer() - start
        srht_output["sketch time"][idx] += sketch_time
        print("SRHT sketch time on ({0:2d},{1:3d}): {2:4f}".format(num_rows, dimension,\
                                                               sketch_time))
        start = timer()
        sketch_product = S_A@random_vector
        product_time = timer() - start
        srht_output["product time"][idx] += product_time
        sketch_norm = np.linalg.norm(sketch_product, ord=2)**2
        srht_output["norms"][idx] = sketch_norm
        srht_output["distortion"][idx] += (sketch_norm/true_norm)
        print("SRHT prod time  on ({0:2d},{1:3d}): {2:4f}".format(num_rows, dimension, product_time))
        print("SRHT distortion on ({0:2d},{1:3d}): {2:4f}".format(num_rows, dimension,sketch_norm/true_norm))

    
        # CWT sketching
        tidy_data = sort_row_order(matrix)

        start = timer()
        hashed_rows1, sketched_data1 = countSketch(tidy_data[0], tidy_data[2], matrix.nnz, reduced_rows)
        sketch_time = timer() - start
        countSketch_output["sketch time"][idx] += sketch_time 
        print("CWT sketch time on ({0:2d},{1:3d}): {2:4f}".format(num_rows, dimension,\
                                                                         sketch_time))
        S_A = scipy.sparse.coo_matrix((sketched_data1, (hashed_rows1,matrix.col))).todense()
        start = timer()
        sketch_product = S_A@random_vector
        product_time = timer() - start
        countSketch_output["product time"][idx] += product_time
        sketch_norm = np.linalg.norm(sketch_product, ord=2)**2
        countSketch_output["distortion"][idx] += (sketch_norm/true_norm)
        print("CWT prod time  on ({0:2d},{1:3d}): {2:4f}".format(num_rows, dimension, product_time))
        print("CWT distortion on ({0:2d},{1:3d}): {2:4f}".format(num_rows, dimension,sketch_norm/true_norm))



# Iteration wasn't working so do manually
countSketch_output = {k : v/n_samples for k,v in countSketch_output.items()}
srht_output = {k : v/n_samples for k,v in srht_output.items()}
true_output = {k : v/n_samples for k,v in true_output.items()}

#for the_dict in [countSketch_output, srht_output, true_output]:
#    print(the_dict)
#    the_dict = {k : v/n_samples for k,v in the_dict.items()}
#    print(the_dict)    

pprint("CWT: {}".format(countSketch_output), width=1)
pprint("SRHT: {}".format(srht_output),width=1)
pprint("No sketch: {}".format(true_output), width=1)

countSketch_fname = "countSketch_output.npy"
srht_fname = "srht_output.npy"
no_sketch_fname = "no_sketch.npy"

for fname in [countSketch_fname, srht_fname, no_sketch_fname]:
    if fname == srht_fname:
         savedict = srht_output
    elif fname == countSketch_fname:
        savedict = countSketch_output
    else:
        savedict = true_output
        
    if not os.path.exists(os.path.join(OUT_DIR,fname)):
        #os.mkdir(os.path.join(OUT_DIR, inputs_filename))
        np.save(os.path.join(OUT_DIR, fname), savedict)
else:
    np.save(os.path.join(OUT_DIR, fname), savedict)
