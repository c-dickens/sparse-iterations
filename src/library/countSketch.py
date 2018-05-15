import numpy as np
import numpy.random as npr
import scipy.sparse as sparse
from scipy.sparse import coo_matrix
import numba
from numba import jit


@jit(nopython=True) # comment this if want vanilla
def countSketch(input_rows, input_data,
                       input_nnz,
                       sketch_size, seed=None):
    '''
    input_matrix: sparse - coo_matrix type
    sketch_size: int
    seed=None : random seed
    '''
    hashed_rows = np.empty(input_rows.shape,dtype=np.int32)
    current_row = 0
    hash_val = npr.choice(sketch_size)
    sign_val = npr.choice(np.array([-1.0,1.0]))
    hashed_rows[0] = hash_val
    for idx in range(input_nnz):
        row_id = input_rows[idx]
        data_val = input_data[idx]
        if row_id == current_row:
            hashed_rows[idx] = hash_val
            input_data[idx] = sign_val*data_val
        else:
            # make new hashes
            hash_val = npr.choice(sketch_size)
            sign_val = npr.choice(np.array([-1.0,1.0]))
            hashed_rows[idx] = hash_val
            input_data[idx] = sign_val*data_val
    return hashed_rows, input_data
3


def sort_row_order(input_data):
    sorted_row_column = np.array((input_data.row,
                                  input_data.col,
                                  input_data.data))

    idx  = np.argsort(sorted_row_column[0])
    sorted_rows = np.array(sorted_row_column[0,idx], dtype=np.int32)
    sorted_cols = np.array(sorted_row_column[1,idx], dtype=np.int32)
    sorted_data = np.array(sorted_row_column[2,idx], dtype=np.float64)
    return sorted_rows, sorted_cols, sorted_data


if __name__=="__main__":
    import time
    from tabulate import tabulate


    matrix = sparse.random(1000, 50, 0.1)
    x = np.random.randn(matrix.shape[1])
    true_norm = np.linalg.norm(matrix@x,ord=2)**2
    tidy_data =  sort_row_order(matrix)

    sketch_size = 300
    start = time.time()
    hashed_rows, sketched_data = countSketch(tidy_data[0],\
                                            tidy_data[2], matrix.nnz,sketch_size)
    duration_slow = time.time() - start
    S_A = sparse.coo_matrix((sketched_data, (hashed_rows,matrix.col)))
    approx_norm_slow = np.linalg.norm(S_A@x,ord=2)**2
    rel_error_slow = approx_norm_slow/true_norm
    #print("Sketch time: {}".format(duration_slow))
    start = time.time()
    hashed_rows, sketched_data = countSketch(tidy_data[0],\
                                            tidy_data[2], matrix.nnz,sketch_size)
    duration = time.time() - start
    #print("Sketch time: {}".format(duration))
    S_A = sparse.coo_matrix((sketched_data, (hashed_rows,matrix.col)))
    approx_norm = np.linalg.norm(S_A@x,ord=2)**2
    rel_error = approx_norm/true_norm
    #print("Relative norms: {}".format(approx_norm/true_norm))
    print(tabulate([[duration_slow, rel_error_slow, 'Yes'],
                    [duration, rel_error, 'No']],
                    headers=['Sketch Time', 'Relative Error', 'Dry Run'],
                    tablefmt='orgtbl'))
