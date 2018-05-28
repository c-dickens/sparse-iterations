import numpy as np
import numpy.random as npr
import scipy.sparse as sparse
from scipy.sparse import coo_matrix
import numba
from numba import jit

@jit(nopython=True)
def countSketch_elt_stream(matrixA, sketch_size):
    n,d = matrixA.shape
    sketch = np.zeros((sketch_size,d))
    nonzero_rows, nonzero_cols = np.nonzero(matrixA)
    hashedIndices = np.random.choice(sketch_size, n, replace=True)
    randSigns = np.random.choice(2, n, replace=True) * 2 - 1
    for ii,jj in zip(nonzero_rows,nonzero_cols):
        bucket = hashedIndices[ii]
        sketch[bucket, jj] += randSigns[ii]*matrixA[ii,jj]
    #for ii in range(nonzero_rows):
    #    for jj in range(nonzero_cols):
    #        bucket = hashedIndices[ii]
    #        sketch[bucket, jj] += randSigns[ii]*matrixA[ii,jj]
    # Above commented code might be marginally faster
    return sketch





@jit(nopython=True) # comment this if want just numpy
def countSketch(input_rows, input_data,
                       input_nnz,
                       sketch_size, seed=None):
    '''
    input_rows: row indices for data (can be repeats)
    input_data: values seen in row location,
    input_nnz : number of nonzers in the data (can replace with
    len(input_data) but avoided here for speed)
    sketch_size: int
    seed=None : random seed
    '''
    hashed_rows = np.empty(input_rows.shape,dtype=np.int32)
    current_row = 0
    hash_val = npr.choice(sketch_size)
    sign_val = npr.choice(np.array([-1.0,1.0]))
    #print(hash_val)
    hashed_rows[0] = hash_val
    #print(hash_val)
    for idx in np.arange(input_nnz):
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

@jit(nopython=True) # comment this if want vanilla
def countSketchStreaming(matrixA, m):
    n, d = matrixA.shape
    sketch = np.zeros((m, d))
    hashedIndices = np.random.choice(m, n, replace=True)
    randSigns = np.random.choice(2, n, replace=True) * 2 - 1
    for j in range(n):
        a = matrixA[j, :]
        h = hashedIndices[j]
        g = randSigns[j]
        sketch[h,:] += g * a
    return sketch


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


    matrix = sparse.random(50000, 1000, 1)
    x = np.random.randn(matrix.shape[1])
    true_norm = np.linalg.norm(matrix@x,ord=2)**2
    cov_mat = matrix.T.toarray()@matrix.toarray()
    matrix_norm = np.linalg.norm(cov_mat, ord='fro')**2
    tidy_data =  sort_row_order(matrix)

    sketch_size = 2500
    start = time.time()
    #hashed_rows, sketched_data = countSketch(tidy_data[0],\
    #                                        tidy_data[2], matrix.nnz,sketch_size)
    duration_slow = time.time() - start
    #S_A = sparse.coo_matrix((sketched_data, (hashed_rows,matrix.col)))
    #approx_norm_slow = np.linalg.norm(S_A@x,ord=2)**2
    #rel_error_slow = approx_norm_slow/true_norm
    #print("Sketch time: {}".format(duration_slow))
    start = time.time()
    #hashed_rows, sketched_data = countSketch(tidy_data[0],\
    #                                        tidy_data[2], matrix.nnz,sketch_size)
    duration = time.time() - start
    #print("Sketch time: {}".format(duration))
    #S_A = sparse.coo_matrix((sketched_data, (hashed_rows,matrix.col))).toarray()
    #approx_norm = np.linalg.norm(S_A@x,ord=2)**2
    #rel_error = approx_norm/true_norm
    #print("Relative norms: {}".format(approx_norm/true_norm))
    #sketched_cov_mat = S_A.T@S_A
    start = time.time()
    #approx_matrix_norm = np.linalg.norm(sketched_cov_mat, ord='fro')**2
    #matrix_duration = time.time() - start
    #diff = np.linalg.norm(sketched_cov_mat - cov_mat, ord='fro')
    #mat_rel_error = diff/(3*matrix_norm)
    #mat_rel_error = approx_matrix_norm / matrix_norm

    # Streaming approach
    # Dry run
    A = matrix.toarray()
    S_A = countSketchStreaming(A, sketch_size)
    start = time.time()
    S_A = countSketchStreaming(A, sketch_size)
    second_time = time.time() - start
    new_approx_norm = np.linalg.norm(S_A@x, ord=2)**2
    new_rel_error = new_approx_norm/true_norm
    sketch_mat_norm = np.linalg.norm(S_A.T@S_A,ord=2)**2
    new_mat_rel_error = sketch_mat_norm/matrix_norm

    print("Time: {}".format(second_time))
    print("Mat-vec rel error: {}".format(new_rel_error))
    print("Mat-mat rel error: {}".format(new_mat_rel_error))
    #print(tabulate([[duration_slow, rel_error_slow, 'Yes'],
    #                [duration, rel_error, 'No'],
    #                [matrix_duration, mat_rel_error, 'No'],
    #                [second_time, new_rel_error, 'Yes'],
    #                [second_time, new_mat_rel_error, 'Yes']],
    #                headers=['Sketch Time', 'Relative Error', 'Dry Run'],
    #                tablefmt='orgtbl'))
