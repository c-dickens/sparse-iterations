# Time and Space tradeoffs for Constrained Least Squares Regression

These experiments aim to indicate whether there is a threshold at which,
once the sparsity of a data matrix surpasses this threshold, it is
noticeably better to use a _countSketch_ transform (CWT) rather than a _Subsampled
Randomized Hadamard transform_ (SRHT).

Considerations:
- The SRHT achieves a subspace embedding with O(d/eps^2) space but the CWT
requires O(d^2/eps^2).  It will be interesting to see whether there is a cutoff
inbetween for which using the CWT loses performance compared to the SRHT and testing
at a fixed sketch size will clearly incur higher empirical variation in the distortion
between the true norm and the sketched norm for the CWT.  As a result, care needs to
be taken to set the sketch size.
