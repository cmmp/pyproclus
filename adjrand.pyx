import numpy as np
cimport numpy as np

DTYPE = np.double

ctypedef np.long_t DTYPE_L
ctypedef np.double_t DTYPE_D

def computeAdjustedRandIndex(np.ndarray[DTYPE_D, ndim = 1] pred, np.ndarray[DTYPE_L, ndim = 1] expect):
	cdef double ARI = 0.0

	if pred.shape[0] != expect.shape[0]:
		raise Exception("pred and expect must have the same length.")

	cdef int N = pred.shape[0]

	cdef int M = N * (N - 1) / 2

	cdef int a = 0 # number of pairs that belong to the same class and the same cluster
	cdef b = 0 # number of pairs that belong to the same class but different clusters
	cdef c = 0 # number of pairs that belong to different classes but the same cluster

	cdef int i = 0
	cdef int j = 0

	for i in range(N - 1):
		for j in range(i + 1, N):
			if expect[i] == expect[j] and pred[i] == pred[j]:
				a += 1
			elif expect[i] == expect[j] and pred[i] != pred[j]:
				b += 1
			elif expect[i] != expect[j] and pred[i] == pred[j]:
				c += 1

	ARI =  (a - ((a+c)*(a+b) / M)) / ( (((a+c)+(a+b)) / 2.0) - ((a+c)*(a+b) / M) )

	return ARI