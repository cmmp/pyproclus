# pyproclus
A python implementation of PROCLUS: PROjected CLUStering algorithm.

You will need [NumPy](http://www.numpy.org/) and [SciPy](http://www.scipy.org/) to run the program.

For running the examples you will also need [Matplotlib](http://matplotlib.org/).

Check out the paper [here](http://dl.acm.org/citation.cfm?id=304188).

One of the evaluation measures is written in [Cython](http://cython.org/) for efficiency. The generated C code is already included in this distribution, along with a compiled 64-bit linux shared library. This is not required to run the algorithm, only for computing the [Adjusted Rand Index](http://en.wikipedia.org/wiki/Rand_index#Adjusted_Rand_index). To build the shared library simply type `make` on a Unix system.

Reference:

Charu C. Aggarwal, Joel L. Wolf, Philip S. Yu, Cecilia Procopiuc, and Jong Soo Park. 1999. Fast algorithms for projected clustering. In Proceedings of the 1999 ACM SIGMOD international conference on Management of data (SIGMOD '99). ACM, New York, NY, USA, 61-72. DOI=10.1145/304182.304188 
