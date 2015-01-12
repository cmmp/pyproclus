import proclus as prc
import plotter
import arffreader as ar
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import adjrand

# Test to see if there is a correlation between the
# proclus objective function and the adjusted rand index

X, sup = ar.readarff("data/highdataproclus.arff")

NRUNS = 15

objs = np.zeros(NRUNS) # objective function results
adjs = np.zeros(NRUNS) # adjusted rand index results

print "Beginning runs..."

for i in xrange(NRUNS):
	rseed = np.random.randint(low = 0, high = 1239831)
	print ">>> Run %d/%d using seed %d" % (i + 1, NRUNS, rseed)
	M, D, A = prc.proclus(X, k = 7, l = 2, seed = rseed)
	objs[i] = prc.evaluateClusters(X, A, D, M)
	adjs[i] = adjrand.computeAdjustedRandIndex(A, sup)

print "Finished runs..."

sidx = np.argsort(objs)

plt.clf()
plt.plot(objs[sidx], adjs[sidx], 'bo-')
plt.xlabel("Objective function results")
plt.ylabel("Adjusted Rand Index")
plt.show()
plt.draw()

print "Pearson correlation: %.4f" % pearsonr(objs[sidx], adjs[sidx])[0]
