import matplotlib.pyplot as plt
import numpy as np
import ipdb

def plotDataset(X, D):
	plt.clf()
	plt.plot(X[:,D[0]], X[:,D[1]], 'bo')
	plt.xlabel('X' + str(D[0]))
	plt.ylabel('X' + str(D[1]))
	plt.show()
	plt.draw()

def plotClustering(X, M, A, D):
	""" Plot a proclus clustering result.
		X: the data matrix
		M: medoid indices
		A: cluster assignments
		D: dimensions to plot
	"""
	plt.clf()
	plt.xlabel('X' + str(D[0]))
	plt.ylabel('X' + str(D[1]))
	d1, d2 = D
	colors = np.empty(X.shape[0], dtype = 'object')
	colors[np.where(A == -1)[0]] = "0.7" # gray for outliers
	picks = ['b', 'g', 'r', 'c', 'm', 'k', 'y']

	print 'cluster counts:'
	print np.unique(A)
	print [len(np.where(A == i)[0]) for i in np.unique(A)]

	i = 0
	for c in np.setdiff1d(np.unique(A), [-1]):
		if i >= len(picks):
			raise Exception("used more colors than i have...")
		colors[np.where(A == c)[0]] = picks[i]
		i += 1
	
	plt.scatter(X[:,d1], X[:,d2], c = colors.tolist(), marker = 'o', s = 40)
	# plot medoids as orange diamonds:
	plt.plot(X[M,d1], X[M,d2], marker = 'D', mfc = '#FFFF4D', ms = 7, ls = '')
	plt.show()
	plt.draw()