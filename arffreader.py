# A simple arff reader
#
# Author: Cassio M. M. Pereira

import numpy as np
import re

def readarff(fileName, supervision = True):
	""" Basic reader for arff files. If supervision = true,
		treat the last dimension as the supervision and return
		it separately.
	"""

	f = open(fileName, 'r')
	ndim = len(re.findall("@attribute", f.read())) # try to guess the number of dimensions
	f.close()

	if supervision:
		X = customload(fileName, usecols = np.arange(ndim - 1))
		try:
			supervision = customload(fileName, usecols = [ndim - 1]).astype(np.int)
		except:
			supervision = customload(fileName, usecols = [ndim - 1], dtype = np.str)
			supu = np.unique(supervision)
			supervision = np.array([np.where(supu == i)[0] for i in supervision]).astype(np.int)
		return (X, supervision)
	else:
		return customload(fileName, usecols = None)

def customload(fileName, usecols, dtype = np.double):
	return np.loadtxt(fileName, dtype = dtype, comments = '@', delimiter = ',', usecols = usecols)
