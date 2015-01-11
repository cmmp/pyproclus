from __future__ import print_function
import numpy as np

class Metadata:
	def __init__(self, relation_name, attNames):
		self.name = relation_name
		self.names = attNames

	def names(self):
		return self.names

def write_arff(instances, supervision, metadata):
        print("@relation", metadata.name)
        classVals = '{' + ",".join([a for a in np.unique(supervision)]) + '}'
        for attName in metadata.names:
                if not attName == 'class':
                        print("@attribute", attName, "numeric")
                else:
                        print("@attribute", "class", classVals)
        print("@data")
        for i in xrange(instances.shape[0]):
                for val in instances[i,:]:
                        print(val, ",", end = '', sep = '')
                print(supervision[i])
