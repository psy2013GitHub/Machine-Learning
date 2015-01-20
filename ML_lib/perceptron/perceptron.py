
import numpy as np


class perception(object):
	def __init__(self,eta):
        self.eta = eta

    def fit(self,train_set,train_label):
    	'''
    	train_set,  np array
    	train_label, np array
    	'''
    	nSample, nFeature = np.shape(train_set)
    	# init w, b
        w = np.zeros([nFeature,1])
        b = 0.0
        # update w, b
        nEval = 0
        while 1:
        	if nEval > 100000:
        		print "Linear Unseparable"
        		return
            nWrongSeparate = 0
        	for i in xrange(nSample):
        		x = train_set[i,:]
        		y = train_label[i]
        	    # sanple i 
        	    if y * (w * x + b) < 0:
        	    	nWrongSeparate += 1
        	    	w += self.eta * y * x
        	        b += self.eta * y
            if nWrongSeparate == 0:
               break
            nEval += 1

        self.w = w
        self.b = b

    def predict(test_sample):
        y = self.w * test_sample + self.b
        if y > 0:
          return 1
        else:
          return -1    


