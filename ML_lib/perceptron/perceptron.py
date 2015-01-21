
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
            if y * (np.dot(w.transpose() * x + b)) < 0:
               nWrongSeparate += 1
               w += self.eta * y * x
               b += self.eta * y
            if nWrongSeparate == 0:
               break
            nEval += 1

        self.w = w
        self.b = b

    def predict(self, test_sample):
        y = np.dot(self.w.transpose() * test_sample) + self.b
        if y > 0:
          return 1
        else:
          return -1
         

if __name__ == "__main__":
   X = np.array([[0,1],[0.5,0.7],[0.5,0.2],[1,0]])
   Y = np.array([1,1,-1,-1])
   clf = perceptron(0.3)
   clf.fit(X,Y)
   print "w: ", clf.w
   print "b: ", clf.b
  

