
from __future__ import division
import numpy as np

class GMM(object):
      def __init__(self, thresh_Q = 0.00001):
          self.thresh_Q = thresh_Q

      def init_para(self):
          self.alpha = np.zeros(self.nModels, dtype=np.float)          
          self.alpha[0:self.nModels-1] = np.random.uniform(0.001, 1/self.nModels, self.nModels-1); self.alpha[self.nModels-1] = 1 - np.sum(self.alpha[0:self.nModels-1])
          self.mv    = np.zeros(self.nModels, dtype=np.float)          
          self.sigma2 = np.zeros(self.nModels, dtype=np.float) + 1 # sigma squared
          self.E_gamma = np.zeros((self.nSamples, self.nModels), dtype=np.float)
          
      def E_step(self):
          for j in xrange(self.nSamples):
             for k in xrange(self.nModels):
                p = Gaussian_p(self.mv[k], self.sigma2[k], self.y[j])
                #print "p, ", p
                self.E_gamma[j, k] = self.alpha[k] * p  
             sum_E_gamma = np.sum(self.E_gamma[j, :])
             self.E_gamma[j, :] = self.E_gamma[j, :] / sum_E_gamma
          self.E_Q = self.Q_func_val()

      def M_step(self):
          # init alpha, mv, sigma
          alpha_new = np.zeros(self.nModels, dtype=np.float)
          mv_new = np.zeros(self.nModels, dtype=np.float)
          sigma2_new = np.zeros(self.nModels, dtype=np.float)
          Q = 0.0
          E_gamma_across_samples = np.sum(self.E_gamma, axis=0)
          for k in xrange(self.nModels):
             alpha_new[k]  = np.sum(self.E_gamma[:, k]) 
             mv_new[k]     = np.sum(np.dot(self.E_gamma[:, k], self.y)) 
             sigma2_new[k]  = np.sum(np.dot(self.E_gamma[:, k], np.power(self.y - self.mv[k], 2)))
          alpha_new /= self.nSamples
          mv_new    /= E_gamma_across_samples
          sigma2_new /= E_gamma_across_samples
          self.last_alpha = self.alpha
          self.alpha = alpha_new
          self.last_mv = self.mv
          self.mv    = mv_new
          self.last_sigma2 = self.sigma2
          self.sigma2 = sigma2_new
          self.M_Q = self.Q_func_val()
     
      def Q_func_val(self):
          Q = 0.0
          for k in xrange(self.nModels): 
             for j in xrange(self.nSamples):
               Q += self.E_gamma[j, k] * np.log(self.alpha[k] + 1) + self.E_gamma[j, k] * np.log(Gaussian_p(self.mv[k], self.sigma2[k], self.y[j]) + 1)
          return Q

      def fit(self, train_X, nModels):
          self.y = train_X
          self.nSamples = np.shape(train_X)[0]; self.nModels = nModels
          self.init_para()
          itera = 0
          while 1:
             itera += 1
             self.E_step()
             self.M_step()
             # # print self.E_Q, self.M_Q
             # print "M_Q: ", self.M_Q, "E_Q: ", self.E_Q
             if itera > 10000: # or np.abs(self.M_Q - self.E_Q) <= self.thresh_Q: # EM is very easily converge to local maximum
                self.itera = itera
                break
             
def Gaussian_p(mv, sigma2, y):
    p = np.exp(-1 * np.power(y - mv, 2) / (2 * sigma2)) / np.sqrt(2 * np.pi * sigma2)
    p += np.spacing(1) # in case of p approxiamtes 0
    # print mv, sigma, y, p
    return p
		

if __name__ == "__main__":
  # 20 from N(1,4); 20 from N(10,9); 20 from N(30,16)
  data = []
  for j in xrange(600):
     m = np.random.choice(3)
     if m == 0:
         num = np.random.randn(1) * 2 + 1
     elif m == 1:
         num = np.random.randn(1) * 3 + 10
     else:
         num = np.random.randn(1) * 4 + 30
     data.append(num)
  data = np.array(data, dtype=np.float)
  print data.shape
  gmm = GMM()
  gmm.fit(data, 3)
  print "new\n"
  print gmm.alpha
  print gmm.mv
  print gmm.sigma2
  print "last\n"
  print gmm.last_alpha
  print gmm.last_mv
  print gmm.last_sigma2
  print gmm.E_Q
  print gmm.M_Q


  print gmm.itera
