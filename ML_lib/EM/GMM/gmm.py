
import numpy as np
from future import division



class GMM(object):
      def __init__(self, thresh_Q = 0.00001):
          self.thresh_Q = thresh_Q

      def init_para(self):
          self.alpha = np.zeros(self.nModels, dtype=np.float)          
          self.mv    = np.zeros(self.nModels, dtype=np.float)          
          self.sigma = np.zeros(self.nModels, dtype=np.float)          
          self.E_gamma = np.zeros((self.nSamples, self.nModels), dtype=np.float)
          
      def E_step_E_gamma(self):
          for j in xrange(self.nSamples):
             sum_E_gamma = 0.0
             for k in xrange(self.nModels):
                self.E_gamma[j, k] = self.alpha[k] * Gaussian_p(self.mv[k], self.sigma[k], self.y[j])  
                sum_E_gamma += self.E_gamma[j, k]
             self.E_gamma[j, k] /= sum_E_gamma 

      def M_step(self):
          alpha_new = np.zeros(self.nModels, dtype=np.float); nv_new = np.zeros(self.nModels, dtype=np.float); sigma_new = np.zeros(self.nModels, dtype=np.float)
          for k in xrange(self.nModels): 
             for j in xrange(self.nSamples):
               alpha_new[k] += self.E_gamma[j, k] 
               mv_new[k]    += self.E_gamma[j, k] * self.y[j] 
               sigma_new[k] += self.E_gamma[j, k] * np.power(self.y[j] - self.mv[k], 2)
         E_gamma_across_models = np.sum(self.E_gamma, axis=0)
         alpha_new /= self.nSamples
         mv_new    /= E_gamma_across_models
         sigma_new /= E_gamma_across_models
         self.alpha = alpha_new
         self.mv    = mv_new
         self.sigma = sigma_new
 
      def E_step_Q_func_value(self):
          Q = 0.0
          for j in xrange(self.nSamples):
             for k in xrange(self.nModels):
                Q += self.E_sigma[j, k] * np.log(self.alpha[k]) + self.E_sigma[j, k] * np.log(Gaussian_p(self.mv[k], self.sigma[k], self.y[j])) 
          return Q

      def fit(self, train_X):
          self.y = self.train_X
          sef.nSamples, self.nModels = np.shape(train_X)
          self.init_para()
          self.E_step_E_gamma()
          last_Q = self.E_step_Q_func_value()
          Q = None
          while not Q or (Q - last_Q > self.thresh_Q):
             self.E_step()
             self.M_step()
             Q = self.E_step_Q_func_value()
             
             
def Gaussian_p(mv, sigma, y):
    p = np.exp(-1 * (y - mv) * (y - mv) / (2 * np.power(sigma, 2))) / (np.sqrt(2 * np.pi) * sigma)
    return p
		

if __name__ == "__main__":
  # 20 from N(1,4); 20 from N(2,9); 20 from N(13,16)
  data1 = np.random.randn(20) * 2 + 1
  data2 = np.random.randn(20) * 3 + 2
  data3 = np.random.randn(20) * 4 + 13
  data = np.hstack((data1, data2, data3))
  gmm = GMM()
  gmm.fit(data) 
  print self.alpha
