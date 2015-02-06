
from __future__ import division
import numpy as np


class HMM(object):
      '''
         main app of hmm:
                         1, probability of p(O|Pi, A, B)
                         2, supervised learning of Pi, A, B
                         3, unsupervised learning of Pi, A, B
      '''
      def __init__(self, Pi, A, B):
          '''
            Pi:  column vector of hidden variables
            A :  matrix of hidden variables, symetry
            B :  matrix of hidden variables [row] and outputs [column]
          '''
          self.A = A
          self.B = B

      def proba_calc(self, O_idx_lst, methods=["forward"]):
          '''
            O_idx_lst: output idx lst
            method:
                   "forward":  "alpha_t(i)" means "p(O1, O2, ..., Ot|I_t=q_i)";
                   "backward": "alpha_t(i)" means "p(O_t+1, O_t+2, ..., O_T|I_t=q_i)";
          '''
          for method in methods:
              T = len(O_idx_lst) # length of total series
              if method == "forward":
                 self.alpha = np.zeros((T,self.nHidden), dtype=np.float)
                 for t in xrange(T):
                    if not t:
                      for i in xrange(self.nHidden): # i, hidden variables
                         self.alpha[t,i] = self.Pi(i) * self.B(i, O_idx_lst[0]) 
                    else:
                      for i in xrange(self.nHidden):
                         self.alpha[t,i] = np.sum(np.dot(alpha[t-1, :], self.A(:, i)) * self.B(i, O_idx_lst[t]))
                 
                 # "self.O_given_lambda" means "p(O|lambda)"
                 self.O_given_lambda = np.sum(self.alpha[T-1, :])  
                 
              elif method == "backward":
                 self.beta = np.zeros((T,self.nHidden), dtype=np.float)  
                 for t in xrange(T):
                    if not t:
                      self.beta[T-t-1, :] = 1 # set vector to a num in numpy 
                    else:
                      for i in xrange(self.nHidden):
                         self.beta[T-t-1, i] = np.sum(np.dot(np.dot(self.A[i, :], self.beta[T-t, :]) ,  self.B[:, O_idx_lst[T-t]] ))
                  
                 # "self.O_given_lambda" means "p(O|lambda)"
                 self.O_given_lambda = np.sum(np.dot(np.dot(self.Pi[:], self.beta[0, :]), self.B[:, O_idx_lst[0]]) )

