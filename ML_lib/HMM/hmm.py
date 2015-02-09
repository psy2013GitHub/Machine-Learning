
from __future__ import division
import numpy as np


class HMM(object):
      '''
         main app of hmm:
                         1, probability of p(O|Pi, A, B)
                         2, supervised learning of Pi, A, B
                         3, unsupervised learning of Pi, A, B
      '''
      def __init__(self, Pi=None, A=None, B=None):
          '''
            Pi:  column vector of hidden variables
            A :  matrix of hidden variables, symetry
            B :  matrix of hidden variables [row] and outputs [column]
          '''
          self.Pi = Pi
          self.A = A
          self.B = B
          self.nHidden, self.nOutput = self.B.shape if self.B.shape[0] else (None, None)

      def proba_calc(self, O_idx_lst, method=["forward"]):
          '''
            O_idx_lst: output idx lst
            method:
                   "forward":  "alpha_t(i)" means "p(O1, O2, ..., Ot|I_t=q_i)";
                   "backward": "alpha_t(i)" means "p(O_t+1, O_t+2, ..., O_T|I_t=q_i)";
          '''
          for m in method:
              T = len(O_idx_lst) # length of total series
              if m == "forward":
                 self.alpha = np.zeros((T,self.nHidden), dtype=np.float)
                 for t in xrange(T):
                    if t==0:
                      self.alpha[0, :] = self.Pi * self.B[:, O_idx_lst[0]] 
                    else:
                      for j in xrange(self.nHidden):
                         self.alpha[t, j] = np.sum(self.alpha[t-1, :] * self.A[:, j]) * self.B[j, O_idx_lst[t]]
                 
              elif m == "backward":
                 self.beta = np.zeros((T,self.nHidden), dtype=np.float)  
                 for t in xrange(T):
                    if t==0:
                      self.beta[T-1, :] = 1 # set vector to a num in numpy 
                    else:
                      for j in xrange(self.nHidden):
                         self.beta[T-t-1, j] = np.sum(self.A[j, :] * self.beta[T-t, :] *  self.B[:, O_idx_lst[T-t]])


      def model_calc(self, nHidden, nOutput, O_idx_lst, method="unsuperwised"):
          '''
            Learning problem of hmm:
                                    supervised
                                    unsupervised, i.e. Baum-Welch Algorithm
          '''
          if method == "unsuperwised":
             # init parameters
             def init_para(self): 
                 self.init_para(nHidden, nOutput, method="random")
            
             T = len(O_idx_lst) # length of total series
             while 1:  
                 # get | update self.alpha & self.beta
                 self.proba_calc(O_idx_lst, methods=["forward", "backward"])
                 # E-step by draft
                 # M-step maximize & update
                 # update Pi
                 self.Pi[j1] = self.alpha[0, :] * self.beta[0, :] / np.sum(self.alpha[0, :] * self.beta[0, :]) 
                 for j1 in xrange(self.nHidden):
                     # update A
                     tmp2 = 0.0 # denominator
                     for t in xrange(T-1):
                        if t==0:
                           tmp1 = self.alpha[t, j1] * self.beta[t+1, :] * self.A[j1, :] * self.B[:, O_idx_lst[t+1]] 
                        else:
                           tmp1 += self.alpha[t, j1] * self.beta[t+1, :] * self.A[j1, :] * self.B[:, O_idx_lst[t+1]] 
                           tmp2 += self.alpha[t, j1], self.beta[t,j1]
                        self.A[j1, :] = tmp1 / tmp2
                     for ok in self.nOutput:
                        # update B
                        tmp1 = 0.0; tmp2 = 0.0
                        for t in xrange(T-1):
                           tmp1 += self.alpha[t, j1] if O_idx_lst[t]==ok else 0      
                           self.B[j1, ok] = tmp1 / (tmp2 + self.alpha[T-1, j1] * self.beta[T-1, j1]) # use tmp2 caculated previously in updating A
                    
      def init_para(self, method="random"):
          self.Pi = np.zeros(self.nHidden, dtype=np.float); self.A = np.zeros((self.nHidden, self.nHidden), dtype=np.float); self.B = np.zeros((self.nHidden, self.nOutput), dtype=np.float)
          if method == "random":
             # Pi
             self.Pi = get_sum1_proba(self.nHidden) 
             # A & B
             for j in xrange(self.nHidden):
                self.A[:, j] = get_sum1_proba(self.nHidden) 
                self.B[j, :] = get_sum1_proba(self.nOutput)
              
def get_sum1_proba(vec_len):
    p = np.zeros(vec_len)
    p[0:vec_len-1] = np.random.uniform(0.0001, 1.0/vec_len, vec_len-1)
    p[vec_len-1] = np.sum(p[0:vec_len-1])
    return p




if __name__ == "__main__":
   Pi= np.array([0.2,0.4,0.4], dtype=np.float)
   A = np.array([
                 [0.5,0.2,0.3],
                 [0.3,0.5,0.2],
                 [0.2,0.3,0.5],
                ], dtype=np.float)
   B = np.array([
                 [0.5,0.5],
                 [0.4,0.6],
                 [0.7,0.3],
                ], dtype=np.float)
 
   hmm = HMM(Pi,A,B)
   hmm.proba_calc([0,1,0,1], method=["forward", "backward"])




