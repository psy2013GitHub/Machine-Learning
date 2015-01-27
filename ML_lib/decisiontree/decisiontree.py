
from __future__ import division
import numpy as np

class BTree_Node(object):
	def __init__(self, parent=None, is_leaf=0, feature=None, threshold=None, is_categoral=0, sample_idx=None, value=None, proba=None):
		self.children_left  = None
		self.children_right = None
		self.parent = parent
		self.is_leaf = is_leaf		 
		self.threshold = threshold
		self.feature = feature
		self.sample_idx = sample_idx
		self.proba = proba
		self.is_categoral = 0


class DecisionTree(object):
	def __init__(self, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, \
		min_samples_leaf=1, max_features=None, random_state=None, \
		compute_importances=None, max_leaf_nodes=None):

		self.max_features = max_features
		self.criterion = criterion
		self.spliter = splitter

		self.max_depth = max_depth
		self.min_samples_split = min_samples_split if min_samples_split > min_samples_leaf else min_samples_leaf # ensure 'min_samples_split' > 'min_samples_leaf'
		self.min_samples_leaf = min_samples_leaf 
		self.max_leaf_nodes = max_leaf_nodes

		self.random_state = random_state
		self.compute_importances = compute_importances

		# if best split but samples @ leaf < self.min_samples_leaf
		self.stratege = 1 # 1, set as leaf directly; 2, use non-best split instead 

		# depth
		self.depth = 0

		# n_leaf_nodes
		self.n_leaf_nodes = 0 


	def fit(self, train_X, train_Y, category_var_idx=None):
		nsample, nfeature = train_X.shape
		self.nfeature = nfeature
		self.classes = np.unique(train_Y)
		self.train_X = train_X
		self.train_Y = train_Y
		self.category_var_idx = category_var_idx
		# split node
		root = BTree_Node(sample_idx=range(nsample))
		self.split_node = [root]
		while 1:

			if not self.split_node or (self.max_leaf_nodes and self.max_leaf_nodes <= self.n_leaf_nodes + len(self.split_node)) \
			 or (self.max_depth and self.depth >= self.max_depth): 
				for node in self.split_node:
					node.is_leaf = 1
					node.proba = self.proba_calc(self.train_Y[node.sample_idx])
					node.n_leaf_nodes += 1
				break

			new_split_node = []
			for node in self.split_node:
				# min_samples_split
				if len(node.sample_idx) <= self.min_samples_split:
					node.is_leaf = 1
					node.proba = self.proba_calc(self.train_Y[node.sample_idx])
					self.n_leaf_nodes += 1
					continue
				# feature & cut
				if self.stratege == 1:
					minFeatureIdx, minCut, minGini, minChildren_left, minChildren_right = self.Gini_select(node.sample_idx)
				else:
					minFeatureIdx, minCut, minGini, minChildren_left, minChildren_right = self.Gini_select(node.sample_idx, constraint=True)
				# update info @ leaf
				if not minFeatureIdx:
					node.is_leaf = 1
					node.proba = self.proba_calc(self.train_Y[node.sample_idx])
					self.n_leaf_nodes += 1
					continue
				else:
					node.feature = minFeatureIdx
					# print "feature: ", node.feature, " cut: ", minCut, " gini: ", minGini
					node.threshold = minCut
					node.Gini = minGini
					if minFeatureIdx in self.category_var_idx:
						node.is_categoral = 1

				if (self.stratege == 1 and (len(minChildren_left) < self.min_samples_leaf or len(minChildren_right) < self.min_samples_leaf)) \
					or (self.stratege == 2 and not minFeatureIdx):
					node.is_leaf = 1
					node.proba = self.proba_calc(self.train_Y[node.sample_idx])
					self.n_leaf_nodes += 1
					continue

				# update leaves

				L = BTree_Node(parent=node, feature=None, is_leaf=0, threshold=None, is_categoral=0, sample_idx=minChildren_left, value=None)
				node.children_left = L
				new_split_node.append(L)
					
				R = BTree_Node(parent=node, feature=None, is_leaf=0, threshold=None, is_categoral=0, sample_idx=minChildren_right, value=None)
				node.children_right = R
				new_split_node.append(R)

			# update self.leaf & depth
			self.split_node = new_split_node
			self.depth += 1

		
	def Gini_select(self, p_sample_idx, constraint=False):
		'''
		select the feature
		p_sample_idx: sample_idx in parent node
		feature_col: feature column
		'''
		minFeatureIdx=None; minCut=None; minGini=None; minChildren_left=None; minChildren_right=None

		p_sample_idx = np.array(p_sample_idx)

		X = self.train_X[p_sample_idx,:]
		Y = self.train_Y[p_sample_idx]

		if len(np.unique(Y)) == 1: # only one category left
			return minFeatureIdx, minCut, minGini, minChildren_left, minChildren_right		
		
		p_sample_len = len(p_sample_idx)

		minFeatureIdx = []
		for f in xrange(self.nfeature):
			cuts = np.unique(X[:,f])
			col = X[:,f]
			for cut in cuts:
				if f in self.category_var_idx:
					p_sample_idx1 = np.nonzero(col==cut)[0]; p_sample_len1 = len(p_sample_idx1)
					p_sample_idx2 = np.nonzero(col!=cut)[0]; p_sample_len2 = len(p_sample_idx2)
				else:
					p_sample_idx1 = np.nonzero(col<=cut)[0]; p_sample_len1 = len(p_sample_idx1) 
					p_sample_idx2 = np.nonzero(col> cut)[0]; p_sample_len2 = len(p_sample_idx2)

				if constraint and (p_sample_len1 < self.min_samples_leaf or p_sample_len2 < self.min_samples_leaf): # fix me about constraint
					continue

				gini1 = 0 # gini after split @ feature f @ cut
				for c in self.classes:
					if p_sample_len1 != 0:
						p = len(np.nonzero(Y[p_sample_idx1]==c)[0]) / p_sample_len1
					else:
						p = 0
					gini1 += p * (1-p)
				gini2 = 0
				for c in self.classes:
					if p_sample_len2 != 0:
						p = len(np.nonzero(Y[p_sample_idx2]==c)[0]) / p_sample_len2
					else:
						p = 0
					gini2 += p * (1-p)
				gini_sum = (p_sample_len1 / p_sample_len) * gini1 + (p_sample_len2 / p_sample_len) * gini2
				
				if not minFeatureIdx or minGini > gini_sum:
					minFeatureIdx = f
					minGini = gini_sum
					minCut = cut
					minChildren_left  = p_sample_idx[p_sample_idx1]
					minChildren_right = p_sample_idx[p_sample_idx2]

		return minFeatureIdx, minCut, minGini, minChildren_left, minChildren_right 

	def proba_calc(self, Y):
		P = []
		for c in self.classes:
			p = sum(np.nonzero(Y==c)[0]) / len(Y)
			P.append(p)
		return P


if __name__ == "__main__":
	# var1, yong 0, middle 1, old 2
	# var2, jobed 1, unjobed 0
	# var3, housed 1, unhoused 0
	# var4, very good 2, good 1, usual 0
	train_X = np.array([    \
	           			[0,0,0,0], \
	           			[0,0,0,1], \
	           			[0,1,0,1], \
	           			[0,1,1,0], \
	           			[0,0,0,0], \
	           			[1,0,0,0], \
	           			[1,0,0,1], \
	           			[1,1,1,1], \
	           			[1,0,1,2], \
	           			[1,0,1,2], \
	           			[2,0,1,2], \
	           			[2,0,1,1], \
	           			[2,1,0,1], \
	           			[2,1,0,2], \
	           			[2,0,0,0], \
	          		   ])
	category_var_idx = range(len(train_X))
	train_Y = np.array([0,0,1,1,0,0,0,1,1,1,1,1,1,1,0])
	clf = DecisionTree(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, \
		min_samples_leaf=1, max_features=None, random_state=None, \
		compute_importances=None, max_leaf_nodes=None)
	clf.fit(train_X, train_Y, category_var_idx)
	# print clf.depth
	print clf.n_leaf_nodes

