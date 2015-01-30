
from __future__ import division
import numpy as np
import copy

class BTree_Node(object):
	def __init__(self, parent=None, is_leaf=0, feature=None, threshold=None, is_categoral=0, sample_idx=None, value=None, proba=None):
		self.children_left  = None
		self.children_right = None
		self.parent = parent
		self.is_leaf = is_leaf
		self.is_root = 0		 
		self.threshold = threshold
		self.feature = feature
		self.sample_idx = sample_idx
		self.proba = proba
		self.is_categoral = 0
		self.Ct = None
		self.CT = None
		self.T_size = None # leaf_nodes belong to self


class DecisionTree(object):
	'''
	CART: classfication and regression tree
	Breiman, 1984
	'''
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
		self.root = BTree_Node(sample_idx=range(nsample)); self.root.is_root = 1
		self.split_node = [self.root]
		while 1:
			
			if not self.split_node or (self.max_leaf_nodes and self.max_leaf_nodes <= self.n_leaf_nodes + len(self.split_node)) \
			 or (self.max_depth and self.depth >= self.max_depth): 
				for node in self.split_node:
					node.is_leaf = 1
					node.n_leaf_nodes += 1
					node.proba, node.Ct = self.proba_gini_calc(self.train_Y[node.sample_idx])
					node.CT = node.Ct
				break

			new_split_node = []
			for node in self.split_node:
				node.proba, node.Ct = self.proba_gini_calc(self.train_Y[node.sample_idx])
				# min_samples_split
				if len(node.sample_idx) <= self.min_samples_split:
					node.is_leaf = 1
					self.n_leaf_nodes += 1
					node.CT = node.Ct
					continue

				# feature & cut
				if self.stratege == 1:
					minFeatureIdx, minCut, minGini, minChildren_left, minChildren_right = self.Gini_select(node.sample_idx)
				else:
					minFeatureIdx, minCut, minGini, minChildren_left, minChildren_right = self.Gini_select(node.sample_idx, constraint=True)
				# update info @ leaf
				if not minFeatureIdx:
					node.is_leaf = 1
					self.n_leaf_nodes += 1
					node.CT = node.Ct
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
					self.n_leaf_nodes += 1
					node.CT = node.Ct
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

	def score(self, test_set):
		# omitted
		pass

	def prune(self, validation_set):
		'''
		two steps:
		      1, get leaf_nodes
		      2, calcu alpha
		'''
		alpha_lst = []; Tree_lst = [copy.deepcopy(self)]; Score_lst = [None] # None instead of +Inf
		min_alpha = None; minInternalNode = None
		k = 0
		nEval = None
		while 1:

			# get Tk
			T = copy.deepcopy(Tree_lst[-1]) 

			# if only root left: i.e. nEval == 0 , then break
			if (not T.root.children_left) and (not T.root.children_right):
				break

			T.get_leaf(); T.curr_nodes = T.leaf_nodes
			# from bottom to up
			depth = 0
			nEval = 0
			while 1:

				if not T.curr_nodes:
					break

				new_nodes = []
				for node in T.curr_nodes:

					p = node.parent
					if not p: # root's parent
						T.curr_nodes = None
						minInternalNode = T.root # i.e. only root remained
						break
					if node.is_prune_visited:
						continue
					else:
						# Ct_alpha, preserved in node 
						# p.Ct
						# CT_alpha
						p.CT = 0.0; p.T_size = 0.0
						if p.children_left:
							p.children_left.is_prune_visited = 1
							p.CT += p.children_left.CT
							p.T_size += 1 if p.children_left.is_leaf else p.children_left.T_size
						if p.children_right:
							p.children_right.is_prune_visited = 1
							p.CT += p.children_right.CT
							p.T_size += 1 if p.children_right.is_leaf else p.children_left.T_size
						# alpha
						p.alpha = (p.CT - p.Ct) / (p.T_size - 1)
						if not min_alpha or min_alpha > p.alpha:
							minInternalNode = p
							min_alpha = p.alpha

						new_nodes.append(p)

					T.curr_nodes = new_nodes

			# from up to bottom
			minInternalNode.is_leaf = 1; minInternalNode.children_left = None; minInternalNode.children_right = None
			alpha_lst.append(min_alpha); 
			T.depth = None; Tree_lst.append(T) # depth set to none to fix in future

			# cross validation
			# scr = self.score(validation_set)
			# Score_lst.append(scr)

			# find mininum score



	def get_leaf(self):
		self.leaf_nodes = []
		def add_leaf(node):
			node.is_prune_visited = 0
			if node.is_leaf:
				self.leaf_nodes.append(node)

		self.Travel_first_Recur(self.root, add_leaf)


	def Travel_first_Recur(self, node, func, *func_args_tuple):
		'''
			func: function handle
			func_args_tuple: argument of func, return val included as reference as C++
		'''
		# print func_args_tuple
		# if node.is_leaf:
		func(node, *func_args_tuple)
		if node.children_left:
			self.Travel_first_Recur(node.children_left, func, *func_args_tuple)
		if node.children_left:
			self.Travel_first_Recur(node.children_right, func, *func_args_tuple)
		return


	def proba_gini_calc(self, Y):
		P = []; valGini = 0
		for c in self.classes:
			p = sum(np.nonzero(Y==c)[0]) / len(Y)
			valGini += p * (1 - p)
			P.append(p)
		return P, valGini


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
	# print clf.n_leaf_nodes
	# clf.get_leaf()
	# print len(clf.leaf_nodes)
	clf.prune([])

