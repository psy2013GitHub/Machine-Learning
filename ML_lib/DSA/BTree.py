








class BTree_Node(object):
	def __init__(self, parent=None, value=None):
		self.value = value
		self.children_left  = None
		self.children_right = None
		self.parent = parent		 



class BTree(object):
	def __init__(self):
		self.root = BTree_Node()
		self.leaf = self.root

	def add(value):
		