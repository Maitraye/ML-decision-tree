class Node:
  def __init__(self):	
    self.label = None # label - is the output class label, can be multiple values, not just 0 or 1
    self.children = {} # children - a dictionary (key=attribute value, value=node)
    self.name = None # name - name of the attribute being split on
    self.missing_attr_children = {} # a dictionary (key=attribute value, value=probability of occuring that value among examples at the current node)
    self.major = None # most common value of class label at this node
    self.pruning_gain = -100 # increase in accuracy if this node (and the subtree below it) is pruned
    self.is_pruning = False # flag to calculate the pruning gain of the node