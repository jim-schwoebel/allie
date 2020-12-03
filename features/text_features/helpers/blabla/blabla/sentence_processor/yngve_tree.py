from anytree import NodeMixin, RenderTree


class YngveNode(NodeMixin):
    """The class to represent a node in the Yngve tree of a sentence"""

    def __init__(self, name, score, parent=None, children=None):
        """The initialization method to initialize a Yngve node with a name, score, parent node and child node
			Args:
				name (str): the name of the node
				score (str): the yngve score of a node
				parent (Yngve): the parent node of the current node
				children (Yngve): the child Yngve node of the current node
			Returns:
				None
		"""
        super(YngveNode, self).__init__()
        self.name = name
        self.score = score
        self.parent = parent
        if children:  # set children only if given
            self.children = children
