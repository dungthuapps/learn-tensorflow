"""Manual Tensorflow basic syntax from scratch.

It includes OOP, Graph, Operations, PlaceHolder, Variables concept
Reference:
    https://www.udemy.com/course/complete-guide-to-tensorflow-for-deep-learning-with-python
"""

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np


class Operation():
    def __init__(self, input_nodes=[]):
        self.input_nodes = input_nodes
        self.output_nodes = []

        # add node to graph
        for node in input_nodes:
            node.output_nodes.append(self)

        # add the operations into graph
        _default_graph.operations.append(self)

    def compute(self):
        pass


class add(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x, y):
        self.inputs = [x, y]
        return x + y


class multiply(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x, y):
        self.inputs = [x, y]
        return x * y


class matmul(Operation):
    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x, y):
        self.inputs = [x, y]

        # for numpy multiplication
        return x.dot(y)


class PlaceHolder():
    """Place Holder ~ empty node needs a value
        to be provided to computeoutput.
    """

    def __init__(self):
        self.output_nodes = []

        # add placeholder to global graph
        _default_graph.placeholders.append(self)


class Variable():
    """Variables ~ changeable parameter of Graph"""

    def __init__(self, initial_value=None):
        self.value = initial_value
        self.output_nodes = []

        # add variables to global graph
        _default_graph.variables.append(self)


class Graph():
    """A graph will stores all input and operations.

    A Global variable connecting variables and place holders specific
    operations
    A Graph - a global variable
        n1 + n2 := (n1 ~constant) -> (n3~operation) <- (n2 ~ constant)

    """

    def __init__(self):
        self.operations = []
        self.placeholders = []
        self.variables = []

    def set_as_default(self):
        # declare this graph as global
        global _default_graph
        _default_graph = self


class Session():
    """Session to
        (1) execute all ooperations within a session
        (2) PostOrder Tree Traveseral to execute nodes in correct order.
    """
    @staticmethod
    def traverse_postorder(operation):
        """PostOrder Traversal of Nodes.

        Make sure correct order of for e.g. Ax + b
            then Ax first, then Ax + b
        """
        nodes_postorder = []

        def recurse(node):
            if isinstance(node, Operation):
                for input_node in node.input_nodes:
                    recurse(input_node)
            nodes_postorder.append(node)
        recurse(operation)
        return nodes_postorder

    def run(self, operation, feed_dict={}):
        """run a session.

        :param feed_dict: provide data to placeholders
        """
        nodes_postorder = self.traverse_postorder(operation)
        for node in nodes_postorder:
            # Placeholder
            if type(node) == PlaceHolder:
                node.output = feed_dict[node]
            # Variable
            elif type(node) == Variable:
                node.output = node.value
            # Operation
            else:
                node.inputs = [input_node.output
                               for input_node in node.input_nodes]

                node.output = node.compute(*node.inputs)

            if type(node.output) == list:
                node.output = np.array(node.output)

        return operation.output


class Sigmoid(Operation):
    def __init__(self, z):
        super().__init__([z])

    def compute(self, z):
        return 1 / (1 + np.exp(-z))


# Demo 1
#   (1) Define
#       example z = ax + b, here x is placeholder, a, b is varialbe
g = Graph()
g.set_as_default()
a = Variable(10)  # set a = 10
b = Variable(1)    # set b = 1
x = PlaceHolder()  # will provide later when run in sesssion
y = multiply(a, x)  # y = ax
z = add(y, b)  # z = ax + b

# currently no thing happens because empty x
# (2) feed data into placeholder


sess = Session()
result = sess.run(operation=z, feed_dict={x: 10})

# Demo 2
g = Graph()
g.set_as_default()
a = Variable([[10, 20], [30, 40]])
b = Variable([1, 2])
x = PlaceHolder()
y = matmul(a, x)
z = add(y, b)

sess = Session()
result = sess.run(operation=z, feed_dict={x: 10})

# Demo 3a
# Activation functions

# create samples with 2 classes
n_classes = 2
seed = 75
data = make_blobs(n_samples=50, n_features=2,
                  centers=n_classes, random_state=seed)

features = data[0]
labels = data[1]

# plot 2-features column
plt.scatter(features[:, 0], features[:, 1], c=labels, cmap="coolwarm")
plt.show()

# draw points with splitter line
x = np.linspace(0, 11, 10)
y = -x + 5  # example
plt.scatter(features[:, 0], features[:, 1], c=labels, cmap="coolwarm")
plt.plot(x, y)
plt.show()

# now we have an example of classification with line of spliter
# assume the line is y = mx + b (y, x are features, m, b are parameters)
#   --convert-to->  y -mx - b = 0
#   --to-matrix-> Y - MX - b = 0

# Demo 3b - 1*y + 1*x - 5 = 0  (assume m = -1 like example of 3a,j y = -x + 5)
#   -> hence y + x - 5 = 0
g = Graph()
g.set_as_default()
x = PlaceHolder()
w = Variable([1, 1])
b = Variable(-5)
z = add(matmul(w, x), b)
# with activation operation
a = Sigmoid(z)

sess = Session()
result = sess.run(operation=a, feed_dict={x: [8, 10]})
