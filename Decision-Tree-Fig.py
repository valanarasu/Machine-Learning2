import pandas as pd
import matplotlib.pyplot as pld
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
iris=load_iris()
classifer=DecisionTreeClassifier()
classifer.fit(iris.data,iris.target)
pld.figure(figsize=(15,10))
tree.plot_tree(classifer,filled=True)
pld.show()