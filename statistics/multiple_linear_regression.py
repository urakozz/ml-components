import numpy as np

# Multiple Linear Regression
# https://onlinecourses.science.psu.edu/stat501/node/311
# http://www.stat.yale.edu/Courses/1997-98/101/linmult.htm

# Calculate Regression Parameters
# X = [[1,2]  (3x2)
#      [3,4]
#      [5,6]] 
# y = [[3]    (3x1)
#      [4]
#      [5]]
def regression_parameters(X, y):
  # X -> (3x2)
  # X.T -> (2x3)
  # X.T.dot(X) -> (2x2)
  # X.T.dot(y) -> (2x1)
  # np.linalg.inv(X.T.dot(X)) -> (2x2)
  # B (2x1)
  B = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))
  return B

# Predict values
def predict(B, X):
  return np.dot(X, B)

# example
# X = np.array([[1,2],[3,4],[5,6]])
# y = np.array([[3],[4],[5]])
# X_test = np.array([[0.5,1],[2.1,3]])
# 
# B = regression_parameters(X, y)
# print(predict(X_test, B))
