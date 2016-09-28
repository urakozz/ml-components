from __future__ import print_function

# Linear regression parameters
# http://www.stat.yale.edu/Courses/1997-98/101/linregin.htm
#
# Returns (slope, intercept)
#
# equivalent to:
#   scipy.stats.linregress()[:2]
def slope_interception(x,y): #parameter of linear regression
	n = len(x)
	intercept = ((n * sum(x[i] * y[i] for i in range(n)) - sum(x)*sum(y)) / ( n * sum(v**2 for v in x) - sum(x) ** 2))
	slope = sum(y)/n - interception * sum(x) / n
	return slope, intercept

# Regression prediction
# http://onlinestatbook.com/2/regression/intro.html
# Y' = bX + A
def regression_prediction(slope, intercept, x):
  return slope * x + intercept
  
# example
# x = [15 , 12 , 8 ,  8 ,  7 ,  7  , 7  , 6 ,  5,   3]
# y = [10 , 25 , 17 , 11 , 13 , 17 , 20 , 13 , 9 ,  15]
# b, A = slope_interception(x,y)
# predicted = regression_prediction(b, A, x)
# >>> 15.5
