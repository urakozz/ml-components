# equivalent to 
#
# from sklearn.preprocessing import PolynomialFeatures
# poly = PolynomialFeatures(3)
# poly.fit_transform(X)
def polynomial_features(x, order=3):
    x = np.asarray(x).T[np.newaxis]
    n = x.shape[1]
    power_matrix = np.tile(np.arange(order + 1), (n, 1)).T[..., np.newaxis]
    X = np.power(x, power_matrix)
    I = np.indices((order + 1, ) * n).reshape((n, (order + 1) ** n)).T
    F = np.product(np.diagonal(X[I], 0, 1, 2), axis=2)
    return F.T
