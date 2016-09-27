import math

#Karl Pearson’s coefficient of correlation
def corellation(A,B):
    n = float(len(A))
    muA = sum(A)/n
    muB = sum(B)/n
    diffA = map(lambda x: x - muA, A)
    diffB = map(lambda x: x - muB, B)
    stdA = math.sqrt((1/(n-1))* sum([d*d for d in diffA]))
    stdB = math.sqrt((1/(n-1))* sum([d*d for d in diffB]))
    return (sum([A[i]*B[i] for i in range(int(n))]) - n * muA * muB) / ((n-1) * stdA * stdB)
