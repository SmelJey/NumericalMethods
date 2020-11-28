# a - main diag, b - upper diag, c - lower diag
def tridiagonalMatrix(a, b, c, f):
    n = len(a)

    curAlpha = 0
    curBetta = 0
    alpha = []
    betta = []

    for i in range(0, n):
        denominator = c[i] * curAlpha + a[i]
        curBetta = (f[i] - c[i] * curBetta) / denominator
        curAlpha = -b[i] / denominator

        alpha.append(curAlpha)
        betta.append(curBetta)

    curY = curBetta
    res = []
    for i in range(n-1, -1, -1):
        curY = alpha[i] * curY + betta[i]
        res.append(curY)

    return res[n::-1]
