import itertools
from functools import reduce


def lagrange(x, xArr, yArr):
    res = 0
    for i in range(len(xArr)):
        mult = 1
        for j in range(len(xArr)):
            if i != j:
                mult *= (x - xArr[j]) / (xArr[i] - xArr[j])
        res += yArr[i] * mult
    return res


def lagrangeDerivative(x, xArr, yArr):
    r = 0

    for i in range(0, len(xArr)):
        a = 0
        b = 1
        w = []

        for j in range(0, len(xArr)):
            if i != j:
                b *= xArr[i] - xArr[j]
                w.append(xArr[j])

        for item in itertools.combinations(w, len(xArr) - 2):
            a += reduce(lambda cur, prev: cur * prev, [x - it for it in item])

        r += yArr[i] * a / b
    return r


def omegaN(x, xArr):
    res = 1
    for i in range(0, len(xArr)):
        res *= x - xArr[i]
    return res


def __calculateDiff(diffMatrix, yArr):
    curDiffs = []
    for i in range(1, len(yArr)):
        curDiffs.append(yArr[i] - yArr[i - 1])
    diffMatrix.append(curDiffs)
    if len(curDiffs) > 1:
        __calculateDiff(diffMatrix, curDiffs)


def dividedDiffs(yArr):
    diffMatrix = [yArr]
    __calculateDiff(diffMatrix, yArr)
    return diffMatrix


def newton1(xStar, h, xArr, diffMatrix):
    res = 0
    mult = 1
    t = (xStar - xArr[0]) / h
    if not (0 < t < 1):
        raise Exception('xStar must be in the start of the table')

    for i in range(0, len(diffMatrix)):
        res += diffMatrix[i][0] * mult
        mult *= (t - i) / (i + 1)

    return res


def newton2(xStar, h, xArr, diffMatrix):
    res = 0
    mult = 1
    t = (xStar - xArr[-1]) / h
    if not -1 < t < 0:
        raise Exception('xStar must be in the end of the table')

    for i in range(0, len(diffMatrix)):
        res += diffMatrix[i][-1] * mult
        mult *= (t + i) / (i + 1)

    return res


def getClosestIdx(xArr, x):
    for i in range(0, len(xArr)):
        if x < xArr[i]:
            return i - 1
    return -1


def gauss12(xStar, h, xArr, diffMatrix):
    closestIdx = getClosestIdx(xArr, xStar)
    t = (xStar - xArr[closestIdx]) / h
    if not (-0.5 < t < 0.5):
        raise Exception('incorrect xStar')
    mult = 1
    res = 0
    if t > 0:
        for i in range(0, len(diffMatrix)):
            res += diffMatrix[i][closestIdx] * mult
            mult *= (t + ((-1) ** i) * ((i + 1) // 2)) / (i + 1)
            if i % 2 == 1:
                closestIdx -= 1
    else:
        for i in range(0, len(diffMatrix)):
            res += diffMatrix[i][closestIdx] * mult
            mult *= (t + ((-1) ** i) * (i + 1) // 2) / (i + 1)
            if i % 2 == 1:
                closestIdx -= 1
    return res
