import algebra.matrix as matrix


def euler(a, b, y0, z0, yf, zf, n):
    h = (b - a) / n
    curX = a
    curY = y0
    curZ = z0

    x = [a]
    y = [y0]
    z = [z0]

    for i in range(n):
        newZ = curZ + h * zf(curX, curY, curZ)
        newY = curY + h * yf(curX, curY, curZ)
        curX += h
        curZ = newZ
        curY = newY
        x.append(curX)
        y.append(curY)
        z.append(curZ)

    return x, y, z


def rungeKutta(a, b, y0, z0, yf, zf, n):
    h = (b - a) / n
    curX = a
    curY = y0
    curZ = z0

    x = [a]
    y = [y0]
    z = [z0]

    for i in range(n):
        k1 = yf(curX, curY, curZ)
        m1 = zf(curX, curY, curZ)

        k2 = yf(curX + h / 2, curY + h * k1 / 2, curZ + h * m1 / 2)
        m2 = zf(curX + h / 2, curY + h * k1 / 2, curZ + h * m1 / 2)

        k3 = yf(curX + h / 2, curY + h * k2 / 2, curZ + h * m2 / 2)
        m3 = zf(curX + h / 2, curY + h * k2 / 2, curZ + h * m2 / 2)

        k4 = yf(curX + h, curY + h * k3, curZ + h * m3)
        m4 = zf(curX + h, curY + h * k3, curZ + h * m3)

        curX += h
        curY += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        curZ += h * (m1 + 2 * m2 + 2 * m3 + m4) / 6

        x.append(curX)
        y.append(curY)
        z.append(curZ)

    return x, y, z


def search(x, delta, f, eps):
    x0 = x
    x1 = x0 + delta
    diff0 = f(x0)
    diff1 = f(x1)

    if diff1 > diff0:
        delta *= -1

    while diff0 > eps:
        x1 = x0 + delta
        diff1 = f(x1)

        if diff1 < diff0:
            x0 = x1
            diff0 = diff1
        else:
            x0 = x0 - delta
            delta /= 10
            diff0 = f(x0)

    return x0


def shootingMethodZ(a, b, z0, z1, zf, n, eps):
    def yf(x, y, z):
        return z

    def searchFunc(y):
        newX, newY, newZ = rungeKutta(a, b, y, z0, yf, zf, n)
        return abs(z1 - newZ[-1])

    res = search(0, 10, searchFunc, eps)
    resX, resY, resZ = rungeKutta(a, b, res, z0, yf, zf, n)
    return resX, resY, res


def shootingMethodY(a, b, y0, y1, zf, n, eps):
    def yf(x, y, z):
        return z

    def searchFunc(z):
        newX, newY, newZ = rungeKutta(a, b, y0, z, yf, zf, n)
        return abs(y1 - newY[-1])

    res = search(0, 10, searchFunc, eps)
    resX, resY, resZ = rungeKutta(a, b, y0, res, yf, zf, n)

    return resX, resY, res


def secantMethodZ(a, b, z0, z1, zf, n):
    def yf(x, y, z):
        return z

    y0 = -10
    yDelta = 10

    newX0, newY0, newZ0 = rungeKutta(a, b, y0, z0, yf, zf, n)
    y1 = y0 + yDelta
    newX1, newY1, newZ1 = rungeKutta(a, b, y1, z0, yf, zf, n)

    yRes = y0 + (y1 - y0) * (z1 - newZ0[-1]) / (newZ1[-1] - newZ0[-1])
    resX0, resY0, resZ0 = rungeKutta(a, b, yRes, z0, yf, zf, n)
    return resX0, resY0, yRes


def secantMethodY(a, b, y0, y1, zf, n):
    def yf(x, y, z):
        return z

    z0 = -10
    zDelta = 10

    newX0, newY0, newZ0 = rungeKutta(a, b, y0, z0, yf, zf, n)
    z1 = z0 + zDelta
    newX1, newY1, newZ1 = rungeKutta(a, b, y0, z1, yf, zf, n)

    zRes = z0 + (z1 - z0) * (y1 - newY0[-1]) / (newY1[-1] - newY0[-1])
    resX0, resY0, resZ0 = rungeKutta(a, b, y0, zRes, yf, zf, n)
    return resX0, resY0, zRes


def finiteDiffMethod1(a, b, n, p, q, oldF, alpha, beta, gamma):
    def f(x):
        return oldF(x)

    h = (b - a) / n
    x = [a + i * h for i in range(n + 1)]
    aDiag = [alpha[0] * h - beta[0]]
    bDiag = [beta[0]]
    cDiag = [0]
    fVec = [gamma[0] * h]

    for i in range(1, n):
        aDiag.append(q(x[i]) * h**2 - 2)
        bDiag.append(1 + h/2 * p(x[i]))
        cDiag.append(1 - h/2 * p(x[i]))
        fVec.append(h * h * f(x[i]))

    aDiag.append(alpha[1] * h + beta[1])
    bDiag.append(0)
    cDiag.append(-beta[1])
    fVec.append(gamma[1] * h)

    res = matrix.tridiagonalMatrix(aDiag, bDiag, cDiag, fVec)
    return x, res


def finiteDiffMethod2(a, b, n, p, q, f, alpha, beta, gamma):
    h = (b - a) / n
    x = [a + i * h for i in range(n + 1)]
    denom = (1 + h / 2 * p(x[1]))
    aDiag = [2 * alpha[0] * h - 3 * beta[0] + beta[0] * (1 - h / 2 * p(x[1])) / denom]
    bDiag = [4 * beta[0] + beta[0] * (q(x[1]) * h ** 2 - 2) / denom]
    cDiag = [0]
    fVec = [2 * gamma[0] * h + beta[0] * h ** 2 * f(x[1]) / denom]

    for i in range(1, n):
        aDiag.append(q(x[i]) * h ** 2 - 2)
        bDiag.append(1 + h / 2 * p(x[i]))
        cDiag.append(1 - h / 2 * p(x[i]))
        fVec.append(h * h * f(x[i]))

    denom = (1 - h / 2 * p(x[n - 1]))
    aDiag.append(2 * alpha[1] * h + 3 * beta[1] - (beta[1] * (1 + h / 2 * p(x[n - 1]))) / denom)
    bDiag.append(0)
    cDiag.append(beta[1] * (2 - q(x[n - 1]) * h ** 2) / denom - 4 * beta[1])
    fVec.append(2 * gamma[1] * h - (beta[1] * h ** 2 * f(x[n - 1])) / denom)

    res = matrix.tridiagonalMatrix(aDiag, bDiag, cDiag, fVec)
    return x, res


