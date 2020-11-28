import random
import math


def integrate(method, f, a, b, startN, eps):
    res = method(f, a, b, startN)
    startN *= 2
    newRes = method(f, a, b, startN)

    while abs(res - newRes) > eps:
        res = newRes
        startN *= 2
        newRes = method(f, a, b, startN)

    print("Achieved eps:", eps, "for", startN, "steps", "using method", method.__name__)
    print("Result:", newRes)
    return newRes


def middleRectangles(f, a, b, n):
    h = (b - a) / n
    res = 0
    curX = a
    for i in range(n):
        res += f(curX + h / 2)
        curX += h
    res *= h
    return res


def newtonCotes3(f, a, b, n):
    if n > 3:
        # recursion split for [a; (b + a) / 2] and [(b + a) / 2; b] segments
        m = (b + a) / 2
        return newtonCotes3(f, a, m, n / 2) + newtonCotes3(f, m, b, n / 2)

    c = [1/8, 3/8, 3/8, 1/8]
    curX = a
    res = 0
    h = (b - a) / 3
    for i in range(4):
        res += c[i] * f(curX)
        curX += h
    res *= b - a
    return res


def chebyshev3(f, a, b, n):
    if n > 3:
        # recursion split for [a; (b + a) / 2] and [(b + a) / 2; b] segments
        m = (b + a) / 2
        return chebyshev3(f, a, m, n / 2) + chebyshev3(f, m, b, n / 2)

    t = [-math.sqrt(1/2), 0, math.sqrt(1/2)]
    res = 0
    for i in range(3):
        curX = (b + a) / 2 + (b - a) / 2 * t[i]
        res += f(curX)
    res *= (b - a) / n
    return res


def monteKarlo(f, a, b, n):
    xArr = [random.uniform(a, b) for x in range(n)]

    res = 0
    for x in xArr:
        res += f(x)
    res *= (b - a) / n
    return res
