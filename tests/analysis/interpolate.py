import unittest
import math
import util
import analysis.interpolate as interpol


def f(x):
    return x * x + math.log(x) - 4


def firstDer(x):
    return 2 * x + 1 / x


def secondDer(x):
    return 2 - 1 / (x ** 2)


def thirdDer(x):
    return 2.0 / (x ** 3)


def elevenDer(x):
    return 3628800 / (x**11)


def fact(n):
    res = 1
    for i in range(2, n + 1):
        res *= i
    return res


class InterpolationTest(unittest.TestCase):
    def prepare_test(self, a, b, xStar, n):
        xArr, yArr = util.createPlot(f, a, b, n)
        idx = interpol.getClosestIdx(xArr, xStar)
        return idx, xArr, yArr

    def test_lagrange1(self):
        a = 1.45
        b = 1.95
        xStar = 1.52
        idx, xArr, yArr = self.prepare_test(a, b, xStar, 10)

        lg1 = interpol.lagrange(xStar, [xArr[idx], xArr[idx + 1]], [yArr[idx], yArr[idx + 1]])
        minR1 = secondDer(xArr[idx]) * interpol.omegaN(xStar, [xArr[idx], xArr[idx + 1]]) / 2
        maxR1 = secondDer(xArr[idx + 1]) * interpol.omegaN(xStar, [xArr[idx], xArr[idx + 1]]) / 2
        r1 = abs(lg1 - f(xStar))

        self.assertGreaterEqual(r1, abs(minR1))
        self.assertGreaterEqual(abs(maxR1), r1)

    def test_lagrange2(self):
        a = 1.45
        b = 1.95
        xStar = 1.52

        idx, xArr, yArr = self.prepare_test(a, b, xStar, 10)

        localX = [xArr[idx + i] for i in range(-1, 2)]
        localY = [yArr[idx + i] for i in range(-1, 2)]

        lg2 = interpol.lagrange(xStar, localX, localY)
        minR2 = thirdDer(xArr[idx + 1]) * interpol.omegaN(xStar, localX) / 6
        maxR2 = thirdDer(xArr[idx - 1]) * interpol.omegaN(xStar, localX) / 6
        r2 = abs(lg2 - f(xStar))
        self.assertGreaterEqual(r2, abs(minR2))
        self.assertGreaterEqual(abs(maxR2), r2)

    def test_newton1(self):
        a = 1.5
        b = 2
        xStar = 1.52

        idx, xArr, yArr = self.prepare_test(a, b, xStar, 10)
        h = (b - a) / 10

        diffMatrix = interpol.dividedDiffs(yArr)

        nwt1 = interpol.newton1(xStar, h, xArr, diffMatrix)
        nwt1Rmin = elevenDer(xArr[-1]) * interpol.omegaN(xStar, xArr) / fact(len(xArr))
        nwt1Rmax = elevenDer(xArr[0]) * interpol.omegaN(xStar, xArr) / fact(len(xArr))
        nwt1R = abs(f(xStar) - nwt1)
        self.assertGreaterEqual(nwt1R, abs(nwt1Rmin))
        self.assertGreaterEqual(abs(nwt1Rmax), nwt1R)

    def test_newton2(self):
        a = 1.5
        b = 2
        xStar = 1.97

        idx, xArr, yArr = self.prepare_test(a, b, xStar, 10)
        h = (b - a) / 10

        diffMatrix = interpol.dividedDiffs(yArr)

        nwt2 = interpol.newton2(xStar, h, xArr, diffMatrix)
        nwt2Rmin = elevenDer(xArr[-1]) * interpol.omegaN(xStar, xArr) / fact(len(xArr))
        nwt2Rmax = elevenDer(xArr[0]) * interpol.omegaN(xStar, xArr) / fact(len(xArr))
        nwt2R = abs(f(xStar) - nwt2)
        self.assertGreaterEqual(nwt2R, abs(nwt2Rmin))
        self.assertGreaterEqual(abs(nwt2Rmax), nwt2R)

    def test_gauss1(self):
        a = 1.5
        b = 2
        xStar = 1.77

        idx, xArr, yArr = self.prepare_test(a, b, xStar, 10)
        h = (b - a) / 10

        diffMatrix = interpol.dividedDiffs(yArr)

        gss1 = interpol.gauss12(xStar, h, xArr, diffMatrix)
        gss1Rmin = elevenDer(xArr[-1]) * interpol.omegaN(xStar, xArr) / fact(len(xArr))
        gss1Rmax = elevenDer(xArr[0]) * interpol.omegaN(xStar, xArr) / fact(len(xArr))
        gss1R = abs(f(xStar) - gss1)
        self.assertGreaterEqual(gss1R, abs(gss1Rmin))
        self.assertGreaterEqual(abs(gss1Rmax), gss1R)

    def test_lagrange_der(self):
        a = 1.5
        b = 1.5 + 0.05 * 3
        xm = 1.6

        idx, xArr, yArr = self.prepare_test(a, b, xm, 3)
        r = abs(interpol.lagrangeDerivative(xm, xArr, yArr) - firstDer(xm))
        self.assertGreater(1e-4, r)


if __name__ == '__main__':
    unittest.main()
