import unittest
import math
import util
import ode.ode2 as ode
import matplotlib.pyplot as plt


def zfTest(x, y, z):
    return x - y - 2 * z


def solTest(x):
    return x + math.exp(-x) * (x + 2) - 2


def zfTest2(x, y, z):
    return -math.sinh(x) * z - x * y - 1


class Ode2Test(unittest.TestCase):
    def checkOde2(self, method, yf, zf, sol, y0, z0, a, b, n, order):
        h = (b - a) / n
        eps = h ** order
        resX, resY, resZ = method(a, b, y0, z0, yf, zf, n)
        expX, expY = util.createPlot(sol, a, b, n)
        diffX = max([abs(x - y) for x, y in zip(resX, expX)])
        diffY = max([abs(x - y) for x, y in zip(resY, expY)])
        print("method:", method.__name__, "error:", diffY, "expected:", eps)

        self.assertEqual(diffX, 0)
        self.assertAlmostEqual(diffY, 0, delta=eps)

    def test_euler(self):
        self.checkOde2(ode.euler, lambda x, y, z: z, zfTest, solTest, 0, 0, 0, 1, 1000, 1)

    def test_rk(self):
        self.checkOde2(ode.rungeKutta, lambda x, y, z: z, zfTest, solTest, 0, 0, 0, 1, 100, 4)

    def test_bp_shooting(self):
        x, y, y0 = ode.shootingMethodY(0, 1, 2, 2, zfTest2, 1000, 1e-5)
        self.assertAlmostEqual(y0, 0.8666736566040598, delta=1e-5)

    def test_bp_secant(self):
        x, y, y0 = ode.secantMethodY(0, 1, 2, 2, zfTest2, 1000)
        self.assertAlmostEqual(y0, 0.8666736566040598, delta=1e-12)

    def test_finite_diff_method1(self):
        def p(x):
            return math.sinh(x)

        def q(x):
            return x

        def f(x):
            return -1

        # y'' =  -sinh(x)y' - xy - 1, y'(0) = 2, y'(1) = 2
        x, y = ode.finiteDiffMethod2(0, 1, 100, p, q, f, [0, 0], [1, 1], [2, 2])
        testX, testY, y0 = ode.secantMethodZ(0, 1, 2, 2, zfTest2, 100)
        for y1, y2 in zip(y, testY):
            self.assertAlmostEqual(y1, y2, delta=1e-2)

    def test_finite_diff_method2(self):
        def p(x):
            return math.sinh(x)
        def q(x):
            return x
        def f(x):
            return -1

        # y'' =  -sinh(x)y' - xy - 1, y(0) = 2, y(1) = 2
        x, y = ode.finiteDiffMethod2(0, 1, 1000, p, q, f, [1, 1], [0, 0], [2, 2])
        testX, testY, y0 = ode.secantMethodY(0, 1, 2, 2, zfTest2, 1000)

        for y1, y2 in zip(y, testY):
            self.assertAlmostEqual(y1, y2, delta=1e-3)

    def test_finite_diff_method3(self):
        # №753 fillipov e^x - 2
        def sol(x):
            return math.exp(x) - 2

        # y'' - y' = 0, y(0) = -1, y'(1) - y(1) = 2
        x, y = ode.finiteDiffMethod2(0, 1, 10000, lambda x: -1, lambda x: 0, lambda x: 0, [1, -1], [0, 1], [-1, 2])
        exactSol = [sol(xi) for xi in x]
        for y1, y2 in zip(y, exactSol):
            self.assertAlmostEqual(y1, y2, delta=1e-4)

    def test_finite_diff_method4(self):
        # 751 fillipov
        def sol(x):
            return math.sinh(x) / math.sinh(1) - 2 * x

        # y'' - y = 2x, y(0) = 0, y(1) = -1
        x, y = ode.finiteDiffMethod2(0, 1, 10000, lambda x: 0, lambda x: -1, lambda x: 2 * x, [1, 1], [0, 0], [0, -1])
        exactSol = [sol(xi) for xi in x]
        for y1, y2 in zip(y, exactSol):
            self.assertAlmostEqual(y1, y2, delta=1e-4)

    def test_finite_diff_method_3rd_condition(self):
        # 702 fillipov
        # solution for desmos (tex)
        # \left(\frac{x}{2}+1\right)\ln x+\frac{3}{2}+C_{1}\left(x+2\right)+\frac{1}{x}C_{2}
        # C2 = 0, C1 = -1
        def sol(x):
            return (x / 2 + 1) * math.log(x) - x - 1/2

        # equation
        # (x+1)xy'' + (x+2)y' - y = x + 1/x

        # f(1) = - 3 / 2
        # f(e) = -e/2 + 1/2
        # f'(1) = 1/2
        # f'(e) = 1/e

        # boundary problem
        # f(1) - f'(1) = -2
        # f(e) + f'(e) = 1/e - e/2 + 1/2
        x, y = ode.finiteDiffMethod2(1, math.e, 100, lambda x: (x + 2) / (x ** 2 + x), lambda x: -1 / (x ** 2 + x), lambda x: (x + 1 / x) / (x ** 2 + x), [1, 1], [-1, 1], [-2, 1 / math.e - math.e / 2 + 1/2])
        exactSol = [sol(xi) for xi in x]
        for y1, y2 in zip(y, exactSol):
            self.assertAlmostEqual(y1, y2, delta=1e-4)


if __name__ == '__main__':
    unittest.main()
